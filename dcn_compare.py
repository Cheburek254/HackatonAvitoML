"""Compare low-rank DCN-V2 and a parameter-matched MLP on the same Avito feeds."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter, sleep

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import Subset

from evaluation import attach_scores, ndcg, zscore
from local_hypotheses import CONFIRM, EARLY, blend, metrics, report, xl_pair_signals
from training_queue import ROOT, atomic_text, keep_awake, note
from v4_followup import EARLY_SPLIT, LATE_SPLIT, SEED, SIZE, VARIANT, adaptation_data, init_neural, write_scores


EPOCHS = 5
BATCH_SIZE = 32
WEIGHTS = (0., .05, .10, .15, .20)
NAMES = ("dcn-v2", "mlp")
BASELINE = .3768931070503751
ROUND = "avito-followups"
REPORT_NAME = "dcn-v2-vs-mlp-200k"


class Ranker(nn.Module):
    variant = VARIANT  # Use the same BF16 inference path for both models.

    def __init__(self, name):
        super().__init__()
        self.name = name
        width = 168 + 16 + 128  # V4 numeric features, category embedding, user embedding.
        self.category = nn.Embedding(256, 16, padding_idx=0)
        self.normalize = nn.LayerNorm(width)
        self.compress = nn.ModuleList([nn.Linear(width, 32, bias=False) for _ in range(2)])
        self.expand = nn.ModuleList([nn.Linear(32, width) for _ in range(2)])
        self.head = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Dropout(.10),
                                  nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, features, category, user, padding, history, events, history_padding):
        users = user[:, None, :].expand(-1, features.shape[1], -1)
        initial = self.normalize(torch.cat([features, self.category(category), users], dim=-1))
        hidden = initial
        for down, up in zip(self.compress, self.expand, strict=True):
            reduced = down(hidden)
            update = initial * up(reduced) if self.name == "dcn-v2" else up(torch.nn.functional.gelu(reduced))
            hidden = hidden + update
        return self.head(hidden).squeeze(-1)


def train_epoch(neural, models, optimizers, schedules, batches, epoch):
    started = perf_counter()
    for model in models.values():
        model.train()
    for step, (_, _, inputs, target) in enumerate(batches, 1):
        candidates, categories, users, padding = (tensor.cuda() for tensor in inputs[:4])
        target = target.cuda()
        losses = []
        for name, model in models.items():
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                values = model(candidates, categories, users, padding, None, None, None)
            loss = neural.loss_for(VARIANT, values.float(), target, padding)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            schedules[name].step()
            losses.append(loss.detach())
        if step == 1 or step % 100 == 0 or step == len(batches):
            dcn, mlp = (float(value) for value in losses)
            print(f"dcn-mlp epoch={epoch} batch={step}/{len(batches)} loss={dcn:.6f}  mlp={mlp:.6f}", flush=True)
    return perf_counter() - started


def score_mix(data, weight):
    if weight == 0:
        return data.with_columns(pl.col("control").alias("score"))
    return data.with_columns(((1-weight)*zscore("control") + weight*pl.col("new_z")).alias("score"))


def attach_model(data, predictions):
    return (attach_scores(data, predictions).with_columns(zscore("score").alias("new_z"))
            .rename({"score": "new_raw"}))


def read_epoch(path):
    # Older runs wrote NumPy float64 values into the validation trials.
    from numpy.core.multiarray import scalar

    safe_types = [(scalar, "numpy.core.multiarray.scalar"), np.dtype, type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(safe_types):
        return torch.load(path, map_location="cpu", weights_only=True)


def prepare_validation(args, neural):
    frame = blend(xl_pair_signals(args), pl.col("gru_with_xl")).rename({"score": "control"})
    early = frame.filter(pl.col("timestamp") < LATE_SPLIT)
    n_early = [early.filter(mask)["feed_id"].n_unique() for mask in (EARLY, CONFIRM)]
    if n_early != [3636, 3654]:
        raise ValueError("Unexpected early validation split")
    frame_late = frame.filter(pl.col("timestamp") >= LATE_SPLIT)
    if abs(ndcg(score_mix(frame_late, 0))["ndcg"] - BASELINE) > 1e-8:
        raise ValueError("Did not reproduce the 0.376893 reference")
    reference = [ndcg(score_mix(early.filter(mask), 0))["ndcg"] for mask in (EARLY, CONFIRM)]
    note(f"Verified baseline {BASELINE:.9f} and early split {n_early}")

    val = neural.Feeds(args.data/"val", 0, True, 64, True, rich_features=True,
                       v3_features=True, v4_features=True)
    return frame, early, reference, val


def train(args):
    neural, _ = init_neural(args)
    frame, early, reference, val = prepare_validation(args, neural)
    fit, selection, stats_feeds = adaptation_data(args, neural, args.previous/"train-feeds.parquet")
    early_idx = np.flatnonzero(val.feeds["timestamp"].to_numpy() < LATE_SPLIT).tolist()
    train_batches = neural.loader(fit, BATCH_SIZE, shuffle=True, prefetch=True)
    early_batches = neural.loader(Subset(val, early_idx), BATCH_SIZE, prefetch=True)
    models = {}
    for name in NAMES:
        torch.manual_seed(SEED)
        models[name] = Ranker(name).cuda()
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
                  for name, model in models.items()}
    schedules = {name: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS*len(train_batches))
                 for name, opt in optimizers.items()}
    torch.cuda.reset_peak_memory_stats()
    trials = {name: [] for name in NAMES}
    times = []
    for epoch in range(1, EPOCHS + 1):
        path = args.folder/f"epoch-{epoch}.pt"
        fit.feeds.rng = np.random.default_rng(SEED+epoch)
        if path.exists():
            state = read_epoch(path)
            for name in NAMES:
                models[name].load_state_dict(state["models"][name])
                optimizers[name].load_state_dict(state["optimizers"][name])
                schedules[name].load_state_dict(state["schedules"][name])
            torch.set_rng_state(state["rng_cpu"])
            torch.cuda.set_rng_state(state["rng_cuda"])
            trials = state["trials"]
            times = state["times"]
            note(f"Reused epoch {epoch}")
            continue

        times.append(train_epoch(neural, models, optimizers, schedules, train_batches, epoch))
        for name, model in models.items():
            data = attach_model(early, neural.scores(model, early_batches, "cuda"))
            folds = [data.filter(mask) for mask in (EARLY, CONFIRM)]
            for weight in WEIGHTS:
                values = [float(ndcg(score_mix(fold, weight))["ndcg"]) for fold in folds]
                gain = min(value-base for value, base in zip(values, reference, strict=True))
                trials[name].append(dict(epoch=epoch, weight=weight, early_ndcg=values, min_gain=gain))
            best = max(trials[name][-len(WEIGHTS):], key=lambda row: row["min_gain"])
            note(f"{name} epoch {epoch}, best early weight {best['weight']:.2f}, "
                 f"minimum gain {best['min_gain']:+.6f}")
        print(f"dcn-mlp epoch={epoch} training_seconds={times[-1]:.2f}", flush=True)
        temp = path.with_suffix(".tmp")
        torch.save(dict(models={name: model.state_dict() for name, model in models.items()},
                        optimizers={name: opt.state_dict() for name, opt in optimizers.items()},
                        schedules={name: sched.state_dict() for name, sched in schedules.items()},
                        rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(),
                        trials=trials, times=times,
                        training_peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20), temp)
        temp.replace(path)

    complete_evaluation(args, neural, models, val, frame, reference, trials, times, selection, stats_feeds,
                        torch.cuda.max_memory_allocated()/2**20)


def finalize(args):
    state = read_epoch(args.folder/f"epoch-{EPOCHS}.pt")
    if any(len(state["trials"][name]) != EPOCHS*len(WEIGHTS) for name in NAMES) or len(state["times"]) != EPOCHS:
        raise ValueError("Cannot finalize training without all five completed epochs")
    selection = pl.read_parquet(args.previous/"train-feeds.parquet")
    original = json.loads((args.previous/"report.json").read_text(encoding="utf-8"))
    neural, _ = init_neural(args)
    frame, _, reference, val = prepare_validation(args, neural)
    models = {name: Ranker(name).cuda() for name in NAMES}
    torch.cuda.reset_peak_memory_stats()
    note(f"Recovered {EPOCHS} completed epochs; evaluating selected checkpoints only")
    complete_evaluation(args, neural, models, val, frame, reference, state["trials"], state["times"], selection,
                        original["past_stats_feeds"], state.get("training_peak_cuda_mib"))


def complete_evaluation(args, neural, models, val, frame, reference, trials, times, selection, stats_feeds, train_peak):
    from ensemble import Slates
    from pair_blend import paired_interval

    selected = {name: max(trials[name], key=lambda row: row["min_gain"]) for name in NAMES}
    challenger = {name: max((row for row in trials[name] if row["weight"]), key=lambda row: row["min_gain"])
                  for name in NAMES}
    atomic_text(args.folder/"selection.json", json.dumps(dict(selected=selected, challenger=challenger), indent=2))
    users = (pl.scan_parquet(str(args.data/"val/impressions/part-*.parquet"))
             .filter(pl.col("timestamp") >= LATE_SPLIT).select("feed_id", "user_id").sort("feed_id").collect())
    def per_feed(scored):
        slates = Slates(scored.filter(pl.col("timestamp") >= LATE_SPLIT), "cpu")
        ids = slates.frame["feed_id"].unique(maintain_order=True).to_numpy()
        if not np.array_equal(ids, users["feed_id"].to_numpy()):
            raise ValueError("User rows do not match the validation feeds")
        return slates.per_feed_ndcg(slates.pack("score")).numpy().astype(np.float64)

    baseline = score_mix(frame, 0)
    base_values = per_feed(baseline)
    rows, intervals, candidates = [metrics(baseline, "fixed-control")], {}, {}
    for name, choice in challenger.items():
        model = models[name]
        state = read_epoch(args.folder/f"epoch-{choice['epoch']}.pt")
        model.load_state_dict(state["models"][name])
        path = args.folder/f"{name}-val.parquet"
        write_scores(neural.scores(model, neural.loader(val, BATCH_SIZE, prefetch=True), "cuda"), path)
        data = attach_model(frame, pl.read_parquet(path))
        scored = score_mix(data, choice["weight"])
        candidates[name] = per_feed(scored)
        intervals[name] = paired_interval(candidates[name]-base_values, users["user_id"].to_numpy())
        rows.extend([metrics(scored, f"{name}-in-blend"),
                     metrics(data.with_columns(pl.col("new_raw").alias("score")), f"{name}-standalone")])
        torch.save(dict(name=name, state_dict=state["models"][name], epoch=choice["epoch"],
                        weight=choice["weight"]), args.folder/f"{name}.pt")
    intervals["dcn-v2-minus-mlp"] = paired_interval(candidates["dcn-v2"]-candidates["mlp"], users["user_id"].to_numpy())
    report(args, rows, train_feeds=SIZE, epochs=EPOCHS, batch_size=BATCH_SIZE, early_reference=reference,
           holdout_feeds=users.height, early_trials=trials, selected=selected, challenger=challenger,
           early_split=EARLY_SPLIT, holdout_paired_bootstrap=intervals, epoch_seconds=times,
           peak_cuda_mib=train_peak, observed_cuda_mib=torch.cuda.max_memory_allocated()/2**20, stats_feeds=stats_feeds,
           train_timestamps=[selection["timestamp"].min(), selection["timestamp"].max()],
           parameters={name: sum(p.numel() for p in model.parameters()) for name, model in models.items()})


def queue(args):
    from team_reports import sync

    recover = args.stage == "recover-queue"
    if recover:
        wait_for = args.runs/"gru-remaining-200k/neural-ndcg/run.status"
        note("Waiting for the existing GRU queue to release the GPU")
        while not wait_for.exists():
            sleep(15)

    log, watch, result, status = (args.folder/name for name in ("train.log", "watch.json", "report.json", "run.status"))
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    status.unlink(missing_ok=True)
    atomic_text(watch, json.dumps(dict(report=str(result), log=str(log), status=str(status), data=str(args.data),
                                       models=["dcn-mlp"], iterations=1, epochs=EPOCHS,
                                       label="DCN-V2 и MLP · проверка чекпоинтов" if recover
                                       else "DCN-V2 и MLP · одинаковые 200 000 лент")))
    if not recover:
        subprocess.Popen([sys.executable, str(ROOT/"dashboard.py"), "--watch", str(watch)], cwd=ROOT,
                         creationflags=flags)
    with log.open("ab") as stream:
        stream.write(b"Starting dcn-mlp\n")
        stream.flush()
        action = "finalize" if recover else "train"
        with subprocess.Popen([sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve()), action,
                               "--data", str(args.data), "--runs", str(args.runs), "--output", str(args.folder),
                               "--gpu", args.gpu], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                              creationflags=flags) as child:
            while child.poll() is None:
                keep_awake()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
            code = child.returncode or int(not result.exists())
    atomic_text(status, str(code))
    if code:
        raise RuntimeError(f"Training exited with status {code}")
    instructions = args.folder/"REPRODUCE.md"
    atomic_text(instructions, "# DCN-V2 и MLP на 200 000 лент\n\n"
                "Нужны Avito train и val, исходные прогнозы V4 и robust, сохранённый список train-feeds.parquet "
                "из v4-next-200k/adapt и расшифрованные XL №002 и №003 в runs/gru-din-xl-team-exchange/parts. "
                "Рабочая копия valishinqa-v4-worktree использует b47fdcb и исходные глобальные статистики.\n\n"
                "Обе модели получают одинаковые признаки, батчи и попарный лосс с весом ΔNDCG. Два низкоранговых "
                "слоя DCN используют поэлементное пересечение с исходным входом. Два слоя MLP используют GELU. "
                "Подбор эпохи и веса максимизирует меньший из двух ранних приростов к контролю. "
                "Вес указывает долю новой модели в финальном Z-бленде.\n\n"
                "```powershell\n.\\.venv\\Scripts\\python.exe dcn_compare.py train --data C:/study/AvitoData/competition "
                "--runs runs/avito-real --output runs/avito-real/dcn-v2-vs-mlp-200k --gpu 0\n```\n\n"
                "```json\n" + result.read_text(encoding="utf-8") + "\n```\n")
    sync(ROUND, REPORT_NAME, result, revision=args.revision, instructions=instructions,
         checkpoints=[f"{name}.pt" for name in NAMES])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "finalize", "queue", "recover-queue"))
    for key in ("data", "runs", "output"):
        parser.add_argument("--"+key, required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--revision", default="HEAD")
    args = parser.parse_args()
    args.previous = args.runs/"v4-next-200k/adapt"
    args.prior = args.runs/"local-hypotheses-200k"
    args.reference = args.runs/"valishinqa-v4-repro"
    args.v4_code = args.runs/"valishinqa-v4-worktree"
    args.folder = args.output
    args.folder.mkdir(parents=True, exist_ok=True)
    {"train": train, "finalize": finalize, "queue": queue, "recover-queue": queue}[args.stage](args)
