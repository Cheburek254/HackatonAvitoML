"""Test a single candidate self-attention layer on the DCN-V2 control."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import sleep

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import Subset

from dcn_compare import BATCH_SIZE, EPOCHS, ROUND, Ranker, attach_model, prepare_validation, train_epoch
from evaluation import attach_scores, ndcg, zscore
from local_hypotheses import CONFIRM, EARLY, metrics, report
from training_queue import ROOT, atomic_text, keep_awake, note
from v4_followup import EARLY_SPLIT, LATE_SPLIT, SEED, SIZE, adaptation_data, init_neural, scores_into, write_scores


NAME = "dcn-v2-attention"
REPORT_NAME = f"{NAME}-200k"
BASELINE = .377325410772094
DCN_WEIGHT = .15
SHARES = (0., .25, .5, .75, 1.)


class AttentionRanker(Ranker):
    def __init__(self):
        super().__init__("dcn-v2")
        with torch.random.fork_rng(devices=[]):
            self.attention_norm = nn.LayerNorm(64)
            self.attention = nn.MultiheadAttention(64, 4, batch_first=True)
            nn.init.zeros_(self.attention.out_proj.weight)
            nn.init.zeros_(self.attention.out_proj.bias)

    def forward(self, features, category, user, padding, history, events, history_padding):
        hidden = self.cross_features(features, category, user)
        for index in range(len(self.head)-1):
            hidden = self.head[index](hidden)
        query = self.attention_norm(hidden)
        context, _ = self.attention(query, query, query, key_padding_mask=padding, need_weights=False)
        return self.head[-1](hidden + context).squeeze(-1)


def score_mix(data, share):
    dcn = pl.col("baseline_dcn")
    if share:
        dcn = (1-share)*dcn + share*pl.col("new_z")
    return data.with_columns(((1-DCN_WEIGHT)*zscore("control") + DCN_WEIGHT*dcn).alias("score"))


def train(args):
    neural, _ = init_neural(args)
    frame, _, _, val = prepare_validation(args, neural)
    frame = scores_into(frame, args.runs/"dcn-v2-vs-mlp-200k/dcn-v2-val.parquet", "baseline_dcn")
    late = pl.col("timestamp") >= LATE_SPLIT
    control = score_mix(frame, 0.)
    if abs(ndcg(control.filter(late))["ndcg"] - BASELINE) > 1e-8:
        raise ValueError("Did not reproduce the 0.377325 DCN blend")
    reference = [ndcg(control.filter(mask))["ndcg"] for mask in (EARLY, CONFIRM)]
    early = frame.filter(~late)
    note(f"Verified DCN blend control: {BASELINE:.9f}")
    fit, selection, stats_feeds = adaptation_data(args, neural, args.previous/"train-feeds.parquet")
    early_indices = np.flatnonzero(val.feeds["timestamp"].to_numpy() < LATE_SPLIT).tolist()
    batches = neural.loader(fit, BATCH_SIZE, shuffle=True, prefetch=True)
    early_batches = neural.loader(Subset(val, early_indices), BATCH_SIZE, prefetch=True)
    torch.manual_seed(SEED)
    model = AttentionRanker().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS*len(batches))
    torch.cuda.reset_peak_memory_stats()
    trials, times = [], []
    for epoch in range(1, EPOCHS+1):
        path = args.folder/f"epoch-{epoch}.pt"
        fit.feeds.rng = np.random.default_rng(SEED+epoch)
        if path.is_file():
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state["state_dict"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["schedule"])
            torch.set_rng_state(state["rng_cpu"])
            torch.cuda.set_rng_state(state["rng_cuda"])
            trials, times = state["trials"], state["times"]
            note(f"Reused epoch {epoch}")
            continue
        times.append(train_epoch(neural, {NAME: model}, {NAME: optimizer}, {NAME: scheduler}, batches, epoch, label=NAME))
        scored = attach_model(early, neural.scores(model, early_batches, "cuda"))
        folds = [scored.filter(mask) for mask in (EARLY, CONFIRM)]
        for share in SHARES:
            values = [float(ndcg(score_mix(fold, share))["ndcg"]) for fold in folds]
            gain = min(value-base for value, base in zip(values, reference, strict=True))
            trials.append(dict(epoch=epoch, share=share, early_ndcg=values, min_gain=gain))
        best = max(trials[-len(SHARES):], key=lambda row: row["min_gain"])
        note(f"Epoch {epoch} finished in {times[-1]:.0f}s, best attention share {best['share']:.2f}, "
             f"minimum early gain {best['min_gain']:+.6f}")
        tmp = path.with_suffix(".tmp")
        torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                        schedule=scheduler.state_dict(), rng_cpu=torch.get_rng_state(),
                        rng_cuda=torch.cuda.get_rng_state(), trials=trials, times=times), tmp)
        tmp.replace(path)

    selected = max(trials, key=lambda row: row["min_gain"])
    challenger = max((row for row in trials if row["share"]), key=lambda row: row["min_gain"])
    atomic_text(args.folder/"selection.json", json.dumps(dict(selected=selected, challenger=challenger), indent=2))
    state = torch.load(args.folder/f"epoch-{challenger['epoch']}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["state_dict"])
    predictions = neural.scores(model, neural.loader(val, BATCH_SIZE, prefetch=True), "cuda")
    write_scores(predictions, args.folder/"val_predictions.parquet")
    scored = score_mix(attach_model(frame, predictions), challenger["share"])
    torch.save(dict(state_dict=state["state_dict"], variant=NAME, epoch=challenger["epoch"],
                    share=challenger["share"]), args.folder/"model.pt")

    from ensemble import Slates
    from pair_blend import paired_interval

    users = (pl.scan_parquet(str(args.data/"val/impressions/part-*.parquet"))
             .filter(late).select("feed_id", "user_id").sort("feed_id").collect())
    slates = [Slates(item.filter(late), "cpu") for item in (control, scored)]
    if not np.array_equal(slates[0].frame["feed_id"].unique(maintain_order=True).to_numpy(),
                          users["feed_id"].to_numpy()):
        raise ValueError("User rows do not match the validation feeds")
    plain, attention = [s.per_feed_ndcg(s.pack("score")).numpy().astype(np.float64) for s in slates]
    old_raw = attach_scores(frame, pl.read_parquet(args.runs/"dcn-v2-vs-mlp-200k/dcn-v2-val.parquet"))
    rows = [metrics(control, "fixed-control"), metrics(scored, "dcn-v2-attention-in-blend"),
            metrics(old_raw, "dcn-v2-standalone"), metrics(attach_scores(frame, predictions), f"{NAME}-standalone")]
    report(args, rows, train_feeds=SIZE, epochs=EPOCHS, batch_size=BATCH_SIZE, selected=selected, challenger=challenger,
           early_trials=trials, early_reference=reference, dcn_weight=DCN_WEIGHT,
           absolute_attention_weight=DCN_WEIGHT*challenger["share"], split1_end=EARLY_SPLIT,
           holdout_feeds=users.height, holdout_paired_bootstrap=paired_interval(attention-plain, users["user_id"].to_numpy()),
           parameters=sum(p.numel() for p in model.parameters()), stats_feeds=stats_feeds, epoch_seconds=times,
           train_timestamps=[selection["timestamp"].min(), selection["timestamp"].max()],
           peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20)


def queue(args):
    from team_reports import publish

    log, watch, result, status = (args.folder/name for name in ("train.log", "watch.json", "report.json", "run.status"))
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    status.unlink(missing_ok=True)
    atomic_text(watch, json.dumps(dict(report=str(result), log=str(log), status=str(status), data=str(args.data),
                                       models=[NAME], iterations=1, epochs=EPOCHS,
                                       label="DCN‑V2 · один слой внимания · 200 000 лент")))
    subprocess.Popen([sys.executable, str(ROOT/"dashboard.py"), "--watch", str(watch)], cwd=ROOT,
                     creationflags=flags)
    with log.open("ab") as stream:
        stream.write(f"Starting {NAME}\n".encode())
        stream.flush()
        command = [sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve()), "train",
                   "--data", str(args.data), "--runs", str(args.runs), "--output", str(args.folder),
                   "--gpu", args.gpu]
        with subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, creationflags=flags) as child:
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
    atomic_text(instructions, "# DCN-V2 с self-attention по кандидатам\n\n"
                "Использованы те же 200 000 лент, признаки и параметры обучения, что в тесте DCN-V2 и MLP. "
                "Один остаточный слой внимания с четырьмя головами добавлен перед выходным слоем. "
                "Доля замены DCN и эпоха выбраны по меньшему из приростов на двух ранних отрезках.\n\n"
                "Нужны артефакты dcn-v2-vs-mlp-200k, двух XL, V4 и robust, а также исходная рабочая копия V4 b47fdcb.\n\n"
                "```powershell\n.\\.venv\\Scripts\\python.exe dcn_attention.py train --data C:/study/AvitoData/competition "
                "--runs runs/avito-real --output runs/avito-real/dcn-v2-attention-200k --gpu 0\n```\n\n"
                "```json\n" + result.read_text(encoding="utf-8") + "\n```\n")
    for attempt in range(3):
        try:
            publish(ROUND, REPORT_NAME, result, revision=args.revision,
                    instructions=instructions, checkpoints=["model.pt"])
            return
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            note(f"GitHub publication attempt {attempt+1}: {error}")
            if attempt < 2:
                sleep(20)
    raise RuntimeError(f"Report saved at {result}; GitHub publication needs a retry")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "queue"))
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
    {"train": train, "queue": queue}[args.stage](args)
