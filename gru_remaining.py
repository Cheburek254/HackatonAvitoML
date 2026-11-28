"""Run the remaining matched GRU v4 comparisons in sequence on one GPU."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter, sleep

import numpy as np
import polars as pl
import torch
from torch.utils.data import ConcatDataset

from evaluation import attach_scores, ndcg, zscore
from local_hypotheses import CONFIRM, EARLY, blend, metrics, report, xl_pair_signals
from training_queue import ROOT, atomic_text, keep_awake, note
from v4_followup import (ADAPT_EPOCHS, EARLY_SPLIT, LATE_SPLIT, SEED, SIZE, VARIANT,
                         adaptation_data, init_neural, materialize_adaptation, scores_into, write_scores)


STAGES = ("epochs-by-period", "mixed-periods", "neural-ndcg")
ROUND = "avito-followups"
BASELINE = .3768931070503751
SINKHORN_STEPS = 10
TEMPERATURE = 1.


def neural_ndcg(scores, target, padding, temperature=TEMPERATURE):
    """Masked deterministic NeuralSort with a fixed number of Sinkhorn updates."""
    valid = ~padding
    size = scores.shape[1]
    ranks = torch.arange(size, device=scores.device)
    length = valid.sum(1, keepdim=True)
    valid_ranks = ranks[None, :] < length
    valid_square = valid_ranks[..., None] & valid[:, None, :]
    scores = scores.masked_fill(padding, 0)
    distances = (scores[:, :, None] - scores[:, None, :]).abs()
    penalties = (distances * valid[:, None, :]).sum(2)
    slope = length - 1 - 2*ranks[None, :]
    logits = (slope[..., None]*scores[:, None, :] - penalties[:, None, :])/temperature
    logits = logits.masked_fill(~valid_square, -1e9)
    soft_ranks = (logits.softmax(dim=2) + 1e-8)*valid_square
    for _ in range(SINKHORN_STEPS):
        columns = soft_ranks.sum(1, keepdim=True).clamp_min(1e-8).masked_fill(~valid[:, None, :], 1)
        soft_ranks = soft_ranks/columns
        rows = soft_ranks.sum(2, keepdim=True).clamp_min(1e-8).masked_fill(~valid_ranks[..., None], 1)
        soft_ranks = soft_ranks/rows
    gains = (2.**target - 1).masked_fill(padding, 0)
    discounts = (ranks + 2).float().log2().reciprocal()
    ideal = (gains.sort(descending=True).values*discounts).sum(dim=1)
    dcg = ((soft_ranks*gains[:, None, :]).sum(dim=2)*discounts).sum(dim=1)
    nonempty = ideal > 0
    return ((1 - dcg/ideal.clamp_min(1e-8))*nonempty).sum()/nonempty.sum().clamp_min(1)


def validate(neural, args):
    return neural.Feeds(args.data/"val", 0, True, 64, True,
                        rich_features=True, v3_features=True, v4_features=True)


def evaluate_epoch_transfer(args, neural):
    from ensemble import Slates
    from pair_blend import paired_interval

    frame = xl_pair_signals(args)
    val = validate(neural, args)
    model = neural.Ranker(VARIANT).cuda()
    sources = ((2, args.prior/"period"), (3, args.runs/"local-evening-200k/third-period"))
    late = pl.col("timestamp") >= LATE_SPLIT
    users = (pl.scan_parquet(str(args.data/"val/impressions/part-*.parquet"))
             .filter(late).select("feed_id", "user_id").sort("feed_id").collect())
    rows, details, intervals = [], {}, {}
    for period, folder in sources:
        baseline = f"period{period}_e6"
        frame = scores_into(frame, folder/"val_predictions.parquet", baseline)
        per_feed = {}
        for epoch in (5, 6):
            name = f"period-{period}-epoch-{epoch}"
            if epoch == 5:
                path = args.folder/f"{name}-val.parquet"
                if not path.is_file():
                    model.load_state_dict(torch.load(folder/"epoch-5.pt", map_location="cpu",
                                                     weights_only=True)["state_dict"])
                    note(f"Scoring {name}")
                    write_scores(neural.scores(model, neural.loader(val, 16, prefetch=True), "cuda"), path)
                frame = scores_into(frame, path, "epoch5_other")
            other = pl.col("epoch5_other" if epoch == 5 else baseline)
            gru = .75*.5*(pl.col("fifth") + other) + .25*zscore("xl_pair")
            scored = blend(frame, gru)
            rows.append(metrics(scored, name))
            details[name] = [ndcg(scored.filter(mask))["ndcg"] for mask in (EARLY, CONFIRM, late)]
            slates = Slates(scored.filter(late), "cpu")
            if not np.array_equal(users["feed_id"].to_numpy(),
                                  slates.frame["feed_id"].unique(maintain_order=True).to_numpy()):
                raise ValueError("User rows do not match the validation feeds")
            per_feed[epoch] = slates.per_feed_ndcg(slates.pack("score")).numpy().astype(np.float64)
        intervals[f"period-{period}"] = paired_interval(per_feed[5]-per_feed[6], users["user_id"].to_numpy())
    if abs(details["period-2-epoch-6"][2] - BASELINE) > 1e-8:
        raise ValueError("Did not reproduce the 0.376893 reference")
    report(args, rows, train_feeds=SIZE, split_scores=details, split_order=["early1", "early2", "late"],
           split1_end=EARLY_SPLIT, holdout_feeds=users.height, epochs=6, qsm_trees=800,
           holdout_epoch5_minus_epoch6=intervals)


def mixed_datasets(args, neural):
    files = sorted((args.data/"train/impressions").glob("part-*.parquet"))
    periods = ((args.previous, 100_000), (args.prior/"period", 60_000),
               (args.runs/"local-evening-200k/third-period", 40_000))
    datasets, samples, details = [], [], []
    for index, (folder, amount) in enumerate(periods):
        original = pl.read_parquet(folder/"train-feeds.parquet")
        if original["feed_id"].n_unique() != SIZE:
            raise ValueError(f"Unexpected original period size in {folder}")
        chosen = original.sample(n=amount, seed=SEED + index).sort(["timestamp", "feed_id"])
        cutoff = original["timestamp"].min()
        fit, count = materialize_adaptation(args, neural, files, chosen, history_cutoff=cutoff)
        datasets.append(fit)
        samples.append(chosen.with_columns(pl.lit(index+1).alias("period")))
        details.append(dict(feeds=amount, history_cutoff=cutoff, past_stats_feeds=count, seed=SEED+index,
                            first_timestamp=chosen["timestamp"].min(), last_timestamp=chosen["timestamp"].max()))
    selection = pl.concat(samples)
    if selection.height != SIZE or selection["feed_id"].n_unique() != SIZE:
        raise ValueError("Mixed periods must contain exactly 200,000 disjoint feeds")
    selection.write_parquet(args.folder/"train-feeds.parquet")
    return datasets, details


def fit_new(args, neural):
    listwise = args.stage == "neural-ndcg"
    if listwise:
        fit, selection, stats_feeds = adaptation_data(args, neural, args.previous/"train-feeds.parquet")
        datasets, details = [fit], [dict(feeds=SIZE, past_stats_feeds=stats_feeds)]
        selection.write_parquet(args.folder/"train-feeds.parquet")
    else:
        datasets, details = mixed_datasets(args, neural)
    fit = datasets[0] if listwise else ConcatDataset(datasets)
    val = validate(neural, args)
    batches = neural.loader(fit, 16, shuffle=True, prefetch=True)
    model = neural.Ranker(VARIANT).cuda()
    original = args.reference/"valishinqa-gru-din-v4-300k/gru-din-v4/model.pt"
    source = torch.load(original, map_location="cpu", weights_only=True)
    model.load_state_dict(source["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ADAPT_EPOCHS*len(batches))
    if listwise:
        # This subprocess replaces only the loss used by the original V4 training loop.
        neural.loss_for = lambda variant, scores, target, padding: neural_ndcg(scores, target, padding)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for epoch in range(1, ADAPT_EPOCHS+1):
        for index, dataset in enumerate(datasets):
            dataset.feeds.rng = np.random.default_rng(SEED+epoch+index*100)
        path = args.folder/f"epoch-{epoch}.pt"
        if path.is_file():
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state["state_dict"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["schedule"])
            torch.set_rng_state(state["rng_cpu"])
            torch.cuda.set_rng_state(state["rng_cuda"])
            times.append(state["seconds"])
            note(f"Reused epoch {epoch}")
        else:
            seconds = neural.train_epoch(model, optimizer, batches, "cuda", VARIANT, epoch, scheduler)
            times.append(seconds)
            tmp = path.with_suffix(".tmp")
            torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                            schedule=scheduler.state_dict(), rng_cpu=torch.get_rng_state(),
                            rng_cuda=torch.cuda.get_rng_state(), seconds=seconds), tmp)
            tmp.replace(path)
    torch.save(dict(state_dict={key: value.cpu() for key, value in model.state_dict().items()},
                    variant=VARIANT, history_window=64, experiment=args.stage), args.folder/"model.pt")
    observe(args, neural, model, val, details, times)


def observe(args, neural, model, validation, details, times):
    from ensemble import Slates
    from pair_blend import paired_interval

    frame = xl_pair_signals(args)
    def mix(data, weight):
        first = (1-weight)*pl.col("fifth") + weight*pl.col("new_gru")
        return blend(data, .75*.5*(first + pl.col("second")) + .25*zscore("xl_pair"))

    from torch.utils.data import Subset

    early_indices = np.flatnonzero(validation.feeds["timestamp"].to_numpy() < LATE_SPLIT).tolist()
    predictions = neural.scores(model, neural.loader(Subset(validation, early_indices), 16, prefetch=True), "cuda")
    early = (attach_scores(frame.filter(pl.col("timestamp") < LATE_SPLIT), predictions)
             .with_columns(zscore("score").alias("new_gru")).drop("score"))
    folds = [early.filter(mask) for mask in (EARLY, CONFIRM)]
    trials = {weight: [ndcg(mix(fold, weight))["ndcg"] for fold in folds] for weight in (0., .25, .5, 1.)}
    gains = {weight: min(x-y for x, y in zip(values, trials[0.], strict=True)) for weight, values in trials.items()}
    selected = max(gains, key=gains.get)
    challenger = max((weight for weight in gains if weight), key=gains.get)
    atomic_text(args.folder/"selection.json", json.dumps(dict(selected=selected, challenger=challenger,
                                                             early_trials=trials), indent=2))
    path = args.folder/"val_predictions.parquet"
    write_scores(neural.scores(model, neural.loader(validation, 16, prefetch=True), "cuda"), path)
    frame = attach_scores(frame, pl.read_parquet(path)).with_columns(zscore("score").alias("new_gru")).rename(
        {"score": "new_raw"})
    frame = attach_scores(frame, pl.read_parquet(args.previous/"val_predictions.parquet")).rename(
        {"score": "plain_raw"})
    old = mix(frame, 0.)
    trial = mix(frame, challenger)
    rows = [metrics(old, "fixed-control"), metrics(trial, "new-gru-in-blend"),
            metrics(frame.with_columns(pl.col("plain_raw").alias("score")), "fresh-pairwise-epoch6"),
            metrics(frame.with_columns(pl.col("new_raw").alias("score")), f"{args.stage}-epoch6")]
    if abs(rows[0]["holdout"]["ndcg"] - BASELINE) > 1e-8:
        raise ValueError("Did not reproduce the 0.376893 reference")
    late = pl.col("timestamp") >= LATE_SPLIT
    users = (pl.scan_parquet(str(args.data/"val/impressions/part-*.parquet"))
             .filter(late).select("feed_id", "user_id").sort("feed_id").collect())
    slates = [Slates(scored.filter(late), "cpu") for scored in (old, trial)]
    if not np.array_equal(users["feed_id"].to_numpy(), slates[0].frame["feed_id"].unique(maintain_order=True).to_numpy()):
        raise ValueError("User rows do not match the validation feeds")
    values = [s.per_feed_ndcg(s.pack("score")).numpy().astype(np.float64) for s in slates]
    settings = dict(temperature=TEMPERATURE, sinkhorn_steps=SINKHORN_STEPS) if args.stage == "neural-ndcg" else {}
    report(args, rows, train_feeds=SIZE, epochs=ADAPT_EPOCHS, epoch_seconds=times, periods=details,
           early_trials=trials, selected_new_gru_share=selected, challenger_new_gru_share=challenger,
           holdout_paired_bootstrap=paired_interval(values[1]-values[0], users["user_id"].to_numpy()),
           holdout_feeds=users.height, peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20, **settings)


def queue(args):
    from team_reports import publish

    wait_for = args.runs/"dcn-v2-vs-mlp-200k/run.status"
    note("Waiting for DCN-V2 and MLP to release the GPU")
    while not wait_for.exists():
        keep_awake()
        sleep(15)
    note(f"DCN/MLP exited with status {wait_for.read_text(encoding='utf-8').strip()}")
    watch = args.output/"watch.json"
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen([sys.executable, str(ROOT/"dashboard.py"), "--watch", str(watch)], cwd=ROOT,
                     creationflags=flags)
    failures = []
    for name in STAGES:
        folder = args.output/name
        folder.mkdir(exist_ok=True)
        result, status, log = (folder/file for file in ("report.json", "run.status", "train.log"))
        if not result.exists():
            status.unlink(missing_ok=True)
            atomic_text(watch, json.dumps(dict(report=str(result), log=str(log), status=str(status),
                                               data=str(args.data), models=[VARIANT], iterations=1,
                                               epochs=ADAPT_EPOCHS, label=name)))
            started = perf_counter()
            note(f"Starting {name}")
            with log.open("ab") as stream, subprocess.Popen(
                [sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve()), name, "--data", str(args.data),
                 "--runs", str(args.runs), "--output", str(args.output), "--gpu", args.gpu],
                stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT, creationflags=flags,
            ) as child:
                while child.poll() is None:
                    keep_awake()
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        pass
                code = child.returncode or int(not result.exists())
            atomic_text(status, str(code))
            note(f"{name} finished in {perf_counter()-started:.0f}s, exit={code}")
            if code:
                failures.append(name)
                continue
        instructions = folder/"REPRODUCE.md"
        atomic_text(instructions, f"# {name}\n\nРабочая копия V4 b47fdcb и артефакты предыдущих прогонов, "
                    "включая XL №002 и №003, используются для контроля 0,376893.\n\n"
                    f"```powershell\n.\\.venv\\Scripts\\python.exe gru_remaining.py {name} --data C:/study/AvitoData/competition "
                    "--runs runs/avito-real --output runs/avito-real/gru-remaining-200k --gpu 0\n```\n\n"
                    "```json\n" + result.read_text(encoding="utf-8") + "\n```\n")
        for attempt in range(2):
            try:
                publish(ROUND, name, result, revision=args.revision, instructions=instructions,
                        checkpoints=["model.pt"] if (folder/"model.pt").is_file() else [])
                break
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                note(f"GitHub publication attempt {attempt+1} for {name}: {error}")
                if attempt == 0:
                    sleep(10)
    if failures:
        raise RuntimeError(f"Failed stages: {failures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(*STAGES, "queue"))
    for key in ("data", "runs", "output"):
        parser.add_argument("--"+key, required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--revision", default="HEAD")
    args = parser.parse_args()
    args.previous = args.runs/"v4-next-200k/adapt"
    args.prior = args.runs/"local-hypotheses-200k"
    args.reference = args.runs/"valishinqa-v4-repro"
    args.v4_code = args.runs/"valishinqa-v4-worktree"
    args.folder = args.output/args.stage
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "queue":
        queue(args)
    else:
        args.folder.mkdir(exist_ok=True)
        neural, _ = init_neural(args)
        (evaluate_epoch_transfer if args.stage == "epochs-by-period" else fit_new)(args, neural)
