"""Run the 200k-feed GRU adaptation and the matched history-cleaning control."""

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import polars as pl

from evaluation import KEYS, attach_scores, ndcg, zscore
from training_queue import ROOT, atomic_text, keep_awake, note


SIZE = 200_000
BASE_SIZE = 300_000
EARLY_SPLIT = 1790073856
LATE_SPLIT = 1790107648
VARIANT = "gru-din-v4"
SEED = 42
ADAPT_EPOCHS = 6


def scores_into(frame, path, name):
    return attach_scores(frame, pl.read_parquet(path)).with_columns(zscore("score").alias(name)).drop("score")


def labels_and_blend(data, reference):
    frame = (pl.scan_parquet(str(data / "val/impressions/part-*.parquet"))
             .select(*KEYS, "timestamp", "target", pl.col("history__item_id").list.len().alias("history_length"))
             .explode("item_id", "target").collect())
    sources = dict(qsm=reference / "qsm_val_predictions.parquet",
                   gru4=reference / "gru4-published_val_predictions.parquet",
                   gru3=reference / "retrain/gru-din-v3/val_predictions.parquet",
                   lgbm=reference / "retrain/lightgbm-v3/val_predictions.parquet")
    for name, path in sources.items():
        frame = scores_into(frame, path, name)
    return frame.with_columns((.40 * pl.col("qsm") + .30 * pl.col("gru4") +
                              .15 * pl.col("gru3") + .15 * pl.col("lgbm")).alias("base"))


def select_feeds(data, before=None):
    files = sorted((data / "train/impressions").glob("part-*.parquet"))
    keys = scan_train(files, ["feed_id", "timestamp"]).collect()
    if keys["feed_id"].n_unique() != len(keys):
        raise ValueError("Duplicate train feed IDs")
    eligible = keys.slice(BASE_SIZE)
    if before is not None:
        eligible = eligible.filter(pl.col("timestamp") < before)
    candidates = (eligible.sort(["timestamp", "feed_id"], descending=True).head(SIZE)
                  .sort(["timestamp", "feed_id"]))
    if len(candidates) != SIZE or candidates["feed_id"].is_in(keys["feed_id"].head(BASE_SIZE)).any():
        raise ValueError("Adaptation selection overlaps the pretrained rows")
    first_val = pl.scan_parquet(str(data / "val/impressions/part-*.parquet")).select(pl.col("timestamp").min()).collect().item()
    if candidates["timestamp"].max() >= first_val:
        raise ValueError("Adaptation feeds must end before val")
    return files, candidates


def scan_train(files, columns):
    return pl.concat([pl.scan_parquet(path).select(columns) for path in files])


def historic_stats(files, candidates, cutoff):
    ids = candidates["item_id"].explode().drop_nulls().unique()
    columns = ["timestamp", "item_id", "target", "pos", "history__item_id"]
    counts, histories = [], []
    total_clicks = total_shows = feeds = 0
    for i in range(0, len(files), 16):
        past = (scan_train(files[i:i + 16], columns)
                .filter(pl.col("timestamp") < cutoff).collect())
        feeds += len(past)
        shown = past.select("item_id", "target", "pos").explode("item_id", "target", "pos")
        total_clicks += int(shown["target"].sum() or 0)
        total_shows += len(shown)
        counts.append(shown.filter((pl.col("item_id") > 0) & pl.col("item_id").is_in(ids))
                      .group_by("item_id").agg(
                          pl.len().cast(pl.Int64).alias("shows"), pl.col("target").sum().cast(pl.Int64).alias("clicks"),
                          pl.col("pos").cast(pl.Int64).sum().alias("positions")))
        histories.append(past.select(pl.col("history__item_id").list.head(64).alias("item_id")).explode("item_id")
                         .filter((pl.col("item_id") > 0) & pl.col("item_id").is_in(ids))
                         .group_by("item_id").agg(pl.len().cast(pl.Int64).alias("history")))
        if len(counts) == 5:
            counts = [pl.concat(counts).group_by("item_id").agg(pl.all().exclude("item_id").sum())]
            histories = [pl.concat(histories).group_by("item_id").agg(pl.col("history").sum())]
        if i % 64 == 0:
            note(f"Temporal statistics {min(i + 16, len(files))}/{len(files)} train files")
    counts = pl.concat(counts).group_by("item_id").agg(pl.all().exclude("item_id").sum())
    hist = pl.concat(histories).group_by("item_id").agg(pl.col("history").sum())
    stats = counts.join(hist, on="item_id", how="full", coalesce=True).fill_null(0).sort("item_id")
    if not total_shows or stats.is_empty():
        raise ValueError("No earlier item statistics")
    return (*[stats[col].to_numpy() for col in ("item_id", "shows", "clicks", "positions", "history")],
            total_clicks / total_shows), feeds


class ShuffledRows:
    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        self.height = rows.height
        self.rng = np.random.default_rng(SEED)

    def row(self, index, *, named):
        feed = self.rows.row(index, named=named)
        indices = self.rng.permutation(len(feed["item_id"]))
        for column in self.columns:
            values = feed[column]
            feed[column] = [values[i] for i in indices]
        return feed


def init_neural(args):
    from accelerator import select_gpu

    gpu = select_gpu(args.gpu)
    note(f"RTX: {gpu.name}")
    # Use the original V4 implementation and its saved statistics for validation.
    sys.path.insert(0, str(args.v4_code))
    import neural
    import torch

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(SEED)
    return neural, torch


def write_scores(predictions, path):
    stage = path.with_suffix(".tmp")
    predictions.write_parquet(stage)
    stage.replace(path)


def checkpoint_model(checkpoint, model):
    return {**checkpoint, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}


def adaptation_data(args, neural, expected_selection=None, before=None, *, window=64, v4_features=True):
    files, selection = select_feeds(args.data, before)
    if expected_selection is not None and not selection.equals(pl.read_parquet(expected_selection)):
        raise ValueError("The selected training feeds differ from the original adaptation")
    fit, stats_feeds = materialize_adaptation(args, neural, files, selection, window=window,
                                             v4_features=v4_features)
    return fit, selection, stats_feeds


def materialize_adaptation(args, neural, files, selection, *, window=64, v4_features=True, history_cutoff=None):
    first, last = selection["timestamp"].min(), selection["timestamp"].max()
    fit = neural.Feeds(args.data / "train", 1, True, window, True, rich_features=True, v3_features=True)
    columns = fit.feeds.columns
    rows = (scan_train(files, columns).filter(pl.col("timestamp").is_between(first, last))
            .join(selection.lazy().select("feed_id"), on="feed_id", how="semi").collect())
    if len(rows) != selection.height or rows["feed_id"].n_unique() != selection.height:
        raise ValueError("Wrong number of adaptation feeds")
    cutoff = first if history_cutoff is None else history_cutoff
    if cutoff > first:
        raise ValueError("History statistics must precede the first training feed")
    stats, stats_feeds = historic_stats(files, rows, cutoff) if v4_features else (None, 0)
    fit.feeds = ShuffledRows(rows, [*neural.CANDIDATE_COLUMNS, "target"])
    fit.v4_features = v4_features
    fit.global_stats = stats
    fit.is_train = False  # Earlier statistics already exclude every adaptation label.
    note(f"Prepared {len(fit):,} adaptation feeds, timestamps {first}–{last}; stats from {stats_feeds:,} earlier feeds")
    return fit, stats_feeds


def adapt(args):
    neural, torch = init_neural(args)
    from experiments import score_summary

    fit, selection, stats_feeds = adaptation_data(args, neural)
    first, last = selection["timestamp"].min(), selection["timestamp"].max()

    from torch.utils.data import Subset

    validation = neural.Feeds(args.data / "val", 0, True, 64, True,
                              rich_features=True, v3_features=True, v4_features=True)
    early_indices = np.flatnonzero(validation.feeds["timestamp"].to_numpy() < EARLY_SPLIT).tolist()
    if len(early_indices) != 3636 or len(validation) != 14423:
        raise ValueError("Validation does not match the published split")
    train_batches = neural.loader(fit, 16, shuffle=True, prefetch=True)
    early_batches = neural.loader(Subset(validation, early_indices), 16, prefetch=True)
    frame = labels_and_blend(args.data, args.reference)
    early = frame.filter(pl.col("timestamp") < EARLY_SPLIT)
    path = args.reference / "valishinqa-gru-din-v4-300k/gru-din-v4/model.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint["variant"], checkpoint["history_window"]) != (VARIANT, 64):
        raise ValueError("Unexpected pretrained GRU checkpoint")
    model = neural.Ranker(VARIANT).to("cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ADAPT_EPOCHS * len(train_batches))
    selection.write_parquet(args.output / "train-feeds.parquet")
    best = (ndcg(early.with_columns(pl.col("base").alias("score")))["ndcg"], 0, 0.0)
    curve = []
    for epoch in range(1, ADAPT_EPOCHS + 1):
        fit.feeds.rng = np.random.default_rng(SEED + epoch)
        saved = args.output / f"epoch-{epoch}.pt"
        predictions = args.output / f"epoch-{epoch}-early.parquet"
        if saved.exists() and predictions.exists():
            resume = torch.load(saved, map_location="cpu", weights_only=True)
            model.load_state_dict(resume["state_dict"])
            optimizer.load_state_dict(resume["optimizer"])
            schedule.load_state_dict(resume["schedule"])
            torch.set_rng_state(resume["rng_cpu"])
            torch.cuda.set_rng_state(resume["rng_cuda"])
            seconds = resume["seconds"]
            note(f"Reused epoch {epoch}")
        else:
            seconds = neural.train_epoch(model, optimizer, train_batches, "cuda", VARIANT, epoch, schedule)
            stage = saved.with_suffix(".tmp")
            write_scores(neural.scores(model, early_batches, "cuda"), predictions)
            torch.save({**checkpoint_model(checkpoint, model), "optimizer": optimizer.state_dict(),
                        "schedule": schedule.state_dict(), "rng_cpu": torch.get_rng_state(),
                        "rng_cuda": torch.cuda.get_rng_state(), "seconds": seconds}, stage)
            stage.replace(saved)
        trial = scores_into(early, predictions, "adapted")
        for alpha in (.5, 1.0):
            expr = pl.col("base") + (.30 * alpha) * (pl.col("adapted") - pl.col("gru4"))
            metric = ndcg(trial.with_columns(expr.cast(pl.Float32).alias("score")))["ndcg"]
            curve.append(dict(epoch=epoch, alpha=alpha, ndcg=metric, seconds=seconds))
            if metric > best[0]:
                best = (metric, epoch, alpha)
        print(f"{VARIANT} epoch={epoch} tuning_ndcg={max(row['ndcg'] for row in curve if row['epoch']==epoch):.6f}", flush=True)
    _, epoch, alpha = best
    rows = [dict(model="v4-base", **score_summary(frame.with_columns(pl.col("base").alias("score")), LATE_SPLIT))]
    late_rows = frame.filter(pl.col("timestamp") >= LATE_SPLIT)
    confirmation = frame.filter(pl.col("timestamp").is_between(EARLY_SPLIT, LATE_SPLIT, closed="left"))
    confirm_baseline = ndcg(confirmation.with_columns(pl.col("base").alias("score")))["ndcg"]
    if epoch:
        winner = torch.load(args.output / f"epoch-{epoch}.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(winner["state_dict"])
        torch.save(checkpoint_model(checkpoint, model), args.output / "model.pt")
        predicted = neural.scores(model, neural.loader(validation, 16, prefetch=True), "cuda")
        write_scores(predicted, args.output / "val_predictions.parquet")
        scored = scores_into(frame, args.output / "val_predictions.parquet", "adapted")
        expr = pl.col("base") + (.30 * alpha) * (pl.col("adapted") - pl.col("gru4"))
        scored = scored.with_columns(expr.cast(pl.Float32).alias("score"))
        rows.append(dict(model="v4-adapted", **score_summary(scored, LATE_SPLIT)))
        confirmation = scored.filter(pl.col("timestamp").is_between(EARLY_SPLIT, LATE_SPLIT, closed="left"))
        confirm_score = ndcg(confirmation)["ndcg"]
    else:
        confirm_score = confirm_baseline
    report = dict(complete=True, train_feeds=SIZE, val_feeds=len(validation),
                  tuning_cutoff_timestamp=LATE_SPLIT, early_selection_cutoff=EARLY_SPLIT, best_epoch=epoch,
                  adapted_fraction=alpha, curve=curve, fit_timestamps=[first, last], past_stats_feeds=stats_feeds,
                  confirmation_ndcg=confirm_score, confirmation_baseline=confirm_baseline,
                  previous_fine_tune_late=.37538676302808566, late_feeds=late_rows["feed_id"].n_unique(), models=rows)
    atomic_text(args.output / "report.json", json.dumps(report, indent=2))


def control(args):
    neural, _ = init_neural(args)
    from experiments import score_summary

    fit = args.output / "fit"
    if not (fit / "metrics.json").is_file():
        neural.train(SimpleNamespace(variant=VARIANT, device="cuda", data=args.data, output=fit, train_feeds=SIZE,
                                     val_feeds=0, train_offset=0, history_window=64, batch_size=16, prefetch=True,
                                     tune_before=LATE_SPLIT, epochs=5))
    gc.collect()
    frame = labels_and_blend(args.data, args.reference)
    for name, folder in (("control", fit), ("end", args.pilots / "end4rec")):
        frame = scores_into(frame, folder / "val_predictions.parquet", name)
    rows = []
    for name, expr in (("gru-v4-200k", pl.col("control")), ("gru-v4-end4rec-200k", pl.col("end")),
                       ("v4-with-gru-v4-200k", pl.col("base") + .30 * (pl.col("control") - pl.col("gru4"))),
                       ("v4-with-end4rec-200k", pl.col("base") + .30 * (pl.col("end") - pl.col("gru4")))):
        rows.append(dict(model=name, **score_summary(frame.with_columns(expr.cast(pl.Float32).alias("score")), LATE_SPLIT)))
    atomic_text(args.output / "report.json", json.dumps(dict(complete=True, train_feeds=SIZE,
                val_feeds=frame["feed_id"].n_unique(), tuning_cutoff_timestamp=LATE_SPLIT, models=rows), indent=2))


def queue(args):
    from team_reports import sync

    args.output.mkdir(parents=True, exist_ok=True)
    watch = args.output / "watch.json"
    subprocess.Popen([sys.executable, "-u", str(ROOT / "dashboard.py"), "--watch", str(watch)], cwd=ROOT,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    forwarded = [piece for key in ("data", "v4_code", "reference", "pilots", "gpu")
                 for piece in (f"--{key.replace('_', '-')}", str(getattr(args, key)))]
    failures = []
    for stage, epochs, label in (("adapt", ADAPT_EPOCHS, "GRU v4 · дообучение 200 000"),
                                 ("control", 5, "GRU v4 · контроль END4Rec 200 000")):
        folder = args.output / stage
        folder.mkdir(exist_ok=True)
        report, log, status = folder / "report.json", folder / "run.log", folder / "run.status"
        if not report.is_file():
            status.unlink(missing_ok=True)
            atomic_text(watch, json.dumps(dict(report=str(report), log=str(log), status=str(status), data=str(args.data),
                                              models=[VARIANT], label=label, iterations=0, epochs=epochs)))
            note(f"Starting {stage}")
            with log.open("ab") as handle:
                command = [sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve()), stage, *forwarded,
                           "--output", str(folder)]
                start = perf_counter()
                with subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT) as child:
                    while child.poll() is None:
                        keep_awake()
                        try:
                            child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            pass
                code = child.returncode or (0 if report.is_file() else 1)
                note(f"Finished {stage} after {perf_counter() - start:.1f}s with exit {code}")
                atomic_text(status, str(code))
                if code:
                    failures.append(stage)
                    continue
        sync("avito-v4-next-200k", stage, report, revision=args.revision)
    if failures:
        raise RuntimeError(f"Failed stages: {', '.join(failures)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("adapt", "control", "queue"))
    for name in ("data", "output", "v4-code", "reference", "pilots"):
        parser.add_argument(f"--{name}", type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--revision", default="HEAD")
    args = parser.parse_args()
    {"adapt": adapt, "control": control, "queue": queue}[args.stage](args)


if __name__ == "__main__":
    main()
