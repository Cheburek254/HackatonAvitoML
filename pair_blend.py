"""Compare two completed rankers, fitting RRF and per-feed Z-blend weights on early val."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
import torch

from accelerator import select_gpu
from ensemble import Slates, add_model_scores
from evaluation import zscore
from experiments import read_frame, validation


def paired_interval(delta, users):
    # Resample users with replacement, giving each feed equal weight in the mean.
    # Each anonymous feed forms its own cluster.
    clusters = np.where(users > 0, users, -1 - np.arange(len(users)))
    _, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=delta)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(42)
    means = []
    for _ in range(40):
        sample = rng.integers(len(sums), size=(50, len(sums)))
        means.extend(sums[sample].sum(axis=1) / counts[sample].sum(axis=1))
    return {"delta": float(delta.mean()), "ci95": np.quantile(means, [.025, .975]).tolist(),
            "user_clusters": len(counts), "resamples": len(means)}


def compare(args, output):
    gpu = select_gpu(args.gpu) if args.gpu is not None else None
    device = "cuda" if gpu else "cpu"
    labels, cutoff = validation(args.data, 0)
    paths = [args.left, args.right]
    names = [path.name for path in paths]
    if len(set([*names, "rrf", "z-blend"])) != 4:
        raise ValueError("Choose two distinct base model names")
    feeds = read_frame(args.data, "val", ["feed_id", "user_id", "timestamp"], 0).sort("feed_id")
    budgets = {}
    for name, path in zip(names, paths, strict=True):
        source = json.loads((path.parent / "report.json").read_text(encoding="utf-8"))
        if (not source["complete"] or source["val_feeds"] != len(feeds)
                or source["tuning_cutoff_timestamp"] != cutoff or name not in {r["model"] for r in source["models"]}):
            raise ValueError(f"Source report for {name} does not match the full validation split")
        budgets[name] = source["train_feeds"]
    frame = add_model_scores(labels.select("feed_id", "item_id", "timestamp", "target"),
                             [pl.read_parquet(path / "val_predictions.parquet") for path in paths], names)
    frame = frame.with_columns(
        *[zscore(name).alias("z-blend_" + name) for name in names],
        *[(1 / (60 + pl.col(name).rank("average", descending=True).over("feed_id"))).alias("rrf_" + name)
          for name in names])
    early = Slates(frame.filter(pl.col("timestamp") < cutoff), device)
    full = Slates(frame, device)
    if not np.array_equal(full.frame["feed_id"].to_numpy()[full.rows[:, 0]], feeds["feed_id"].to_numpy()):
        raise ValueError("Feed metadata and candidate labels differ")
    tune = feeds["timestamp"].to_numpy() < cutoff
    rows, per_feed, weights, search = [], {}, {}, {}

    def record(name, values):
        scores = full.per_feed_ndcg(values).cpu().numpy().astype(np.float64)
        per_feed[name] = scores
        rows.append({"model": name, "tuning": {"ndcg": float(scores[tune].mean())},
                     "holdout": {"ndcg": float(scores[~tune].mean())}, "val": {"ndcg": float(scores.mean())}})

    for name in names:
        record(name, full.pack(name))
    for method in ("rrf", "z-blend"):
        print(f"Starting {method}", flush=True)
        columns = [f"{method}_{name}" for name in names]
        left, right = map(early.pack, columns)
        grid = [{"right_weight": w / 10, "early_ndcg": early.metric(((1 - w / 10) * left + w / 10 * right).float())}
                for w in range(11)]
        weight = max(grid, key=lambda row: row["early_ndcg"])["right_weight"]
        search[method], weights[method] = grid, weight
        left, right = map(full.pack, columns)
        values = ((1 - weight) * left + weight * right).float()
        record(method, values)
        full.predictions(values).write_parquet(output / f"{method}_val_predictions.parquet")
        print(f"Finished {method}: right_weight={weight:.1f}, holdout_ndcg={rows[-1]['holdout']['ndcg']:.8f}",
              flush=True)
    selected = max(weights, key=lambda method: max(row["early_ndcg"] for row in search[method]))
    users = feeds["user_id"].to_numpy()[~tune]
    comparisons = {}
    for lhs, rhs in [("rrf", "z-blend"), *[(method, names[0]) for method in weights]]:
        comparisons[f"{lhs} minus {rhs}"] = paired_interval((per_feed[lhs] - per_feed[rhs])[~tune], users)
    report = {"complete": True, "gpu": vars(gpu) if gpu else None, "members": names, "train_feeds_by_model": budgets,
              "val_feeds": len(feeds), "val_feed_limit": 0, "val_candidates": frame.height,
              "tuning_cutoff_timestamp": cutoff, "early_feeds": int(tune.sum()), "holdout_feeds": int((~tune).sum()),
              "right_weights": weights, "early_grid": search, "rrf_offset": 60, "selected_on_early": selected,
              "holdout_paired_bootstrap": comparisons, "models": sorted(rows, key=lambda r: -r["holdout"]["ndcg"])}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("data", "left", "right", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--gpu", help="NVIDIA index or UUID, or omit for CPU")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as staging:
        pending = Path(staging) / "result"
        pending.mkdir()
        compare(args, pending)
        pending.rename(args.output)
