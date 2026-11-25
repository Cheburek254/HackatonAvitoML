"""Compare fixed CatBoost/GRU and early-tuned LightGBM Z-blends."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
import torch

from accelerator import select_gpu
from ensemble import Slates, add_model_scores, weighted
from experiments import read_frame, validation
from pair_blend import paired_interval, signals


CAT, LGB, GRU = "din-events-v2", "lgbm-din-events-v2", "gru-din-plus"
MEMBERS = (CAT, LGB, GRU)
BASE_WEIGHTS = (0.7, 0.0, 0.3)
BASE_NAME = "cat-gru"
NEW_NAMES = ("cat-lightgbm", "cat-lightgbm-gru")


def grid(include_gru):
    # Twenty equal units give a 5% step and include the existing 70/30 split.
    for cat in range(21):
        for gru in range(21 - cat) if include_gru else (0,):
            yield cat / 20, (20 - cat - gru) / 20, gru / 20


def compare(args, output):
    gpu = select_gpu(args.gpu) if args.gpu is not None else None
    device = "cuda" if gpu else "cpu"
    if not gpu:
        torch.set_num_threads(4)
    labels, cutoff = validation(args.data, 0)
    feeds = read_frame(args.data, "val", ["feed_id", "user_id", "timestamp"], 0).sort("feed_id")
    metadata = json.loads((args.lgbm / "metrics.json").read_text(encoding="utf-8"))
    if (metadata["model"] != LGB or metadata["tuning_cutoff_timestamp"] != cutoff
            or metadata["val_feeds"] != len(feeds)):
        raise ValueError("LightGBM was evaluated on a different split")
    paths = (args.catboost, args.lgbm, args.gru)
    model_reports = [json.loads((path.parent / "report.json").read_text(encoding="utf-8"))
                     for path in (args.catboost, args.gru)]
    if any(not report["complete"] or report["val_feeds"] != len(feeds)
           or report["tuning_cutoff_timestamp"] != cutoff for report in model_reports):
        raise ValueError("CatBoost or GRU was evaluated on a different split")
    train = dict(zip(MEMBERS, [model_reports[0]["train_feeds"], metadata["train_feeds"], model_reports[1]["train_feeds"]],
                     strict=True))
    frame = signals(add_model_scores(labels.select("feed_id", "item_id", "timestamp", "target"),
                                     [pl.read_parquet(path / "val_predictions.parquet") for path in paths],
                                     MEMBERS), MEMBERS, "z-blend")
    early = Slates(frame.filter(pl.col("timestamp") < cutoff), device)
    full = Slates(frame, device)
    if not np.array_equal(full.frame["feed_id"].to_numpy()[full.rows[:, 0]], feeds["feed_id"].to_numpy()):
        raise ValueError("Feed metadata and candidate keys differ")
    tuning = feeds["timestamp"].to_numpy() < cutoff
    users = feeds["user_id"].to_numpy()[~tuning]
    cols = [f"z-blend_{name}" for name in MEMBERS]
    early_values = [early.pack(col) for col in cols]
    full_values = [full.pack(col) for col in cols]
    weights = {BASE_NAME: BASE_WEIGHTS}
    trials = {}
    for name, include_gru in zip(NEW_NAMES, (False, True), strict=True):
        print(f"Starting {name}", flush=True)
        trials[name] = [{"weights": candidate, "early_ndcg": early.metric(weighted(candidate, early_values))}
                        for candidate in grid(include_gru)]
        # Prefer the smaller LightGBM contribution if tuning NDCG ties.
        weights[name] = max(trials[name], key=lambda row: (row["early_ndcg"], -row["weights"][1]))["weights"]
    results, points = [], {}
    for name, candidate in weights.items():
        values = weighted(candidate, full_values)
        points[name] = scores = full.per_feed_ndcg(values).cpu().numpy().astype(np.float64)
        row = {"model": name, "weights": dict(zip(MEMBERS, candidate, strict=True)),
               "tuning": {"ndcg": float(scores[tuning].mean())}, "holdout": {"ndcg": float(scores[~tuning].mean())},
               "val": {"ndcg": float(scores.mean())}}
        results.append(row)
        full.predictions(values).write_parquet(output / f"{name}_val_predictions.parquet")
        print(f"Finished {name}: holdout_ndcg={row['holdout']['ndcg']:.8f}", flush=True)
    tests = [(name, BASE_NAME) for name in NEW_NAMES] + [(NEW_NAMES[1], NEW_NAMES[0])]
    report = {"complete": True, "gpu": vars(gpu) if gpu else None,
              "models": sorted(results, key=lambda row: -row["holdout"]["ndcg"]),
              "val_feeds": len(feeds), "val_candidates": frame.height, "val_feed_limit": 0,
              "tuning_cutoff_timestamp": cutoff, "early_feeds": int(tuning.sum()), "holdout_feeds": int((~tuning).sum()),
              "train_feeds_by_model": train, "lightgbm": metadata, "early_grid": trials,
              "holdout_paired_bootstrap": {
                  f"{a} minus {b}": paired_interval((points[a] - points[b])[~tuning], users) for a, b in tests}}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("data", "catboost", "lgbm", "gru", "output"):
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
