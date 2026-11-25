"""Train and apply a LightGBM LambdaRank model on Avito feeds using ranker.py features."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import polars as pl

from evaluation import KEYS, attach_scores, ndcg, peak_ram_mib
from ranker import VARIANTS, candidate_batches, feed_batches, read_candidates


def group_sizes(feed_ids):
    # feed_ids from read_candidates are grouped contiguously by feed_id, preserve exact order:
    arr = np.asarray(feed_ids)
    boundaries = np.r_[0, np.flatnonzero(arr[1:] != arr[:-1]) + 1, len(arr)]
    return np.diff(boundaries)


def to_numpy_features(df, features):
    # Ensure category_id is int32/float32 cleanly for LightGBM
    return df.select(features).to_numpy().astype(np.float32)


def train(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    start = perf_counter()
    features = VARIANTS[args.variant]
    cat_idx = [features.index("category_id")] if "category_id" in features else []

    print(f"Preparing features for LightGBM ({args.variant})", flush=True)
    training = read_candidates(args.data / "train", features=features, labeled=True,
                               max_feeds=args.train_feeds, offset=args.train_offset)
    train_feeds = training["feed_id"].n_unique()
    X_train = to_numpy_features(training, features)
    y_train = training["target"].to_numpy().astype(np.int32)
    g_train = group_sizes(training["feed_id"].to_numpy())
    del training

    val = read_candidates(args.data / "val", features=features, labeled=True, max_feeds=args.val_feeds)
    val_feeds_df = pl.concat(list(feed_batches(args.data / "val", ["feed_id", "timestamp"], args.val_feeds)))
    timestamps = val_feeds_df["timestamp"].unique().sort()
    cutoff = args.tune_before if args.tune_before is not None else int(timestamps[len(timestamps) // 2])
    early_ids = val_feeds_df.filter(pl.col("timestamp") < cutoff)["feed_id"]
    late_ids = val_feeds_df.filter(pl.col("timestamp") >= cutoff)["feed_id"]

    tuning = val.filter(pl.col("feed_id").is_in(early_ids))
    X_tune = to_numpy_features(tuning, features)
    y_tune = tuning["target"].to_numpy().astype(np.int32)
    g_tune = group_sizes(tuning["feed_id"].to_numpy())
    # A cutoff at the longest tuning feed covers all candidates.
    full_cutoff = int(g_tune.max())

    dtrain = lgb.Dataset(X_train, label=y_train, group=g_train, feature_name=features,
                         categorical_feature=cat_idx, free_raw_data=True)
    dtune = lgb.Dataset(X_tune, label=y_tune, group=g_tune, reference=dtrain,
                        feature_name=features, categorical_feature=cat_idx, free_raw_data=True)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [full_cutoff],
        "learning_rate": args.lr,
        "num_leaves": args.num_leaves,
        "min_child_samples": 100,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "num_threads": args.threads,
        "seed": 42,
        "verbose": -1,
    }
    print(f"Training LightGBM: {len(y_train):,} candidates ({train_feeds:,} feeds), {args.iterations} rounds", flush=True)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=args.iterations,
        valid_sets=[dtune],
        valid_names=["early_val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    X_val = to_numpy_features(val, features)
    scores = model.predict(X_val, num_iteration=model.best_iteration, num_threads=args.threads).astype(np.float32)
    predictions = val.select(KEYS).with_columns(pl.Series("score", scores, dtype=pl.Float32))

    joined = attach_scores(val.select(*KEYS, "target"), predictions)
    m_all = ndcg(joined)
    m_early = ndcg(joined.filter(pl.col("feed_id").is_in(early_ids)))
    m_late = ndcg(joined.filter(pl.col("feed_id").is_in(late_ids)))

    args.output.mkdir(parents=True)
    model.save_model(str(args.output / "model.txt"))
    predictions.write_parquet(args.output / "val_predictions.parquet")
    report = {
        "model": f"lgbm-{args.variant}",
        "variant": args.variant,
        "best_iteration": model.best_iteration,
        "train_feeds": train_feeds,
        "val_feeds": val["feed_id"].n_unique(),
        "tuning_cutoff_timestamp": cutoff,
        "options": {"train_offset": args.train_offset, "iterations": args.iterations, "learning_rate": args.lr,
                    "num_leaves": args.num_leaves, "threads": args.threads, "early_stop_eval_at": full_cutoff,
                    "lightgbm": lgb.__version__},
        "tuning": m_early,
        "holdout": m_late,
        "val": m_all,
        "seconds": round(perf_counter() - start, 2),
        "peak_ram_mib": peak_ram_mib(),
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Finished lgbm-{args.variant}: holdout_ndcg={m_late['ndcg']:.8f}", flush=True)


def predict(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    model = lgb.Booster(model_file=str(args.model))
    features = model.feature_name()
    template = pl.read_parquet(args.data / "sample_submission.parquet", columns=KEYS)
    chunks = []
    for batch in candidate_batches(args.data / "test", features=features):
        scores = model.predict(to_numpy_features(batch, features), num_threads=args.threads).astype(np.float32)
        chunks.append(batch.select(KEYS).with_columns(pl.Series("score", scores, dtype=pl.Float32)))
    submission = attach_scores(template, pl.concat(chunks))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.write_parquet(args.output)
    print(f"Saved {submission.height:,} scores to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--variant", choices=VARIANTS, default="din-events-v2")
    tr.add_argument("--train-feeds", type=int, default=50_000)
    tr.add_argument("--train-offset", type=int, default=0)
    tr.add_argument("--val-feeds", type=int, default=0)
    tr.add_argument("--iterations", type=int, default=500)
    tr.add_argument("--lr", type=float, default=0.05)
    tr.add_argument("--num-leaves", type=int, default=63)
    tr.add_argument("--tune-before", type=int)
    tr.set_defaults(run=train)
    pr = sub.add_parser("predict")
    pr.add_argument("--model", type=Path, required=True)
    pr.set_defaults(run=predict)
    for cmd in (tr, pr):
        cmd.add_argument("--data", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
