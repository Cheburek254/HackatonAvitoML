"""Train and apply a GPU XGBoost Ranker (rank:ndcg) on Avito feeds using ranker.py features."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
import xgboost as xgb

from evaluation import KEYS, attach_scores, ndcg, peak_ram_mib
from ranker import VARIANTS, candidate_batches, feed_batches, read_candidates


def to_numpy_features(df, features):
    return df.select(features).to_numpy().astype(np.float32)


def make_dmatrix(df, features, labeled=True):
    X = to_numpy_features(df, features)
    qid = df["feed_id"].to_numpy().astype(np.int32)
    y = df["target"].to_numpy().astype(np.float32) if labeled else None
    dm = xgb.QuantileDMatrix(X, label=y, qid=qid, feature_names=features) if labeled else xgb.DMatrix(X, qid=qid, feature_names=features)
    return dm


def train(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    start = perf_counter()
    features = VARIANTS[args.variant]

    cold_drop = args.cold_drop if args.cold_drop is not None else (0.15 if args.variant == "din-events-v3-robust" else 0.0)
    print(f"Preparing features for XGBoost GPU ({args.variant}, cold_drop={cold_drop:.2f})", flush=True)
    training = read_candidates(args.data / "train", features=features, labeled=True,
                               max_feeds=args.train_feeds, offset=args.train_offset,
                               cold_drop_prob=cold_drop).sort("feed_id")
    train_feeds = training["feed_id"].n_unique()
    dtrain = make_dmatrix(training, features, labeled=True)
    n_train = training.height
    del training

    val = read_candidates(args.data / "val", features=features, labeled=True, max_feeds=args.val_feeds).sort("feed_id")
    val_feeds_df = pl.concat(list(feed_batches(args.data / "val", ["feed_id", "timestamp"], args.val_feeds)))
    timestamps = val_feeds_df["timestamp"].unique().sort()
    cutoff = args.tune_before if args.tune_before is not None else int(timestamps[len(timestamps) // 2])
    early_ids = val_feeds_df.filter(pl.col("timestamp") < cutoff)["feed_id"]
    late_ids = val_feeds_df.filter(pl.col("timestamp") >= cutoff)["feed_id"]

    tuning = val.filter(pl.col("feed_id").is_in(early_ids)).sort("feed_id")
    dtune = xgb.QuantileDMatrix(
        to_numpy_features(tuning, features),
        label=tuning["target"].to_numpy().astype(np.float32),
        qid=tuning["feed_id"].to_numpy().astype(np.int32),
        ref=dtrain,
        feature_names=features,
    )

    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@30",
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": args.depth,
        "learning_rate": args.lr,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 50,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "seed": 42,
    }
    print(f"Training XGBoost GPU: {n_train:,} candidates ({train_feeds:,} feeds), {args.iterations} rounds", flush=True)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=args.iterations,
        evals=[(dtune, "early_val")],
        early_stopping_rounds=50,
        verbose_eval=50,
    )

    best_iter = model.best_iteration + 1
    dval = xgb.DMatrix(to_numpy_features(val, features), feature_names=features)
    scores = model.predict(dval, iteration_range=(0, best_iter)).astype(np.float32)
    predictions = val.select(KEYS).with_columns(pl.Series("score", scores, dtype=pl.Float32))

    joined = attach_scores(val.select(*KEYS, "target"), predictions)
    m_all = ndcg(joined)
    m_early = ndcg(joined.filter(pl.col("feed_id").is_in(early_ids)))
    m_late = ndcg(joined.filter(pl.col("feed_id").is_in(late_ids)))

    args.output.mkdir(parents=True)
    model.save_model(str(args.output / "model.json"))
    predictions.write_parquet(args.output / "val_predictions.parquet")
    report = {
        "model": f"xgb-{args.variant}",
        "variant": args.variant,
        "best_iteration": best_iter,
        "train_feeds": train_feeds,
        "val_feeds": val["feed_id"].n_unique(),
        "tuning": m_early,
        "holdout": m_late,
        "val": m_all,
        "seconds": round(perf_counter() - start, 2),
        "peak_ram_mib": peak_ram_mib(),
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    try:
        import notify_tg
        notify_tg.send(
            f"Готов XGBoost GPU ({args.variant}, {train_feeds} лент):\n"
            f"• early={m_early['ndcg']:.5f} | holdout={m_late['ndcg']:.5f} | full={m_all['ndcg']:.5f}\n"
            f"• trees={best_iter} | {report['seconds']:.0f}с | RAM={report['peak_ram_mib']/1024:.1f} ГБ"
        )
    except Exception:
        pass


def predict(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    model = xgb.Booster()
    model.load_model(str(args.model))
    model.set_param({"device": "cuda"})
    features = model.feature_names
    template = pl.read_parquet(args.data / "sample_submission.parquet", columns=KEYS)
    chunks = []
    for batch in candidate_batches(args.data / "test", features=features):
        dm = xgb.DMatrix(to_numpy_features(batch, features), feature_names=features)
        scores = model.predict(dm).astype(np.float32)
        chunks.append(batch.select(KEYS).with_columns(pl.Series("score", scores, dtype=pl.Float32)))
    submission = attach_scores(template, pl.concat(chunks))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.write_parquet(args.output)
    print(f"Saved {submission.height:,} scores to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--variant", choices=VARIANTS, default="din-events-v3")
    tr.add_argument("--train-feeds", type=int, default=200_000)
    tr.add_argument("--train-offset", type=int, default=0)
    tr.add_argument("--val-feeds", type=int, default=0)
    tr.add_argument("--iterations", type=int, default=800)
    tr.add_argument("--lr", type=float, default=0.05)
    tr.add_argument("--depth", type=int, default=7)
    tr.add_argument("--cold-drop", type=float, default=None)
    tr.add_argument("--tune-before", type=int)
    tr.set_defaults(run=train)
    pr = sub.add_parser("predict")
    pr.add_argument("--model", type=Path, required=True)
    pr.set_defaults(run=predict)
    for cmd in (tr, pr):
        cmd.add_argument("--data", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
