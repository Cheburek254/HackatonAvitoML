"""Export the highest late-val NDCG model or blend from the local CatBoost/GRU experiment."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from catboost import CatBoostRanker
import numpy as np
import polars as pl
import torch

from accelerator import select_gpu
from ensemble import add_model_scores
from evaluation import KEYS, attach_scores, zscore
from experiments import read_frame
from pair_blend import combine, signals
from ranker import VARIANTS, candidate_batches, make_pool, raw_scores
from sharded_runs import part_dir, test_predictions


CAT, GRU = "din-events-v2", "gru-din-plus"


def tree_predictions(split, model_paths, threads, device, limit=0):
    models = [CatBoostRanker().load_model(str(path)) for path in model_paths]
    features = VARIANTS[CAT]
    if not models or any(model.feature_names_ != features for model in models):
        raise ValueError("Expected din-events-v2 checkpoints with the same feature order")
    columns = [f"part_{i}" for i in range(len(models))]
    chunks = []
    for index, batch in enumerate(candidate_batches(split, features=features, max_feeds=limit), 1):
        # Every model sees exactly the same deterministic features for these complete feeds.
        pool = make_pool(batch, features, threads)
        raw = batch.select(KEYS).with_columns(
            pl.Series(name, raw_scores(model, pool, threads, device), dtype=pl.Float32)
            for name, model in zip(columns, models, strict=True))
        score = sum(zscore(name).cast(pl.Float64) for name in columns) / len(columns)
        chunks.append(raw.select(*KEYS, score.cast(pl.Float32).alias("score")))
        if index % 10 == 0:
            print(f"CatBoost: scored {index} batches with {len(models)} checkpoints", flush=True)
    return pl.concat(chunks)


def fused_scores(template, left, right, report, method, device):
    names = report["members"]
    frame = signals(add_model_scores(template.select(KEYS), [left, right], names), names, method, report["rrf_offset"])
    values = [torch.as_tensor(frame[f"{method}_{name}"].to_numpy().copy(), device=device) for name in names]
    scores = combine(*values, report["right_weights"][method])
    return frame.select(KEYS).with_columns(pl.Series("score", scores.cpu().numpy(), dtype=pl.Float32))


def export(args, directory):
    select_gpu(args.gpu)
    comparison_dir = args.queue / "comparison"
    report = json.loads((comparison_dir / "report.json").read_text(encoding="utf-8"))
    if not report["complete"] or report["members"] != [CAT, GRU]:
        raise ValueError("Expected a completed CatBoost / GRU comparison")
    # Prefer the single model in an exact tie.
    winner = max(report["models"], key=lambda row: (row["holdout"]["ndcg"], row["model"] in (CAT, GRU)))
    method = winner["model"]
    weight = report["right_weights"].get(method)
    if method not in (CAT, GRU, "rrf", "z-blend"):
        raise ValueError(f"Unexpected winner: {method}")
    cat_root, gru_root = args.catboost / "merged", args.queue / "gru-fit"
    sources = {}
    for name, root in ((CAT, cat_root), (GRU, gru_root)):
        source = json.loads((root / "report.json").read_text(encoding="utf-8"))
        base = next(r for r in source["models"] if r["model"] == name)
        compared = next(r for r in report["models"] if r["model"] == name)
        if (not source["complete"] or source["train_feeds"] != report["train_feeds_by_model"][name]
                or source["val_feeds"] != report["val_feeds"] or source["tuning_cutoff_timestamp"] != report["tuning_cutoff_timestamp"]
                or abs(base["holdout"]["ndcg"] - compared["holdout"]["ndcg"]) > 1e-6):
            raise ValueError(f"Source for {name} differs from the model in the comparison")
        sources[name] = source
    print(f"Winner: {method}, late NDCG={winner['holdout']['ndcg']:.8f}, GRU weight={weight}", flush=True)
    if weight is not None:
        a, b = [pl.read_parquet(root / name / "val_predictions.parquet") for root, name in ((cat_root, CAT), (gru_root, GRU))]
        saved = pl.read_parquet(comparison_dir / f"{method}_val_predictions.parquet")
        keys = read_frame(args.data, "val", KEYS, 0).explode("item_id")
        check = attach_scores(saved.rename({"score": "saved"}), fused_scores(keys, a, b, report, method, "cuda"))
        if not np.allclose(check["saved"].to_numpy(), check["score"].to_numpy(), rtol=1e-5, atol=2e-6):
            raise ValueError("The submission blend does not reproduce the saved validation scores")
        del a, b, check, saved
    sample = pl.read_parquet(args.data / "sample_submission.parquet")
    if method != GRU:
        tree_report = sources[CAT]
        if tree_report["method"] != "mean-feed-zscore":
            raise ValueError("Unrecognized CatBoost shard aggregation")
        paths = [part_dir(args.catboost, part) / "fit" / CAT / "model.cbm" for part in tree_report["parts"]]
        cat = tree_predictions(args.data / "test", paths, args.threads, "gpu")
        result = cat
    if method != CAT:
        test_predictions(args, gru_root, [GRU])
        gru = pl.read_parquet(gru_root / GRU / "test_predictions.parquet")
        result = gru
    if weight is not None:
        result = fused_scores(sample, cat, gru, report, method, "cuda")
    submission = attach_scores(sample.select(KEYS), result).select(sample.columns).cast(sample.schema)
    if not submission.select(KEYS).equals(sample.select(KEYS)) or not submission["score"].is_finite().all():
        raise ValueError("Submission rows or scores differ from the sample schema")
    path = directory / "submission.parquet"
    submission.write_parquet(path)
    saved = pl.read_parquet(path)
    if saved.schema != sample.schema or not saved.equals(submission):
        raise ValueError("Submission verification failed after writing")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    recipe = {"complete": True, "selected_by": "holdout.ndcg", "winner": method, "gru_weight": weight,
              "rrf_offset": report["rrf_offset"] if method == "rrf" else None, "metrics": winner,
              "catboost_shards": len(sources[CAT]["parts"]), "train_feeds_by_model": report["train_feeds_by_model"],
              "test_feeds": submission["feed_id"].n_unique(), "rows": len(submission), "sha256": digest}
    (directory / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(submission):,} candidates, SHA-256 {digest}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("data", "catboost", "queue", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--wait", action="store_true", help="Wait for a running training queue to finish the comparison")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.wait:
        from training_queue import keep_awake
        if not args.queue.is_dir():
            parser.error("--wait requires the existing queue directory")
        print("Waiting for the local validation comparison", flush=True)
        while not (args.queue / "comparison/report.json").is_file():
            for stage in ("queue", "gru", "blends"):
                status = args.queue / f"{stage}.status"
                if status.is_file() and status.read_text(encoding="utf-8").strip() not in ("", "0"):
                    raise RuntimeError(f"Training queue stopped at {stage}. See the local log.")
            keep_awake()
            time.sleep(15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as staging:
        pending = Path(staging) / "submission"
        pending.mkdir()
        export(args, pending)
        pending.rename(args.output)
