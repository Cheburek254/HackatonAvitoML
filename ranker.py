"""Train and apply a CatBoost ranker to Avito competition feeds."""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from catboost import CatBoost, CatBoostRanker, Pool

import behavior
import embedding_clusters
from evaluation import KEYS, NDCG, attach_scores, ndcg, peak_ram_mib

BASE_FEATURES = ["f1", "f2", "f3", "f4", "f5"]
CANDIDATE_COLUMNS = ["item_id", "category_id", *BASE_FEATURES, "item_id__embedding_idx"]
HISTORY_COLUMNS = ["history__item_id", "history__category_id", "history__item_id__embedding_idx"]
AVITO_FEATURES = ["category_id", *BASE_FEATURES, "user_item_dot", "user_item_cosine", "item_embedding_missing"]
PERCENTILES = [f"{name}_feed_percentile" for name in BASE_FEATURES]
HISTORY_FEATURES = ["history_length", "recent_category_affinity", "recent_item_seen", "recent_items_cosine"]
CATEGORY_TREND_FEATURES = ["category_share_last8", "category_share_previous56", "category_trend"]
WIDE_TREND_FEATURES = ["category_share_last16", "category_share_previous112", "category_trend_wide"]
CLUSTER_FEATURES = ["embedding_cluster", "cluster_history_share", "cluster_cosine"]
DIN_FEATURES = ["din_max_cosine", "din_top3_cosine", "din_best_recency", "din_match_available"]
SIM_FEATURES = ["sim_top16_cosine", "sim_recent_cosine", "sim_matching_category_share"]
MIND_FEATURES = ["mind_best_cosine", "mind_category_cosine", "mind_interest_count"]
FEATURES = ["category_id", *BASE_FEATURES, *PERCENTILES, "user_item_cosine", "item_embedding_missing", *HISTORY_FEATURES]
VARIANTS = {
    "avito-features": AVITO_FEATURES,
    "history": FEATURES,
    "din": [*FEATURES, *DIN_FEATURES],
    "sim": [*FEATURES, *SIM_FEATURES],
    "multi-interest": [*FEATURES, *MIND_FEATURES],
    "lambda": [*FEATURES, *DIN_FEATURES],
    **{kind: [*FEATURES, *names] for kind, names in behavior.FEATURES.items()},
    "category-trend": [*FEATURES, *CATEGORY_TREND_FEATURES],
    "category-trend-wide": [*FEATURES, *WIDE_TREND_FEATURES],
    "embedding-clusters": [*FEATURES, *CLUSTER_FEATURES],
    "history-distill": FEATURES,
    "category-trend-pixie": [*FEATURES, *CATEGORY_TREND_FEATURES, *behavior.FEATURES["pixie"]],
}
RECENT_EVENTS = 32
HISTORY_WEIGHTS = np.exp2(-np.arange(RECENT_EVENTS, dtype=np.float32) / 8)


def cosine(items, vector):
    dot = np.einsum("ij,j->i", items, vector)
    scale = np.linalg.norm(items, axis=1) * np.linalg.norm(vector)
    return np.divide(dot, scale, out=np.zeros_like(dot), where=scale > 0)


def distinct_events(item_ids):
    known_positions = np.flatnonzero(item_ids > 0)
    _, first_positions = np.unique(item_ids[known_positions], return_index=True)
    unique_events = item_ids == 0
    unique_events[known_positions[first_positions]] = True
    return unique_events


def category_trend(categories, history, recent_count=8, total=64):
    recent = np.asarray(history[:recent_count])
    earlier = np.asarray(history[recent_count:total])
    recent = recent[recent != 0]
    earlier = earlier[earlier != 0]
    recent_share = (categories[:, None] == recent).sum(axis=1) / max(len(recent), 1)
    earlier_share = (categories[:, None] == earlier).sum(axis=1) / max(len(earlier), 1)
    trend = recent_share - earlier_share if recent.size and earlier.size else np.zeros_like(recent_share)
    return recent_share, earlier_share, trend


def pair_cosine(items, history):
    dot = np.einsum("ik,jk->ij", items, history)
    scale = np.linalg.norm(items, axis=1)[:, None] * np.linalg.norm(history, axis=1)
    return np.divide(dot, scale, out=np.zeros_like(dot), where=scale > 0)


def matching_features(items, categories, history_items, history_categories, positions, mode):
    valid = np.linalg.norm(history_items, axis=1) > 0
    history, history_categories, positions = history_items[valid], history_categories[valid], positions[valid]
    present = np.linalg.norm(items, axis=1) > 0
    if mode == "multi-interest":
        interests, counts = np.unique(history_categories[history_categories != 0], return_counts=True)
        interests = interests[np.argsort(-counts, kind="stable")[:4]]
        if not len(interests):
            return np.zeros((len(MIND_FEATURES), len(items)), dtype=np.float32)
        profiles = np.stack([history[history_categories == interest].mean(axis=0) for interest in interests])
        similarities = pair_cosine(items, profiles)
        return np.stack([similarities.max(axis=1), (similarities * (categories[:, None] == interests)).sum(axis=1),
                         np.full(len(items), len(interests), dtype=np.float32)])

    names = DIN_FEATURES if mode == "din" else SIM_FEATURES
    if not len(history):
        return np.zeros((len(names), len(items)), dtype=np.float32)
    similarities = pair_cosine(items, history)
    if mode == "din":
        neighbors = min(3, len(history))
        best = np.argmax(similarities, axis=1)
        top = np.partition(similarities, -neighbors, axis=1)[:, -neighbors:]
        return np.stack([top.max(axis=1), top.mean(axis=1), np.where(present, 1 / (1 + positions[best]), 0), present])

    neighbors = min(16, len(history))
    top = np.argpartition(similarities, -neighbors, axis=1)[:, -neighbors:]
    scores = np.take_along_axis(similarities, top, axis=1)
    weights = np.exp2(-positions[top] / 16)
    category_match = (categories[:, None] != 0) & (categories[:, None] == history_categories[top])
    return np.stack([scores.mean(axis=1), (scores * weights).sum(axis=1) / weights.sum(axis=1),
                     np.where(present, category_match.mean(axis=1), 0)])


def vector_features(feed, embeddings, feature_names, reference, clusters=None):
    item_ids = np.asarray(feed["item_id"])
    categories = np.asarray(feed["category_id"])
    item_indices = np.asarray(feed["item_id__embedding_idx"], dtype=np.intp)
    present = item_indices >= 0
    items = np.zeros((len(item_ids), embeddings.shape[1]), dtype=np.float32)
    items[present] = embeddings[item_indices[present]]
    user = np.asarray(feed["user_embedding"], dtype=np.float32)
    result = {"user_item_cosine": cosine(items, user), "item_embedding_missing": ~present}
    if "user_item_dot" in feature_names:
        result["user_item_dot"] = np.einsum("ij,j->i", items, user)
    if "history_length" not in feature_names:
        return result

    window = 256 if SIM_FEATURES[0] in feature_names else RECENT_EVENTS
    history_ids = np.asarray(feed["history__item_id"][:window], dtype=np.int32)
    history_categories = np.asarray(feed["history__category_id"][:window])
    history_indices = np.asarray(feed["history__item_id__embedding_idx"][:window], dtype=np.intp)
    unique_events = distinct_events(history_ids)
    recent = slice(0, RECENT_EVENTS)
    weights = HISTORY_WEIGHTS[:len(history_ids[recent])] * unique_events[recent]
    category_matches = (categories[:, None] == history_categories[recent]) & (categories[:, None] != 0)

    history_present = (history_indices >= 0) & unique_events
    history_items = embeddings[history_indices[history_present]].astype(np.float32)
    positions = np.flatnonzero(history_present)
    recent_vectors = positions < RECENT_EVENTS
    history_vector = np.einsum("i,ij->j", weights[positions[recent_vectors]], history_items[recent_vectors])
    result.update(zip(HISTORY_FEATURES, (
        np.full(len(item_ids), len(feed["history__item_id"]), dtype=np.float32),
        np.einsum("ij,j->i", category_matches, weights),
        ((item_ids > 0) & np.isin(item_ids, history_ids[recent])).astype(np.float32),
        cosine(items, history_vector),
    ), strict=True))
    if CATEGORY_TREND_FEATURES[0] in feature_names:
        result.update(zip(CATEGORY_TREND_FEATURES, category_trend(categories, feed["history__category_id"]), strict=True))
    if WIDE_TREND_FEATURES[0] in feature_names:
        result.update(zip(WIDE_TREND_FEATURES, category_trend(categories, feed["history__category_id"], 16, 128), strict=True))
    if clusters is not None:
        values = embedding_clusters.history_match(item_indices, history_indices[unique_events], clusters)
        result.update(zip(CLUSTER_FEATURES, values, strict=True))
    for mode, names in (("din", DIN_FEATURES), ("sim", SIM_FEATURES), ("multi-interest", MIND_FEATURES)):
        if names[0] in feature_names:
            values = matching_features(items, categories, history_items, history_categories[history_present], positions, mode)
            result.update(zip(names, values, strict=True))
    if reference is not None:
        result.update(zip(behavior.FEATURES[reference.kind], reference.score(feed), strict=True))
    return result


def expand_batch(feeds, embeddings, labeled, features, reference, clusters=None):
    per_feed = [vector_features(feed, embeddings, features, reference, clusters) for feed in feeds.iter_rows(named=True)]
    arrays = [*CANDIDATE_COLUMNS, *(["target"] if labeled else [])]
    candidates = feeds.select("feed_id", *arrays).explode(arrays).drop("item_id__embedding_idx")
    return candidates.with_columns(
        *[
            pl.Series(name, np.concatenate([columns[name] for columns in per_feed]).astype(np.float32))
            for name in per_feed[0]
        ],
        *[
            (pl.col(name).rank("average").over("feed_id") / pl.len().over("feed_id"))
            .cast(pl.Float32).alias(f"{name}_feed_percentile")
            for name, percentile in zip(BASE_FEATURES, PERCENTILES, strict=True) if percentile in features
        ],
    )


def feed_batches(split, columns, max_feeds=0):
    parts = sorted((split / "impressions").glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No part-*.parquet files under {split / 'impressions'}")

    processed = 0
    for part in parts:
        with pq.ParquetFile(part) as parquet:
            for batch in parquet.iter_batches(batch_size=512, columns=columns):
                if max_feeds:
                    batch = batch.slice(0, max_feeds - processed)
                yield pl.from_arrow(batch)
                processed += batch.num_rows
                if max_feeds and processed >= max_feeds:
                    return


def open_embeddings(split):
    directory = split / "item_embeddings"
    metadata = json.loads((directory / "data.meta").read_text(encoding="utf-8"))
    return np.memmap(directory / "data.memmap", mode="r", dtype=metadata["dtype"],
                     shape=tuple(metadata["shape"]), order=metadata["order"])


def candidate_batches(split, *, features=FEATURES, labeled=False, max_feeds=0,
                      reference=None, after=None, before=None, centers=None, device="cpu"):
    columns = ["feed_id", "user_embedding", *CANDIDATE_COLUMNS]
    if "history_length" in features:
        columns.extend(HISTORY_COLUMNS)
    if reference is not None:
        columns.append("user_id")
    if labeled:
        columns.append("target")
    if after is not None or before is not None:
        columns.append("timestamp")
    embeddings = open_embeddings(split)
    clusters = embedding_clusters.assign(embeddings, centers, device) if centers is not None else None
    for batch in feed_batches(split, columns, max_feeds):
        if after is not None:
            batch = batch.filter(pl.col("timestamp") >= after)
        if before is not None:
            batch = batch.filter(pl.col("timestamp") < before)
        if batch.height:
            yield expand_batch(batch, embeddings, labeled, features, reference, clusters)


def read_candidates(split, **kwargs):
    batches = list(candidate_batches(split, **kwargs))
    if not batches:
        raise ValueError(f"No feeds in {split}")
    return pl.concat(batches, rechunk=True)


def make_pool(candidates, features, threads, labeled=False):
    return Pool(candidates.select(features), label=candidates["target"] if labeled else None,
                group_id=candidates["feed_id"] if labeled else None, cat_features=["category_id"], thread_count=threads)


def load_training(args, features, centers=None):
    split = args.data / "train"
    kind = behavior.reference_kind(features)
    if kind is None:
        return read_candidates(split, features=features, labeled=True, max_feeds=args.train_feeds,
                               centers=centers, device=args.device), None

    columns = ["timestamp", "user_id", "item_id", "category_id", "target", *HISTORY_COLUMNS[:2]]
    past = pl.concat(list(feed_batches(split, columns, args.train_feeds)))
    if not past.height:
        raise ValueError("No training feeds")
    times = np.sort(past["timestamp"].to_numpy())
    boundaries = sorted(set(int(times[len(times) * index // 4]) for index in (1, 2, 3)) - {int(times[0])})
    reference = behavior.Reference(kind)
    chunks = []
    for after, before in zip([None, *boundaries], [*boundaries, None], strict=True):
        chunks.append(read_candidates(split, features=features, labeled=True, max_feeds=args.train_feeds,
                                      reference=reference, after=after, before=before))
        if before is not None:
            reference = behavior.fit(kind, past.filter(pl.col("timestamp") < before), args.threads, args.device)
    return pl.concat(chunks, rechunk=True), behavior.fit(kind, past, args.threads, args.device)


def train_centers(args):
    split = args.data / "train"
    indices = "item_id__embedding_idx"
    used = pl.concat(list(feed_batches(split, [indices], args.train_feeds)))[indices].explode().unique().sort().to_numpy()
    return embedding_clusters.fit(open_embeddings(split), used[used >= 0], args.device)


def new_model(args, iterations):
    return CatBoostRanker(
        loss_function="LambdaMart:metric=NDCG" if args.variant == "lambda" else "YetiRank",
        eval_metric="RMSE" if args.device == "gpu" else NDCG,
        task_type=args.device.upper(), iterations=iterations, depth=6, learning_rate=0.1,
        random_seed=42, thread_count=args.threads, allow_writing_files=False,
        **({"devices": "0", "gpu_ram_part": .65} if args.device == "gpu" else {}),
    )


def raw_scores(model, pool, threads, device):
    return CatBoost.predict(model, pool, thread_count=threads, task_type=device.upper())


def select_gpu_checkpoint(model, tuning, val_pool, args):
    from accelerator import GpuNDCG

    metric = GpuNDCG(tuning)
    step = max(1, min(5, args.iterations // 20))
    checkpoints = sorted({1, model.tree_count_, *range(step, model.tree_count_, step)})
    best_ndcg, best_tree, curve = -1, 1, []
    for trees in checkpoints:
        # The CatBoost GPU evaluator applies all trees regardless of ntree_end.
        checkpoint = model.copy()
        checkpoint.shrink(trees)
        prediction = raw_scores(checkpoint, val_pool, args.threads, args.device)
        value = metric(prediction)["ndcg"]
        curve.append([trees, value])
        print(f"{args.variant} trees={trees} tuning_ndcg={value:.8f}", flush=True)
        if value > best_ndcg:
            best_ndcg, best_tree = value, trees
    model.shrink(best_tree)
    return curve


def distillation_targets(args, training):
    features = VARIANTS["category-trend"]
    split = args.data / "train"
    teacher_data = read_candidates(split, features=features, labeled=True, max_feeds=args.train_feeds)
    times = pl.concat(list(feed_batches(split, ["feed_id", "timestamp"], args.train_feeds)))
    stamps = times["timestamp"].unique().sort().to_numpy()
    cuts = np.unique([stamps[len(stamps) * index // 4] for index in (1, 2, 3)])
    soft_scores = []
    for before, after in zip(cuts, [*cuts[1:], int(stamps[-1]) + 1], strict=True):
        past_ids = times.filter(pl.col("timestamp") < int(before))["feed_id"]
        next_ids = times.filter(pl.col("timestamp").is_between(int(before), int(after), closed="left"))["feed_id"]
        if not len(past_ids) or not len(next_ids):
            continue
        print(f"Distillation teacher: {len(past_ids)} training feeds, {len(next_ids)} scoring feeds", flush=True)
        past = teacher_data.filter(pl.col("feed_id").is_in(past_ids))
        future = teacher_data.filter(pl.col("feed_id").is_in(next_ids))
        teacher = new_model(args, min(args.iterations, 50))
        teacher.fit(make_pool(past, features, args.threads, labeled=True), verbose=False)
        soft_scores.append(score_batch(teacher, future, args.threads, args.device).with_columns(
            ((pl.col("score").rank("average").over("feed_id") - 0.5) / pl.len().over("feed_id")).alias("teacher")
        ).drop("score"))
    if not soft_scores:
        return training
    return training.join(pl.concat(soft_scores), on=KEYS, how="left", validate="1:1").with_columns(
        (0.7 * pl.col("target") + 0.3 * pl.col("teacher").fill_null(pl.col("target"))).cast(pl.Float32).alias("target")
    ).drop("teacher")


def train(args):
    if args.output.exists():
        raise FileExistsError(args.output)

    features = VARIANTS[args.variant]
    print(f"Preparing features for {args.variant} ({args.device})", flush=True)
    centers = train_centers(args) if args.variant == "embedding-clusters" else None
    training, reference = load_training(args, features, centers)
    if args.variant == "history-distill":
        training = distillation_targets(args, training)
    train_pool = make_pool(training, features, args.threads, labeled=True)
    del training
    val = read_candidates(args.data / "val", features=features, labeled=True, max_feeds=args.val_feeds,
                          reference=reference, centers=centers, device=args.device)
    tuning = val
    if args.tune_before is not None:
        tune_ids = pl.concat(list(feed_batches(args.data / "val", ["feed_id", "timestamp"], args.val_feeds)))
        tuning = val.filter(pl.col("feed_id").is_in(tune_ids.filter(pl.col("timestamp") < args.tune_before)["feed_id"]))
        if not tuning.height or tuning.height == val.height:
            raise ValueError("--tune-before must divide the selected validation feeds into two nonempty sets")
    val_pool = make_pool(tuning, features, args.threads, labeled=True)
    model = new_model(args, args.iterations)
    metric = ndcg
    curve = []
    if args.device == "gpu":
        from accelerator import gpu_ndcg

        metric = gpu_ndcg
        model.fit(train_pool, verbose=max(1, min(10, args.iterations // 10)), use_best_model=False)
        curve = select_gpu_checkpoint(model, tuning, val_pool, args)
    else:
        model.fit(train_pool, eval_set=val_pool, use_best_model=True, early_stopping_rounds=50, verbose=10)
    predictions = score_batch(model, val, args.threads, args.device)
    metrics = metric(attach_scores(val.select(*KEYS, "target"), predictions))
    report = {
        **{f"val_{name}": value for name, value in metrics.items()},
        "trees": model.tree_count_,
        "train_candidates": train_pool.num_row(),
        "val_candidates": val.height,
        "train_feed_limit": args.train_feeds,
        "val_feed_limit": args.val_feeds,
        "device": args.device,
        "variant": args.variant,
        "tune_before": args.tune_before,
        "tuning_curve": curve,
    }
    args.output.mkdir(parents=True)
    model.save_model(str(args.output / "model.cbm"))
    predictions.write_parquet(args.output / "val_predictions.parquet")
    if reference is not None:
        reference.save(args.output / "behavior.npz")
    if centers is not None:
        np.save(args.output / "centers.npy", centers)
    report["peak_ram_mib"] = peak_ram_mib()
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "tuning_curve"}, indent=2))


def score_batch(model, candidates, threads, device="cpu"):
    pool = make_pool(candidates, model.feature_names_, threads)
    scores = raw_scores(model, pool, threads, device)
    return candidates.select(KEYS).with_columns(pl.Series("score", scores, dtype=pl.Float32))


def predict(args):
    if args.output.exists():
        raise FileExistsError(args.output)

    model = CatBoostRanker().load_model(str(args.model))
    template = pl.read_parquet(args.data / "sample_submission.parquet", columns=KEYS)
    kind = behavior.reference_kind(model.feature_names_)
    reference = behavior.Reference.load(kind, args.model.with_name("behavior.npz")) if kind else None
    centers = np.load(args.model.with_name("centers.npy"), allow_pickle=False) if CLUSTER_FEATURES[0] in model.feature_names_ else None
    predictions = pl.concat([
        score_batch(model, batch, args.threads, args.device)
        for batch in candidate_batches(args.data / "test", features=model.feature_names_, reference=reference,
                                       centers=centers, device=args.device)
    ])
    submission = attach_scores(template, predictions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.write_parquet(args.output)
    print(f"Saved {submission.height:,} scores to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    training = commands.add_parser("train")
    training.add_argument("--train-feeds", type=int, default=50_000, help="0 uses the full train split")
    training.add_argument("--val-feeds", type=int, default=10_000, help="0 uses the full val split")
    training.add_argument("--iterations", type=int, default=500)
    training.add_argument("--variant", choices=VARIANTS, default="history")
    training.add_argument("--tune-before", type=int, help="Use only validation feeds strictly before this Unix time for early stopping")
    training.set_defaults(run=train)
    scoring = commands.add_parser("predict")
    scoring.add_argument("--model", type=Path, required=True)
    scoring.set_defaults(run=predict)
    for command in (training, scoring):
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--threads", type=int, default=4)
        command.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
        command.add_argument("--gpu-id", help="NVIDIA index or UUID from nvidia-smi")

    args = parser.parse_args()
    if getattr(args, "train_feeds", 0) < 0 or getattr(args, "val_feeds", 0) < 0:
        parser.error("Feed limits must be nonnegative")
    if args.threads < 1 or getattr(args, "iterations", 1) < 1:
        parser.error("Threads and iterations must be positive")
    if getattr(args, "variant", None) == "lambda" and args.device != "cpu":
        parser.error("CatBoost LambdaMart supports CPU training")
    if args.gpu_id:
        from accelerator import select_gpu

        if args.device != "gpu":
            parser.error("--gpu-id requires --device gpu")
        select_gpu(args.gpu_id)
    args.run(args)


if __name__ == "__main__":
    main()
