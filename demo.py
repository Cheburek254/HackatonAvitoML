"""Generate a reproducible Avito-shaped dataset and run synthetic comparisons."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


BASE_FEATURES = [f"f{index}" for index in range(1, 6)]
SCHEMA = {
    "feed_id": pa.int32(), "timestamp": pa.int32(), "user_id": pa.int32(),
    "category_id": pa.list_(pa.int8()), **dict.fromkeys(BASE_FEATURES, pa.list_(pa.float32())),
    "pos": pa.list_(pa.int16()), "target": pa.list_(pa.int8()),
    "item_id__embedding_idx": pa.list_(pa.int32()), "item_id": pa.list_(pa.int32()),
    "user_embedding": pa.list_(pa.int8()), "history__item_id": pa.list_(pa.int32()),
    "history__category_id": pa.list_(pa.int8()), "history__event_type": pa.list_(pa.int8()),
    "history__event_source": pa.list_(pa.int8()), "history__item_id__embedding_idx": pa.list_(pa.int32()),
}
SPLITS = ("train", "val", "test")
TIMES = (1788652800, 1790035200, 1790380800)


class Catalog:
    def __init__(self, rng):
        self.rng = rng
        self.topics = 48
        self.topic_vectors = rng.standard_normal((self.topics, 128)).astype(np.float32)
        self.topic = np.repeat(np.arange(self.topics), 240)
        self.category = (self.topic // 3 + 1).astype(np.int8)
        self.category[rng.random(len(self.topic)) < 0.01] = 0
        self.quality = rng.standard_normal(len(self.topic))
        self.embeddings = self.quantize(self.topic_vectors[self.topic] + 0.32 * rng.standard_normal((len(self.topic), 128)))
        self.item_age = np.tile(np.repeat(np.arange(3), 80), self.topics)
        self.interests = rng.integers(0, self.topics, size=(2200, 2))

    @staticmethod
    def quantize(vectors):
        return np.rint(vectors * 32).clip(-128, 127).astype(np.int8)

    def feed(self, split, index, feed_id, timestamp, available, groups, index_of):
        rng = self.rng
        user = int(rng.integers(0, 1400)) if rng.random() < 0.87 else int(rng.integers(1400 + split * 250, 1650 + split * 250))
        topics = self.interests[user]
        active, alternative = map(int, topics if rng.random() < 0.55 else topics[::-1])
        user_vector = self.quantize(self.topic_vectors[topics].mean(axis=0) + 0.45 * rng.standard_normal(128))
        if index < 2:
            count, history_size = ((5, 0), (301, 256))[index]
        else:
            count = int(np.clip(rng.lognormal(4.0, 0.72), 5, 301))
            history_size = int(rng.choice([0, 1, 6, 24, 64, 128, 256], p=[.08, .04, .13, .27, .25, .17, .06]))
        history_topics = np.where(rng.random(history_size) < .62, active, alternative)
        drift = rng.random(history_size) < .16
        history_topics[drift] = rng.integers(self.topics, size=int(drift.sum()))
        history = np.array([rng.choice(groups[topic]) for topic in history_topics], dtype=np.int32)
        if history_size > 3 and rng.random() < .5:
            history[3] = history[0]

        relevant_pool = np.union1d(groups[active], groups[alternative])
        focus = min(count - 2, max(2, count // 3), len(relevant_pool))
        relevant = rng.choice(relevant_pool, size=focus, replace=False)
        candidates = np.r_[relevant, rng.choice(np.setdiff1d(available, relevant), size=count - focus, replace=False)]
        candidate_topics = self.topic[candidates]
        relevance = (candidate_topics == active) + 0.35 * (candidate_topics == alternative)
        raw = rng.standard_normal((5, count))
        raw[0] += .6 * relevance
        raw[1] += .5 * self.quality[candidates]
        raw[2] += .35 * (self.item_age[candidates] == split)
        raw[3] += .35 * (candidate_topics == alternative)
        raw[4] += .25 * np.log1p(history_size) / 5
        presented = np.argsort(-(raw[0] + .45 * raw[1])) if split == 0 else rng.permutation(count)
        positions = np.empty(count, dtype=np.int16)
        positions[presented] = np.arange(count, dtype=np.int16)
        if count < 290 and rng.random() < .25:
            step = max(11, (count - 1) // (301 - count) + 1)
            positions += positions // step

        best_match = np.zeros(count)
        if history_size:
            width = min(history_size, 24)
            history_vectors = self.embeddings[history[:width]].astype(np.float32)
            item_vectors = self.embeddings[candidates].astype(np.float32)
            cosine = np.einsum("ij,kj->ik", item_vectors, history_vectors) / (np.linalg.norm(item_vectors, axis=1)[:, None] * np.linalg.norm(history_vectors, axis=1))
            best_match = (cosine * np.exp2(-np.arange(width) / 16)).max(axis=1)
        scores = 3.1 * best_match + 1.05 * relevance + .65 * raw[0] + .35 * self.quality[candidates]
        scores += (.58 if split == 0 else .23) * np.exp(-positions / 14)
        scores += rng.gumbel(0, .58, size=count)
        positives = min(count - 1, 1 + rng.poisson(0.65))
        target = np.zeros(count, dtype=np.int8)
        target[np.argsort(scores)[-positives:]] = 1
        order = presented if split == 0 else rng.permutation(count)
        candidate_indices = index_of[candidates].copy()
        candidate_indices[rng.random(count) < .025] = -1
        history_indices = index_of[history].copy()
        history_indices[rng.random(history_size) < .04] = -1
        history_ids = history + 1
        history_ids[rng.random(history_size) < .02] = 0
        events = rng.choice([0, 1, 2, 3, 4, 5], size=history_size, p=[.01, .45, .27, .13, .09, .05])
        sources = rng.choice([0, 1, 2, 3, 4], size=history_size, p=[.02, .50, .25, .15, .08])
        row = {
            "feed_id": feed_id, "timestamp": timestamp, "user_id": user + 1 if rng.random() >= .025 else 0,
            "category_id": self.category[candidates][order].tolist(),
            "item_id": (candidates[order] + 1).tolist(), "item_id__embedding_idx": candidate_indices[order].tolist(),
            "user_embedding": user_vector.tolist(), "history__item_id": history_ids.tolist(),
            "history__category_id": self.category[history].tolist(), "history__event_type": events.tolist(),
            "history__event_source": sources.tolist(), "history__item_id__embedding_idx": history_indices.tolist(),
            **dict(zip(BASE_FEATURES, np.tanh(raw[:, order]).astype(np.float32).tolist(), strict=True)),
        }
        if split == 0:
            row["pos"] = positions[order].tolist()
        if split < 2:
            row["target"] = target[order].tolist()
        return row


def generate(root, counts, seed):
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    catalog = Catalog(rng)
    submission_schema = pa.schema({"feed_id": pa.int32(), "item_id": pa.int32(), "score": pa.float32()})
    feed_id = 1
    summaries = {}
    with pq.ParquetWriter(root / "sample_submission.parquet", submission_schema, compression="zstd") as submission:
        for split, (name, count, start) in enumerate(zip(SPLITS, counts, TIMES, strict=True)):
            directory = root / name
            (directory / "impressions").mkdir(parents=True)
            (directory / "item_embeddings").mkdir()
            available = np.flatnonzero(catalog.item_age <= split)
            if split:
                available = available[(catalog.item_age[available] == split) | (available % 11 != 0)]
            groups = [available[catalog.topic[available] == topic] for topic in range(catalog.topics)]
            permutation = rng.permutation(available)
            index_of = np.full(len(catalog.topic), -1, dtype=np.int32)
            index_of[permutation] = np.arange(len(permutation), dtype=np.int32)
            table = catalog.embeddings[permutation]
            table.tofile(directory / "item_embeddings" / "data.memmap")
            metadata = {"dtype": "int8", "shape": list(table.shape), "order": "C"}
            (directory / "item_embeddings" / "data.meta").write_text(json.dumps(metadata), encoding="utf-8")
            exclusions = set() if split == 0 else {"pos"} if split == 1 else {"pos", "target"}
            schema = pa.schema({key: value for key, value in SCHEMA.items() if key not in exclusions})
            candidates = 0
            for first in range(0, count, 512):
                rows = []
                for index in range(first, min(first + 512, count)):
                    timestamp = start + index * (13 if split == 0 else 3) * 86400 // count
                    timestamp -= timestamp % 60
                    row = catalog.feed(split, index, feed_id, timestamp, available, groups, index_of)
                    candidates += len(row["item_id"])
                    rows.append(row)
                    feed_id += 1
                pq.write_table(pa.Table.from_pylist(rows, schema=schema), directory / "impressions" / f"part-{first // 512:05}.parquet", compression="zstd")
                if split == 2:
                    # Store keys in a different order to exercise key-based validation.
                    keys = [(row["feed_id"], item) for row in reversed(rows) for item in reversed(row["item_id"])]
                    feeds, items = zip(*keys, strict=True)
                    submission.write_table(pa.Table.from_pydict({"feed_id": feeds, "item_id": items, "score": np.zeros(len(keys), dtype=np.float32)}, schema=submission_schema))
            (directory / "impressions" / "_SUCCESS").touch()
            summaries[name] = {"feeds": count, "candidates": candidates}
    (root / "readme.md").write_text(f"Synthetic data generated by demo.py with seed {seed}.\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("runs/synthetic/competition"))
    parser.add_argument("--train-feeds", type=int, default=6000)
    parser.add_argument("--val-feeds", type=int, default=1500)
    parser.add_argument("--test-feeds", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    counts = [args.train_feeds, args.val_feeds, args.test_feeds]
    if min(counts) < 2:
        parser.error("Each split requires at least two feeds")
    generate(args.data, counts, args.seed)


if __name__ == "__main__":
    main()
