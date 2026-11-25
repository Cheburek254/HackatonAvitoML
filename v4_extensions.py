"""Train and compare history denoising and temporal graph scoring for the V4 blend."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import polars as pl

from evaluation import KEYS, attach_scores, ndcg, zscore
from training_queue import atomic_text


CUTOFF = 1790107648
V4_WEIGHTS = dict(qsm=.40, gru4=.30, gru3=.15, lgbm=.15)


def train_end(args):
    from accelerator import select_gpu

    gpu = select_gpu(args.gpu)
    sys.path.insert(0, str(args.v4_code.resolve()))
    import neural
    import torch
    from torch import nn

    class DenoisedRanker(neural.Ranker):
        def __init__(self, variant):
            super().__init__(variant)
            self.reliability = nn.Sequential(nn.Linear(128 * 2, 32), nn.GELU(), nn.Linear(32, 1))
            nn.init.zeros_(self.reliability[-1].weight)
            nn.init.constant_(self.reliability[-1].bias, 2.0)

        def history_tokens(self, history, events, padding):
            tokens, mask, lengths = super().history_tokens(history, events, padding)
            body = tokens[:, 1:].masked_fill(padding[..., None], 0)
            recent_positions = torch.arange(body.shape[1], device=body.device) >= (lengths[:, None] - 8)
            recent = (body * recent_positions[..., None]).sum(dim=1, keepdim=True)
            recent = recent / lengths.clamp(1, 8)[:, None, None]
            gate = self.reliability(torch.cat((body, recent.expand_as(body)), dim=-1)).sigmoid()
            return torch.cat((tokens[:, :1], body * gate), dim=1), mask, lengths

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    print(f"PyTorch GPU: {gpu.name} ({gpu.uuid})", flush=True)
    neural.Ranker = DenoisedRanker
    neural.train(SimpleNamespace(variant="gru-din-v4", device="cuda", data=args.data, output=args.output,
                                 train_feeds=args.train_feeds, val_feeds=0, train_offset=0, history_window=64,
                                 batch_size=16, prefetch=True, tune_before=CUTOFF, epochs=args.epochs))
    checkpoint_path = args.output / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["extension"] = "end4rec-soft-gate"
    staged = checkpoint_path.with_suffix(".tmp")
    torch.save(checkpoint, staged)
    staged.replace(checkpoint_path)


def tensor_vectors(values):
    # The shared 128-D embeddings use the same deterministic 32-D projection for each side.
    vectors = np.asarray(values, dtype=np.float32).reshape(-1, 32, 4)
    vectors = vectors @ np.array([.5, -.5, .5, .5], dtype=np.float32)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-6)


class TemporalProfiles:
    def __init__(self, train, val):
        ids = np.concatenate([frame["item_id"].explode().to_numpy() for frame in (train, val)])
        self.items = np.unique(ids)
        self.sums = np.zeros((len(self.items), 32), dtype=np.float32)
        self.counts = np.zeros(len(self.items), dtype=np.int32)
        edges = (train.select("row", "user_id", pl.col("history__item_id").list.head(32).alias("item_id"))
                 .explode("item_id").filter((pl.col("item_id") > 0) & (pl.col("user_id") > 0))
                 .group_by("item_id", "user_id").agg(pl.col("row").min()).sort("row"))
        edge_ids = edges["item_id"].to_numpy()
        index = np.searchsorted(self.items, edge_ids).clip(max=len(self.items) - 1)
        known = self.items[index] == edge_ids
        self.edge_rows = edges["row"].to_numpy()[known]
        self.edge_items = index[known]
        self.train_users = tensor_vectors(train["user_embedding"].list.to_array(128).to_numpy())

    def update(self, first, last):
        start, stop = np.searchsorted(self.edge_rows, (first, last))
        index = self.edge_items[start:stop]
        rows = self.edge_rows[start:stop]
        for offset in range(0, len(rows), 131072):
            block = slice(offset, offset + 131072)
            np.add.at(self.sums, index[block], self.train_users[rows[block]])
        np.add.at(self.counts, index, 1)

    def score(self, frame, embeddings):
        counts = frame["item_id"].list.len().to_numpy()
        users = tensor_vectors(frame["user_embedding"].list.to_array(128).to_numpy())
        history = np.zeros_like(users)
        for row, indices in enumerate(frame["history__item_id__embedding_idx"].list.head(32)):
            index = indices.to_numpy()
            valid = index >= 0
            if valid.any():
                history[row] = np.exp2(-np.arange(len(index), dtype=np.float32)[valid] / 8) @ tensor_vectors(embeddings[index[valid]])
        history /= np.maximum(np.linalg.norm(history, axis=1, keepdims=True), 1e-6)
        index = np.searchsorted(self.items, frame["item_id"].explode().to_numpy())
        neighbors = self.counts[index]
        profiles = self.sums[index]
        strength = np.linalg.norm(profiles, axis=1)
        profiles /= np.maximum(strength[:, None], 1e-6)
        owner = np.repeat(np.arange(len(frame)), counts)
        return np.column_stack((np.einsum("ij,ij->i", users[owner], profiles),
                                np.einsum("ij,ij->i", history[owner], profiles), np.log1p(neighbors).astype(np.float32),
                                strength / np.maximum(neighbors, 1).astype(np.float32)))


def train_graph(args):
    from accelerator import select_gpu

    gpu = select_gpu(args.gpu)
    from catboost import CatBoostRanker, Pool
    from ranker import feed_batches, open_embeddings

    print(f"CatBoost GPU: {gpu.name} ({gpu.uuid})", flush=True)
    columns = (*KEYS, "timestamp", "target", "user_id", "user_embedding",
               "history__item_id", "history__item_id__embedding_idx")
    start = perf_counter()
    train = (pl.concat(list(feed_batches(args.data / "train", columns, args.train_feeds)))
             .sort(["timestamp", "feed_id"]).with_row_index("row")
             .with_columns((pl.col("timestamp") // 86400).alias("day")))
    val = pl.concat(list(feed_batches(args.data / "val", columns)))
    if train["timestamp"].max() >= val["timestamp"].min():
        raise ValueError("Graph snapshot must end before validation starts")
    graph = TemporalProfiles(train, val)
    training_embeddings = open_embeddings(args.data / "train")
    blocks = []
    for day in train.partition_by("day", maintain_order=True):
        for batch in day.iter_slices(512):
            blocks.append(graph.score(batch, training_embeddings))
        graph.update(day["row"][0], day["row"][-1] + 1)
        print(f"Graph features {day['row'][-1] + 1:,}/{len(train):,} training feeds", flush=True)
    training = np.concatenate(blocks)
    del blocks
    if not training[:, 2].any():
        raise ValueError("No temporal graph coverage for training")
    validation_embeddings = open_embeddings(args.data / "val")
    validation = np.concatenate([graph.score(batch, validation_embeddings) for batch in val.iter_slices(512)])
    del graph
    pool = Pool(training, label=train["target"].explode().to_numpy(),
                group_id=np.repeat(train["feed_id"].to_numpy(), train["item_id"].list.len().to_numpy()))
    ranker = CatBoostRanker(task_type="GPU", devices="0", loss_function="QuerySoftMax", iterations=args.iterations,
                            depth=4, learning_rate=.1, random_seed=42, verbose=20, allow_writing_files=False)
    ranker.fit(pool)
    args.output.mkdir(parents=True, exist_ok=False)
    ranker.save_model(str(args.output / "model.cbm"))
    scores = (val.select(KEYS).explode("item_id").with_columns(
        pl.Series("score", ranker.predict(validation).astype(np.float32)), pl.Series("supported", validation[:, 2] > 0)))
    scores.write_parquet(args.output / "val_predictions.parquet")
    atomic_text(args.output / "metrics.json", json.dumps(dict(train_feeds=len(train), val_feeds=len(val),
                coverage=float((validation[:, 2] > 0).mean()), seconds=round(perf_counter() - start, 2)), indent=2))


def compare(args):
    from accelerator import select_gpu
    from ensemble import Slates, add_model_scores

    gpu = select_gpu(args.gpu)
    paths = dict(qsm=args.retrain.parent / "qsm_val_predictions.parquet",
                 gru4=args.retrain.parent / "gru4-published_val_predictions.parquet",
                 gru3=args.retrain / "gru-din-v3/val_predictions.parquet",
                 lgbm=args.retrain / "lightgbm-v3/val_predictions.parquet",
                 end=args.output / "end4rec/val_predictions.parquet", graph=args.output / "edg4rec/val_predictions.parquet")
    labels = (pl.scan_parquet(str(args.data / "val/impressions/part-*.parquet"))
              .select(*KEYS, "timestamp", "target").explode("item_id", "target").collect())
    sources = {key: pl.read_parquet(path) for key, path in paths.items()}
    frame = add_model_scores(labels, list(sources.values()), list(sources)).join(
        sources["graph"].select(*KEYS, "supported"), on=KEYS, validate="1:1")
    frame = frame.with_columns(pl.when("supported").then(pl.col("graph")).alias("graph_supported"))
    frame = frame.with_columns(*[zscore(name).alias(f"z_{name}") for name in (*V4_WEIGHTS, "end")],
                               zscore("graph_supported").fill_null(0).alias("z_graph"))
    frame = frame.with_columns(sum(weight * pl.col(f"z_{name}") for name, weight in V4_WEIGHTS.items()).alias("control"),
                               (.30 * (pl.col("z_end") - pl.col("z_gru4"))).alias("end_delta"))
    early = Slates(frame.filter(pl.col("timestamp") < CUTOFF), "cuda")
    early_control = early.pack("control")
    grids = {"v4-end4rec": ("end_delta", [i / 20 for i in range(21)]),
             "v4-edg4rec": ("z_graph", [i / 100 for i in range(31)])}
    alternatives = {"v4-reconstructed-base": pl.col("control")}
    selected = {}
    for name, (feature, grid) in grids.items():
        increment = early.pack(feature)
        selected[name] = max(grid, key=lambda weight: early.metric(early_control + weight * increment))
        alternatives[name] = pl.col("control") + selected[name] * pl.col(feature)
    del early, early_control, increment
    rows = []
    for name, expr in alternatives.items():
        scores = frame.select(*KEYS, "target", "timestamp", expr.alias("score"))
        rows.append(dict(model=name, tuning=ndcg(scores.filter(pl.col("timestamp") < CUTOFF)),
                         holdout=ndcg(scores.filter(pl.col("timestamp") >= CUTOFF)), val=ndcg(scores)))
    rows.sort(key=lambda row: row["holdout"]["ndcg"], reverse=True)
    base_report = json.loads((args.retrain / "report.json").read_text(encoding="utf-8"))
    fine = next((row for row in base_report["models"] if row["model"] == "v4-fine-tune-early"), None)
    graph_report = json.loads((args.output / "edg4rec/metrics.json").read_text(encoding="utf-8"))
    atomic_text(args.output / "report.json", json.dumps(dict(complete=True, train_feeds=graph_report["train_feeds"],
                val_feeds=labels["feed_id"].n_unique(), cutoff=CUTOFF, models=rows, weights=selected,
                graph_coverage=graph_report["coverage"], fine_tune_reference_late=fine["holdout"]["ndcg"] if fine else None,
                gpu={"name": gpu.name}, settings={"epochs": args.epochs, "iterations": args.iterations}), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("end", "graph", "compare"))
    for name in ("data", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("v4-code", "retrain"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--train-feeds", default=200000, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--iterations", default=200, type=int)
    args = parser.parse_args()
    if args.stage == "end" and args.v4_code is None or args.stage == "compare" and args.retrain is None:
        parser.error("end requires --v4-code and compare requires --retrain")
    {"end": train_end, "graph": train_graph, "compare": compare}[args.stage](args)


if __name__ == "__main__":
    main()
