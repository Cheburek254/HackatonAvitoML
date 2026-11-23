"""Train a CUDA neural ranker with independent recent and earlier history encoders."""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from accelerator import GpuNDCG, cuda_tensor, select_gpu
from evaluation import KEYS, attach_scores, peak_ram_mib
from ranker import FEATURES, candidate_batches, feed_batches, open_embeddings


class Dataset:
    def __init__(self, split, limit, labeled):
        tensor = cuda_tensor()
        self.rows = pl.concat(list(candidate_batches(split, max_feeds=limit, labeled=labeled)))
        columns = ["feed_id", "timestamp", "item_id__embedding_idx", "history__item_id__embedding_idx"]
        self.feeds = pl.concat(list(feed_batches(split, columns, limit)))
        embeddings = open_embeddings(split)
        indices = self.feeds["item_id__embedding_idx"].explode().to_numpy()
        self.items = np.zeros((len(indices), embeddings.shape[1]), dtype=np.int8)
        self.items[indices >= 0] = embeddings[indices[indices >= 0]]
        history = np.full((len(self.feeds), 64), -1, dtype=np.int32)
        for row, values in enumerate(self.feeds["history__item_id__embedding_idx"]):
            history[row, :min(len(values), 64)] = values[:64].to_numpy()
        summaries = []
        for batch in np.array_split(history, (len(history) + 127) // 128):
            vectors = np.zeros((*batch.shape, embeddings.shape[1]), dtype=np.int8)
            vectors[batch >= 0] = embeddings[batch[batch >= 0]]
            raw = tensor(vectors, device="CUDA").cast("float")
            raw = raw / raw.square().sum(axis=2, keepdim=True).sqrt().maximum(1e-6)
            windows = []
            for window in (slice(0, 8), slice(8, 64)):
                total = raw[:, window].sum(axis=1)
                windows.append(total / total.square().sum(axis=1, keepdim=True).sqrt().maximum(1e-6))
            summaries.append(tensor.stack(*windows, dim=1).numpy())
        self.profiles = np.concatenate(summaries)
        self.feed_rows = np.repeat(np.arange(len(self.feeds)), self.feeds["item_id__embedding_idx"].list.len().to_numpy())
        self.floats = np.nan_to_num(self.rows.select(FEATURES[1:]).to_numpy().astype(np.float32), nan=0.0)
        self.floats[:, FEATURES[1:].index("history_length")] /= 256
        self.floats[:, FEATURES[1:].index("recent_category_affinity")] /= 8
        self.categories = self.rows["category_id"].to_numpy().astype(np.uint8).astype(np.int32)
        self.targets = self.rows["target"].to_numpy().astype(np.float32) if labeled else None

    def inputs(self, indices):
        tensor = cuda_tensor()
        return tuple(tensor(array, device="CUDA") for array in (
            self.floats[indices], self.categories[indices], self.items[indices], self.profiles[self.feed_rows[indices]],
        ))


class DualHistory:
    def __init__(self, input_dim):
        from tinygrad import nn

        self.category = nn.Embedding(256, 8)
        self.item = nn.Linear(128, 24)
        self.short = nn.Linear(128, 24)
        self.long = nn.Linear(128, 24)
        self.mixer = nn.Linear(input_dim + 8 + 72, 64)
        self.output = nn.Linear(64, 1)

    def __call__(self, floats, category, items, profiles):
        from tinygrad import Tensor

        items = items.cast("float")
        items = items / items.square().sum(axis=1, keepdim=True).sqrt().maximum(1e-6)
        candidate = self.item(items).tanh()
        recent = self.short(profiles[:, 0]).tanh()
        earlier = self.long(profiles[:, 1]).tanh()
        joint = Tensor.cat(floats, self.category(category), candidate, candidate * recent, candidate * earlier, dim=1)
        return self.output(self.mixer(joint).relu()).flatten()


def batches(indices, size):
    for start in range(0, len(indices), size):
        yield indices[start:start + size]


def score(model, data, ids, size):
    return np.concatenate([model(*data.inputs(part)).numpy() for part in batches(ids, size)])


def train(args):
    from tinygrad import TinyJit, nn
    from tinygrad.helpers import Context
    from tinygrad.nn.state import get_state_dict, load_state_dict, safe_save

    tensor = cuda_tensor()
    tensor.manual_seed(42)
    rng = np.random.default_rng(42)
    training = Dataset(args.data / "train", args.train_feeds, True)
    val = Dataset(args.data / "val", args.val_feeds, True)
    all_val = np.arange(len(val.rows))
    tune = all_val if args.tune_before is None else np.flatnonzero(
        val.feeds["timestamp"].to_numpy()[val.feed_rows] < args.tune_before
    )
    if args.tune_before is not None and not 0 < len(tune) < len(all_val):
        raise ValueError("--tune-before must divide validation into two nonempty sets")
    metric = GpuNDCG(val.rows[tune])
    model = DualHistory(training.floats.shape[1])
    load_state_dict(model, {key: value.to("CUDA") for key, value in get_state_dict(model).items()}, verbose=False)
    optimizer = nn.optim.AdamW(list(get_state_dict(model).values()), lr=.002)

    @TinyJit
    def step(floats, category, items, profiles, targets, valid):
        optimizer.zero_grad()
        prediction = model(floats, category, items, profiles)
        loss = ((prediction.maximum(0) - prediction * targets + (1 + (-prediction.abs()).exp()).log()) * valid).sum() / valid.sum()
        loss.backward()
        optimizer.step()
        return loss.realize()

    print("CUDA: prepared dual history inputs", flush=True)
    best, best_epoch, best_state, curve = -1, 0, None, []
    block = args.batch_size
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(training.rows))
        with tensor.train(), Context(JIT=2):
            for batch_index, ids in enumerate(batches(order, block), start=1):
                valid = np.arange(block) < len(ids)
                indices = np.resize(ids, block)
                loss = step(*training.inputs(indices), tensor(training.targets[indices], device="CUDA"),
                            tensor(valid.astype(np.float32), device="CUDA"))
                total_batches = (len(order) + block - 1) // block
                if batch_index == 1 or batch_index == total_batches or batch_index % 20 == 0:
                    print(f"dual-history epoch={epoch} batch={batch_index}/{total_batches} loss={loss.item():.6f}", flush=True)
        value = metric(score(model, val, tune, block))["ndcg"]
        curve.append([epoch, value])
        print(f"dual-history epoch={epoch} tuning_ndcg={value:.8f}", flush=True)
        if value > best:
            best, best_epoch = value, epoch
            best_state = {key: value.detach().numpy().copy() for key, value in get_state_dict(model).items()}
    load_state_dict(model, {key: tensor(value, device="CUDA") for key, value in best_state.items()}, verbose=False)
    predictions = val.rows.select(KEYS).with_columns(pl.Series("score", score(model, val, all_val, block)))
    args.output.mkdir(parents=True)
    safe_save(get_state_dict(model), str(args.output / "model.safetensors"))
    predictions.write_parquet(args.output / "val_predictions.parquet")
    report = {"variant": "dual-history", "device": "gpu", "best_epoch": best_epoch, "tuning_ndcg": best,
              "tuning_curve": curve, "peak_ram_mib": peak_ram_mib()}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def predict(args):
    from tinygrad.nn.state import load_state_dict, safe_load

    data = Dataset(args.data / "test", 0, False)
    model = DualHistory(data.floats.shape[1])
    load_state_dict(model, {key: value.to("CUDA") for key, value in safe_load(str(args.model)).items()}, verbose=False)
    predictions = data.rows.select(KEYS).with_columns(pl.Series("score", score(model, data, np.arange(len(data.rows)), args.batch_size)))
    template = pl.read_parquet(args.data / "sample_submission.parquet", columns=KEYS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    attach_scores(template, predictions).write_parquet(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    verbs = parser.add_subparsers(required=True)
    training, scoring = verbs.add_parser("train"), verbs.add_parser("predict")
    training.add_argument("--epochs", type=int, default=5)
    training.add_argument("--train-feeds", type=int, default=0)
    training.add_argument("--val-feeds", type=int, default=0)
    training.add_argument("--tune-before", type=int)
    training.set_defaults(action=train)
    scoring.add_argument("--model", type=Path, required=True)
    scoring.set_defaults(action=predict)
    for command in (training, scoring):
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--gpu-id")
        command.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch_size < 1 or getattr(args, "epochs", 1) < 1 or min(getattr(args, "train_feeds", 0), getattr(args, "val_feeds", 0)) < 0:
        parser.error("Batch size and epochs must be positive, feed limits nonnegative")
    if args.gpu_id:
        select_gpu(args.gpu_id)
    # Tinygrad needs the default set before nn layers allocate their parameters.
    cuda_tensor()
    from tinygrad import Device

    Device.DEFAULT = "CUDA"
    args.action(args)


if __name__ == "__main__":
    main()
