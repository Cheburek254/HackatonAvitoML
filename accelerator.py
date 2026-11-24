"""NVIDIA selection and CUDA tensor computations used by the benchmark."""

import csv
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl


@dataclass(frozen=True)
class GPU:
    index: str
    uuid: str
    name: str
    memory: str


def nvidia_gpus():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [GPU(*(field.strip() for field in row)) for row in csv.reader(result.stdout.splitlines())]


def select_gpu(selection):
    selected = next((gpu for gpu in nvidia_gpus() if selection in (gpu.uuid, gpu.index)), None)
    if selected is None:
        raise ValueError(f"NVIDIA GPU {selection!r} unavailable; see nvidia-smi")
    os.environ["CUDA_VISIBLE_DEVICES"] = selected.uuid
    return selected


@lru_cache(maxsize=1)
def cuda_tensor():
    if os.name == "nt" and not os.environ.get("CUDA_PATH", "").lower().endswith("nvcuda.dll"):
        driver = Path(os.environ["SystemRoot"]) / "System32" / "nvcuda.dll"
        os.environ["CUDA_PATH"] = str(driver)
    os.environ["CUDA_PTX"] = "1"
    os.environ.setdefault("CACHELEVEL", "0")
    try:
        from tinygrad import Tensor

        Tensor([1, 2], device="CUDA").sum().item()
    except (ImportError, OSError, RuntimeError, AttributeError) as error:
        raise RuntimeError("CUDA initialization failed. Verify the NVIDIA driver and install tinygrad==0.12.0.") from error
    return Tensor


def category_coefficients(inputs, targets, indices):
    if has_torch_cuda() and not cuda_tensor.cache_info().currsize:
        import torch

        design = torch.as_tensor(inputs, device="cuda")
        target = torch.as_tensor(targets, device="cuda")
        codes = torch.as_tensor(indices.astype(np.int64), device="cuda")
        width = inputs.shape[1]
        weights = design.new_zeros(width + 256)
        for _ in range(80):
            residual = (design @ weights[:width] + weights[width:][codes]).sigmoid() - target
            by_category = torch.bincount(codes, weights=residual, minlength=256)
            gradient = torch.cat([design.T @ residual, by_category]) / len(targets)
            weights -= .6 * (gradient + .002 * weights)
        return weights.cpu().numpy()
    inputs = np.column_stack([inputs, np.eye(256, dtype=np.float32)[indices]])
    tensor = cuda_tensor()
    design = tensor(inputs, device="CUDA").realize()
    target = tensor(targets, device="CUDA").realize()
    weights = tensor.zeros(inputs.shape[1], device="CUDA")
    for _ in range(80):
        residual = (design @ weights).sigmoid() - target
        weights = (weights - .6 * ((design.T @ residual) / len(targets) + .002 * weights)).realize()
    return weights.numpy()


class GpuNDCG:
    """Compute pessimistic tie ranks and all three NDCG cutoffs on the selected GPU."""

    def __init__(self, frame):
        tensor = cuda_tensor()
        order = frame.with_row_index("_row").sort("feed_id")
        feeds = order["feed_id"].to_numpy()
        if not len(feeds):
            raise ValueError("Cannot evaluate an empty validation split")
        boundaries = np.r_[0, np.flatnonzero(feeds[1:] != feeds[:-1]) + 1, len(feeds)]
        lengths = np.diff(boundaries)
        self.groups = len(lengths)
        self.batches = []
        targets = order["target"].to_numpy()
        row_ids = order["_row"].to_numpy()
        # Group similar lengths to limit padding when the dataset includes long feeds.
        for positions in np.array_split(np.argsort(lengths), (self.groups + 63) // 64):
            size = int(lengths[positions].max())
            columns = np.arange(size)
            valid = columns < lengths[positions, None]
            indices = np.where(valid, boundaries[positions, None] + columns, 0)
            labels = tensor(np.where(valid, targets[indices], 0).astype(np.int32), device="CUDA").realize()
            mask = tensor(valid, device="CUDA").realize()
            ix = tensor.arange(size, device="CUDA").realize()
            discount = (ix.cast("float") + 2).log2().reciprocal().realize()
            self.batches.append((row_ids[indices], labels, mask, ix, discount))

    def __call__(self, scores):
        tensor = cuda_tensor()
        partials = []
        for rows, labels, mask, ix, discount in self.batches:
            values = tensor(np.asarray(scores[rows], dtype=np.float32), device="CUDA")
            candidate, opponent = values.unsqueeze(2), values.unsqueeze(1)
            left, right = labels.unsqueeze(2), labels.unsqueeze(1)
            ahead_on_tie = (right < left) | ((right == left) & (ix[None, :] < ix[:, None]))
            ranks = (((opponent > candidate) | ((opponent == candidate) & ahead_on_tie))
                     & mask.unsqueeze(1)).sum(axis=2)
            gains = (ranks.cast("float") + 2).log2().reciprocal() * labels * mask
            ideal = ix < labels.sum(axis=1, keepdim=True)
            metrics = []
            for cutoff in (len(ix), 10, 30):
                dcg = (gains * (ranks < cutoff)).sum(axis=1)
                idcg = (discount * ideal * (ix < cutoff)).sum(axis=1)
                metrics.append((idcg > 0).where(dcg / idcg.maximum(1e-12), 1).sum())
            partials.append(tensor.stack(*metrics).realize())
        values = (tensor.stack(*partials).sum(axis=0) / self.groups).numpy()
        return dict(zip(("ndcg", "ndcg_at_10", "ndcg_at_30"), map(float, values), strict=True))


@lru_cache(maxsize=1)
def has_torch_cuda():
    try:
        import torch

        return torch.cuda.is_available()
    except (ImportError, OSError):
        return False


def gpu_ndcg_evaluator(frame):
    # Tinygrad uses a driver-created context. Avoid switching to PyTorch in the same process after initializing it.
    if cuda_tensor.cache_info().currsize or not has_torch_cuda():
        return GpuNDCG(frame)
    import torch
    from ensemble import Slates

    indexed = frame.with_row_index("_original_row")
    if "item_id" not in indexed:
        indexed = indexed.with_columns(pl.col("_original_row").alias("item_id"))
    slates = Slates(indexed, "cuda")
    original = slates.frame["_original_row"].to_numpy()[slates.rows]

    def score(values):
        packed = torch.as_tensor(values[original], dtype=torch.float32, device="cuda")
        return slates.metrics(packed)

    return score


def gpu_ndcg(frame: pl.DataFrame):
    return gpu_ndcg_evaluator(frame)(frame["score"].to_numpy())
