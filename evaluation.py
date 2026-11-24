"""Evaluate and combine candidate scores on one validation set."""

import sys

import numpy as np
import polars as pl

KEYS = ["feed_id", "item_id"]
NDCG = "NDCG:type=Base"


def zscore(column):
    values = pl.col(column)
    deviation = values.std(ddof=0).over("feed_id")
    return ((values - values.mean().over("feed_id")) / (deviation + 1e-6)).cast(pl.Float32)


def peak_ram_mib():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class MemoryCounters(ctypes.Structure):
            _fields_ = [("size", wintypes.DWORD), ("faults", wintypes.DWORD), *[
                (name, ctypes.c_size_t) for name in (
                    "peak", "working", "quota_peak_paged", "quota_paged", "quota_peak_nonpaged",
                    "quota_nonpaged", "pagefile", "peak_pagefile",
                )
            ]]

        counters = MemoryCounters()
        counters.size = ctypes.sizeof(counters)
        current = ctypes.windll.kernel32.GetCurrentProcess
        current.restype = wintypes.HANDLE
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = (wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD)
        if not query(current(), ctypes.byref(counters), counters.size):
            return None
        return round(counters.peak / 2**20, 1)
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(value / (2**20 if sys.platform == "darwin" else 2**10), 1)


def attach_scores(labels, predictions):
    if not labels.height or labels.height != predictions.height:
        raise ValueError("Predictions and labels must contain the same nonempty set of candidates")
    if predictions["score"].null_count() or not predictions["score"].is_finite().all():
        raise ValueError("Scores must be finite")
    joined = labels.join(predictions.select(*KEYS, "score"), on=KEYS, how="left", validate="1:1")
    if joined["score"].null_count():
        raise ValueError("Prediction keys differ from evaluation keys")
    return joined


def ndcg(frame):
    ordered = frame.sort("feed_id")
    feeds = ordered["feed_id"].to_numpy()
    target = ordered["target"].to_numpy()
    score = ordered["score"].to_numpy()
    if not len(feeds):
        raise ValueError("Cannot evaluate an empty validation split")
    boundaries = np.r_[0, np.flatnonzero(feeds[1:] != feeds[:-1]) + 1, len(feeds)]
    discounts = 1 / np.log2(np.arange(np.diff(boundaries).max()) + 2)
    totals = np.zeros(3)
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        labels = target[start:end]
        ranked = labels[np.lexsort((labels, -score[start:end]))]
        for index, cutoff in enumerate((end - start, 10, 30)):
            size = min(cutoff, end - start)
            ideal = discounts[:min(size, int(labels.sum()))].sum()
            totals[index] += ranked[:size] @ discounts[:size] / ideal if ideal else 1
    return dict(zip(("ndcg", "ndcg_at_10", "ndcg_at_30"), totals / (len(boundaries) - 1), strict=True))


def blend(left, right, weight, offset=60):
    def reciprocal(frame, name):
        rank = pl.col("score").rank("average", descending=True).over("feed_id")
        return frame.select(*KEYS, (1 / (offset + rank)).alias(name))

    ranks = reciprocal(left, "left").join(reciprocal(right, "right"), on=KEYS, validate="1:1")
    if ranks.height != left.height or ranks.height != right.height:
        raise ValueError("Models must rank the same candidates")
    return ranks.select(
        *KEYS, (weight * pl.col("left") + (1 - weight) * pl.col("right")).cast(pl.Float32).alias("score")
    )
