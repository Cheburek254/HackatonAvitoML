"""Fit three-family rank fusion and a repetition penalty on early validation."""

import numpy as np
import polars as pl
import torch

from evaluation import KEYS, attach_scores, zscore


FAMILIES = {
    "gbdt": ("strong-gbdt",),
    "listwise": ("avito-joint", "set-transformer", "neural-pairwise"),
    "sequence": ("gru-din", "sasrec"),
}


def add_model_scores(frame, predictions, columns):
    for name, prediction in zip(columns, predictions, strict=True):
        checked = attach_scores(frame.select(KEYS), prediction).rename({"score": name})
        frame = frame.join(checked, on=KEYS, validate="1:1")
    return frame


class Slates:
    def __init__(self, frame, device):
        # Stable ties rank all negatives ahead of positives, as in Avito's scorer.
        self.frame = frame.sort(["feed_id", "target", "item_id"])
        feeds = self.frame["feed_id"].to_numpy()
        starts = np.r_[0, np.flatnonzero(feeds[1:] != feeds[:-1]) + 1, len(feeds)]
        lengths = np.diff(starts)
        if not len(feeds):
            raise ValueError("Need nonempty labeled feeds")
        size = int(lengths.max())
        columns = np.arange(size)
        valid = columns < lengths[:, None]
        self.rows = np.where(valid, starts[:-1, None] + columns, 0)
        self.device = device
        self.valid = torch.as_tensor(valid, device=device)
        self.targets = self.pack("target").masked_fill(~self.valid, 0)
        self.discount = 1 / torch.log2(torch.arange(size, device=device, dtype=torch.float32) + 2)
        self.positives = self.targets.sum(dim=1, dtype=torch.long)
        self.cumulative = torch.cat([self.discount.new_zeros(1), self.discount.cumsum(0)])
        self.ideal = self.cumulative[self.positives]

    def pack(self, column):
        return torch.as_tensor(self.frame[column].to_numpy()[self.rows], device=self.device)

    def per_feed_ndcg(self, values):
        order = values.masked_fill(~self.valid, -torch.inf).argsort(dim=1, descending=True, stable=True)
        dcg = (self.targets.gather(1, order) * self.discount).sum(dim=1)
        return torch.where(self.ideal > 0, dcg / self.ideal.clamp_min(1e-8), 1)

    def metric(self, values):
        return float(self.per_feed_ndcg(values).mean().item())

    def metrics(self, values):
        order = values.masked_fill(~self.valid, -torch.inf).argsort(dim=1, descending=True, stable=True)
        gain = self.targets.gather(1, order) * self.discount
        result = {}
        for name, cutoff in (("ndcg", values.shape[1]), ("ndcg_at_10", 10), ("ndcg_at_30", 30)):
            ideal = self.cumulative[self.positives.clamp(max=cutoff)]
            dcg = gain[:, :cutoff].sum(dim=1)
            result[name] = float(torch.where(ideal > 0, dcg / ideal.clamp_min(1e-8), 1).mean().item())
        return result

    def predictions(self, values):
        return self.frame.select(KEYS).with_columns(
            pl.Series("score", values[self.valid].cpu().numpy(), dtype=pl.Float32)
        )


def fit_weights(slates, signals, trials):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    parameters = [f"w_{index}" for index in range(len(signals))]
    for values in ([1] * len(signals), *np.eye(len(signals)).tolist()):
        study.enqueue_trial(dict(zip(parameters, values, strict=True)))

    def objective(trial):
        weights = [trial.suggest_float(parameter, 0, 1) for parameter in parameters]
        total = sum(weights)
        return slates.metric(sum(weight * score for weight, score in zip(weights, signals, strict=True)) / max(total, 1e-8))

    study.optimize(objective, n_trials=max(trials, len(parameters) + 1))
    weights = np.array([study.best_params[name] for name in parameters])
    return (weights / weights.sum()).tolist()


def weighted(weights, scores):
    return sum(weight * score for weight, score in zip(weights, scores, strict=True))


def diversify(slates, scores, penalty, depth=20):
    if penalty == 0:
        return scores
    categories = slates.pack("category_id").long().remainder(256)
    valid = slates.valid
    length = valid.sum(dim=1, keepdim=True).clamp_min(1)
    mean = scores.masked_fill(~valid, 0).sum(dim=1, keepdim=True) / length
    centered = (scores - mean).masked_fill(~valid, 0)
    deviation = (centered.square().sum(dim=1, keepdim=True) / length).sqrt()
    normalized = centered / (deviation + 1e-6)
    ceiling = normalized.masked_fill(~valid, -torch.inf).amax(dim=1, keepdim=True)
    remaining = valid.clone()
    output = normalized.clone()
    counts = scores.new_zeros((len(scores), 256))
    for rank in range(min(depth, scores.shape[1])):
        charge = counts.gather(1, categories)
        best = (normalized - penalty * charge).masked_fill(~remaining, -torch.inf).argmax(dim=1, keepdim=True)
        selected = remaining.gather(1, best)
        category = categories.gather(1, best)
        counts.scatter_add_(1, category, (selected & (category != 0)).to(scores.dtype))
        output.scatter_(1, best, torch.where(selected, ceiling + depth - rank, output.gather(1, best)))
        remaining.scatter_(1, best, False)
    return output


def fit_families(labels, cutoff, results, files, device, trials):
    by_model = {row["model"]: row for row in results}
    models = [
        max((model for model in candidates if model in files), key=lambda name: by_model[name]["tuning"]["ndcg"])
        for candidates in FAMILIES.values()
    ]
    print(f"Fitting three-family blends: {', '.join(models)}", flush=True)
    raw = add_model_scores(labels, [pl.read_parquet(files[name]) for name in models], models)
    frame = raw.with_columns(
        *[zscore(name).alias(f"z_{name}") for name in models],
        *[(1 / (60 + pl.col(name).rank("average", descending=True).over("feed_id"))).cast(pl.Float32)
          .alias(f"rrf_{name}") for name in models],
    )
    early = Slates(frame.filter(pl.col("timestamp") < cutoff), device)
    full = Slates(frame, device)
    blends = []
    for scheme in ("z", "rrf"):
        columns = [f"{scheme}_{name}" for name in models]
        signals = [early.pack(column) for column in columns]
        weights = fit_weights(early, signals, trials)
        values = weighted(weights, [full.pack(column) for column in columns])
        early_values = weighted(weights, signals)
        metadata = {"members": models, "weights": weights, "optuna_trials": trials}
        blends.append((f"three-{scheme}", full.predictions(values), metadata))
        if scheme == "z":
            z_scores, z_early = values, early_values
    print("Tuning category repetition penalty", flush=True)
    penalties = (0, .025, .05, .10, .20, .40)
    penalty = max(penalties, key=lambda value: early.metric(diversify(early, z_early, value)))
    blends.append(("three-z-mmr", full.predictions(diversify(full, z_scores, penalty)),
                   {"base": "three-z", "category_penalty": penalty, "depth": 20}))
    return blends
