"""Train candidate-set and history-aware neural rankers on Avito feeds."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset

from evaluation import KEYS, attach_scores, ndcg, peak_ram_mib
from ranker import BASE_FEATURES, CANDIDATE_COLUMNS, HISTORY_COLUMNS, feed_batches, open_embeddings

VARIANTS = ("transformer", "transformer-listmle", "avito-joint", "neural-din", "bst", "gru-din", "sasrec",
            "set-transformer", "neural-pairwise")
HISTORY_VARIANTS = {"avito-joint", "neural-din", "bst", "gru-din", "sasrec", "neural-pairwise"}


def id_tensor(values):
    return torch.from_numpy(np.asarray(values, dtype=np.int8).astype(np.uint8).astype(np.int64))


class Feeds(Dataset):
    def __init__(self, split, max_feeds, labeled, window, uses_history):
        columns = ["feed_id", "timestamp", "user_embedding", *CANDIDATE_COLUMNS]
        if labeled:
            columns.append("target")
        if uses_history:
            columns.extend([*HISTORY_COLUMNS, "history__event_type", "history__event_source"])
        self.feeds = pl.concat(list(feed_batches(split, columns, max_feeds)))
        self.embeddings = open_embeddings(split)
        self.labeled = labeled
        self.window = window if uses_history else 0

    def __len__(self):
        return self.feeds.height

    def vectors(self, indices):
        indices = np.asarray(indices, dtype=np.intp)
        valid = indices >= 0
        values = np.zeros((len(indices), 128), dtype=np.float32)
        values[valid] = self.embeddings[indices[valid]]
        return F.normalize(torch.from_numpy(values), dim=-1), torch.from_numpy(~valid)

    def __getitem__(self, index):
        feed = self.feeds.row(index, named=True)
        items, missing = self.vectors(feed["item_id__embedding_idx"])
        floats = torch.tensor([feed[name] for name in BASE_FEATURES], dtype=torch.float32).T.nan_to_num()
        candidates = torch.cat([floats, items, missing[:, None]], dim=-1)
        user = F.normalize(torch.tensor(feed["user_embedding"], dtype=torch.float32), dim=0)
        if self.window:
            # Reverse only the available prefix so recurrent layers see chronological order.
            indices = feed["history__item_id__embedding_idx"][:self.window][::-1]
            vectors, history_missing = self.vectors(indices)
            events = torch.stack([id_tensor(feed[name][:self.window][::-1]) for name in
                                  ("history__category_id", "history__event_type", "history__event_source")], dim=-1)
            history_features = torch.cat([vectors, history_missing[:, None]], dim=-1)
        else:
            history_features, events = torch.empty((0, 129)), torch.empty((0, 3), dtype=torch.long)
        target = torch.tensor(feed["target"], dtype=torch.float32) if self.labeled else torch.zeros(len(items))
        return feed["feed_id"], feed["item_id"], candidates, id_tensor(feed["category_id"]), user, history_features, events, target


def collate(rows):
    feeds, items, candidates, categories, users, history, events, target = zip(*rows, strict=True)
    mask = torch.arange(max(map(len, items)))[None, :] >= torch.tensor(list(map(len, items)))[:, None]
    history_mask = torch.arange(max(map(len, history)))[None, :] >= torch.tensor(list(map(len, history)))[:, None]
    inputs = (pad_sequence(candidates, batch_first=True), pad_sequence(categories, batch_first=True), torch.stack(users),
              mask, pad_sequence(history, batch_first=True), pad_sequence(events, batch_first=True), history_mask)
    return feeds, items, inputs, pad_sequence(target, batch_first=True)


class Ranker(nn.Module):
    def __init__(self, variant, width=64):
        super().__init__()
        self.variant = variant
        self.category = nn.Embedding(256, 8, padding_idx=0)
        self.candidate = nn.Linear(142, width)
        self.user = nn.Linear(128, width)
        self.output = nn.Linear(width, 1)
        if variant == "set-transformer":
            self.inducing = nn.Parameter(torch.randn(8, width) / width ** .5)
            self.summarize = nn.MultiheadAttention(width, 4, batch_first=True)
            self.recover = nn.MultiheadAttention(width, 4, batch_first=True)
            self.set_norm = nn.LayerNorm(width)
        else:
            self.encoder = nn.TransformerEncoder(self.layer(width), 2, enable_nested_tensor=False)
        if variant in HISTORY_VARIANTS:
            self.event_type = nn.Embedding(256, 8, padding_idx=0)
            self.event_source = nn.Embedding(256, 8, padding_idx=0)
            self.position = nn.Embedding(256, width)
            self.history = nn.Linear(153, width)
            self.empty_history = nn.Parameter(torch.zeros(1, 1, width))
        if variant in HISTORY_VARIANTS - {"avito-joint", "sasrec"}:
            self.history_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        if variant in HISTORY_VARIANTS - {"avito-joint"}:
            self.history_norm = nn.LayerNorm(width)
        if variant in {"bst", "sasrec"}:
            self.history_encoder = nn.TransformerEncoder(self.layer(width), 1, enable_nested_tensor=False)
        if variant == "gru-din":
            self.history_encoder = nn.GRU(width, width, batch_first=True)

    @staticmethod
    def layer(width):
        return nn.TransformerEncoderLayer(width, 4, width * 2, dropout=.1, batch_first=True, norm_first=True)

    def history_tokens(self, history, events, padding):
        inputs = torch.cat([history, self.category(events[..., 0]), self.event_type(events[..., 1]),
                            self.event_source(events[..., 2])], dim=-1)
        lengths = (~padding).sum(dim=1)
        positions = (lengths[:, None] - 1 - torch.arange(history.shape[1], device=history.device)).clamp_min(0)
        tokens = self.history(inputs) + self.position(positions)
        tokens = torch.cat([self.empty_history.expand(tokens.shape[0], -1, -1), tokens], dim=1)
        mask = F.pad(padding, (1, 0), value=False)
        return tokens, mask, lengths

    def condition_on_history(self, candidates, history, events, padding):
        tokens, mask, lengths = self.history_tokens(history, events, padding)
        if self.variant == "gru-din":
            tokens, _ = self.history_encoder(tokens)
        if self.variant in {"bst", "sasrec"}:
            causal = torch.ones((tokens.shape[1], tokens.shape[1]), dtype=torch.bool, device=tokens.device).triu(1) if self.variant == "sasrec" else None
            tokens = self.history_encoder(tokens, mask=causal, src_key_padding_mask=mask)
        if self.variant == "sasrec":
            profile = tokens[torch.arange(len(lengths), device=tokens.device), lengths][:, None, :]
            return self.history_norm(candidates + profile * candidates)
        context, _ = self.history_attention(candidates, tokens, tokens, key_padding_mask=mask, need_weights=False)
        return self.history_norm(candidates + context)

    def forward(self, features, category, user, padding, history, events, history_padding):
        items = self.candidate(torch.cat([features, self.category(category)], dim=-1))
        if self.variant == "avito-joint":
            history, history_mask, _ = self.history_tokens(history, events, history_padding)
            prefix = torch.cat([self.user(user)[:, None, :], history], dim=1)
            tokens = torch.cat([prefix, items], dim=1)
            mask = torch.cat([F.pad(history_mask, (1, 0), value=False), padding], dim=1)
            attention_mask = torch.zeros((tokens.shape[1], tokens.shape[1]), dtype=torch.bool, device=tokens.device)
            attention_mask[:prefix.shape[1], prefix.shape[1]:] = True
            tokens = self.encoder(tokens, mask=attention_mask, src_key_padding_mask=mask)
            return self.output(tokens[:, prefix.shape[1]:]).squeeze(-1)
        if self.variant in HISTORY_VARIANTS:
            items = self.condition_on_history(items, history, events, history_padding)
        tokens = torch.cat([self.user(user)[:, None, :], items], dim=1)
        mask = F.pad(padding, (1, 0), value=False)
        if self.variant == "set-transformer":
            inducing = self.inducing.unsqueeze(0).expand(len(tokens), -1, -1)
            context, _ = self.summarize(inducing, tokens, tokens, key_padding_mask=mask, need_weights=False)
            refined, _ = self.recover(tokens, context, context, need_weights=False)
            tokens = self.set_norm(tokens + refined)
        else:
            tokens = self.encoder(tokens, src_key_padding_mask=mask)
        return self.output(tokens[:, 1:]).squeeze(-1)


def loss_for(variant, scores, target, padding):
    if variant in {"transformer-listmle", "avito-joint"}:
        # Randomize equally relevant candidates before the stable sort by label.
        order = (target + torch.rand_like(target)).masked_fill(padding, -1).argsort(dim=1, descending=True)
        ordered = scores.gather(1, order)
        valid = ~padding.gather(1, order)
        remaining = torch.logcumsumexp(ordered.masked_fill(~valid, -torch.finfo(scores.dtype).max).flip(1), dim=1).flip(1)
        return ((remaining - ordered).masked_fill(~valid, 0).sum(dim=1) / valid.sum(dim=1)).mean()
    if variant != "neural-pairwise":
        loss = F.binary_cross_entropy_with_logits(scores, target, reduction="none").masked_fill(padding, 0)
        return (loss.sum(dim=1) / (~padding).sum(dim=1)).mean()
    ranks = scores.detach().masked_fill(padding, -torch.inf).argsort(dim=1, descending=True).argsort(dim=1)
    discounts = 1 / torch.log2(torch.arange(scores.shape[1], device=scores.device) + 2)
    actual = discounts[ranks]
    ideal = discounts.cumsum(0)[target.sum(dim=1).long().clamp_min(1) - 1]
    pairs = (target[:, :, None] > target[:, None, :]) & ~padding[:, None, :]
    weights = (actual[:, :, None] - actual[:, None, :]).abs() / ideal[:, None, None]
    losses = F.softplus(scores[:, None, :] - scores[:, :, None]) * weights
    return (losses.masked_fill(~pairs, 0).sum(dim=(1, 2)) / pairs.sum(dim=(1, 2)).clamp_min(1)).mean()


def loader(dataset, batch_size, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def scores(model, batches, device):
    model.eval()
    chunks = []
    with torch.inference_mode():
        for feeds, items, inputs, _ in batches:
            values = model(*(tensor.to(device) for tensor in inputs)).cpu()
            chunks.append(pl.DataFrame({"feed_id": [feed for feed, keys in zip(feeds, items, strict=True) for _ in keys],
                                        "item_id": [key for keys in items for key in keys],
                                        "score": values[~inputs[3]].numpy()},
                                       schema={"feed_id": pl.Int32, "item_id": pl.Int32, "score": pl.Float32}))
    return pl.concat(chunks)


def labels(dataset):
    return dataset.feeds.select(*KEYS, "target").explode("item_id", "target")


def train_epoch(model, optimizer, batches, device, variant, epoch):
    start = perf_counter()
    model.train()
    for step, (_, _, inputs, target) in enumerate(batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        inputs = tuple(tensor.to(device) for tensor in inputs)
        loss = loss_for(variant, model(*inputs), target.to(device), inputs[3])
        loss.backward()
        optimizer.step()
        if step == 1 or step == len(batches) or step % 100 == 0:
            print(f"{variant} epoch={epoch} batch={step}/{len(batches)} loss={loss.detach().item():.6f}", flush=True)
    seconds = round(perf_counter() - start, 2)
    print(f"{variant} epoch={epoch} training_seconds={seconds:.2f}", flush=True)
    return seconds


def train(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(42)
    use_history = args.variant in HISTORY_VARIANTS
    train_set = Feeds(args.data / "train", args.train_feeds, True, args.history_window, use_history)
    val_set = Feeds(args.data / "val", args.val_feeds, True, args.history_window, use_history)
    train_batches = loader(train_set, args.batch_size, shuffle=True)
    val_labels = labels(val_set)
    tune_labels, tune_batches = val_labels, loader(val_set, args.batch_size)
    if args.tune_before is not None:
        indices = np.flatnonzero(val_set.feeds["timestamp"].to_numpy() < args.tune_before)
        if not 0 < len(indices) < len(val_set):
            raise ValueError("--tune-before must divide the selected validation feeds into two nonempty sets")
        tune_labels = val_labels.filter(pl.col("feed_id").is_in(val_set.feeds["feed_id"][indices]))
        tune_batches = loader(Subset(val_set, indices.tolist()), args.batch_size)
    model = Ranker(args.variant).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    best, best_epoch, best_state, curve = -1, 0, None, []
    epoch_seconds = []
    tuning_slates = None
    if args.device == "cuda":
        from ensemble import Slates

        tuning_slates = Slates(tune_labels, args.device)
    for epoch in range(1, args.epochs + 1):
        epoch_seconds.append(train_epoch(model, optimizer, train_batches, args.device, args.variant, epoch))
        predicted = scores(model, tune_batches, args.device)
        if tuning_slates is None:
            metric = ndcg(attach_scores(tune_labels, predicted))["ndcg"]
        else:
            aligned = attach_scores(tuning_slates.frame, predicted).sort(["feed_id", "target", "item_id"])
            values = torch.as_tensor(aligned["score"].to_numpy()[tuning_slates.rows], device=args.device)
            metric = tuning_slates.metric(values)
        curve.append([epoch, metric])
        print(f"{args.variant} epoch={epoch} tuning_ndcg={metric:.6f}", flush=True)
        if metric > best:
            best, best_epoch = metric, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    predictions = scores(model, loader(val_set, args.batch_size), args.device)
    report = {f"val_{name}": value for name, value in ndcg(attach_scores(val_labels, predictions)).items()}
    report.update(variant=args.variant, best_epoch=best_epoch, tune_before=args.tune_before, tuning_curve=curve,
                  epoch_seconds=epoch_seconds, peak_ram_mib=peak_ram_mib())
    args.output.mkdir(parents=True)
    torch.save({"state_dict": best_state, "variant": args.variant, "history_window": args.history_window}, args.output / "model.pt")
    predictions.write_parquet(args.output / "val_predictions.parquet")
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def oof(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    dataset = Feeds(args.data / "train", args.train_feeds, True, args.history_window, args.variant in HISTORY_VARIANTS)
    timestamps = dataset.feeds["timestamp"].to_numpy()
    unique = np.unique(timestamps)
    if len(unique) < 3:
        raise ValueError("OOF needs at least three distinct timestamps")
    bounds = (*[int(unique[len(unique) * index // 3]) for index in (1, 2)], int(unique[-1]) + 1)
    chunks = []
    for first, end in zip(bounds[:-1], bounds[1:], strict=True):
        fit_rows = np.flatnonzero(timestamps < first)
        score_rows = np.flatnonzero((timestamps >= first) & (timestamps < end))
        print(f"OOF {args.variant}: {len(fit_rows)} past feeds, {len(score_rows)} future feeds", flush=True)
        torch.manual_seed(42)
        model = Ranker(args.variant).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        training = loader(Subset(dataset, fit_rows.tolist()), args.batch_size, shuffle=True)
        for epoch in range(1, args.epochs + 1):
            train_epoch(model, optimizer, training, args.device, args.variant, epoch)
        predictions = loader(Subset(dataset, score_rows.tolist()), args.batch_size)
        chunks.append(scores(model, predictions, args.device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pl.concat(chunks).write_parquet(args.output)


def predict(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    variant = checkpoint["variant"]
    model = Ranker(variant).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    test = Feeds(args.data / "test", 0, False, checkpoint["history_window"], variant in HISTORY_VARIANTS)
    template = pl.read_parquet(args.data / "sample_submission.parquet", columns=KEYS)
    submission = attach_scores(template, scores(model, loader(test, args.batch_size), args.device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.write_parquet(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    training = commands.add_parser("train")
    training.add_argument("--variant", choices=VARIANTS, default="neural-din")
    training.add_argument("--epochs", type=int, default=5)
    training.add_argument("--train-feeds", type=int, default=50_000)
    training.add_argument("--val-feeds", type=int, default=0)
    training.add_argument("--history-window", type=int, default=32)
    training.add_argument("--tune-before", type=int)
    training.set_defaults(run=train)
    oof_command = commands.add_parser("oof")
    oof_command.add_argument("--variant", choices=VARIANTS, required=True)
    oof_command.add_argument("--epochs", type=int, default=2)
    oof_command.add_argument("--train-feeds", type=int, default=50_000)
    oof_command.add_argument("--history-window", type=int, default=32)
    oof_command.set_defaults(run=oof)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--model", type=Path, required=True)
    prediction.set_defaults(run=predict)
    for command in (training, oof_command, prediction):
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--batch-size", type=int, default=16)
        command.add_argument("--threads", type=int, default=4)
        command.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if min(args.batch_size, args.threads, getattr(args, "epochs", 1)) < 1:
        parser.error("Batch size, threads and epochs must be positive")
    if not 1 <= getattr(args, "history_window", 1) <= 256 or min(getattr(args, "train_feeds", 0), getattr(args, "val_feeds", 0)) < 0:
        parser.error("History window must be 1–256 and feed limits must be nonnegative")
    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA PyTorch is required for --device cuda")
        torch.set_float32_matmul_precision("high")
    args.run(args)


if __name__ == "__main__":
    main()
