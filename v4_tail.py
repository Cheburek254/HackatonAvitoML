"""Compare candidate-specific retrieval from old history with regular GRU v4."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset

from evaluation import KEYS, attach_scores, ndcg, zscore
from local_hypotheses import CONFIRM, EARLY, blend, epoch5_period_signals, metrics, report
from training_queue import ROOT, atomic_text, keep_awake, note
from v4_followup import (ADAPT_EPOCHS, EARLY_SPLIT, LATE_SPLIT, SEED, SIZE, VARIANT,
                         adaptation_data, init_neural, scores_into, write_scores)


NAME = "gru-v4-tail-top8"
TOP_K = 8


class TailFeeds:
    def __init__(self, base, neural):
        self.base, self.neural = base, neural
        self.rows = base.feeds if isinstance(base.feeds, pl.DataFrame) else base.feeds.rows

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        base = self.base[index]
        raw = self.rows.row(index, named=True)
        vectors, missing = self.base.vectors(raw["history__item_id__embedding_idx"][64:256])
        # Keep original positions even when some event embeddings are missing.
        positions = torch.arange(64, 64 + len(vectors))[~missing]
        events = torch.stack([self.neural.id_tensor(raw[name][64:256])[~missing] for name in
                              ("history__category_id", "history__event_type", "history__event_source")], dim=-1)
        extra = (vectors[~missing], events, positions)
        return base, extra


class TailRanker(nn.Module):
    variant = VARIANT

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.query = nn.Linear(128, 32)
        self.key = nn.Linear(152, 32)
        self.value = nn.Linear(152, 32)
        self.position = nn.Embedding(256, 32)
        self.output = nn.Linear(32, 128, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, features, category, user, padding, history, events, history_padding,
                tail, event_ids, positions, tail_padding):
        base = self.base
        items = base.candidate(torch.cat([features, base.category(category)], dim=-1))
        items = base.condition_on_history(items, history, events, history_padding)
        if tail.shape[1]:
            with torch.autocast(device_type=tail.device.type, enabled=False):
                cosine = features[..., -129:-1].float() @ tail.float().transpose(1, 2)
            top_scores, top = cosine.masked_fill(tail_padding[:, None, :], -2.).topk(min(TOP_K, tail.shape[1]))
            valid = ~tail_padding[:, None, :].expand_as(cosine).gather(2, top)
            valid = valid & ~padding[..., None] & (features[..., -1:] == 0)
            tail_info = torch.cat([tail, base.category(event_ids[..., 0]), base.event_type(event_ids[..., 1]),
                                   base.event_source(event_ids[..., 2])], dim=-1)
            row = torch.arange(len(items), device=items.device)[:, None, None]
            position = self.position(positions)
            keys = (self.key(tail_info) + position)[row, top]
            values = (self.value(tail_info) + position)[row, top]
            logits = (self.query(items)[:, :, None] * keys).sum(-1).float() / (32 ** .5) + top_scores
            weights = logits.masked_fill(~valid, -1e4).softmax(-1) * valid
            context = (weights[..., None].to(values.dtype) * values).sum(dim=2)
            items = items + self.output(context)
        tokens = torch.cat([base.user(user)[:, None, :], items], dim=1)
        return base.output(base.encoder(tokens, src_key_padding_mask=F.pad(padding, (1, 0), value=False))
                           [:, 1:]).squeeze(-1)


def tail_loader(neural, data, *, shuffle=False):
    def collate(rows):
        original, tail = zip(*rows, strict=True)
        feeds, items, inputs, targets = neural.collate(original)
        vectors, events, positions = zip(*tail, strict=True)
        lengths = torch.tensor(list(map(len, vectors)))
        mask = torch.arange(int(lengths.max()))[None, :] >= lengths[:, None]
        return feeds, items, (*inputs, pad_sequence(vectors, batch_first=True),
                              pad_sequence(events, batch_first=True), pad_sequence(positions, batch_first=True),
                              mask), targets

    return neural.PrefetchBatches(DataLoader(data, batch_size=16, shuffle=shuffle, collate_fn=collate))


def initial_model(args, neural):
    path = args.reference/"valishinqa-gru-din-v4-300k/gru-din-v4/model.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint["variant"], checkpoint["history_window"]) != (VARIANT, 64):
        raise ValueError("Unexpected GRU v4 checkpoint")
    base = neural.Ranker(VARIANT)
    base.load_state_dict(checkpoint["state_dict"])
    rng = torch.get_rng_state()
    model = TailRanker(base).cuda()
    torch.set_rng_state(rng)
    return model


def train(args):
    neural, _ = init_neural(args)
    fit, selection, stats_feeds = adaptation_data(args, neural, args.previous/"train-feeds.parquet")
    validation = neural.Feeds(args.data/"val", 0, True, 64, True,
                              rich_features=True, v3_features=True, v4_features=True)
    training = tail_loader(neural, TailFeeds(fit, neural), shuffle=True)
    early_idx = np.flatnonzero(validation.feeds["timestamp"].to_numpy() < LATE_SPLIT).tolist()
    if len(early_idx) != 7290 or len(validation) != 14423:
        raise ValueError("Validation does not match published split")
    val_tail = TailFeeds(validation, neural)
    early_batches = tail_loader(neural, Subset(val_tail, early_idx))
    model = initial_model(args, neural)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ADAPT_EPOCHS * len(training))
    for epoch in range(1, ADAPT_EPOCHS + 1):
        saved = args.folder/f"epoch-{epoch}.pt"
        fit.feeds.rng = np.random.default_rng(SEED + epoch)
        if saved.is_file():
            state = torch.load(saved, map_location="cpu", weights_only=True)
            model.load_state_dict(state["state_dict"])
            optimizer.load_state_dict(state["optimizer"])
            schedule.load_state_dict(state["schedule"])
            torch.set_rng_state(state["rng_cpu"])
            torch.cuda.set_rng_state(state["rng_cuda"])
            note(f"Reused epoch {epoch}")
        else:
            neural.train_epoch(model, optimizer, training, "cuda", VARIANT, epoch, schedule)
            temp = saved.with_suffix(".tmp")
            torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                            schedule=schedule.state_dict(), rng_cpu=torch.get_rng_state(),
                            rng_cuda=torch.cuda.get_rng_state(), variant=NAME, history_window=64), temp)
            temp.replace(saved)

    # Compare both models at epoch 6. Tune only the share of its prediction inside period 1.
    early_path = args.folder/"epoch-6-early.parquet"
    write_scores(neural.scores(model, early_batches, "cuda"), early_path)
    frame = epoch5_period_signals(args)
    early = scores_into(frame.filter(pl.col("timestamp") < LATE_SPLIT), early_path, "tail")
    def mix(data, share):
        first = (1-share)*pl.col("fifth") + share*pl.col("tail")
        return blend(data, .5*(first + pl.col("second")))

    folds = [early.filter(mask) for mask in (EARLY, CONFIRM)]
    trials = {share: [ndcg(mix(fold, share))["ndcg"] for fold in folds] for share in (0., .25, .5, 1.)}
    selected = max(trials, key=lambda share: min(new - old for new, old in
                   zip(trials[share], trials[0.], strict=True)))
    challenger = max((share for share in trials if share), key=lambda share: min(new - old for new, old in
                     zip(trials[share], trials[0.], strict=True)))
    atomic_text(args.folder/"selection.json", json.dumps(dict(selected=selected, challenger=challenger,
                                                             early_trials=trials), indent=2))
    path = args.folder/"val_predictions.parquet"
    write_scores(neural.scores(model, tail_loader(neural, val_tail), "cuda"), path)
    frame = attach_scores(frame, pl.read_parquet(path)).with_columns(zscore("score").alias("tail")).rename(
        {"score": "tail_raw"})
    frame = attach_scores(frame, pl.read_parquet(args.previous/"val_predictions.parquet")).rename(
        {"score": "plain_raw"})
    rows = [metrics(frame.with_columns(pl.col("plain_raw").alias("score")), "plain-gru-v4-epoch6"),
            metrics(frame.with_columns(pl.col("tail_raw").alias("score")), NAME),
            metrics(mix(frame, 0.), "fixed-control"),
            metrics(mix(frame, challenger), "tail-blend-challenger")]
    if abs(rows[2]["holdout"]["ndcg"] - .37657187723743313) > 1e-8:
        raise ValueError("Did not reproduce the 0.376572 reference")
    coverage = (pl.scan_parquet(str(args.data/"val/impressions/part-*.parquet"))
                .select("feed_id", "user_id", pl.col("history__item_id__embedding_idx").list.slice(64, 192)
                        .list.eval(pl.element() >= 0).list.any().fill_null(False).alias("has_tail")).collect())
    late = frame.filter(pl.col("timestamp") >= LATE_SPLIT).join(coverage, on="feed_id", validate="m:1")
    from ensemble import Slates
    from pair_blend import paired_interval

    user_ids = late.select("feed_id", "user_id").unique().sort("feed_id")["user_id"].to_numpy()
    def per_feed(scored):
        slate = Slates(scored, "cpu")
        return slate.per_feed_ndcg(slate.pack("score")).numpy().astype(np.float64)

    interval = paired_interval(per_feed(mix(late, challenger)) - per_feed(mix(late, 0.)), user_ids)
    plain_interval = paired_interval(per_feed(late.with_columns(pl.col("tail_raw").alias("score"))) -
                                     per_feed(late.with_columns(pl.col("plain_raw").alias("score"))), user_ids)
    coverage_rows = {}
    for present in (True, False):
        subset = late.filter(pl.col("has_tail") == present)
        coverage_rows[str(present)] = dict(feeds=subset["feed_id"].n_unique(), ndcg={
            "plain": ndcg(subset.with_columns(pl.col("plain_raw").alias("score")))["ndcg"],
            "tail": ndcg(subset.with_columns(pl.col("tail_raw").alias("score")))["ndcg"],
            "control_blend": ndcg(mix(subset, 0.))["ndcg"], "tail_blend": ndcg(mix(subset, challenger))["ndcg"]})
    torch.save(dict(state_dict={key: value.cpu() for key, value in model.state_dict().items()},
                    variant=NAME, history_window=64, top_k=TOP_K), args.folder/"model.pt")
    report(args, rows, train_feeds=SIZE, epochs=ADAPT_EPOCHS, selected_tail_share=selected,
           challenger_tail_share=challenger, early_trials=trials, early_split=EARLY_SPLIT,
           holdout_feeds=len(user_ids), holdout_paired_bootstrap={"blend": interval, "model": plain_interval},
           holdout_by_available_tail=coverage_rows, top_k=TOP_K, history_positions=[65, 256],
           stats_feeds=stats_feeds, selection_timestamps=[selection["timestamp"].min(), selection["timestamp"].max()])


def queue(args):
    from team_reports import sync

    watch, log = args.folder/"watch.json", args.folder/"train.log"
    result, status = args.folder/"report.json", args.folder/"run.status"
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    status.unlink(missing_ok=True)
    atomic_text(watch, json.dumps(dict(report=str(result), log=str(log), status=str(status), data=str(args.data),
                                       models=[VARIANT], iterations=1, epochs=ADAPT_EPOCHS,
                                       label="GRU v4 · 200 000 лент · отбор старых событий")))
    subprocess.Popen([sys.executable, str(ROOT/"dashboard.py"), "--watch", str(watch)], cwd=ROOT,
                     creationflags=flags)
    started = perf_counter()
    with log.open("ab") as stream:
        stream.write(f"Starting {VARIANT}\n".encode())
        stream.flush()
        with subprocess.Popen([sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve()), "train",
                               "--data", str(args.data), "--runs", str(args.runs), "--output", str(args.folder),
                               "--gpu", args.gpu], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                              creationflags=flags) as child:
            while child.poll() is None:
                keep_awake()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
            code = child.returncode or int(not result.exists())
    atomic_text(status, str(code))
    if code:
        raise RuntimeError(f"Training exited with status {code}")
    result_data = json.loads(result.read_text(encoding="utf-8"))
    instructions = args.folder/"REPRODUCE.md"
    atomic_text(instructions, "# GRU v4 с отбором старых событий\n\n"
                "Нужны Avito train и val, исходная GRU v4 из valishinqa-v4-repro, прогнозы и train-feeds.parquet "
                "из v4-next-200k/adapt, прогнозы epochs и period из local-hypotheses-200k, исходные сигналы V4 "
                "и robust. Рабочая копия valishinqa-v4-worktree использует b47fdcb и исходные глобальные статистики.\n\n"
                "Сравнение отдельных моделей проведено на шестой эпохе. Доля модели в первом периоде "
                "выбрана по наименьшему приросту на двух ранних отрезках. Поздняя val использована для сравнения "
                "контроля с лучшей положительной долей.\n\n"
                "```powershell\n.\\.venv\\Scripts\\python.exe v4_tail.py train --data C:/study/AvitoData/competition "
                "--runs runs/avito-real --output runs/avito-real/gru-v4-tail-200k --gpu 0\n```\n\n"
                "```json\n" + json.dumps(result_data, indent=2) + "\n```\n")
    note(f"Training and evaluation finished in {perf_counter()-started:.0f}s")
    sync("avito-followups", "gru-v4-tail-200k", result, revision=args.revision, instructions=instructions,
         checkpoints=["model.pt"], extra={"Доля tail по ранней val": str(result_data["selected_tail_share"]),
                                      "Проверенная ненулевая доля": str(result_data["challenger_tail_share"])})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "queue"))
    for key in ("data", "runs", "output"):
        parser.add_argument("--"+key, required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--revision", default="HEAD")
    args = parser.parse_args()
    args.previous = args.runs/"v4-next-200k/adapt"
    args.prior = args.runs/"local-hypotheses-200k"
    args.reference = args.runs/"valishinqa-v4-repro"
    args.v4_code = args.runs/"valishinqa-v4-worktree"
    args.folder = args.output
    args.folder.mkdir(parents=True, exist_ok=True)
    {"train": train, "queue": queue}[args.stage](args)
