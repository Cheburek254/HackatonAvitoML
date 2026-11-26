"""Train candidate-set and history-aware neural rankers on Avito feeds."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch.utils.data import DataLoader, Dataset, Subset

from evaluation import KEYS, attach_scores, ndcg, peak_ram_mib
from ranker import BASE_FEATURES, CANDIDATE_COLUMNS, HISTORY_COLUMNS, feed_batches, open_embeddings

TIE_VARIANTS = ("neural-din-plus-ties", "gru-din-plus-ties")
VARIANTS = ("transformer", "transformer-listmle", "avito-joint", "neural-din", "bst", "gru-din", "sasrec",
            "set-transformer", "neural-pairwise", "neural-din-plus", "gru-din-plus", "gru-din-v3", "gru-din-v4",
            "gru-din-v4-listnet", "gru-din-v5", "gru-din-xl", *TIE_VARIANTS)
HISTORY_VARIANTS = {"avito-joint", "neural-din", "bst", "gru-din", "sasrec", "neural-pairwise",
                    "neural-din-plus", "gru-din-plus", "gru-din-v3", "gru-din-v4", "gru-din-v4-listnet", "gru-din-v5", "gru-din-xl", *TIE_VARIANTS}
RICH_VARIANTS = {"neural-din-plus", "gru-din-plus", "gru-din-v3", "gru-din-v4", "gru-din-v4-listnet", "gru-din-v5", "gru-din-xl", *TIE_VARIANTS}
GRU_VARIANTS = {"gru-din", "gru-din-plus", "gru-din-v3", "gru-din-v4", "gru-din-v4-listnet", "gru-din-v5", "gru-din-xl", "gru-din-plus-ties"}
V3_VARIANTS = {"gru-din-v3", "gru-din-v4", "gru-din-v4-listnet", "gru-din-v5", "gru-din-xl"}
V4_VARIANTS = {"gru-din-v4", "gru-din-v4-listnet", "gru-din-v5", "gru-din-xl"}
V5_VARIANTS = {"gru-din-v5"}
XL_VARIANTS = {"gru-din-xl"}


def id_tensor(values):
    cleaned = [0 if v is None else v for v in values]
    return torch.from_numpy(np.asarray(cleaned, dtype=np.int8).astype(np.uint8).astype(np.int64))


def candidate_percentiles(values, *, average_ties=False):
    size = values.shape[0]
    if not size:
        return values
    if not average_ties:
        return (values.argsort(dim=0).argsort(dim=0).float() + 0.5) / size

    ordered, indices = values.sort(dim=0)
    positions = torch.arange(size, device=values.device).unsqueeze(1).expand_as(indices)
    distinct = ordered[1:] != ordered[:-1]
    boundary = torch.ones_like(ordered[:1], dtype=torch.bool)
    starts = torch.cat((boundary, distinct))
    ends = torch.cat((distinct, boundary))
    first = torch.where(starts, positions, 0).cummax(dim=0).values
    last = torch.where(ends, positions, size - 1).flip((0,)).cummin(dim=0).values.flip((0,))
    midpoints = ((first + last).float() * 0.5 + 0.5) / size
    return torch.empty_like(values).scatter_(0, indices, midpoints)


def load_global_stats_for_neural(include_val=False):
    stats_dir = Path(__file__).resolve().parent / "runs" / "global_stats"
    with_val_path = stats_dir / "item_stats_with_val.parquet"
    if include_val and with_val_path.is_file():
        path = with_val_path
        meta_path = stats_dir / "meta_with_val.json"
        print(f"Using train+val global item stats for neural: {path.name}", flush=True)
    else:
        path = stats_dir / "item_stats.parquet"
        meta_path = stats_dir / "meta.json"
    if not path.is_file():
        raise FileNotFoundError(f"Global stats not found at {path}; run build_global_stats.py first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    df = pl.read_parquet(path)
    pos_shows = df["item_pos_shows"].to_numpy() if "item_pos_shows" in df.columns else df["item_shows"].to_numpy()
    hist_col = df["item_hist_count"].to_numpy() if "item_hist_count" in df.columns else np.zeros(df.height, dtype=np.int32)
    r_shows = df["recent_shows"].to_numpy() if "recent_shows" in df.columns else np.zeros(df.height, dtype=np.int32)
    r_clicks = df["recent_clicks"].to_numpy() if "recent_clicks" in df.columns else np.zeros(df.height, dtype=np.int32)
    prior = float(meta["prior"])
    recent_prior = float(meta.get("recent_prior", prior))
    recent_cutoff = int(meta.get("recent_cutoff", 0))
    return (
        df["item_id"].to_numpy(),
        df["item_shows"].to_numpy(),
        df["item_clicks"].to_numpy(),
        df["item_pos_sum"].to_numpy(),
        pos_shows,
        hist_col,
        r_shows,
        r_clicks,
        prior,
        recent_prior,
        recent_cutoff,
    )


class Feeds(Dataset):
    def __init__(self, split, max_feeds, labeled, window, uses_history, *, offset=0, rich_features=False,
                 v3_features=False, v4_features=False, v5_features=False, average_ties=False, cold_drop_prob=0.0,
                 global_stats=None, embeddings=None):
        columns = ["feed_id", "timestamp", "user_embedding", *CANDIDATE_COLUMNS]
        if labeled:
            columns.append("target")
        if uses_history:
            columns.extend([*HISTORY_COLUMNS, "history__event_type", "history__event_source"])
        self.is_train = split.name == "train"
        if (v4_features or v5_features) and self.is_train:
            columns.append("pos")
        batches = list(feed_batches(split, columns, max_feeds, offset=offset))
        self.feeds = pl.concat(batches) if batches else pl.DataFrame()
        self.embeddings = embeddings if embeddings is not None else open_embeddings(split)
        self.labeled = labeled
        self.window = window if uses_history else 0
        self.rich_features = rich_features
        self.v3_features = v3_features or v4_features or v5_features
        self.v4_features = v4_features or v5_features
        self.v5_features = v5_features
        self.cold_drop_prob = cold_drop_prob if self.is_train else 0.0
        if self.v4_features:
            self.global_stats = global_stats if global_stats is not None else load_global_stats_for_neural(include_val=(split.name == "test"))
        else:
            self.global_stats = None
        self.average_ties = average_ties

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
        user = F.normalize(torch.tensor(feed["user_embedding"], dtype=torch.float32), dim=0)
        cat_ids = id_tensor(feed["category_id"])
        if self.window:
            # Reverse only the available prefix so recurrent layers see chronological order.
            indices = feed["history__item_id__embedding_idx"][:self.window][::-1]
            vectors, history_missing = self.vectors(indices)
            events = torch.stack([id_tensor(feed[name][:self.window][::-1]) for name in
                                  ("history__category_id", "history__event_type", "history__event_source")], dim=-1)
            history_features = torch.cat([vectors, history_missing[:, None]], dim=-1)
        else:
            vectors = torch.empty((0, 128))
            history_features, events = torch.empty((0, 129)), torch.empty((0, 3), dtype=torch.long)
        if self.rich_features:
            m = len(items)
            pct = candidate_percentiles(floats, average_ties=self.average_ties)
            u_cos = (items * user).sum(dim=-1, keepdim=True)
            extra = torch.zeros((m, 13), dtype=torch.float32)
            if len(vectors):
                h = len(vectors)
                age = torch.arange(h - 1, -1, -1, dtype=torch.float32)
                sims = items @ vectors.T
                extra[:, 0] = sims.max(dim=1).values
                extra[:, 1] = sims.topk(min(3, h), dim=1).values.mean(dim=1)
                extra[:, 2] = (sims * torch.exp2(-age / 16.0)).max(dim=1).values
                ev_t = events[:, 1]
                ev1 = ev_t == 1
                ev_other = ev_t > 1
                ev_strong = ev_t >= 4
                if ev1.any():
                    extra[:, 3] = sims[:, ev1].max(dim=1).values
                if ev_other.any():
                    extra[:, 4] = sims[:, ev_other].max(dim=1).values
                if ev_strong.any():
                    extra[:, 5] = sims[:, ev_strong].max(dim=1).values
                cat_w = torch.exp2(-age / 8.0)
                cat_m = ((cat_ids[:, None] == events[:, 0][None, :]) & (cat_ids[:, None] != 0)).float()
                extra[:, 6] = (cat_m @ cat_w) / cat_w.sum().clamp_min(1.0)
                recent_mask = age < 8
                earlier_mask = age >= 8
                recent_valid = recent_mask & (events[:, 0] != 0)
                earlier_valid = earlier_mask & (events[:, 0] != 0)
                recent_share = cat_m[:, recent_valid].sum(dim=1) / max(int(recent_valid.sum()), 1)
                earlier_share = cat_m[:, earlier_valid].sum(dim=1) / max(int(earlier_valid.sum()), 1)
                extra[:, 7] = recent_share
                if recent_valid.any() and earlier_valid.any():
                    extra[:, 8] = recent_share - earlier_share
            extra[:, 9] = len(feed["history__item_id"]) / 256.0
            extra[:, 10] = u_cos[:, 0] - u_cos[:, 0].mean()
            extra[:, 11] = extra[:, 0] - extra[:, 0].mean()
            extra[:, 12] = extra[:, 2] - extra[:, 2].mean()
            if self.v3_features:
                v3 = torch.zeros((m, 9), dtype=torch.float32)
                cat_eq = (cat_ids[:, None] == cat_ids[None, :]).float()
                cat_counts = cat_eq.sum(dim=1).clamp_min(1.0)
                v3[:, 0] = cat_counts / max(m, 1)
                f5 = floats[:, 4]
                f3 = floats[:, 2]
                v3[:, 1] = (f5 - f5.mean()) / f5.std(unbiased=False).clamp_min(1e-6)
                v3[:, 2] = f5 - f5.max()
                v3[:, 3] = f5 - (cat_eq @ f5) / cat_counts
                v3[:, 4] = (f3 - f3.mean()) / f3.std(unbiased=False).clamp_min(1e-6)
                v3[:, 5] = m / 300.0
                valid_c = ~missing
                if valid_c.any():
                    valid_idx = torch.nonzero(valid_c, as_tuple=False).squeeze(-1)
                    k = min(5, len(valid_idx))
                    top_idx = valid_idx[f5[valid_idx].topk(k).indices]
                    proto = F.normalize(items[top_idx].mean(dim=0, keepdim=True), dim=-1)
                    v3[:, 6] = (items * proto).sum(dim=-1)
                tail_start = min(self.window, 64)
                tail_idx = feed["history__item_id__embedding_idx"][tail_start:256]
                if len(tail_idx):
                    tail_vecs, tail_miss = self.vectors(tail_idx)
                    tail_valid = ~tail_miss
                    if tail_valid.any():
                        tail_sims = items @ tail_vecs[tail_valid].T
                        v3[:, 7] = tail_sims.max(dim=1).values
                        tail_ev = id_tensor(feed["history__event_type"][tail_start:256])[tail_valid]
                        other_m = tail_ev > 1
                        if other_m.any():
                            v3[:, 8] = tail_sims[:, other_m].max(dim=1).values
                if self.v4_features and self.global_stats is not None:
                    (g_ids, g_shows, g_clicks, g_pos_sum, g_pos_shows, g_hist,
                     g_r_shows, g_r_clicks, prior, recent_prior, recent_cutoff) = self.global_stats
                    i_arr = np.asarray(feed["item_id"], dtype=np.int32)
                    idx = np.searchsorted(g_ids, i_arr)
                    clipped = np.minimum(idx, len(g_ids) - 1)
                    found = (i_arr > 0) & (g_ids[clipped] == i_arr)
                    shows = np.where(found, g_shows[clipped], 0).astype(np.float64)
                    clicks = np.where(found, g_clicks[clipped], 0).astype(np.float64)
                    pos_sum = np.where(found, g_pos_sum[clipped], 0).astype(np.float64)
                    pos_shows = np.where(found, g_pos_shows[clipped], 0).astype(np.float64)
                    hist_c = np.where(found, g_hist[clipped], 0).astype(np.float64)
                    r_shows = np.where(found, g_r_shows[clipped], 0).astype(np.float64)
                    r_clicks = np.where(found, g_r_clicks[clipped], 0).astype(np.float64)
                    if self.is_train and self.labeled:
                        tgt = np.asarray(feed["target"], dtype=np.float64)
                        shows = np.maximum(shows - 1.0, 0.0)
                        clicks = np.maximum(clicks - tgt, 0.0)
                        if "pos" in feed:
                            pos_sum = np.maximum(pos_sum - np.asarray(feed["pos"], dtype=np.float64), 0.0)
                            pos_shows = np.maximum(pos_shows - 1.0, 0.0)
                        if recent_cutoff > 0 and feed["timestamp"] >= recent_cutoff:
                            r_shows = np.maximum(r_shows - 1.0, 0.0)
                            r_clicks = np.maximum(r_clicks - tgt, 0.0)
                    if self.cold_drop_prob > 0.0 and len(i_arr):
                        drop = np.random.random(len(i_arr)) < self.cold_drop_prob
                        shows = np.where(drop, 0.0, shows)
                        clicks = np.where(drop, 0.0, clicks)
                        pos_sum = np.where(drop, 0.0, pos_sum)
                        pos_shows = np.where(drop, 0.0, pos_shows)
                        hist_c = np.where(drop, 0.0, hist_c)
                        r_shows = np.where(drop, 0.0, r_shows)
                        r_clicks = np.where(drop, 0.0, r_clicks)
                    shows_log = np.log1p(shows) / 10.0
                    ctr = (clicks + 30.0 * prior) / (shows + 30.0)
                    mean_pos = (pos_sum + 20.0 * 40.0) / (pos_shows + 20.0)
                    ctr_scaled = (ctr - prior) * 20.0
                    pos_scaled = (mean_pos - 40.0) / 40.0
                    ctr_delta = ctr_scaled - ctr_scaled.mean()
                    pos_delta = pos_scaled - pos_scaled.mean()
                    hist_log = np.log1p(hist_c) / 10.0
                    feats = [shows_log, ctr_scaled, pos_scaled, ctr_delta, pos_delta, hist_log]
                    if self.v5_features:
                        r_shows_log = np.log1p(r_shows) / 10.0
                        r_ctr = (r_clicks + 20.0 * recent_prior) / (r_shows + 20.0)
                        r_ctr_scaled = (r_ctr - recent_prior) * 20.0
                        r_ctr_delta = r_ctr_scaled - r_ctr_scaled.mean()
                        feats.extend([r_shows_log, r_ctr_scaled, r_ctr_delta])
                    v4 = torch.from_numpy(np.stack(feats, axis=1).astype(np.float32))
                    candidates = torch.cat([floats, pct, u_cos, extra, v3, v4, items, missing[:, None]], dim=-1)
                else:
                    candidates = torch.cat([floats, pct, u_cos, extra, v3, items, missing[:, None]], dim=-1)
            else:
                candidates = torch.cat([floats, pct, u_cos, extra, items, missing[:, None]], dim=-1)
        else:
            candidates = torch.cat([floats, items, missing[:, None]], dim=-1)
        target = torch.tensor(feed["target"], dtype=torch.float32) if self.labeled else torch.zeros(len(items))
        return feed["feed_id"], feed["item_id"], candidates, cat_ids, user, history_features, events, target


def collate(rows):
    feeds, items, candidates, categories, users, history, events, target = zip(*rows, strict=True)
    mask = torch.arange(max(map(len, items)))[None, :] >= torch.tensor(list(map(len, items)))[:, None]
    history_mask = torch.arange(max(map(len, history)))[None, :] >= torch.tensor(list(map(len, history)))[:, None]
    inputs = (pad_sequence(candidates, batch_first=True), pad_sequence(categories, batch_first=True), torch.stack(users),
              mask, pad_sequence(history, batch_first=True), pad_sequence(events, batch_first=True), history_mask)
    return feeds, items, inputs, pad_sequence(target, batch_first=True)


class Ranker(nn.Module):
    def __init__(self, variant, width=None):
        super().__init__()
        is_xl = variant in XL_VARIANTS
        width = width or (256 if is_xl else (128 if variant in RICH_VARIANTS else 64))
        heads = 8 if width >= 256 else 4
        self.variant = variant
        self.use_grad_checkpoint = False
        self.category = nn.Embedding(256, 8, padding_idx=0)
        if variant in RICH_VARIANTS:
            in_dim = 179 if variant in V5_VARIANTS else (176 if variant in V4_VARIANTS else (170 if variant in V3_VARIANTS else 161))
            self.candidate = nn.Sequential(
                nn.Linear(in_dim, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(width, width),
            )
        else:
            self.candidate = nn.Linear(142, width)
        self.user = nn.Linear(128, width)
        self.output = nn.Linear(width, 1)
        if variant == "set-transformer":
            self.inducing = nn.Parameter(torch.randn(8, width) / width ** .5)
            self.summarize = nn.MultiheadAttention(width, heads, batch_first=True)
            self.recover = nn.MultiheadAttention(width, heads, batch_first=True)
            self.set_norm = nn.LayerNorm(width)
        else:
            layers = 6 if is_xl else (3 if variant in RICH_VARIANTS else 2)
            self.encoder = nn.TransformerEncoder(self.layer(width, heads), layers, enable_nested_tensor=False)
        if variant in HISTORY_VARIANTS:
            self.event_type = nn.Embedding(256, 8, padding_idx=0)
            self.event_source = nn.Embedding(256, 8, padding_idx=0)
            self.position = nn.Embedding(256, width)
            self.history = nn.Linear(153, width)
            self.empty_history = nn.Parameter(torch.zeros(1, 1, width))
        if variant in HISTORY_VARIANTS - {"avito-joint", "sasrec"}:
            self.history_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        if variant in HISTORY_VARIANTS - {"avito-joint"}:
            self.history_norm = nn.LayerNorm(width)
        if variant in {"bst", "sasrec"}:
            self.history_encoder = nn.TransformerEncoder(self.layer(width, heads), 1, enable_nested_tensor=False)
        if variant in GRU_VARIANTS:
            gru_layers = 2 if is_xl else 1
            self.history_encoder = nn.GRU(width, width, num_layers=gru_layers,
                                          dropout=0.1 if gru_layers > 1 else 0.0, batch_first=True)

    @staticmethod
    def layer(width, heads=4):
        return nn.TransformerEncoderLayer(width, heads, width * 2, dropout=.1, batch_first=True, norm_first=True)

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
        if self.variant in GRU_VARIANTS:
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
            if self.use_grad_checkpoint and self.training:
                for layer in self.encoder.layers:
                    tokens = grad_checkpoint(layer, tokens, None, mask, use_reentrant=False)
                if self.encoder.norm is not None:
                    tokens = self.encoder.norm(tokens)
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
    if variant not in {"neural-pairwise", *RICH_VARIANTS}:
        loss = F.binary_cross_entropy_with_logits(scores, target, reduction="none").masked_fill(padding, 0)
        return (loss.sum(dim=1) / (~padding).sum(dim=1)).mean()
    ranks = scores.detach().masked_fill(padding, -torch.inf).argsort(dim=1, descending=True).argsort(dim=1)
    discounts = 1 / torch.log2(torch.arange(scores.shape[1], device=scores.device) + 2)
    actual = discounts[ranks]
    ideal = discounts.cumsum(0)[target.sum(dim=1).long().clamp_min(1) - 1]
    pairs = (target[:, :, None] > target[:, None, :]) & ~padding[:, None, :]
    weights = (actual[:, :, None] - actual[:, None, :]).abs() / ideal[:, None, None]
    losses = F.softplus(scores[:, None, :] - scores[:, :, None]) * weights
    pairwise = (losses.masked_fill(~pairs, 0).sum(dim=(1, 2)) / pairs.sum(dim=(1, 2)).clamp_min(1)).mean()
    if variant == "gru-din-v4-listnet":
        pos_sum = target.masked_fill(padding, 0).sum(dim=1)
        has_pos = pos_sum > 0
        if has_pos.any():
            log_probs = F.log_softmax(scores.masked_fill(padding, -1e9), dim=1)
            target_prob = target.masked_fill(padding, 0) / pos_sum.clamp_min(1.0)[:, None]
            listnet = -(target_prob * log_probs).sum(dim=1)[has_pos].mean()
            return 0.5 * pairwise + 0.5 * listnet
    return pairwise


class PrefetchBatches:
    """Prepare one CPU batch ahead while CUDA processes the current batch."""

    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        source = iter(self.batches)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="avito-batches") as worker:
            pending = worker.submit(next, source, None)
            while (batch := pending.result()) is not None:
                pending = worker.submit(next, source, None)
                yield batch


def loader(dataset, batch_size, shuffle=False, *, prefetch=False):
    batches = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
    return PrefetchBatches(batches) if prefetch else batches


def scores(model, batches, device):
    model.eval()
    use_bf16 = device == "cuda" and getattr(model, "variant", None) in V4_VARIANTS
    chunks = []
    with torch.inference_mode():
        for feeds, items, inputs, _ in batches:
            dev_inputs = tuple(tensor.to(device) for tensor in inputs)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                values = model(*dev_inputs)
            values = values.float().cpu()
            chunks.append(pl.DataFrame({"feed_id": [feed for feed, keys in zip(feeds, items, strict=True) for _ in keys],
                                        "item_id": [key for keys in items for key in keys],
                                        "score": values[~inputs[3]].numpy()},
                                       schema={"feed_id": pl.Int32, "item_id": pl.Int32, "score": pl.Float32}))
    return pl.concat(chunks)


def labels(dataset):
    return dataset.feeds.select(*KEYS, "target").explode("item_id", "target")


def train_epoch(model, optimizer, batches, device, variant, epoch, scheduler=None, accum_steps=1, label=None):
    start = perf_counter()
    model.train()
    use_bf16 = device == "cuda" and variant in V4_VARIANTS
    tag = label or f"epoch={epoch}"
    optimizer.zero_grad(set_to_none=True)
    total_batches = len(batches)
    for step, (_, _, inputs, target) in enumerate(batches, start=1):
        inputs = tuple(tensor.to(device) for tensor in inputs)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(*inputs)
        loss = loss_for(variant, logits.float(), target.to(device), inputs[3])
        (loss / accum_steps).backward()
        if step % accum_steps == 0 or step == total_batches:
            if variant in RICH_VARIANTS:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        if step == 1 or step == total_batches or step % 200 == 0:
            print(f"{variant} {tag} batch={step}/{total_batches} loss={loss.detach().item():.6f}", flush=True)
    seconds = round(perf_counter() - start, 2)
    print(f"{variant} {tag} training_seconds={seconds:.2f}", flush=True)
    return seconds


def _save_checkpoint(path, state_dict, variant, history_window):
    tmp = path.with_name(f".{path.name}.{uuid4().hex}")
    try:
        torch.save({"state_dict": state_dict, "variant": variant, "history_window": history_window}, tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def train(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(42)
    use_history = args.variant in HISTORY_VARIANTS
    rich = args.variant in RICH_VARIANTS
    v3 = args.variant in V3_VARIANTS
    v4 = args.variant in V4_VARIANTS
    v5 = args.variant in V5_VARIANTS
    is_xl = args.variant in XL_VARIANTS
    average_ties = args.variant in TIE_VARIANTS
    cold_drop = args.cold_drop if args.cold_drop is not None else (0.35 if v5 else 0.0)
    accum_steps = args.accum_steps or (4 if is_xl else 1)
    lr = args.lr or (6e-4 if is_xl else 1e-3)
    chunk_feeds = args.chunk_feeds
    if cold_drop > 0:
        print(f"Cold-item global stats dropout enabled for neural: p={cold_drop:.2f}", flush=True)
    shared_stats = load_global_stats_for_neural(include_val=False) if (v4 or v5) else None
    val_set = Feeds(args.data / "val", args.val_feeds, True, args.history_window, use_history, rich_features=rich,
                    v3_features=v3, v4_features=v4, v5_features=v5, average_ties=average_ties,
                    global_stats=shared_stats)
    val_labels = labels(val_set)
    tune_labels, tune_batches = val_labels, loader(val_set, args.batch_size, prefetch=args.prefetch)
    if args.tune_before is not None:
        indices = np.flatnonzero(val_set.feeds["timestamp"].to_numpy() < args.tune_before)
        if not 0 < len(indices) < len(val_set):
            raise ValueError("--tune-before must divide the selected validation feeds into two nonempty sets")
        tune_labels = val_labels.filter(pl.col("feed_id").is_in(val_set.feeds["feed_id"][indices]))
        tune_batches = loader(Subset(val_set, indices.tolist()), args.batch_size, prefetch=args.prefetch)

    model = Ranker(args.variant).to(args.device)
    if args.resume_from is not None:
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Resumed weights from {args.resume_from}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model {args.variant}: params={n_params:,}, lr={lr}, batch_size={args.batch_size}, accum_steps={accum_steps}, chunk_feeds={chunk_feeds}", flush=True)

    streaming = chunk_feeds > 0 and (args.train_feeds == 0 or chunk_feeds < args.train_feeds)
    if streaming:
        total_target_feeds = args.train_feeds if args.train_feeds > 0 else 3_151_208
        est_batches_per_epoch = max(1, (total_target_feeds + args.batch_size - 1) // args.batch_size)
        est_opt_steps = max(1, (est_batches_per_epoch // accum_steps) * args.epochs)
    else:
        train_set = Feeds(args.data / "train", args.train_feeds, True, args.history_window, use_history,
                          offset=args.train_offset, rich_features=rich, v3_features=v3, v4_features=v4,
                          v5_features=v5, average_ties=average_ties, cold_drop_prob=cold_drop,
                          global_stats=shared_stats)
        train_batches = loader(train_set, args.batch_size, shuffle=True, prefetch=args.prefetch)
        total_target_feeds = len(train_set)
        est_opt_steps = max(1, ((len(train_batches) + accum_steps - 1) // accum_steps) * args.epochs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    if rich:
        warmup = min(500, max(1, est_opt_steps // 10)) if is_xl else 0
        if warmup > 0:
            def lr_lambda(step):
                if step < warmup:
                    return float(step + 1) / float(warmup)
                prog = min(1.0, float(step - warmup) / float(max(1, est_opt_steps - warmup)))
                return 0.02 + 0.98 * 0.5 * (1.0 + np.cos(np.pi * prog))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=est_opt_steps, eta_min=1e-5)
    else:
        scheduler = None

    best, best_step, best_state, curve = -1.0, 0, None, []
    epoch_seconds = []
    tuning_slates = None
    if args.device == "cuda":
        from ensemble import Slates

        tuning_slates = Slates(tune_labels, args.device)

    args.output.mkdir(parents=True, exist_ok=True)
    eval_step = 0
    seen_feeds_per_epoch = 0

    def evaluate_and_checkpoint(step_tag):
        nonlocal best, best_step, best_state, eval_step
        eval_step += 1
        predicted = scores(model, tune_batches, args.device)
        if tuning_slates is None:
            metric = ndcg(attach_scores(tune_labels, predicted))["ndcg"]
        else:
            aligned = attach_scores(tuning_slates.frame, predicted).sort(["feed_id", "target", "item_id"])
            values = torch.as_tensor(aligned["score"].to_numpy()[tuning_slates.rows], device=args.device)
            metric = tuning_slates.metric(values)
        curve.append([eval_step, metric])
        print(f"{args.variant} {step_tag} tuning_ndcg={metric:.6f} (best={max(best, metric):.6f})", flush=True)
        if metric > best:
            best, best_step = metric, eval_step
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            _save_checkpoint(args.output / "model.pt", best_state, args.variant, args.history_window)

    if streaming:
        train_embeddings = open_embeddings(args.data / "train")
        for epoch in range(1, args.epochs + 1):
            seen_this_epoch = 0
            chunk_idx = 0
            while args.train_feeds == 0 or seen_this_epoch < args.train_feeds:
                cur_limit = chunk_feeds if args.train_feeds == 0 else min(chunk_feeds, args.train_feeds - seen_this_epoch)
                cur_offset = args.train_offset + seen_this_epoch
                print(f"{args.variant} epoch={epoch} loading chunk={chunk_idx + 1} (offset={cur_offset:,}, limit={cur_limit:,})...", flush=True)
                chunk_set = Feeds(args.data / "train", cur_limit, True, args.history_window, use_history,
                                  offset=cur_offset, rich_features=rich, v3_features=v3, v4_features=v4,
                                  v5_features=v5, average_ties=average_ties, cold_drop_prob=cold_drop,
                                  global_stats=shared_stats, embeddings=train_embeddings)
                if len(chunk_set) == 0:
                    del chunk_set
                    break
                chunk_idx += 1
                seen_this_epoch += len(chunk_set)
                tag = f"epoch={epoch} chunk={chunk_idx} ({seen_this_epoch:,} feeds)"
                chunk_batches = loader(chunk_set, args.batch_size, shuffle=True, prefetch=args.prefetch)
                epoch_seconds.append(train_epoch(model, optimizer, chunk_batches, args.device, args.variant,
                                                 epoch, scheduler, accum_steps=accum_steps, label=tag))
                del chunk_batches, chunk_set
                gc.collect()
                evaluate_and_checkpoint(tag)
            seen_feeds_per_epoch = seen_this_epoch
    else:
        seen_feeds_per_epoch = len(train_set)
        for epoch in range(1, args.epochs + 1):
            epoch_seconds.append(train_epoch(model, optimizer, train_batches, args.device, args.variant,
                                             epoch, scheduler, accum_steps=accum_steps))
            evaluate_and_checkpoint(f"epoch={epoch}")

    model.load_state_dict(best_state)
    predictions = scores(model, loader(val_set, args.batch_size, prefetch=args.prefetch), args.device)
    report = {f"val_{name}": value for name, value in ndcg(attach_scores(val_labels, predictions)).items()}
    report.update(variant=args.variant, best_epoch=best_step, tune_before=args.tune_before, tuning_curve=curve,
                  train_feeds=seen_feeds_per_epoch, val_feeds=len(val_set),
                  chunk_feeds=chunk_feeds, accum_steps=accum_steps, lr=lr, cold_drop=cold_drop,
                  epoch_seconds=epoch_seconds, peak_ram_mib=peak_ram_mib())
    _save_checkpoint(args.output / "model.pt", best_state, args.variant, args.history_window)
    predictions.write_parquet(args.output / "val_predictions.parquet")
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Finished {args.variant}: full_val_ndcg={report['val_ndcg']:.6f} (best_step={best_step})", flush=True)
    if seen_feeds_per_epoch >= 10_000 and not args.output.name.startswith("_"):
        try:
            import notify_tg
            msg = (f"✅ {args.variant} finished ({args.output.name})\n"
                   f"val_ndcg: {report['val_ndcg']:.6f}\n"
                   f"feeds: {seen_feeds_per_epoch:,} x {args.epochs}ep, best_step={best_step}")
            notify_tg.send(msg)
        except Exception:
            pass


def oof(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    dataset = Feeds(args.data / "train", args.train_feeds, True, args.history_window, args.variant in HISTORY_VARIANTS,
                    offset=args.train_offset, rich_features=args.variant in RICH_VARIANTS,
                    v3_features=args.variant in V3_VARIANTS, v4_features=args.variant in V4_VARIANTS,
                    v5_features=args.variant in V5_VARIANTS, average_ties=args.variant in TIE_VARIANTS)
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
    pending = args.output.with_name(f".oof-{uuid4().hex}.parquet")
    try:
        pl.concat(chunks).write_parquet(pending)
        pending.replace(args.output)
    finally:
        pending.unlink(missing_ok=True)


def predict(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    variant = checkpoint["variant"]
    model = Ranker(variant).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    test = Feeds(args.data / "test", 0, False, checkpoint["history_window"], variant in HISTORY_VARIANTS,
                 rich_features=variant in RICH_VARIANTS, v3_features=variant in V3_VARIANTS,
                 v4_features=variant in V4_VARIANTS, v5_features=variant in V5_VARIANTS,
                 average_ties=variant in TIE_VARIANTS)
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
    training.add_argument("--train-offset", type=int, default=0)
    training.add_argument("--val-feeds", type=int, default=0)
    training.add_argument("--history-window", type=int, default=32)
    training.add_argument("--tune-before", type=int)
    training.add_argument("--cold-drop", type=float, default=None, help="Drop global stats on train items with prob p (default 0.35 for gru-din-v5/xl)")
    training.add_argument("--chunk-feeds", type=int, default=0, help="Stream train in chunks of N feeds to stay within RAM")
    training.add_argument("--accum-steps", type=int, default=0, help="Gradient accumulation steps (default 4 for gru-din-xl, 1 otherwise)")
    training.add_argument("--lr", type=float, default=0.0, help="Override peak learning rate (default 6e-4 for xl, 1e-3 otherwise)")
    training.add_argument("--resume-from", type=Path, default=None, help="Initialize weights from an existing model.pt")
    training.add_argument("--prefetch", action="store_true", help="Prepare one CPU batch ahead")
    training.set_defaults(run=train)
    oof_command = commands.add_parser("oof")
    oof_command.add_argument("--variant", choices=VARIANTS, required=True)
    oof_command.add_argument("--epochs", type=int, default=2)
    oof_command.add_argument("--train-feeds", type=int, default=50_000)
    oof_command.add_argument("--train-offset", type=int, default=0)
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
        command.add_argument("--device", choices=["cpu", "cuda"])
        command.add_argument("--gpu", "--gpu-id", dest="gpu", help="NVIDIA index or UUID")
    args = parser.parse_args()
    if args.gpu is not None and args.device == "cpu":
        parser.error("--gpu cannot be combined with --device cpu")
    args.device = args.device or ("cuda" if args.gpu is not None else "cpu")
    if min(args.batch_size, args.threads, getattr(args, "epochs", 1)) < 1:
        parser.error("Batch size, threads and epochs must be positive")
    if (not 1 <= getattr(args, "history_window", 1) <= 256
            or min(getattr(args, name, 0) for name in ("train_feeds", "val_feeds", "train_offset")) < 0):
        parser.error("History window must be 1–256 and feed limits and offset must be nonnegative")
    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        from accelerator import select_gpu

        selected = select_gpu(args.gpu)
        print(f"PyTorch GPU: {selected.name} ({selected.uuid})", flush=True)
        if not torch.cuda.is_available():
            parser.error("CUDA PyTorch is required for --device cuda")
        torch.set_float32_matmul_precision("high")
    try:
        args.run(args)
    except Exception as exc:
        if (args.run is train
                and (args.train_feeds == 0 or args.train_feeds >= 10_000)
                and not args.output.name.startswith("_")):
            try:
                import notify_tg
                notify_tg.send(f"❌ {args.variant} failed ({args.output.name}): {type(exc).__name__}: {exc}")
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()
