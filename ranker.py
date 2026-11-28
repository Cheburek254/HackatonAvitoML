"""Train and apply a CatBoost ranker to Avito competition feeds."""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from catboost import CatBoost, CatBoostError, CatBoostRanker, Pool

import behavior
import embedding_clusters
from evaluation import KEYS, NDCG, attach_scores, ndcg, peak_ram_mib, zscore

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
EVENT_DIN_FEATURES = [
    "din_decay_max_cosine", "din_best_event_type", "din_best_event_source",
    "din_ev1_max_cosine", "din_ev_other_max_cosine", "din_ev_strong_max_cosine",
]
FEED_CONTEXT_FEATURES = [
    "cat_share_in_feed", "f5_zscore", "f5_minus_max", "f5_cat_delta", "f3_zscore", "feed_size",
    "top5_f5_cosine", "din_long_max_cosine", "din_long_ev_other_max",
]
GLOBAL_STATS_FEATURES = [
    "global_item_shows_log", "global_item_ctr", "global_item_mean_pos",
    "global_item_ctr_delta", "global_item_pos_delta",
]
RECENT_GLOBAL_FEATURES = [
    "global_recent_shows_log", "global_recent_ctr", "global_recent_ctr_delta",
]
GRAPH_FEATURES = [
    "g_hist_u", "g_hist_h", "g_hist_cnt", "g_hist_coh", "g_hist_u_delta",
]
V5_NEW_FEATURES = [
    "top5_f1_cosine", "top5_f3_cosine", "cand_mean_cos",
    "din_src1_max_cosine", "din_src0_max_cosine", "din_src1_ev1_max_cosine",
    "f1_zscore", "f1_minus_max", "f1_cat_delta", "f3_cat_delta",
]
V4_EXTRA_FEATURES = [
    "global_item_hist_log", "global_item_hist_delta", "global_user_feeds_log", "global_user_ctr",
    "cat_din_max_cosine", "f5_x_din_decay", "f5z_x_din_decay", "f1_zscore", "f2_zscore", "f4_zscore",
]
TOP_F5 = "top5_f5_cosine"
TOP_F5_WITH_TIES = "top5_f5_cosine_ties"
SIM_FEATURES = ["sim_top16_cosine", "sim_recent_cosine", "sim_matching_category_share"]
MIND_FEATURES = ["mind_best_cosine", "mind_category_cosine", "mind_interest_count"]
SIM_RANK_SOURCES = ["user_item_cosine", "recent_items_cosine", "din_max_cosine", "din_top3_cosine",
                    "sim_recent_cosine", "mind_best_cosine"]
SIM_RANK_FEATURES = [
    *(f"{name}_feed_percentile" for name in SIM_RANK_SOURCES),
    *(f"{name}_feed_delta" for name in SIM_RANK_SOURCES),
]
FEATURES = ["category_id", *BASE_FEATURES, *PERCENTILES, "user_item_cosine", "item_embedding_missing", *HISTORY_FEATURES]
COMBO_FEATURES = [*FEATURES, *DIN_FEATURES, *SIM_FEATURES, *MIND_FEATURES, *WIDE_TREND_FEATURES]
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
    "combo": COMBO_FEATURES,
    "combo-ranks": [*COMBO_FEATURES, *SIM_RANK_FEATURES],
    "din-events": [*FEATURES, *DIN_FEATURES, *EVENT_DIN_FEATURES],
    "din-events-pos": [*FEATURES, *DIN_FEATURES, *EVENT_DIN_FEATURES, "pos_bias"],
    "din-events-v2": [*FEATURES, *DIN_FEATURES, *EVENT_DIN_FEATURES, *FEED_CONTEXT_FEATURES],
    "din-events-v3": [*FEATURES, *DIN_FEATURES, *EVENT_DIN_FEATURES, *FEED_CONTEXT_FEATURES, *GLOBAL_STATS_FEATURES],
    "din-events-v3-qsm": [*FEATURES, *DIN_FEATURES, *EVENT_DIN_FEATURES, *FEED_CONTEXT_FEATURES, *GLOBAL_STATS_FEATURES],
    "din-events-v3-robust": [*FEATURES, *DIN_FEATURES, *EVENT_DIN_FEATURES, *FEED_CONTEXT_FEATURES, *GLOBAL_STATS_FEATURES, *RECENT_GLOBAL_FEATURES],
    "din-events-v4-graph": [*FEATURES, *DIN_FEATURES, *EVENT_DIN_FEATURES, *FEED_CONTEXT_FEATURES, *GLOBAL_STATS_FEATURES, *GRAPH_FEATURES],
    "din-events-v5-qsm": [*FEATURES, *DIN_FEATURES, *EVENT_DIN_FEATURES, *FEED_CONTEXT_FEATURES, *GLOBAL_STATS_FEATURES, *V5_NEW_FEATURES],
    "din-events-v4-qsm": [*FEATURES, *DIN_FEATURES, *EVENT_DIN_FEATURES, *FEED_CONTEXT_FEATURES, *GLOBAL_STATS_FEATURES, *V4_EXTRA_FEATURES],
}
VARIANTS["din-events-v2-ties"] = [
    TOP_F5_WITH_TIES if name == TOP_F5 else name for name in VARIANTS["din-events-v2"]
]
VARIANTS["strong-gbdt"] = [*FEATURES, *CATEGORY_TREND_FEATURES, *SIM_FEATURES, *behavior.FEATURES["strong-gbdt"]]
STACK_FEATURES = ["stack_joint_z", "stack_gru_z"]
STACK_VARIANTS = ("stack-base", "stack-gbdt")
VARIANTS["stack-base"] = VARIANTS["strong-gbdt"]
VARIANTS["stack-gbdt"] = [*VARIANTS["strong-gbdt"], *STACK_FEATURES]
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


def event_din_features(items, history_items, event_types, event_sources, positions):
    valid = np.linalg.norm(history_items, axis=1) > 0
    history = history_items[valid]
    if not len(history):
        return np.zeros((len(EVENT_DIN_FEATURES), len(items)), dtype=np.float32)
    ev_types = event_types[valid]
    ev_sources = event_sources[valid]
    pos = positions[valid]
    present = np.linalg.norm(items, axis=1) > 0
    similarities = pair_cosine(items, history)
    decayed = similarities * np.exp2(-pos / 16)[None, :]
    best = np.argmax(similarities, axis=1)
    best_type = np.where(present, ev_types[best], 0)
    best_source = np.where(present, ev_sources[best], 0)
    ev1 = ev_types == 1
    ev_other = ev_types > 1
    ev_strong = ev_types >= 4
    ev1_max = similarities[:, ev1].max(axis=1) if ev1.any() else np.zeros(len(items), dtype=np.float32)
    other_max = similarities[:, ev_other].max(axis=1) if ev_other.any() else np.zeros(len(items), dtype=np.float32)
    strong_max = similarities[:, ev_strong].max(axis=1) if ev_strong.any() else np.zeros(len(items), dtype=np.float32)
    return np.stack([decayed.max(axis=1), best_type, best_source, ev1_max, other_max, strong_max]).astype(np.float32)


def top_f5_prototype(items, f5, item_indices, *, include_ties):
    valid = np.flatnonzero(item_indices >= 0)
    if not len(valid):
        return np.zeros(len(items), dtype=np.float32)

    top = valid[np.argsort(-f5[valid])[:5]]
    if include_ties:
        top = valid[f5[valid] >= f5[top[-1]]]
        # Sum normalized vectors in database order for identical output after permutation.
        top = top[np.argsort(item_indices[top])]
    top_vectors = items[top]
    norms = np.linalg.norm(top_vectors, axis=1, keepdims=True)
    prototype = np.divide(top_vectors, norms, out=np.zeros_like(top_vectors), where=norms > 0).mean(axis=0)
    return cosine(items, prototype)


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

    window = 256 if (SIM_FEATURES[0] in feature_names or "din_long_max_cosine" in feature_names) else RECENT_EVENTS
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
            use_all = mode == "sim" or window == RECENT_EVENTS
            mode_items = history_items if use_all else history_items[recent_vectors]
            mode_cats = history_categories[history_present] if use_all else history_categories[history_present][recent_vectors]
            mode_pos = positions if use_all else positions[recent_vectors]
            values = matching_features(items, categories, mode_items, mode_cats, mode_pos, mode)
            result.update(zip(names, values, strict=True))
    if EVENT_DIN_FEATURES[0] in feature_names:
        ev_types = np.asarray([0 if v is None else v for v in feed["history__event_type"][:window]], dtype=np.int32)
        ev_sources = np.asarray([0 if v is None else v for v in feed["history__event_source"][:window]], dtype=np.int32)
        values = event_din_features(
            items, history_items[recent_vectors],
            ev_types[history_present][recent_vectors], ev_sources[history_present][recent_vectors],
            positions[recent_vectors],
        )
        result.update(zip(EVENT_DIN_FEATURES, values, strict=True))
    if "cat_din_max_cosine" in feature_names:
        rec_items = history_items[recent_vectors]
        rec_valid = np.linalg.norm(rec_items, axis=1) > 0
        rec_items = rec_items[rec_valid]
        rec_cats = history_categories[history_present][recent_vectors][rec_valid]
        if len(rec_items):
            sims = pair_cosine(items, rec_items)
            same_cat = (categories[:, None] != 0) & (categories[:, None] == rec_cats[None, :])
            masked = np.where(same_cat, sims, -2.0)
            best_same = masked.max(axis=1)
            result["cat_din_max_cosine"] = np.where(same_cat.any(axis=1), best_same, 0.0).astype(np.float32)
        else:
            result["cat_din_max_cosine"] = np.zeros(len(item_ids), dtype=np.float32)
    top_feature = TOP_F5_WITH_TIES if TOP_F5_WITH_TIES in feature_names else TOP_F5
    if top_feature in feature_names:
        f5 = np.nan_to_num(np.asarray(feed["f5"], dtype=np.float32), nan=-1e9)
        result[top_feature] = top_f5_prototype(items, f5, item_indices, include_ties=top_feature == TOP_F5_WITH_TIES)
    if "top5_f1_cosine" in feature_names:
        f1 = np.nan_to_num(np.asarray(feed["f1"], dtype=np.float32), nan=-1e9)
        result["top5_f1_cosine"] = top_f5_prototype(items, f1, item_indices, include_ties=False)
    if "top5_f3_cosine" in feature_names:
        f3 = np.nan_to_num(np.asarray(feed["f3"], dtype=np.float32), nan=-1e9)
        result["top5_f3_cosine"] = top_f5_prototype(items, f3, item_indices, include_ties=False)
    if "cand_mean_cos" in feature_names:
        c_norms = np.linalg.norm(items, axis=1, keepdims=True)
        unit_c = np.divide(items, c_norms, out=np.zeros_like(items), where=c_norms > 0)
        mean_vec = unit_c.sum(axis=0)
        n_c = len(item_ids)
        if n_c > 1:
            result["cand_mean_cos"] = ((unit_c @ mean_vec) - (c_norms[:, 0] > 0).astype(np.float32)) / (n_c - 1)
        else:
            result["cand_mean_cos"] = np.zeros(n_c, dtype=np.float32)
    if "din_src1_max_cosine" in feature_names:
        rec_items = history_items[recent_vectors]
        rec_valid = np.linalg.norm(rec_items, axis=1) > 0
        rec_items = rec_items[rec_valid]
        if len(rec_items):
            ev_t_all = np.asarray([0 if v is None else v for v in feed["history__event_type"][:window]], dtype=np.int32)
            ev_s_all = np.asarray([0 if v is None else v for v in feed["history__event_source"][:window]], dtype=np.int32)
            r_et = ev_t_all[history_present][recent_vectors][rec_valid]
            r_es = ev_s_all[history_present][recent_vectors][rec_valid]
            sims = pair_cosine(items, rec_items)
            s1 = r_es == 1
            s0 = r_es == 0
            s1e1 = s1 & (r_et == 1)
            result["din_src1_max_cosine"] = sims[:, s1].max(axis=1) if s1.any() else np.zeros(len(item_ids), dtype=np.float32)
            result["din_src0_max_cosine"] = sims[:, s0].max(axis=1) if s0.any() else np.zeros(len(item_ids), dtype=np.float32)
            result["din_src1_ev1_max_cosine"] = sims[:, s1e1].max(axis=1) if s1e1.any() else np.zeros(len(item_ids), dtype=np.float32)
        else:
            z = np.zeros(len(item_ids), dtype=np.float32)
            result["din_src1_max_cosine"] = z
            result["din_src0_max_cosine"] = z
            result["din_src1_ev1_max_cosine"] = z
    if "din_long_max_cosine" in feature_names:
        long_mask = ~recent_vectors
        long_items = history_items[long_mask]
        long_valid = np.linalg.norm(long_items, axis=1) > 0
        long_items = long_items[long_valid]
        if len(long_items):
            ev_types_all = np.asarray([0 if v is None else v for v in feed["history__event_type"][:window]], dtype=np.int32)
            long_ev = ev_types_all[history_present][long_mask][long_valid]
            long_sims = pair_cosine(items, long_items)
            result["din_long_max_cosine"] = long_sims.max(axis=1)
            other_m = long_ev > 1
            result["din_long_ev_other_max"] = long_sims[:, other_m].max(axis=1) if other_m.any() else np.zeros(len(item_ids), dtype=np.float32)
        else:
            result["din_long_max_cosine"] = np.zeros(len(item_ids), dtype=np.float32)
            result["din_long_ev_other_max"] = np.zeros(len(item_ids), dtype=np.float32)
    if reference is not None:
        result.update(zip(behavior.FEATURES[reference.kind], reference.score(feed), strict=True))
    return result


def load_global_item_stats(include_val=False):
    stats_dir = Path(__file__).resolve().parent / "runs" / "global_stats"
    with_val_path = stats_dir / "item_stats_with_val.parquet"
    if include_val and with_val_path.is_file():
        path = with_val_path
        meta_path = stats_dir / "meta_with_val.json"
        print(f"Using train+val global item stats: {path.name}", flush=True)
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
    u_path = stats_dir / "user_stats.parquet"
    if u_path.is_file():
        udf = pl.read_parquet(u_path)
        u_tuple = (udf["user_id"].to_numpy(), udf["user_feeds"].to_numpy(), udf["user_shows"].to_numpy(), udf["user_clicks"].to_numpy())
    else:
        u_tuple = None
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
        u_tuple,
        prior,
        recent_prior,
        recent_cutoff,
    )


def apply_global_item_stats(candidates, global_stats, is_train, features, cold_drop_prob=0.0):
    (g_ids, g_shows, g_clicks, g_pos_sum, g_pos_shows, g_hist,
     g_r_shows, g_r_clicks, u_tuple, prior, recent_prior, recent_cutoff) = global_stats
    item_ids = candidates["item_id"].to_numpy()
    idx = np.searchsorted(g_ids, item_ids)
    clipped = np.minimum(idx, len(g_ids) - 1)
    found = (item_ids > 0) & (g_ids[clipped] == item_ids)
    shows = np.where(found, g_shows[clipped], 0).astype(np.float64)
    clicks = np.where(found, g_clicks[clipped], 0).astype(np.float64)
    pos_sum = np.where(found, g_pos_sum[clipped], 0).astype(np.float64)
    pos_shows = np.where(found, g_pos_shows[clipped], 0).astype(np.float64)
    hist_c = np.where(found, g_hist[clipped], 0).astype(np.float64)
    r_shows = np.where(found, g_r_shows[clipped], 0).astype(np.float64)
    r_clicks = np.where(found, g_r_clicks[clipped], 0).astype(np.float64)
    if is_train and "target" in candidates.columns:
        tgt = candidates["target"].to_numpy().astype(np.float64)
        shows = np.maximum(shows - 1.0, 0.0)
        clicks = np.maximum(clicks - tgt, 0.0)
        if "pos" in candidates.columns:
            pos_sum = np.maximum(pos_sum - candidates["pos"].to_numpy().astype(np.float64), 0.0)
            pos_shows = np.maximum(pos_shows - 1.0, 0.0)
        if "timestamp" in candidates.columns and recent_cutoff > 0:
            is_rec = candidates["timestamp"].to_numpy() >= recent_cutoff
            r_shows = np.where(is_rec, np.maximum(r_shows - 1.0, 0.0), r_shows)
            r_clicks = np.where(is_rec, np.maximum(r_clicks - tgt, 0.0), r_clicks)
    if is_train and cold_drop_prob > 0.0 and len(item_ids):
        seed = (int(item_ids[0]) * 1000003 + len(item_ids) * 97 + 42) & 0xFFFFFFFF
        drop = np.random.default_rng(seed).random(len(item_ids)) < cold_drop_prob
        shows = np.where(drop, 0.0, shows)
        clicks = np.where(drop, 0.0, clicks)
        pos_sum = np.where(drop, 0.0, pos_sum)
        pos_shows = np.where(drop, 0.0, pos_shows)
        hist_c = np.where(drop, 0.0, hist_c)
        r_shows = np.where(drop, 0.0, r_shows)
        r_clicks = np.where(drop, 0.0, r_clicks)
    shows_log = np.log1p(shows).astype(np.float32)
    ctr = ((clicks + 30.0 * prior) / (shows + 30.0)).astype(np.float32)
    mean_pos = ((pos_sum + 20.0 * 40.0) / (pos_shows + 20.0)).astype(np.float32)
    cols = [
        pl.Series("global_item_shows_log", shows_log, dtype=pl.Float32),
        pl.Series("global_item_ctr", ctr, dtype=pl.Float32),
        pl.Series("global_item_mean_pos", mean_pos, dtype=pl.Float32),
    ]
    if "global_recent_shows_log" in features:
        r_shows_log = np.log1p(r_shows).astype(np.float32)
        r_ctr = ((r_clicks + 20.0 * recent_prior) / (r_shows + 20.0)).astype(np.float32)
        cols.extend([
            pl.Series("global_recent_shows_log", r_shows_log, dtype=pl.Float32),
            pl.Series("global_recent_ctr", r_ctr, dtype=pl.Float32),
        ])
    if "global_item_hist_log" in features:
        cols.append(pl.Series("global_item_hist_log", np.log1p(hist_c).astype(np.float32), dtype=pl.Float32))
    if "global_user_ctr" in features and u_tuple is not None and "user_id" in candidates.columns:
        u_ids, u_feeds, u_shows, u_clicks = u_tuple
        cand_u = candidates["user_id"].to_numpy()
        uidx = np.searchsorted(u_ids, cand_u)
        uclip = np.minimum(uidx, len(u_ids) - 1)
        ufound = (cand_u > 0) & (u_ids[uclip] == cand_u)
        uf = np.where(ufound, u_feeds[uclip], 0).astype(np.float64)
        us = np.where(ufound, u_shows[uclip], 0).astype(np.float64)
        uc = np.where(ufound, u_clicks[uclip], 0).astype(np.float64)
        if is_train and "target" in candidates.columns:
            feed_len = candidates.select(pl.len().over("feed_id")).to_series().to_numpy().astype(np.float64)
            feed_hits = candidates.select(pl.col("target").sum().over("feed_id")).to_series().to_numpy().astype(np.float64)
            uf = np.maximum(uf - 1.0, 0.0)
            us = np.maximum(us - feed_len, 0.0)
            uc = np.maximum(uc - feed_hits, 0.0)
        cols.append(pl.Series("global_user_feeds_log", np.log1p(uf).astype(np.float32), dtype=pl.Float32))
        cols.append(pl.Series("global_user_ctr", ((uc + 50.0 * prior) / (us + 50.0)).astype(np.float32), dtype=pl.Float32))
    delta_cols = [
        (pl.col("global_item_ctr") - pl.col("global_item_ctr").mean().over("feed_id")).cast(pl.Float32).alias("global_item_ctr_delta"),
        (pl.col("global_item_mean_pos") - pl.col("global_item_mean_pos").mean().over("feed_id")).cast(pl.Float32).alias("global_item_pos_delta"),
    ]
    if "global_recent_ctr_delta" in features:
        delta_cols.append((pl.col("global_recent_ctr") - pl.col("global_recent_ctr").mean().over("feed_id")).cast(pl.Float32).alias("global_recent_ctr_delta"))
    if "global_item_hist_delta" in features:
        delta_cols.append((pl.col("global_item_hist_log") - pl.col("global_item_hist_log").mean().over("feed_id")).cast(pl.Float32).alias("global_item_hist_delta"))
    return candidates.with_columns(cols).with_columns(delta_cols)


_GRAPH_PROJ = np.array([0.5, -0.5, 0.5, 0.5], dtype=np.float32)


def _project_32d(vectors_128):
    v = np.asarray(vectors_128, dtype=np.float32).reshape(-1, 32, 4) @ _GRAPH_PROJ
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, norms, out=np.zeros_like(v), where=norms > 1e-6)


def load_global_graph(include_val=False):
    graph_dir = Path(__file__).resolve().parent / "runs" / "global_graph"
    with_val = graph_dir / "graph_with_val.npz"
    path = with_val if (include_val and with_val.is_file()) else (graph_dir / "graph_train.npz")
    if not path.is_file():
        raise FileNotFoundError(f"Global graph cache not found at {path}; run build_global_graph.py first")
    print(f"Using global audience graph: {path.name}", flush=True)
    z = np.load(path)
    return (
        z["hist_ids"], z["hist_vec"], z["hist_cnt"], z["hist_coh"],
        z["click_ids"], z["click_vec"], z["click_cnt"], z["click_coh"],
    )


def apply_global_graph(feeds, candidates, embeddings, global_graph, is_train):
    h_ids, h_vec, h_cnt, h_coh, *_ = global_graph
    u32 = _project_32d(feeds["user_embedding"].list.to_array(128).to_numpy())
    h32 = np.zeros_like(u32)
    w_decay = np.exp2(-np.arange(32, dtype=np.float32) / 8.0)
    for row_i, h_idx_s in enumerate(feeds["history__item_id__embedding_idx"].list.head(32)):
        arr = h_idx_s.to_numpy().astype(np.intp)
        valid = arr >= 0
        if valid.any():
            h32[row_i] = w_decay[:len(arr)][valid] @ _project_32d(embeddings[arr[valid]])
    hn = np.linalg.norm(h32, axis=1, keepdims=True)
    h32 = np.divide(h32, hn, out=np.zeros_like(h32), where=hn > 1e-6)

    lens = feeds["item_id"].list.len().to_numpy()
    owner = np.repeat(np.arange(feeds.height, dtype=np.int32), lens)
    cand_ids = candidates["item_id"].to_numpy().astype(np.int32)
    u_cand = u32[owner]
    h_cand = h32[owner]

    # Target-free history audience profile only (no click/target features -> zero target leak)
    hidx = np.minimum(np.searchsorted(h_ids, cand_ids), len(h_ids) - 1)
    hhit = (cand_ids > 0) & (h_ids[hidx] == cand_ids)
    hc = np.where(hhit, h_cnt[hidx], 0).astype(np.float32)
    hcoh = np.where(hhit, h_coh[hidx] / 255.0, 0.0).astype(np.float32)
    hprof = np.where(hhit[:, None], h_vec[hidx].astype(np.float32) / 127.0, 0.0)

    if is_train:
        h64_lists = feeds["history__item_id"].list.head(64).to_list()
        h_sub_mask = np.zeros(len(cand_ids), dtype=bool)
        pos_off = 0
        for r_i, L in enumerate(lens):
            hset = set(h64_lists[r_i])
            if hset:
                sl = cand_ids[pos_off : pos_off + L]
                for j, cid in enumerate(sl):
                    if cid > 0 and cid in hset:
                        h_sub_mask[pos_off + j] = True
            pos_off += L
        h_sub = h_sub_mask & hhit & (hc > 0)
        if h_sub.any():
            hc_new = np.maximum(hc[h_sub] - 1.0, 0.0)
            zeroed = hc_new <= 0
            hc[h_sub] = hc_new
            if zeroed.any():
                sub_idx = np.flatnonzero(h_sub)[zeroed]
                hprof[sub_idx] = 0.0
                hcoh[sub_idx] = 0.0

    g_hist_u = np.einsum("ij,ij->i", u_cand, hprof).astype(np.float32)
    g_hist_h = np.einsum("ij,ij->i", h_cand, hprof).astype(np.float32)

    return candidates.with_columns(
        pl.Series("g_hist_u", g_hist_u, dtype=pl.Float32),
        pl.Series("g_hist_h", g_hist_h, dtype=pl.Float32),
        pl.Series("g_hist_cnt", np.log1p(hc).astype(np.float32), dtype=pl.Float32),
        pl.Series("g_hist_coh", hcoh.astype(np.float32), dtype=pl.Float32),
    ).with_columns(
        (pl.col("g_hist_u") - pl.col("g_hist_u").mean().over("feed_id")).cast(pl.Float32).alias("g_hist_u_delta"),
    )


def expand_batch(feeds, embeddings, labeled, features, reference, clusters=None, global_stats=None, is_train=False, cold_drop_prob=0.0, global_graph=None):
    per_feed = [vector_features(feed, embeddings, features, reference, clusters) for feed in feeds.iter_rows(named=True)]
    has_pos = "pos" in feeds.columns
    need_user = "global_user_ctr" in features and "user_id" in feeds.columns
    has_ts = "timestamp" in feeds.columns and is_train and global_stats is not None
    arrays = [*CANDIDATE_COLUMNS, *(["target"] if labeled else []), *(["pos"] if has_pos else [])]
    select_cols = ["feed_id", *(["user_id"] if need_user else []), *(["timestamp"] if has_ts else []), *arrays]
    candidates = feeds.select(*select_cols).explode(arrays).drop("item_id__embedding_idx")
    if global_stats is not None:
        candidates = apply_global_item_stats(candidates, global_stats, is_train, features, cold_drop_prob=cold_drop_prob)
    if global_graph is not None:
        candidates = apply_global_graph(feeds, candidates, embeddings, global_graph, is_train)
    if has_ts:
        candidates = candidates.drop("timestamp")
    if need_user:
        candidates = candidates.drop("user_id")
    if "pos_bias" in features:
        candidates = candidates.with_columns(
            pl.col("pos").cast(pl.Float32).alias("pos_bias") if has_pos else pl.lit(0.0, dtype=pl.Float32).alias("pos_bias")
        )
    if has_pos:
        candidates = candidates.drop("pos")
    candidates = candidates.with_columns(
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
    extra_ranks = [
        expr
        for name in SIM_RANK_SOURCES
        for expr in (
            (pl.col(name).rank("average").over("feed_id") / pl.len().over("feed_id"))
            .cast(pl.Float32).alias(f"{name}_feed_percentile")
            if f"{name}_feed_percentile" in features else None,
            (pl.col(name) - pl.col(name).mean().over("feed_id"))
            .cast(pl.Float32).alias(f"{name}_feed_delta")
            if f"{name}_feed_delta" in features else None,
        )
        if expr is not None
    ]
    if "cat_share_in_feed" in features:
        extra_ranks.extend([
            (pl.len().over("feed_id", "category_id") / pl.len().over("feed_id")).cast(pl.Float32).alias("cat_share_in_feed"),
            ((pl.col("f5") - pl.col("f5").mean().over("feed_id")) / pl.col("f5").std().over("feed_id").fill_null(1.0).clip(lower_bound=1e-6)).cast(pl.Float32).alias("f5_zscore"),
            (pl.col("f5") - pl.col("f5").max().over("feed_id")).cast(pl.Float32).alias("f5_minus_max"),
            (pl.col("f5") - pl.col("f5").mean().over("feed_id", "category_id")).cast(pl.Float32).alias("f5_cat_delta"),
            ((pl.col("f3") - pl.col("f3").mean().over("feed_id")) / pl.col("f3").std().over("feed_id").fill_null(1.0).clip(lower_bound=1e-6)).cast(pl.Float32).alias("f3_zscore"),
            pl.len().over("feed_id").cast(pl.Float32).alias("feed_size"),
        ])
    if "f1_zscore" in features:
        extra_ranks.append(((pl.col("f1") - pl.col("f1").mean().over("feed_id")) / pl.col("f1").std().over("feed_id").fill_null(1.0).clip(lower_bound=1e-6)).cast(pl.Float32).alias("f1_zscore"))
    if "f1_minus_max" in features:
        extra_ranks.extend([
            (pl.col("f1") - pl.col("f1").max().over("feed_id")).cast(pl.Float32).alias("f1_minus_max"),
            (pl.col("f1") - pl.col("f1").mean().over("feed_id", "category_id")).cast(pl.Float32).alias("f1_cat_delta"),
            (pl.col("f3") - pl.col("f3").mean().over("feed_id", "category_id")).cast(pl.Float32).alias("f3_cat_delta"),
        ])
    if "f2_zscore" in features:
        extra_ranks.extend([
            ((pl.col("f2") - pl.col("f2").mean().over("feed_id")) / pl.col("f2").std().over("feed_id").fill_null(1.0).clip(lower_bound=1e-6)).cast(pl.Float32).alias("f2_zscore"),
            ((pl.col("f4") - pl.col("f4").mean().over("feed_id")) / pl.col("f4").std().over("feed_id").fill_null(1.0).clip(lower_bound=1e-6)).cast(pl.Float32).alias("f4_zscore"),
            (pl.col("f5") * pl.col("din_decay_max_cosine")).cast(pl.Float32).alias("f5_x_din_decay"),
        ])
    if extra_ranks:
        candidates = candidates.with_columns(extra_ranks)
    if "f5z_x_din_decay" in features:
        candidates = candidates.with_columns(
            (pl.col("f5_zscore") * pl.col("din_decay_max_cosine")).cast(pl.Float32).alias("f5z_x_din_decay")
        )
    return candidates


def feed_batches(split, columns, max_feeds=0, *, offset=0):
    parts = sorted((split / "impressions").glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No part-*.parquet files under {split / 'impressions'}")

    processed = 0
    for part in parts:
        with pq.ParquetFile(part) as parquet:
            if offset and offset >= parquet.metadata.num_rows:
                offset -= parquet.metadata.num_rows
                continue
            for batch in parquet.iter_batches(batch_size=512, columns=columns):
                if offset >= batch.num_rows:
                    offset -= batch.num_rows
                    continue
                if offset:
                    batch = batch.slice(offset)
                    offset = 0
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


_worker_embeddings = None


def init_feature_worker(split):
    global _worker_embeddings
    _worker_embeddings = open_embeddings(split)


def expand_in_worker(feeds, labeled, features):
    return expand_batch(feeds, _worker_embeddings, labeled, features, None)


def parallel_candidates(split, batches, labeled, features, workers):
    # Bound outstanding frames and yield them in input order for CatBoost groups.
    pending = deque()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=init_feature_worker, initargs=(split,)) as pool:
        for batch in batches:
            pending.append(pool.submit(expand_in_worker, batch, labeled, features))
            if len(pending) == 2 * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def candidate_batches(split, *, features=FEATURES, labeled=False, max_feeds=0,
                      reference=None, after=None, before=None, centers=None, device="cpu", offset=0,
                      feature_workers=1, cold_drop_prob=0.0):
    columns = ["feed_id", "user_embedding", *CANDIDATE_COLUMNS]
    if "history_length" in features:
        columns.extend(HISTORY_COLUMNS)
    if EVENT_DIN_FEATURES[0] in features:
        columns.extend(["history__event_type", "history__event_source"])
    need_global = GLOBAL_STATS_FEATURES[0] in features
    need_graph = GRAPH_FEATURES[0] in features
    if "pos_bias" in features or (need_global and split.name == "train"):
        first_part = next(iter(sorted((split / "impressions").glob("part-*.parquet"))), None)
        if first_part is not None and "pos" in pq.ParquetFile(first_part).schema_arrow.names:
            columns.append("pos")
    if reference is not None or "global_user_ctr" in features:
        columns.append("user_id")
    if labeled:
        columns.append("target")
    if after is not None or before is not None or (need_global and split.name == "train" and "global_recent_shows_log" in features):
        columns.append("timestamp")
    def selected_batches():
        for batch in feed_batches(split, columns, max_feeds, offset=offset):
            if after is not None:
                batch = batch.filter(pl.col("timestamp") >= after)
            if before is not None:
                batch = batch.filter(pl.col("timestamp") < before)
            if batch.height:
                yield batch

    if feature_workers > 1 and not need_global and not need_graph:
        if reference is not None or centers is not None:
            raise ValueError("Parallel feature workers require features without fitted reference or cluster centers")
        yield from parallel_candidates(split, selected_batches(), labeled, features, feature_workers)
        return

    embeddings = open_embeddings(split)
    global_stats = load_global_item_stats(include_val=(split.name == "test")) if need_global else None
    global_graph = load_global_graph(include_val=(split.name == "test")) if need_graph else None
    is_train = split.name == "train"
    clusters = None
    if centers is not None:
        candidate_key = "item_id__embedding_idx"
        history_key = "history__item_id__embedding_idx"
        batches = feed_batches(split, [candidate_key, history_key], max_feeds, offset=offset)
        indices = np.unique(np.concatenate([
            batch[key].explode().unique().to_numpy() for batch in batches for key in (candidate_key, history_key)
        ]))
        clusters = embedding_clusters.assign(embeddings, centers, device, indices[indices >= 0].astype(np.intp))
    processed_feeds = 0
    next_report = 25_000
    for batch in selected_batches():
        yield expand_batch(batch, embeddings, labeled, features, reference, clusters, global_stats, is_train,
                           cold_drop_prob=cold_drop_prob if is_train else 0.0, global_graph=global_graph)
        processed_feeds += batch.height
        if processed_feeds >= next_report:
            print(f"  prepared {processed_feeds:,} feeds ({split.name})...", flush=True)
            next_report += 25_000


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
    selection = {"max_feeds": args.train_feeds, "offset": args.train_offset}
    cold_drop = args.cold_drop if args.cold_drop is not None else (0.35 if args.variant == "din-events-v3-robust" else 0.0)
    if cold_drop > 0:
        print(f"Cold-item global stats dropout enabled: p={cold_drop:.2f}", flush=True)
    if kind is None:
        return read_candidates(split, features=features, labeled=True, centers=centers,
                               device=args.device, feature_workers=args.feature_workers,
                               cold_drop_prob=cold_drop, **selection), None
    if args.feature_workers > 1:
        raise ValueError("Parallel feature workers require a ranker without a fitted behavior reference")

    columns = ["timestamp", "user_id", "item_id", "category_id", "target", *HISTORY_COLUMNS[:2]]
    feeds = list(feed_batches(split, columns, **selection))
    if not feeds:
        raise ValueError("No training feeds")
    past = pl.concat(feeds)
    times = np.sort(past["timestamp"].to_numpy())
    boundaries = sorted(set(int(times[len(times) * index // 4]) for index in (1, 2, 3)) - {int(times[0])})
    reference = behavior.Reference(kind)
    chunks = []
    for after, before in zip([None, *boundaries], [*boundaries, None], strict=True):
        chunks.append(read_candidates(split, features=features, labeled=True, reference=reference,
                                      after=after, before=before, **selection))
        if before is not None:
            reference = behavior.fit(kind, past.filter(pl.col("timestamp") < before), args.threads, args.device)
    return pl.concat(chunks, rechunk=True), behavior.fit(kind, past, args.threads, args.device)


def train_centers(args):
    split = args.data / "train"
    indices = "item_id__embedding_idx"
    batches = feed_batches(split, [indices], args.train_feeds, offset=args.train_offset)
    used = pl.concat(list(batches))[indices].explode().unique().sort().to_numpy()
    return embedding_clusters.fit(open_embeddings(split), used[used >= 0], args.device)


def new_model(args, iterations):
    if args.variant == "lambda":
        loss = "LambdaMart:metric=NDCG"
    elif args.variant in {"din-events-v3-qsm", "din-events-v3-robust", "din-events-v4-graph", "din-events-v5-qsm", "din-events-v4-qsm"}:
        loss = "QuerySoftMax"
    else:
        loss = "YetiRank"
    depth = 7 if args.variant in {"din-events-v3-qsm", "din-events-v3-robust", "din-events-v4-graph", "din-events-v5-qsm", "din-events-v4-qsm"} else 6
    return CatBoostRanker(
        loss_function=loss,
        eval_metric="RMSE" if args.device == "gpu" else NDCG,
        task_type=args.device.upper(), iterations=iterations, depth=depth, learning_rate=0.1,
        random_seed=42, thread_count=args.threads, allow_writing_files=False,
        **({"devices": "0", "gpu_ram_part": .65} if args.device == "gpu" else {}),
    )


def try_gpu_scores(model, pool, threads):
    try:
        return CatBoost.predict(model, pool, thread_count=threads, task_type="GPU")
    except CatBoostError as error:
        if "Model contains categorical features, GPU evaluation impossible" not in str(error):
            raise
        return None


def raw_scores(model, pool, threads, device):
    if device == "gpu":
        prediction = try_gpu_scores(model, pool, threads)
        if prediction is not None:
            return prediction
        warnings.warn("CatBoost computes unsupported categorical splits on CPU at prediction time.",
                      RuntimeWarning, stacklevel=2)
    return CatBoost.predict(model, pool, thread_count=threads, task_type="CPU")


def select_gpu_checkpoint(model, tuning, val_pool, args):
    from accelerator import gpu_ndcg_evaluator

    metric = gpu_ndcg_evaluator(tuning)
    step = max(1, args.iterations // 20)
    if args.variant not in {"din-events-v2", "din-events-v2-ties", "din-events-v3-qsm", "din-events-v3-robust", "din-events-v4-graph", "din-events-v5-qsm", "din-events-v4-qsm"}:
        step = min(5, step)
    checkpoints = sorted({1, model.tree_count_, *range(step, model.tree_count_, step)})
    full_prediction = try_gpu_scores(model, val_pool, args.threads)
    if full_prediction is None:
        print(f"{args.variant}: CatBoost evaluates categorical splits on CPU; NDCG stays on GPU", flush=True)
    best_ndcg, best_tree, curve = -1, 1, []
    for trees in checkpoints:
        if full_prediction is None:
            prediction = CatBoost.predict(model, val_pool, thread_count=args.threads, ntree_end=trees, task_type="CPU")
        elif trees == model.tree_count_:
            prediction = full_prediction
        else:
            # The CatBoost GPU evaluator applies all trees regardless of ntree_end.
            checkpoint = model.copy()
            checkpoint.shrink(trees)
            prediction = raw_scores(checkpoint, val_pool, args.threads, "gpu")
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
    teacher_data = read_candidates(split, features=features, labeled=True, max_feeds=args.train_feeds,
                                   offset=args.train_offset)
    times = pl.concat(list(feed_batches(split, ["feed_id", "timestamp"], args.train_feeds, offset=args.train_offset)))
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


def with_stack_scores(frame, predictions, *, train=False):
    sources = [pl.read_parquet(path, columns=[*KEYS, "score"]) for path in predictions]
    if train:
        frame = frame.filter(pl.col("feed_id").is_in(sources[0]["feed_id"].unique()))
    for name, source in zip(STACK_FEATURES, sources, strict=True):
        values = attach_scores(frame.select(KEYS), source).with_columns(zscore("score").alias(name)).drop("score")
        frame = frame.join(values, on=KEYS, validate="1:1")
    return frame.sort("feed_id", maintain_order=True)


def train(args):
    if args.output.exists():
        raise FileExistsError(args.output)

    features = VARIANTS[args.variant]
    print(f"Preparing features for {args.variant} ({args.device}, CPU workers={args.feature_workers})", flush=True)
    centers = train_centers(args) if args.variant == "embedding-clusters" else None
    training, reference = load_training(args, features, centers)
    if args.variant == "history-distill":
        training = distillation_targets(args, training)
    if args.variant in STACK_VARIANTS:
        training = with_stack_scores(training, args.stack_train, train=True)
    train_feeds = training["feed_id"].n_unique()
    train_pool = make_pool(training, features, args.threads, labeled=True)
    del training
    val = read_candidates(args.data / "val", features=features, labeled=True, max_feeds=args.val_feeds,
                          reference=reference, centers=centers, device=args.device,
                          feature_workers=args.feature_workers)
    if args.variant == "stack-gbdt":
        val = with_stack_scores(val, args.stack_val)
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
        "train_feeds": train_feeds,
        "val_feeds": val["feed_id"].n_unique(),
        "train_candidates": train_pool.num_row(),
        "val_candidates": val.height,
        "train_feed_limit": args.train_feeds,
        "train_offset": args.train_offset,
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
    report["peak_private_mib"] = peak_ram_mib(private=True)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "tuning_curve"}, indent=2))
    if train_feeds >= 10_000 and not args.output.name.startswith("_"):
        try:
            import notify_tg
            notify_tg.send(notify_tg.format_ranker(args.output, report))
        except Exception:
            pass


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
    batches = candidate_batches(args.data / "test", features=model.feature_names_, reference=reference,
                                centers=centers, device=args.device, feature_workers=args.feature_workers)
    if STACK_FEATURES[0] in model.feature_names_:
        if not args.stack_predictions:
            raise ValueError("stack-gbdt needs avito-joint and gru-din test scores in --stack-predictions")
        candidates = with_stack_scores(pl.concat(list(batches)), args.stack_predictions)
        predictions = score_batch(model, candidates, args.threads, args.device)
    else:
        predictions = pl.concat([score_batch(model, batch, args.threads, args.device) for batch in batches])
    submission = attach_scores(template, predictions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.write_parquet(args.output)
    print(f"Saved {submission.height:,} scores to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    training = commands.add_parser("train")
    training.add_argument("--train-feeds", type=int, default=50_000, help="0 uses the full train split")
    training.add_argument("--train-offset", type=int, default=0, help="Skip the first N training feeds")
    training.add_argument("--val-feeds", type=int, default=10_000, help="0 uses the full val split")
    training.add_argument("--iterations", type=int, default=500)
    training.add_argument("--variant", choices=VARIANTS, default="history")
    training.add_argument("--cold-drop", type=float, default=None, help="Drop global stats on train items with prob p (default 0.35 for din-events-v3-robust)")
    training.add_argument("--tune-before", type=int, help="Use only validation feeds strictly before this Unix time for early stopping")
    training.add_argument("--stack-train", nargs=2, type=Path, metavar=("JOINT", "GRU"))
    training.add_argument("--stack-val", nargs=2, type=Path, metavar=("JOINT", "GRU"))
    training.set_defaults(run=train)
    scoring = commands.add_parser("predict")
    scoring.add_argument("--model", type=Path, required=True)
    scoring.add_argument("--stack-predictions", nargs=2, type=Path, metavar=("JOINT", "GRU"))
    scoring.set_defaults(run=predict)
    for command in (training, scoring):
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--threads", type=int, default=4)
        command.add_argument("--feature-workers", type=int, default=1,
                             help="Parallel CPU feature expansion for models without fitted references")
        command.add_argument("--device", choices=["cpu", "gpu"])
        command.add_argument("--gpu", "--gpu-id", dest="gpu_id", help="NVIDIA index or UUID from nvidia-smi")

    args = parser.parse_args()
    if min(getattr(args, name, 0) for name in ("train_feeds", "val_feeds", "train_offset")) < 0:
        parser.error("Feed limits and offset must be nonnegative")
    if min(args.threads, args.feature_workers, getattr(args, "iterations", 1)) < 1:
        parser.error("Threads, feature workers and iterations must be positive")
    if args.gpu_id is not None and args.device == "cpu":
        parser.error("--gpu cannot be combined with --device cpu")
    args.device = args.device or ("gpu" if args.gpu_id is not None else "cpu")
    if getattr(args, "variant", None) == "lambda" and args.device != "cpu":
        parser.error("CatBoost LambdaMart supports CPU training")
    if getattr(args, "variant", None) in STACK_VARIANTS and (
        not args.stack_train or (args.variant == "stack-gbdt" and not args.stack_val)
    ):
        parser.error("Stack variants require --stack-train; stack-gbdt also requires --stack-val")
    if args.device == "gpu":
        from accelerator import select_gpu

        selected = select_gpu(args.gpu_id)
        print(f"CatBoost GPU: {selected.name} ({selected.uuid})", flush=True)
    try:
        args.run(args)
    except Exception as exc:
        if (args.run is train and not args.output.name.startswith("_")
                and (args.train_feeds == 0 or args.train_feeds >= 10_000)):
            try:
                import notify_tg
                notify_tg.send(f"❌ {args.variant} failed ({args.output.name}): {type(exc).__name__}: {exc}")
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()
