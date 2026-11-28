"""Build a full 3.15M-feed user-item audience graph (history + click profiles) and test blending with sub-v6."""

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
from catboost import CatBoostRanker, Pool

from evaluation import KEYS, attach_scores, ndcg, peak_ram_mib, zscore
from experiments import validation
from ranker import BASE_FEATURES, feed_batches, open_embeddings


PROJ = np.array([0.5, -0.5, 0.5, 0.5], dtype=np.float32)


def project_32d(vectors_128: np.ndarray) -> np.ndarray:
    v = np.asarray(vectors_128, dtype=np.float32).reshape(-1, 32, 4) @ PROJ
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, norms, out=np.zeros_like(v), where=norms > 1e-6)


def collect_target_items(data_dir: Path, train_feeds: int) -> np.ndarray:
    t0 = perf_counter()
    parts = []
    for batch in feed_batches(data_dir / "train", ["item_id"], train_feeds):
        parts.append(batch["item_id"].explode().unique().to_numpy())
    for split in ("val", "test"):
        for batch in feed_batches(data_dir / split, ["item_id"], 0):
            parts.append(batch["item_id"].explode().unique().to_numpy())
    items = np.unique(np.concatenate(parts))
    items = items[items > 0].astype(np.int32)
    print(f"Collected {len(items):,} unique target item_ids (train {train_feeds:,} + val + test) in {perf_counter()-t0:.1f}s", flush=True)
    return items


def accumulate_split_into_graph(
    split_dir: Path,
    target_items: np.ndarray,
    hist_sum: np.ndarray,
    hist_cnt: np.ndarray,
    click_sum: np.ndarray,
    click_cnt: np.ndarray,
    chunk_parts: int = 32,
):
    t0 = perf_counter()
    parts = sorted((split_dir / "impressions").glob("part-*.parquet"))
    n_items = len(target_items)
    total_feeds = 0

    for i in range(0, len(parts), chunk_parts):
        chunk = parts[i : i + chunk_parts]
        df = pl.concat([
            pl.read_parquet(p, columns=["item_id", "target", "user_embedding", "history__item_id"])
            for p in chunk
        ])
        total_feeds += df.height
        u32 = project_32d(df["user_embedding"].list.to_array(128).to_numpy())

        # 1. History edges (first 64 events per feed)
        h_df = (
            df.with_row_index("r")
            .select("r", pl.col("history__item_id").list.head(64).alias("iid"))
            .explode("iid")
            .filter(pl.col("iid") > 0)
        )
        h_ids = h_df["iid"].to_numpy()
        h_rows = h_df["r"].to_numpy()
        idx = np.searchsorted(target_items, h_ids)
        idx_clip = np.minimum(idx, n_items - 1)
        hit = target_items[idx_clip] == h_ids
        hit_idx = idx_clip[hit]
        hit_rows = h_rows[hit]
        for off in range(0, len(hit_idx), 262_144):
            sl = slice(off, off + 262_144)
            np.add.at(hist_sum, hit_idx[sl], u32[hit_rows[sl]])
        np.add.at(hist_cnt, hit_idx, 1)

        # 2. Click edges (target == 1)
        c_df = (
            df.with_row_index("r")
            .select("r", "item_id", "target")
            .explode("item_id", "target")
            .filter((pl.col("target") > 0) & (pl.col("item_id") > 0))
        )
        c_ids = c_df["item_id"].to_numpy()
        c_rows = c_df["r"].to_numpy()
        if len(c_ids):
            cidx = np.searchsorted(target_items, c_ids)
            cclip = np.minimum(cidx, n_items - 1)
            chit = target_items[cclip] == c_ids
            chit_idx = cclip[chit]
            chit_rows = c_rows[chit]
            np.add.at(click_sum, chit_idx, u32[chit_rows])
            np.add.at(click_cnt, chit_idx, 1)

        del df, u32, h_df, h_ids, h_rows, c_df, c_ids, c_rows
        if (i // chunk_parts + 1) % 4 == 0 or i + chunk_parts >= len(parts):
            print(f"  [{split_dir.name}] {min(i + chunk_parts, len(parts))}/{len(parts)} parts ({total_feeds:,} feeds) in {perf_counter()-t0:.1f}s", flush=True)


def extract_features(
    split_dir: Path,
    max_feeds: int,
    labeled: bool,
    is_train: bool,
    target_items: np.ndarray,
    hist_sum: np.ndarray,
    hist_cnt: np.ndarray,
    click_sum: np.ndarray,
    click_cnt: np.ndarray,
) -> pl.DataFrame:
    cols = ["feed_id", "item_id", "category_id", *BASE_FEATURES, "item_id__embedding_idx",
            "user_embedding", "history__item_id", "history__item_id__embedding_idx"]
    if labeled:
        cols.append("target")
    embeddings = open_embeddings(split_dir)
    n_items = len(target_items)
    w_decay = np.exp2(-np.arange(32, dtype=np.float32) / 8.0)

    frames = []
    seen = 0
    for batch in feed_batches(split_dir, cols, max_feeds):
        seen += batch.height
        u_raw = batch["user_embedding"].list.to_array(128).to_numpy().astype(np.float32)
        u32 = project_32d(u_raw)

        # Build 32-D decayed user history vector per feed + 128-D user_item_cosine
        h32 = np.zeros_like(u32)
        h_embs = batch["history__item_id__embedding_idx"].list.head(32)
        for row_i, h_idx_s in enumerate(h_embs):
            arr = h_idx_s.to_numpy().astype(np.intp)
            valid = arr >= 0
            if valid.any():
                vecs = project_32d(embeddings[arr[valid]])
                h32[row_i] = w_decay[:len(arr)][valid] @ vecs
        h_norms = np.linalg.norm(h32, axis=1, keepdims=True)
        h32 = np.divide(h32, h_norms, out=np.zeros_like(h32), where=h_norms > 1e-6)

        lens = batch["item_id"].list.len().to_numpy()
        owner = np.repeat(np.arange(batch.height, dtype=np.int32), lens)
        cand_ids = batch["item_id"].explode().to_numpy().astype(np.int32)
        cand_emb_idx = batch["item_id__embedding_idx"].explode().to_numpy().astype(np.intp)

        # Standard user_item_cosine (128-D)
        u_cos = np.zeros(len(cand_ids), dtype=np.float32)
        valid_c = cand_emb_idx >= 0
        if valid_c.any():
            c_vecs = embeddings[cand_emb_idx[valid_c]].astype(np.float32)
            u_rep = u_raw[owner[valid_c]]
            dot = np.einsum("ij,ij->i", c_vecs, u_rep)
            denom = np.linalg.norm(c_vecs, axis=1) * np.linalg.norm(u_rep, axis=1)
            u_cos[valid_c] = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 1e-6)

        # Lookup graph profiles
        idx = np.searchsorted(target_items, cand_ids)
        idx_clip = np.minimum(idx, n_items - 1)
        hit = (cand_ids > 0) & (target_items[idx_clip] == cand_ids)

        hs = np.where(hit[:, None], hist_sum[idx_clip], 0.0).astype(np.float32)
        hc = np.where(hit, hist_cnt[idx_clip], 0).astype(np.float32)
        cs = np.where(hit[:, None], click_sum[idx_clip], 0.0).astype(np.float32)
        cc = np.where(hit, click_cnt[idx_clip], 0).astype(np.float32)

        # Exact Leave-One-Out on train
        if is_train and labeled:
            tgt = batch["target"].explode().to_numpy()
            c_sub = (tgt > 0) & hit & (cc > 0)
            if c_sub.any():
                cs[c_sub] -= u32[owner[c_sub]]
                cc[c_sub] = np.maximum(cc[c_sub] - 1.0, 0.0)
            # Subtract self from hist profile if candidate appears in this feed's history[:64]
            h64_lists = batch["history__item_id"].list.head(64).to_list()
            h_sub_mask = np.zeros(len(cand_ids), dtype=bool)
            pos_off = 0
            for r_i, L in enumerate(lens):
                hset = set(h64_lists[r_i])
                if hset:
                    sl_ids = cand_ids[pos_off : pos_off + L]
                    for j, cid in enumerate(sl_ids):
                        if cid > 0 and cid in hset:
                            h_sub_mask[pos_off + j] = True
                pos_off += L
            h_sub = h_sub_mask & hit & (hc > 0)
            if h_sub.any():
                hs[h_sub] -= u32[owner[h_sub]]
                hc[h_sub] = np.maximum(hc[h_sub] - 1.0, 0.0)

        h_norm = np.linalg.norm(hs, axis=1)
        h_prof = np.divide(hs, h_norm[:, None], out=np.zeros_like(hs), where=h_norm[:, None] > 1e-6)
        c_norm = np.linalg.norm(cs, axis=1)
        c_prof = np.divide(cs, c_norm[:, None], out=np.zeros_like(cs), where=c_norm[:, None] > 1e-6)

        u_cand = u32[owner]
        h_cand = h32[owner]
        g_hist_u = np.einsum("ij,ij->i", u_cand, h_prof).astype(np.float32)
        g_hist_h = np.einsum("ij,ij->i", h_cand, h_prof).astype(np.float32)
        g_hist_cnt = np.log1p(hc).astype(np.float32)
        g_hist_coh = np.divide(h_norm, np.maximum(hc, 1.0)).astype(np.float32)

        g_click_u = np.einsum("ij,ij->i", u_cand, c_prof).astype(np.float32)
        g_click_h = np.einsum("ij,ij->i", h_cand, c_prof).astype(np.float32)
        g_click_cnt = np.log1p(cc).astype(np.float32)
        g_click_coh = np.divide(c_norm, np.maximum(cc, 1.0)).astype(np.float32)

        exp_cols = ["item_id", "category_id", *BASE_FEATURES, *(["target"] if labeled else [])]
        out_df = batch.select("feed_id", *exp_cols).explode(exp_cols).with_columns(
            pl.Series("user_item_cosine", u_cos, dtype=pl.Float32),
            pl.Series("g_hist_u", g_hist_u, dtype=pl.Float32),
            pl.Series("g_hist_h", g_hist_h, dtype=pl.Float32),
            pl.Series("g_hist_cnt", g_hist_cnt, dtype=pl.Float32),
            pl.Series("g_hist_coh", g_hist_coh, dtype=pl.Float32),
            pl.Series("g_click_u", g_click_u, dtype=pl.Float32),
            pl.Series("g_click_h", g_click_h, dtype=pl.Float32),
            pl.Series("g_click_cnt", g_click_cnt, dtype=pl.Float32),
            pl.Series("g_click_coh", g_click_coh, dtype=pl.Float32),
        ).with_columns(
            (pl.col("g_hist_u") - pl.col("g_hist_u").mean().over("feed_id")).cast(pl.Float32).alias("g_hist_u_delta"),
            (pl.col("g_click_u") - pl.col("g_click_u").mean().over("feed_id")).cast(pl.Float32).alias("g_click_u_delta"),
            ((pl.col("f5") - pl.col("f5").mean().over("feed_id")) / pl.col("f5").std().over("feed_id").fill_null(1.0).clip(lower_bound=1e-6)).cast(pl.Float32).alias("f5_zscore"),
        )
        frames.append(out_df)
        if seen % 50_000 < 512 and seen >= 50_000:
            print(f"  extracted {seen:,} feeds ({split_dir.name})...", flush=True)
    return pl.concat(frames, rechunk=True)


GRAPH_ONLY_FEATURES = [
    "g_hist_u", "g_hist_h", "g_hist_cnt", "g_hist_coh", "g_hist_u_delta",
    "g_click_u", "g_click_h", "g_click_cnt", "g_click_coh", "g_click_u_delta",
]
GRAPH_BASE_FEATURES = [*GRAPH_ONLY_FEATURES, *BASE_FEATURES, "f5_zscore", "user_item_cosine"]


def fit_and_eval(train_df, val_df, early_ids, late_ids, features, name, iterations=400, depth=6):
    t0 = perf_counter()
    tr_pool = Pool(train_df.select(features), label=train_df["target"], group_id=train_df["feed_id"], thread_count=8)
    early_df = val_df.filter(pl.col("feed_id").is_in(early_ids))
    ev_pool = Pool(early_df.select(features), label=early_df["target"], group_id=early_df["feed_id"], thread_count=8)
    model = CatBoostRanker(
        loss_function="QuerySoftMax", eval_metric="RMSE", task_type="GPU", devices="0", gpu_ram_part=0.65,
        iterations=iterations, depth=depth, learning_rate=0.08, random_seed=42, thread_count=8,
        allow_writing_files=False,
    )
    model.fit(tr_pool, verbose=100)
    va_pool = Pool(val_df.select(features), thread_count=8)
    preds = val_df.select(KEYS).with_columns(pl.Series("score", model.predict(va_pool).astype(np.float32)))
    joined = attach_scores(val_df.select(*KEYS, "target"), preds)
    m_all = ndcg(joined)["ndcg"]
    m_early = ndcg(joined.filter(pl.col("feed_id").is_in(early_ids)))["ndcg"]
    m_late = ndcg(joined.filter(pl.col("feed_id").is_in(late_ids)))["ndcg"]
    print(f"[{name}] early={m_early:.6f}  holdout={m_late:.6f}  full={m_all:.6f}  ({perf_counter()-t0:.1f}s)", flush=True)
    return model, preds, {"early": m_early, "holdout": m_late, "full": m_all}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-feeds", type=int, default=200_000)
    parser.add_argument("--iterations", type=int, default=400)
    args = parser.parse_args()

    t_start = perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)

    target_items = collect_target_items(args.data, args.train_feeds)
    n = len(target_items)
    hist_sum = np.zeros((n, 32), dtype=np.float32)
    hist_cnt = np.zeros(n, dtype=np.int32)
    click_sum = np.zeros((n, 32), dtype=np.float32)
    click_cnt = np.zeros(n, dtype=np.int32)

    print("Streaming ALL 3.15M train feeds into collaborative audience graph...", flush=True)
    accumulate_split_into_graph(args.data / "train", target_items, hist_sum, hist_cnt, click_sum, click_cnt)

    print("Extracting graph features for train & val...", flush=True)
    train_df = extract_features(args.data / "train", args.train_feeds, True, True, target_items, hist_sum, hist_cnt, click_sum, click_cnt)
    val_df = extract_features(args.data / "val", 0, True, False, target_items, hist_sum, hist_cnt, click_sum, click_cnt)

    val_hist_cov = float((val_df["g_hist_cnt"] > 0).mean())
    val_click_cov = float((val_df["g_click_cnt"] > 0).mean())
    val_any_cov = float(((val_df["g_hist_cnt"] > 0) | (val_df["g_click_cnt"] > 0)).mean())
    print(f"VAL graph coverage (3.15M train): hist={val_hist_cov:.1%}, click={val_click_cov:.1%}, any={val_any_cov:.1%}", flush=True)

    labels, cutoff = validation(args.data, 0)
    early_ids = labels.filter(pl.col("timestamp") < cutoff)["feed_id"].unique()
    late_ids = labels.filter(pl.col("timestamp") >= cutoff)["feed_id"].unique()
    holdout = labels.filter(pl.col("timestamp") >= cutoff)

    m_only, p_only, r_only = fit_and_eval(train_df, val_df, early_ids, late_ids, GRAPH_ONLY_FEATURES, "graph-only", iterations=args.iterations, depth=5)
    m_base, p_base, r_base = fit_and_eval(train_df, val_df, early_ids, late_ids, GRAPH_BASE_FEATURES, "graph+base", iterations=args.iterations, depth=6)
    p_only.write_parquet(args.output / "val_graph_only.parquet")
    p_base.write_parquet(args.output / "val_graph_base.parquet")
    m_only.save_model(str(args.output / "model_graph_only.cbm"))
    m_base.save_model(str(args.output / "model_graph_base.cbm"))
    del train_df, val_df
    gc.collect()

    # Check blending with sub-v6 (45% rob21 + 55% xl3)
    runs = Path(__file__).resolve().parent / "runs"
    def load_z(path, col="z"):
        return pl.read_parquet(path, columns=[*KEYS, "score"]).with_columns(zscore("score").alias(col)).drop("score")
    def score_sub(sub, pred):
        fids = sub["feed_id"].unique()
        return ndcg(attach_scores(sub, pred.filter(pl.col("feed_id").is_in(fids))))["ndcg"]

    rob21 = load_z(runs / "v3-robust-full/merged/din-events-v3-robust/val_predictions.parquet", "z_rob")
    p1 = load_z(runs / "gru-din-xl-p1/val_predictions.parquet", "z1")
    p2 = load_z(runs / "sub-v6/val_p2.parquet", "z2")
    p3 = load_z(runs / "sub-v6/val_p3.parquet", "z3")
    xl3 = (p1.join(p2, on=KEYS).join(p3, on=KEYS)
             .select(*KEYS, ((pl.col("z1") + pl.col("z2") + pl.col("z3")) / 3.0).alias("raw"))
             .with_columns(zscore("raw").alias("z_xl")).drop("raw"))
    sub_v6 = (rob21.join(xl3, on=KEYS)
                   .select(*KEYS, (0.45 * pl.col("z_rob") + 0.55 * pl.col("z_xl")).alias("raw"))
                   .with_columns(zscore("raw").alias("z_v6")).drop("raw"))

    g_only_z = p_only.with_columns(zscore("score").alias("z_go")).drop("score")
    g_base_z = p_base.with_columns(zscore("score").alias("z_gb")).drop("score")
    m = sub_v6.join(g_only_z, on=KEYS).join(g_base_z, on=KEYS)

    c_go = float(np.corrcoef(m["z_v6"].to_numpy(), m["z_go"].to_numpy())[0, 1])
    c_gb = float(np.corrcoef(m["z_v6"].to_numpy(), m["z_gb"].to_numpy())[0, 1])
    print(f"Corr with sub-v6: graph-only={c_go:.4f}, graph+base={c_gb:.4f}", flush=True)

    base_h = score_sub(holdout, m.select(*KEYS, pl.col("z_v6").alias("score")))
    base_f = score_sub(labels, m.select(*KEYS, pl.col("z_v6").alias("score")))
    print(f"Baseline sub-v6: holdout={base_h:.6f}  full={base_f:.6f}", flush=True)

    best_tag, best_w, best_h, best_f = "none", 0.0, base_h, base_f
    for tag, col in (("graph-only", "z_go"), ("graph+base", "z_gb")):
        for w in (0.03, 0.06, 0.10, 0.15):
            b = m.select(*KEYS, ((1.0 - w) * pl.col("z_v6") + w * pl.col(col)).alias("score"))
            h = score_sub(holdout, b)
            f = score_sub(labels, b)
            print(f"  + {int(w*100):2d}% {tag:10s}: holdout={h:.6f} ({h-base_h:+.6f})  full={f:.6f} ({f-base_f:+.6f})", flush=True)
            if f > best_f:
                best_tag, best_w, best_h, best_f = tag, w, h, f

    # Also add val into graph and check test coverage + save test predictions
    print("Adding val into graph and scoring test...", flush=True)
    accumulate_split_into_graph(args.data / "val", target_items, hist_sum, hist_cnt, click_sum, click_cnt)
    test_df = extract_features(args.data / "test", 0, False, False, target_items, hist_sum, hist_cnt, click_sum, click_cnt)
    test_hist_cov = float((test_df["g_hist_cnt"] > 0).mean())
    test_click_cov = float((test_df["g_click_cnt"] > 0).mean())
    test_any_cov = float(((test_df["g_hist_cnt"] > 0) | (test_df["g_click_cnt"] > 0)).mean())
    print(f"TEST graph coverage (train+val): hist={test_hist_cov:.1%}, click={test_click_cov:.1%}, any={test_any_cov:.1%}", flush=True)

    for tag, model, feats in (("graph_only", m_only, GRAPH_ONLY_FEATURES), ("graph_base", m_base, GRAPH_BASE_FEATURES)):
        tp = Pool(test_df.select(feats), thread_count=8)
        t_pred = test_df.select(KEYS).with_columns(pl.Series("score", model.predict(tp).astype(np.float32)))
        t_pred.write_parquet(args.output / f"test_{tag}.parquet")

    report = {
        "val_cov": {"hist": val_hist_cov, "click": val_click_cov, "any": val_any_cov},
        "test_cov": {"hist": test_hist_cov, "click": test_click_cov, "any": test_any_cov},
        "graph_only": r_only,
        "graph_base": r_base,
        "corr_with_sub_v6": {"graph_only": c_go, "graph_base": c_gb},
        "sub_v6_base": {"holdout": base_h, "full": base_f},
        "best_blend": {"model": best_tag, "weight": best_w, "holdout": best_h, "full": best_f},
        "seconds": round(perf_counter() - t_start, 1),
        "peak_ram_mib": peak_ram_mib(),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)

    try:
        import notify_tg
        notify_tg.send(
            f"✅ 3.15M Graph Ranker finished ({report['seconds']:.0f}s)\n"
            f"Coverage val={val_any_cov:.1%}, test={test_any_cov:.1%}\n"
            f"graph-only: {r_only['full']:.5f} (corr={c_go:.3f})\n"
            f"graph+base: {r_base['full']:.5f} (corr={c_gb:.3f})\n"
            f"sub-v6: {base_f:.5f} -> best (+{int(best_w*100)}% {best_tag}): {best_f:.5f} ({best_f-base_f:+.5f})"
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
