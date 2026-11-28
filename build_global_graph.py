"""Build compact int8-quantized 3.15M collaborative audience graph caches (train-only and train+val)."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl


PROJ = np.array([0.5, -0.5, 0.5, 0.5], dtype=np.float32)


def project_32d(vectors_128: np.ndarray) -> np.ndarray:
    v = np.asarray(vectors_128, dtype=np.float32).reshape(-1, 32, 4) @ PROJ
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, norms, out=np.zeros_like(v), where=norms > 1e-6)


def quantize_profiles(sums: np.ndarray, counts: np.ndarray):
    keep = counts > 0
    s = sums[keep]
    c = counts[keep]
    norms = np.linalg.norm(s, axis=1)
    unit = np.divide(s, norms[:, None], out=np.zeros_like(s), where=norms[:, None] > 1e-6)
    vec_i8 = np.clip(np.rint(unit * 127.0), -127, 127).astype(np.int8)
    coh_u8 = np.clip(np.rint(np.divide(norms, np.maximum(c, 1.0)) * 255.0), 0, 255).astype(np.uint8)
    return keep, vec_i8, c.astype(np.int32), coh_u8


def accumulate(split_dir: Path, hist_ids, hist_sum, hist_cnt, click_ids, click_sum, click_cnt, chunk_parts=32):
    t0 = perf_counter()
    parts = sorted((split_dir / "impressions").glob("part-*.parquet"))
    nh, nc = len(hist_ids), len(click_ids)
    total_feeds = 0
    for i in range(0, len(parts), chunk_parts):
        chunk = parts[i : i + chunk_parts]
        df = pl.concat([
            pl.read_parquet(p, columns=["item_id", "target", "user_embedding", "history__item_id"])
            for p in chunk
        ])
        total_feeds += df.height
        u32 = project_32d(df["user_embedding"].list.to_array(128).to_numpy())

        # History edges
        h_df = (
            df.with_row_index("r")
            .select("r", pl.col("history__item_id").list.head(64).alias("iid"))
            .explode("iid")
            .filter(pl.col("iid") > 0)
        )
        h_i = h_df["iid"].to_numpy()
        h_r = h_df["r"].to_numpy()
        idx = np.minimum(np.searchsorted(hist_ids, h_i), nh - 1)
        hit = hist_ids[idx] == h_i
        hi, hr = idx[hit], h_r[hit]
        for off in range(0, len(hi), 262_144):
            sl = slice(off, off + 262_144)
            np.add.at(hist_sum, hi[sl], u32[hr[sl]])
        np.add.at(hist_cnt, hi, 1)

        # Click edges
        c_df = (
            df.with_row_index("r")
            .select("r", "item_id", "target")
            .explode("item_id", "target")
            .filter((pl.col("target") > 0) & (pl.col("item_id") > 0))
        )
        c_i = c_df["item_id"].to_numpy()
        c_r = c_df["r"].to_numpy()
        if len(c_i):
            cidx = np.minimum(np.searchsorted(click_ids, c_i), nc - 1)
            chit = click_ids[cidx] == c_i
            ci, cr = cidx[chit], c_r[chit]
            np.add.at(click_sum, ci, u32[cr])
            np.add.at(click_cnt, ci, 1)

        del df, u32, h_df, c_df
        if (i // chunk_parts + 1) % 4 == 0 or i + chunk_parts >= len(parts):
            print(f"  [{split_dir.name}] {min(i + chunk_parts, len(parts))}/{len(parts)} ({total_feeds:,} feeds) in {perf_counter()-t0:.1f}s", flush=True)


def save_cache(path: Path, hist_ids, hist_sum, hist_cnt, click_ids, click_sum, click_cnt):
    t0 = perf_counter()
    hk, hv, hc, hcoh = quantize_profiles(hist_sum, hist_cnt)
    ck, cv, cc, ccoh = quantize_profiles(click_sum, click_cnt)
    np.savez(
        path,
        hist_ids=hist_ids[hk].astype(np.int32),
        hist_vec=hv,
        hist_cnt=hc,
        hist_coh=hcoh,
        click_ids=click_ids[ck].astype(np.int32),
        click_vec=cv,
        click_cnt=cc,
        click_coh=ccoh,
    )
    mb = path.stat().st_size / 1024**2
    print(f"Saved {path.name}: hist={int(hk.sum()):,}, click={int(ck.sum()):,}, size={mb:.1f} MiB ({perf_counter()-t0:.1f}s)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--stats", type=Path, default=Path("runs/global_stats/item_stats_with_val.parquet"))
    parser.add_argument("--output", type=Path, default=Path("runs/global_graph"))
    args = parser.parse_args()

    t0 = perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    stats = pl.read_parquet(args.stats, columns=["item_id", "item_hist_count", "item_clicks"])
    hist_ids = stats.filter(pl.col("item_hist_count") >= 2)["item_id"].sort().to_numpy().astype(np.int32)
    click_ids = stats.filter(pl.col("item_clicks") >= 1)["item_id"].sort().to_numpy().astype(np.int32)
    del stats
    print(f"Graph vocab: hist_ids={len(hist_ids):,} (cnt>=2), click_ids={len(click_ids):,} (clicks>=1)", flush=True)

    hist_sum = np.zeros((len(hist_ids), 32), dtype=np.float32)
    hist_cnt = np.zeros(len(hist_ids), dtype=np.int32)
    click_sum = np.zeros((len(click_ids), 32), dtype=np.float32)
    click_cnt = np.zeros(len(click_ids), dtype=np.int32)

    accumulate(args.data / "train", hist_ids, hist_sum, hist_cnt, click_ids, click_sum, click_cnt)
    save_cache(args.output / "graph_train.npz", hist_ids, hist_sum, hist_cnt, click_ids, click_sum, click_cnt)

    accumulate(args.data / "val", hist_ids, hist_sum, hist_cnt, click_ids, click_sum, click_cnt)
    save_cache(args.output / "graph_with_val.npz", hist_ids, hist_sum, hist_cnt, click_ids, click_sum, click_cnt)
    print(f"All done in {perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
