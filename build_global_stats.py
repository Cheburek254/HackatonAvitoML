"""Stream all train (and val) parquet parts to build global item and user statistics with recency."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import polars as pl


def _merge_item_frames(frames: list[pl.DataFrame]) -> pl.DataFrame:
    return (
        pl.concat(frames)
        .group_by("item_id")
        .agg(
            pl.col("item_shows").sum(),
            pl.col("item_clicks").sum(),
            pl.col("item_pos_sum").sum(),
            pl.col("recent_shows").sum(),
            pl.col("recent_clicks").sum(),
        )
    )


def build(data_dir: Path, output_dir: Path, chunk_size: int = 32, recent_days: int = 4):
    start = perf_counter()
    parts = sorted((data_dir / "train" / "impressions").glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No train parts in {data_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_max_ts = int(pl.read_parquet(parts[0], columns=["timestamp"])["timestamp"].max())
    recent_cutoff = train_max_ts - recent_days * 86400
    print(f"train_max_ts={train_max_ts}, recent_cutoff={recent_cutoff} (last {recent_days}d)", flush=True)

    item_aggs = []
    hist_aggs = []
    user_aggs = []
    total_feeds = 0
    total_shows = 0
    total_clicks = 0
    recent_total_shows = 0
    recent_total_clicks = 0

    for i in range(0, len(parts), chunk_size):
        chunk = parts[i : i + chunk_size]
        df = pl.concat([
            pl.read_parquet(p, columns=["user_id", "item_id", "pos", "target", "history__item_id", "timestamp"])
            for p in chunk
        ])
        total_feeds += df.height

        # User-level aggregation
        u_df = (
            df.filter(pl.col("user_id") > 0)
            .select(
                "user_id",
                pl.col("target").list.len().cast(pl.Int64).alias("shows"),
                pl.col("target").list.sum().cast(pl.Int64).alias("clicks"),
            )
            .group_by("user_id")
            .agg(
                pl.len().cast(pl.Int32).alias("user_feeds"),
                pl.col("shows").sum().alias("user_shows"),
                pl.col("clicks").sum().alias("user_clicks"),
            )
        )
        user_aggs.append(u_df)

        # History item popularity (top 64 recent events per feed, target-free!)
        h_df = (
            df.select(pl.col("history__item_id").list.head(64).alias("item_id"))
            .explode("item_id")
            .filter(pl.col("item_id") > 0)
            .group_by("item_id")
            .agg(pl.len().cast(pl.Int32).alias("item_hist_count"))
        )
        hist_aggs.append(h_df)

        # Candidate item-level aggregation (all + recent)
        exp = (
            df.select(
                "item_id", "pos", "target",
                (pl.col("timestamp") >= recent_cutoff).alias("is_recent"),
            )
            .explode("item_id", "pos", "target")
        )
        total_shows += exp.height
        total_clicks += int(exp["target"].sum())
        rec_mask = exp["is_recent"]
        recent_total_shows += int(rec_mask.sum())
        recent_total_clicks += int(exp.filter(pl.col("is_recent"))["target"].sum())

        i_df = (
            exp.filter(pl.col("item_id") > 0)
            .group_by("item_id")
            .agg(
                pl.len().cast(pl.Int32).alias("item_shows"),
                pl.col("target").sum().cast(pl.Int32).alias("item_clicks"),
                pl.col("pos").sum().cast(pl.Int64).alias("item_pos_sum"),
                pl.col("is_recent").sum().cast(pl.Int32).alias("recent_shows"),
                pl.when(pl.col("is_recent")).then(pl.col("target")).otherwise(0).sum().cast(pl.Int32).alias("recent_clicks"),
            )
        )
        item_aggs.append(i_df)
        del df, exp

        if len(item_aggs) >= 4:
            item_aggs = [_merge_item_frames(item_aggs)]
            hist_aggs = [
                pl.concat(hist_aggs)
                .group_by("item_id")
                .agg(pl.col("item_hist_count").sum())
            ]
            user_aggs = [
                pl.concat(user_aggs)
                .group_by("user_id")
                .agg(
                    pl.col("user_feeds").sum(),
                    pl.col("user_shows").sum(),
                    pl.col("user_clicks").sum(),
                )
            ]
        print(f"Processed {min(i + chunk_size, len(parts))}/{len(parts)} parts ({total_feeds:,} feeds) in {perf_counter() - start:.1f}s", flush=True)

    cand_stats = _merge_item_frames(item_aggs)
    hist_stats = (
        pl.concat(hist_aggs)
        .group_by("item_id")
        .agg(pl.col("item_hist_count").sum())
    )
    item_stats = (
        cand_stats.join(hist_stats, on="item_id", how="full", coalesce=True)
        .with_columns(
            pl.col("item_shows").fill_null(0).cast(pl.Int32),
            pl.col("item_clicks").fill_null(0).cast(pl.Int32),
            pl.col("item_pos_sum").fill_null(0).cast(pl.Int64),
            pl.col("item_shows").fill_null(0).cast(pl.Int32).alias("item_pos_shows"),
            pl.col("recent_shows").fill_null(0).cast(pl.Int32),
            pl.col("recent_clicks").fill_null(0).cast(pl.Int32),
            pl.col("item_hist_count").fill_null(0).cast(pl.Int32),
        )
        .sort("item_id")
    )
    user_stats = (
        pl.concat(user_aggs)
        .group_by("user_id")
        .agg(
            pl.col("user_feeds").sum(),
            pl.col("user_shows").sum(),
            pl.col("user_clicks").sum(),
        )
        .sort("user_id")
    )

    prior = total_clicks / max(total_shows, 1)
    recent_prior = recent_total_clicks / max(recent_total_shows, 1)
    item_stats.write_parquet(output_dir / "item_stats.parquet")
    user_stats.write_parquet(output_dir / "user_stats.parquet")
    meta = {
        "total_feeds": total_feeds,
        "total_shows": total_shows,
        "total_clicks": total_clicks,
        "prior": round(prior, 8),
        "recent_prior": round(recent_prior, 8),
        "recent_cutoff": recent_cutoff,
        "recent_days": recent_days,
        "unique_items": item_stats.height,
        "unique_users": user_stats.height,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Saved train item_stats ({item_stats.height:,} items), prior={prior:.5f}, recent_prior={recent_prior:.5f}, time={perf_counter() - start:.1f}s", flush=True)

    # Now build train + val stats for test-time inference (val is 1 day before test!)
    val_parts = sorted((data_dir / "val" / "impressions").glob("part-*.parquet"))
    if val_parts:
        val_df = pl.concat([
            pl.read_parquet(p, columns=["item_id", "target", "history__item_id"])
            for p in val_parts
        ])
        val_hist = (
            val_df.select(pl.col("history__item_id").list.head(64).alias("item_id"))
            .explode("item_id")
            .filter(pl.col("item_id") > 0)
            .group_by("item_id")
            .agg(pl.len().cast(pl.Int32).alias("v_hist"))
        )
        val_exp = val_df.select("item_id", "target").explode("item_id", "target")
        val_shows_total = val_exp.height
        val_clicks_total = int(val_exp["target"].sum())
        val_cand = (
            val_exp.filter(pl.col("item_id") > 0)
            .group_by("item_id")
            .agg(
                pl.len().cast(pl.Int32).alias("v_shows"),
                pl.col("target").sum().cast(pl.Int32).alias("v_clicks"),
            )
        )
        val_stats = (
            val_cand.join(val_hist, on="item_id", how="full", coalesce=True)
            .with_columns(
                pl.col("v_shows").fill_null(0).cast(pl.Int32),
                pl.col("v_clicks").fill_null(0).cast(pl.Int32),
                pl.col("v_hist").fill_null(0).cast(pl.Int32),
            )
        )
        item_stats_with_val = (
            item_stats.join(val_stats, on="item_id", how="full", coalesce=True)
            .with_columns(
                (pl.col("item_shows").fill_null(0) + pl.col("v_shows").fill_null(0)).cast(pl.Int32).alias("item_shows"),
                (pl.col("item_clicks").fill_null(0) + pl.col("v_clicks").fill_null(0)).cast(pl.Int32).alias("item_clicks"),
                pl.col("item_pos_sum").fill_null(0).cast(pl.Int64).alias("item_pos_sum"),
                pl.col("item_pos_shows").fill_null(0).cast(pl.Int32).alias("item_pos_shows"),
                (pl.col("recent_shows").fill_null(0) + pl.col("v_shows").fill_null(0)).cast(pl.Int32).alias("recent_shows"),
                (pl.col("recent_clicks").fill_null(0) + pl.col("v_clicks").fill_null(0)).cast(pl.Int32).alias("recent_clicks"),
                (pl.col("item_hist_count").fill_null(0) + pl.col("v_hist").fill_null(0)).cast(pl.Int32).alias("item_hist_count"),
            )
            .drop("v_shows", "v_clicks", "v_hist")
            .sort("item_id")
        )
        with_val_prior = (total_clicks + val_clicks_total) / max(total_shows + val_shows_total, 1)
        with_val_recent_prior = (recent_total_clicks + val_clicks_total) / max(recent_total_shows + val_shows_total, 1)
        item_stats_with_val.write_parquet(output_dir / "item_stats_with_val.parquet")
        meta_with_val = {
            **meta,
            "total_feeds": total_feeds + val_df.height,
            "total_shows": total_shows + val_shows_total,
            "total_clicks": total_clicks + val_clicks_total,
            "prior": round(with_val_prior, 8),
            "recent_prior": round(with_val_recent_prior, 8),
            "unique_items": item_stats_with_val.height,
            "includes_val": True,
        }
        (output_dir / "meta_with_val.json").write_text(json.dumps(meta_with_val, indent=2) + "\n", encoding="utf-8")
        print(f"Saved train+val item_stats ({item_stats_with_val.height:,} items) in {perf_counter() - start:.1f}s", flush=True)

        # Coverage diagnostics
        val_items = val_exp["item_id"]
        known_train = item_stats["item_id"]
        recent_train = item_stats.filter(pl.col("recent_shows") > 0)["item_id"]
        print(f"val  coverage (train all): {val_items.is_in(known_train).mean():.1%}, (train recent): {val_items.is_in(recent_train).mean():.1%}", flush=True)

        test_parts = sorted((data_dir / "test" / "impressions").glob("part-*.parquet"))
        test_items = pl.concat([pl.read_parquet(p, columns=["item_id"]) for p in test_parts])["item_id"].explode()
        known_with_val = item_stats_with_val["item_id"]
        recent_with_val = item_stats_with_val.filter(pl.col("recent_shows") > 0)["item_id"]
        print(f"test coverage (train only): {test_items.is_in(known_train).mean():.1%} -> (train+val): {test_items.is_in(known_with_val).mean():.1%}", flush=True)
        print(f"test recent   (train only): {test_items.is_in(recent_train).mean():.1%} -> (train+val): {test_items.is_in(recent_with_val).mean():.1%}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--recent-days", type=int, default=4)
    args = parser.parse_args()
    build(args.data, args.output, args.chunk_size, args.recent_days)


if __name__ == "__main__":
    main()
