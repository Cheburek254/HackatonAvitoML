"""Stream all train parquet parts to build compact global item and user statistics."""

import argparse
from pathlib import Path
from time import perf_counter

import polars as pl


def build(data_dir: Path, output_dir: Path, chunk_size: int = 32):
    start = perf_counter()
    parts = sorted((data_dir / "train" / "impressions").glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No train parts in {data_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    item_aggs = []
    hist_aggs = []
    user_aggs = []
    total_feeds = 0
    total_shows = 0
    total_clicks = 0

    for i in range(0, len(parts), chunk_size):
        chunk = parts[i : i + chunk_size]
        df = pl.concat([
            pl.read_parquet(p, columns=["user_id", "item_id", "pos", "target", "history__item_id"])
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

        # Candidate item-level aggregation
        exp = df.select("item_id", "pos", "target").explode("item_id", "pos", "target")
        total_shows += exp.height
        total_clicks += int(exp["target"].sum())
        i_df = (
            exp.filter(pl.col("item_id") > 0)
            .group_by("item_id")
            .agg(
                pl.len().cast(pl.Int32).alias("item_shows"),
                pl.col("target").sum().cast(pl.Int32).alias("item_clicks"),
                pl.col("pos").sum().cast(pl.Int64).alias("item_pos_sum"),
            )
        )
        item_aggs.append(i_df)
        del df, exp

        if len(item_aggs) >= 4:
            item_aggs = [
                pl.concat(item_aggs)
                .group_by("item_id")
                .agg(
                    pl.col("item_shows").sum(),
                    pl.col("item_clicks").sum(),
                    pl.col("item_pos_sum").sum(),
                )
            ]
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

    cand_stats = (
        pl.concat(item_aggs)
        .group_by("item_id")
        .agg(
            pl.col("item_shows").sum(),
            pl.col("item_clicks").sum(),
            pl.col("item_pos_sum").sum(),
        )
    )
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
    item_stats.write_parquet(output_dir / "item_stats.parquet")
    user_stats.write_parquet(output_dir / "user_stats.parquet")
    (output_dir / "meta.json").write_text(
        f'{{"total_feeds": {total_feeds}, "total_shows": {total_shows}, "total_clicks": {total_clicks}, "prior": {prior:.8f}, "unique_items": {item_stats.height}, "unique_users": {user_stats.height}}}\n',
        encoding="utf-8",
    )
    print(f"Saved item_stats ({item_stats.height:,} items) and user_stats ({user_stats.height:,} users), prior={prior:.5f}, time={perf_counter() - start:.1f}s", flush=True)

    known_items = item_stats["item_id"]
    for split in ("val", "test"):
        sp = sorted((data_dir / split / "impressions").glob("part-*.parquet"))
        s_df = pl.concat([pl.read_parquet(p, columns=["item_id"]) for p in sp])
        i_cov = s_df["item_id"].explode().is_in(known_items).mean()
        print(f"Item coverage on {split} (candidates + histories): {i_cov:.1%}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=32)
    args = parser.parse_args()
    build(args.data, args.output, args.chunk_size)


if __name__ == "__main__":
    main()
