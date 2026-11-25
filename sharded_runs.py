"""Fit independent rankers on bounded Avito train shards and average their scores."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import sleep
from types import SimpleNamespace
from uuid import uuid4

import polars as pl
import pyarrow.parquet as pq

from accelerator import gpu_ndcg, select_gpu
from evaluation import KEYS, attach_scores, zscore
from experiments import NEURAL, TREE_MODELS, score_summary, validation
from team_runs import SCRIPTS, run, sample_signature


PEOPLE = ("kirill", "savva", "dimon")
MODELS = (*(name for name in TREE_MODELS if name not in {"lambda", "als"}), *NEURAL)
DEFAULT_SHARDS = 21


def plan(args):
    files = sorted((args.data / "train" / "impressions").glob("part-*.parquet"))
    total = sum(pq.read_metadata(path).num_rows for path in files)
    selected = min(total, args.train_feeds) if args.train_feeds else total
    if not 1 <= args.shards <= selected:
        raise ValueError(f"Choose between 1 and {selected} shards")
    counts, extra = divmod(selected, args.shards)
    settings = {name: getattr(args, name) for name in
                ("train_feeds", "val_feeds", "iterations", "epochs", "history_window", "batch_size", "models", "shards")}
    return {**settings, "selected_train_feeds": selected}, [
        {"index": index + 1, "person": PEOPLE[index % len(PEOPLE)], "offset": index * counts + min(index, extra),
         "feeds": counts + int(index < extra)}
        for index in range(args.shards)
    ]


def part_dir(output, part):
    return output / "parts" / f"part-{part['index']:03d}"


def print_plan(args):
    settings, parts = plan(args)
    print(f"Train: {settings['selected_train_feeds']:,} feeds in {settings['shards']} shards")
    for person in PEOPLE:
        assigned = [part for part in parts if part["person"] == person]
        numbers = ", ".join(str(part["index"]) for part in assigned)
        print(f"{person}: {numbers} ({sum(part['feeds'] for part in assigned):,} feeds)")


def manifest(args, settings, part, val_sample=None):
    sample = sample_signature(args.data, part["feeds"], settings["val_feeds"],
                              train_offset=part["offset"], validation_sample=val_sample)
    return {"part": part, "settings": settings, "sample": sample}


def test_predictions(args, directory, models):
    select_gpu(args.gpu)
    for name in models:
        model = directory / name
        target = model / "test_predictions.parquet"
        if target.is_file():
            continue
        temp = target.with_name(f".test-{uuid4().hex}.parquet")
        if name in NEURAL:
            script, weights = "neural.py", model / "model.pt"
            options = ["--device", "cuda", "--batch-size", args.batch_size]
        else:
            script, weights = "ranker.py", model / "model.cbm"
            options = ["--device", "gpu"]
        try:
            run(script, "predict", "--data", args.data, "--output", temp, "--model", weights,
                "--threads", args.threads, *options)
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)


def monitor(args, report, label, directory):
    root = directory.resolve()
    log = root / "training.log"
    status = root / "training.status"
    status.unlink(missing_ok=True)
    log.write_text("", encoding="utf-8")
    config = {"report": str(report.resolve()), "log": str(log), "status": str(status), "data": str(args.data),
              "models": args.models, "epochs": args.epochs, "iterations": args.iterations,
              "label": label}
    target = args.output / "watch.json"
    pending = target.with_name(f".watch-{uuid4().hex}.json")
    pending.write_text(json.dumps(config), encoding="utf-8")
    pending.replace(target)
    return log, status


def train_part(args, settings, part, val_sample):
    root = part_dir(args.output, part)
    root.mkdir(parents=True, exist_ok=True)
    info = manifest(args, settings, part, val_sample)
    saved = root / "part.json"
    try:
        with saved.open("x", encoding="utf-8") as file:
            json.dump(info, file, indent=2)
    except FileExistsError:
        if json.loads(saved.read_text(encoding="utf-8")) != info:
            raise ValueError(f"{root} contains a different shard or training budget")
    fit = root / "fit"
    report_path = fit / "report.json"
    existing = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None
    print(f"Shard {part['index']}/{settings['shards']} ({part['person']}): "
          f"train rows {part['offset'] + 1:,}–{part['offset'] + part['feeds']:,}", flush=True)
    if existing is None or not existing["complete"]:
        log, status = (monitor(args, report_path, f"Часть {part['index']} из {args.shards}", root)
                       if args.ui else (None, None))
        code = 1
        try:
            run("experiments.py", "benchmark", "--data", args.data, "--output", fit, "--gpu", args.gpu,
                "--models", *settings["models"], "--train-offset", part["offset"], "--train-feeds", part["feeds"],
                "--val-feeds", args.val_feeds, "--iterations", args.iterations, "--epochs", args.epochs,
                "--history-window", args.history_window, "--batch-size", args.batch_size, "--threads", args.threads,
                "--no-rrf", *(["--resume"] if fit.is_dir() else []), log=log)
            code = 0
        finally:
            if status is not None:
                status.write_text(str(code), encoding="utf-8")
    if args.predict_test:
        test_predictions(args, fit, settings["models"])
    if args.report_round:
        from team_reports import sync

        sync(args.report_round, f"part-{part['index']:03d}", report_path, extra={
            "Участник": part["person"], "Часть": f"{part['index']} / {settings['shards']}",
        })
    return info["sample"]["val"]


def train_all(args):
    settings, parts = plan(args)
    assigned = [part for part in parts if args.person == "all" or part["person"] == args.person]
    if not assigned:
        raise ValueError(f"No shards assigned to {args.person}")
    from shard_exchange import key_path

    exchange = (make_exchange(args, settings) if args.parts_key is not None
                or (args.predict_test and args.report_round and key_path(args).is_file()) else None)
    if args.ui:
        subprocess.Popen([sys.executable, str(SCRIPTS / "dashboard.py"), "--watch", str(args.output / "watch.json")],
                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    val_sample = None
    for part in assigned:
        val_sample = train_part(args, settings, part, val_sample)
        if exchange:
            sync_parts(exchange, parts, assigned, collect=args.person in {"kirill", "all"})
    if exchange:
        watch_parts(args, exchange, parts, assigned)
    elif args.person == "all":
        merge_parts(args)
        if args.ui:
            directory = args.output / "merged"
            _, status = monitor(args, directory / "report.json", "Общий результат", directory)
            status.write_text("0", encoding="utf-8")


def make_exchange(args, settings):
    from shard_exchange import Exchange, key_path

    if not args.report_round or not args.predict_test or len(settings["models"]) != 1:
        raise ValueError("Shard exchange requires --report-round, --predict-test and a single model")
    return Exchange(args.output, key_path(args), args.report_round, settings["models"][0])


def sync_parts(exchange, parts, assigned, *, collect):
    try:
        return exchange.sync({part["index"] for part in parts}, {part["index"] for part in assigned}, collect=collect)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Shard exchange will retry: {error}", flush=True)
        return {}


def watch_parts(args, exchange, parts, assigned):
    from team_reports import publish
    from training_queue import keep_awake

    collector = args.person in {"kirill", "all"}
    own = {part["index"] for part in assigned}
    merged = args.output / "merged/report.json"
    previous = None
    while True:
        published = sync_parts(exchange, parts, assigned, collect=collector)
        local = sum(bool(exchange.ready(part["index"])) for part in parts)
        progress = (len(own & published.keys()), local)
        if progress != previous:
            print(f"Shard exchange: {progress[0]}/{len(own)} uploaded, {local}/{len(parts)} ready locally", flush=True)
            previous = progress
        if not collector and own <= published.keys():
            return
        if collector and local == len(parts):
            model = json.loads((exchange.directory(1) / "part.json").read_text(encoding="utf-8"))["settings"]["models"][0]
            submission = merged.parent / model / "submission.parquet"
            if not merged.is_file() or not submission.is_file():
                merge_parts(SimpleNamespace(**(vars(args) | {"report_round": None})))
            if own <= published.keys():
                try:
                    publish(args.report_round, "merged", merged, extra={"Частей": len(parts)})
                    return
                except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                    print(f"Merged report saved locally, publication will retry: {error}", flush=True)
        for _ in range(8):
            keep_awake()
            sleep(15)


def resume_exchange(args):
    manifest_path = next(sorted(args.output.glob("parts/part-*/part.json")), None)
    if manifest_path is None:
        raise ValueError("No training manifest exists to resume the exchange")
    settings = json.loads(manifest_path.read_text(encoding="utf-8"))["settings"]
    expected, parts = plan(SimpleNamespace(data=args.data, **settings))
    if expected != settings:
        raise ValueError("The local data does not match the saved shard plan")
    owned = [part for part in parts if args.person == "all" or part["person"] == args.person]
    watch_parts(args, make_exchange(args, settings), parts, owned)


def average_scores(template, sources):
    merged = template.select(KEYS)
    # Accumulate one part at a time to bound RAM even for dozens of checkpoints.
    merged = merged.with_columns(pl.lit(0.0).alias("sum"))
    for path in sources:
        frame = attach_scores(merged, pl.read_parquet(path, columns=[*KEYS, "score"]))
        merged = frame.select(*KEYS, (pl.col("sum") + zscore("score")).alias("sum"))
    combined = merged.select(*KEYS, (pl.col("sum") / len(sources)).cast(pl.Float32).alias("score"))
    return attach_scores(template.select(KEYS), combined)


def merge_parts(args):
    root = args.output
    first = part_dir(root, {"index": 1}) / "part.json"
    if not first.is_file():
        raise ValueError(f"Missing {first}; collect the part results first")
    settings = json.loads(first.read_text(encoding="utf-8"))["settings"]
    expected_settings, parts = plan(SimpleNamespace(data=args.data, **settings))
    if expected_settings != settings:
        raise ValueError("The current dataset does not match the shard plan")

    select_gpu(args.gpu)
    labels, cutoff = validation(args.data, settings["val_feeds"])
    val_sample = None
    roots, individual = [], {name: [] for name in settings["models"]}
    for part in parts:
        directory = part_dir(root, part)
        fit = directory / "fit"
        if not (fit / "report.json").is_file():
            raise ValueError(f"Missing trained part {part['index']}: {fit}")
        expected = manifest(args, settings, part, val_sample)
        val_sample = expected["sample"]["val"]
        saved = json.loads((directory / "part.json").read_text(encoding="utf-8"))
        report = json.loads((fit / "report.json").read_text(encoding="utf-8"))
        if (saved != expected or not report["complete"] or report["tuning_cutoff_timestamp"] != cutoff
                or report["train_feeds"] != part["feeds"] or report["train_offset"] != part["offset"]
                or report["requested_models"] != settings["models"]):
            raise ValueError(f"Incomplete or incompatible part {part['index']}: {fit}")
        roots.append(fit)
        for row in report["models"]:
            individual[row["model"]].append({"index": part["index"], "holdout_ndcg": row["holdout"]["ndcg"],
                                             "seconds": row["seconds"], "peak_ram_mib": row.get("peak_ram_mib")})

    merged = root / "merged"
    merged.mkdir(exist_ok=True)
    test = pl.read_parquet(args.data / "sample_submission.parquet", columns=KEYS) if args.predict_test else None
    results = []
    for name in settings["models"]:
        scores = average_scores(labels, [fit / name / "val_predictions.parquet" for fit in roots])
        folder = merged / name
        folder.mkdir(exist_ok=True)
        scores.write_parquet(folder / "val_predictions.parquet")
        row = score_summary(attach_scores(labels, scores), cutoff, gpu_ndcg)
        timings = individual[name]
        peaks = [timing["peak_ram_mib"] for timing in timings if timing["peak_ram_mib"] is not None]
        results.append({"model": name, "seconds": round(sum(timing["seconds"] for timing in timings), 2),
                        "peak_ram_mib": max(peaks, default=None), "parts": timings, **row})
        if test is not None:
            submission = average_scores(test, [fit / name / "test_predictions.parquet" for fit in roots])
            submission.write_parquet(folder / "submission.parquet")
    report = {"complete": True, "train_feeds": settings["selected_train_feeds"], "val_feeds": val_sample["rows"],
              "train_feed_limit": settings["train_feeds"], "val_feed_limit": settings["val_feeds"],
              "tuning_cutoff_timestamp": cutoff, "parts": parts, "settings": settings, "method": "mean-feed-zscore",
              "models": sorted(results, key=lambda row: row["holdout"]["ndcg"], reverse=True)}
    path = merged / "report.json"
    temp = path.with_name(f".report-{uuid4().hex}.json")
    temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
    for row in report["models"]:
        print(f"{row['model']}: holdout NDCG {row['holdout']['ndcg']:.6f} / full val {row['val']['ndcg']:.6f}")
    if args.report_round:
        from team_reports import sync

        sync(args.report_round, "merged", path, extra={"Частей": len(parts)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    preview, train, merge, exchange = (commands.add_parser(name) for name in ("plan", "train", "merge", "exchange"))
    for command in (preview, train, merge, exchange):
        command.add_argument("--data", type=Path, required=True)
    for command in (preview, train):
        command.add_argument("--train-feeds", type=int, default=0, help="Total budget across all parts; 0 uses all")
        command.add_argument("--val-feeds", type=int, default=0, help="Common validation limit; 0 uses all")
        command.add_argument("--shards", type=int, default=DEFAULT_SHARDS)
        command.add_argument("--iterations", type=int, default=120)
        command.add_argument("--epochs", type=int, default=3)
        command.add_argument("--history-window", type=int, default=32)
        command.add_argument("--batch-size", type=int, default=16)
        command.add_argument("--models", nargs="+", choices=MODELS, default=["strong-gbdt"])
    for command in (train, merge, exchange):
        command.add_argument("--output", type=Path, required=True, help="Directory containing parts/ and merged/")
        command.add_argument("--gpu", default="0")
        command.add_argument("--predict-test", action="store_true")
        command.add_argument("--report-round", help="Publish reports to GitHub under this round name")
    train.add_argument("--person", choices=("all", *PEOPLE), default="all")
    exchange.add_argument("--person", choices=("all", *PEOPLE), default="kirill")
    for command in (train, exchange):
        command.add_argument("--parts-key", type=Path, help="Shared key file; defaults to .parts-key under --output")
    train.add_argument("--ui", action="store_true", help="Open the progress window and leave it open after training")
    train.add_argument("--threads", type=int, default=6)
    preview.set_defaults(action=print_plan)
    train.set_defaults(action=train_all)
    merge.set_defaults(action=merge_parts)
    exchange.set_defaults(action=resume_exchange)
    args = parser.parse_args()
    if args.action in (print_plan, train_all) and (min(args.train_feeds, args.val_feeds) < 0
            or min(args.shards, args.iterations, args.epochs, args.batch_size, getattr(args, "threads", 1)) < 1
            or not 1 <= args.history_window <= 256 or len(set(args.models)) != len(args.models)):
        parser.error("Invalid training budget or duplicate models")
    if getattr(args, "report_round", None):
        from team_reports import valid_name

        valid_name(args.report_round)
    args.action(args)


if __name__ == "__main__":
    main()
