"""Train independent Avito model families on several computers and merge their scores."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import polars as pl

from accelerator import gpu_ndcg, select_gpu
from evaluation import attach_scores
from experiments import score_summary, validation
from ranker import feed_batches


JOBS = {
    "gbdt": ("history", "category-trend", "pixie-decays", "strong-gbdt"),
    "listwise": ("avito-joint", "set-transformer", "neural-pairwise"),
    "sequence": ("gru-din", "sasrec"),
}
TEACHERS = {"listwise": "avito-joint", "sequence": "gru-din"}
SCRIPTS = Path(__file__).resolve().parent


def run(script, *args):
    subprocess.run([sys.executable, "-u", "-X", "utf8", str(SCRIPTS / script), *map(str, args)], check=True)


def sample_signature(data, train_feeds, val_feeds):
    result = {}
    columns = ("feed_id", "timestamp")
    for name, limit in (("train", train_feeds), ("val", val_feeds)):
        hashes = {column: hashlib.sha256() for column in columns}
        rows = 0
        for batch in feed_batches(data / name, columns, limit):
            rows += batch.height
            for column in columns:
                hashes[column].update(np.asarray(batch[column].to_numpy(), dtype="<i8").tobytes())
        if rows == 0:
            raise ValueError(f"No {name} feeds")
        result[name] = {"rows": rows, **{column: digest.hexdigest() for column, digest in hashes.items()}}
    return result


def benchmark_args(args, settings):
    return [
        "--data", args.data, "--train-feeds", settings["train_feeds"], "--val-feeds", settings["val_feeds"],
        "--iterations", settings["iterations"], "--threads", args.threads,
    ]


def job(args):
    settings = {
        "train_feeds": args.train_feeds, "val_feeds": args.val_feeds, "iterations": args.iterations,
        "epochs": args.epochs, "history_window": args.history_window, "batch_size": args.batch_size,
    }
    signature = sample_signature(args.data, args.train_feeds, args.val_feeds)
    metadata_path = args.output / "job.json"
    metadata = {"role": args.role, "settings": settings, "sample": signature, "oof": False}
    progress_path = args.output / "job-progress.json"
    pending = args.output.with_name(f"{args.output.name}-job.json")
    report = args.output / "report.json"
    if args.resume:
        previous_file = progress_path if progress_path.is_file() else pending
        previous = json.loads(previous_file.read_text(encoding="utf-8"))
        if previous != metadata:
            raise ValueError("Resume requires matching sample, role and settings")
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with pending.open("x", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
    completed = report.is_file() and json.loads(report.read_text(encoding="utf-8"))["complete"]
    if not completed:
        if args.output.exists() and not report.is_file():
            raise ValueError("The benchmark stopped before saving a report. Use a new --output.")
        run("experiments.py", "benchmark", *benchmark_args(args, settings), "--output", args.output,
            "--gpu", args.gpu, "--epochs", args.epochs, "--history-window", args.history_window,
            "--batch-size", args.batch_size, "--models", *JOBS[args.role], "--no-rrf",
            *(["--resume"] if report.is_file() else []))
    if pending.is_file():
        pending.replace(progress_path)
    if args.role in TEACHERS and not args.skip_oof:
        select_gpu(args.gpu)
        teacher = TEACHERS[args.role]
        path = args.output / "oof" / f"{teacher}.parquet"
        if not path.exists():
            run("neural.py", "oof", "--variant", teacher, "--data", args.data, "--output", path,
                "--epochs", args.epochs, "--train-feeds", args.train_feeds, "--history-window", args.history_window,
                "--threads", args.threads, "--batch-size", args.batch_size, "--device", "cuda")
        metadata["oof"] = True
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Team job ready: {args.output}", flush=True)


def merge(args):
    select_gpu(args.gpu)
    jobs = {}
    for path in args.runs:
        metadata = json.loads((path / "job.json").read_text(encoding="utf-8"))
        role = metadata["role"]
        if role not in JOBS or role in jobs:
            raise ValueError(f"Unexpected or duplicate role: {role}")
        jobs[role] = (metadata, path)
    if set(jobs) != set(JOBS):
        raise ValueError("Supply exactly one completed run for each of gbdt, listwise and sequence")
    settings = jobs["gbdt"][0]["settings"]
    signature = sample_signature(args.data, settings["train_feeds"], settings["val_feeds"])
    if any(job["sample"] != signature or job["settings"] != settings for job, _ in jobs.values()):
        raise ValueError("Team jobs used different data samples or experiment budgets")
    if args.stack and not all(jobs[role][0]["oof"] for role in TEACHERS):
        raise ValueError("Stacking needs OOF files from both the listwise and sequence jobs")

    labels, cutoff = validation(args.data, settings["val_feeds"])
    args.output.mkdir(parents=True, exist_ok=False)
    rows, files = [], {}
    for role, (_, directory) in jobs.items():
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        if not report["complete"] or report["tuning_cutoff_timestamp"] != cutoff:
            raise ValueError(f"Incomplete job or validation timestamp mismatch: {directory}")
        for name in JOBS[role]:
            path = directory / name / "val_predictions.parquet"
            scores = score_summary(attach_scores(labels, pl.read_parquet(path)), cutoff, gpu_ndcg)
            rows.append({"model": name, **scores})
            files[name] = path
    from ensemble import fit_families

    for name, predictions, metadata in fit_families(labels, cutoff, rows, files, "cuda", args.optuna_trials):
        scores = score_summary(attach_scores(labels, predictions), cutoff, gpu_ndcg)
        rows.append({"model": name, **metadata, **scores})
        predictions.write_parquet(args.output / f"{name}_val_predictions.parquet")
    if args.stack:
        sources = [jobs[role][1] / "oof" / f"{teacher}.parquet" for role, teacher in TEACHERS.items()]
        validation_scores = [files[teacher] for teacher in TEACHERS.values()]
        for name in ("stack-base", "stack-gbdt"):
            directory = args.output / name
            run("ranker.py", "train", *benchmark_args(args, settings), "--variant", name, "--output", directory,
                "--device", "gpu", "--tune-before", cutoff, "--stack-train", *sources, "--stack-val", *validation_scores)
            predictions = pl.read_parquet(directory / "val_predictions.parquet")
            rows.append({"model": name, **score_summary(attach_scores(labels, predictions), cutoff, gpu_ndcg)})
    report = {"complete": True, "val_feeds": signature["val"]["rows"], "settings": settings, "sample": signature,
              "tuning_cutoff_timestamp": cutoff, "models": sorted(rows, key=lambda item: item["holdout"]["ndcg"], reverse=True)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in report["models"]:
        print(f"{row['model']:<22} {row['holdout']['ndcg']:.6f}  {row['val']['ndcg']:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    worker, collect = commands.add_parser("job"), commands.add_parser("merge")
    for command in (worker, collect):
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--gpu", default="0")
        command.add_argument("--threads", type=int, default=6)
    worker.add_argument("--role", choices=JOBS, required=True)
    worker.add_argument("--train-feeds", type=int, default=50_000)
    worker.add_argument("--val-feeds", type=int, default=0)
    worker.add_argument("--iterations", type=int, default=120)
    worker.add_argument("--epochs", type=int, default=3)
    worker.add_argument("--history-window", type=int, default=32)
    worker.add_argument("--batch-size", type=int, default=16)
    worker.add_argument("--skip-oof", action="store_true")
    worker.add_argument("--resume", action="store_true")
    worker.set_defaults(action=job)
    collect.add_argument("--runs", nargs=3, type=Path, required=True)
    collect.add_argument("--stack", action="store_true")
    collect.add_argument("--optuna-trials", type=int, default=48)
    collect.set_defaults(action=merge)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
