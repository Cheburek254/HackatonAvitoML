"""Run a bounded follow-up on unused train blocks after the candidate-order experiment."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from candidate_order_ablation import (
    BATCH_SIZE, EPOCHS, FEATURE_WORKERS, ITERATIONS, THREADS, TRAIN_FEEDS, read_json,
)
from training_queue import atomic_text, keep_awake, note, run_stage


CAT, GRU, DIN = "din-events-v2-ties", "gru-din-plus-ties", "neural-din-plus-ties"


@dataclass(frozen=True)
class Trial:
    family: str
    window: int
    block: int

    @property
    def name(self):
        label = f"{self.family}-{self.window}" if self.family != "cat" else "cat"
        return f"{label}-b{self.block}"

    @property
    def model(self):
        return {"cat": CAT, "gru": GRU, "din": DIN}[self.family]

    @property
    def offset(self):
        return TRAIN_FEEDS * (self.block - 1)


CONTROL = (Trial("cat", 32, 1), Trial("gru", 32, 1), Trial("din", 32, 1))
QUEUE = (Trial("cat", 32, 2), Trial("cat", 32, 3), Trial("gru", 64, 1), Trial("din", 64, 1),
         Trial("gru", 32, 2), Trial("din", 32, 2), Trial("gru", 64, 2), Trial("din", 64, 2),
         Trial("cat", 32, 4), Trial("cat", 32, 5), Trial("gru", 32, 3), Trial("din", 32, 3))


def seconds(folder, model):
    report = read_json(folder / "report.json")
    return next(row["seconds"] for row in report["models"] if row["model"] == model)


def estimate(args, trial):
    for earlier in reversed(QUEUE[:QUEUE.index(trial)]):
        folder = args.output / earlier.name
        report = folder / "report.json"
        if ((earlier.family, earlier.window) == (trial.family, trial.window)
                and report.is_file() and read_json(report)["complete"]):
            return seconds(folder, earlier.model) * 1.08
    original = seconds(args.prior / trial.model, trial.model)
    return original * (1.17 if trial.window == 64 else 1.08)


def train(args):
    completed, skipped = [], []
    for trial in QUEUE:
        folder = args.output / trial.name
        if not (folder / "report.json").is_file() or not read_json(folder / "report.json")["complete"]:
            budget = (args.deadline - datetime.now().astimezone()).total_seconds()
            expected = estimate(args, trial)
            if budget < expected + 180:
                skipped.append(trial.name)
                note(f"Skipping {trial.name}: budget {budget / 60:.1f}m, estimated {expected / 60:.1f}m")
                continue
            note(f"Estimated {trial.name}: {expected / 60:.1f} minutes")
        run_stage(args, trial.name, folder, [trial.model], f"Вечерний прогон · {trial.name}", [
            "experiments.py", "benchmark", "--data", str(args.data), "--output", str(folder), "--gpu", args.gpu,
            "--models", trial.model, "--train-feeds", str(TRAIN_FEEDS), "--train-offset", str(trial.offset),
            "--val-feeds", "0", "--iterations", str(ITERATIONS), "--epochs", str(EPOCHS),
            "--history-window", str(trial.window), "--batch-size", str(BATCH_SIZE), "--threads", str(THREADS),
            "--no-rrf", "--prefetch-neural",
            *(["--feature-workers", str(FEATURE_WORKERS)] if trial.family == "cat" else []),
            *(["--resume"] if folder.exists() else [])
        ], watch_root=args.prior)
        completed.append(trial.name)
        atomic_text(args.output / "completed.json", json.dumps(completed))
    atomic_text(args.output / "skipped.json", json.dumps(skipped))


def compare(args):
    import numpy as np
    import polars as pl

    from accelerator import select_gpu
    from ensemble import Slates, add_model_scores, weighted
    from evaluation import KEYS, zscore
    from experiments import read_frame, validation
    from pair_blend import paired_interval

    gpu = select_gpu(args.gpu)
    labels, cutoff = validation(args.data, 0)
    feeds = read_frame(args.data, "val", ["feed_id", "user_id", "timestamp"], 0).sort("feed_id")
    early = feeds["timestamp"].to_numpy() < cutoff
    if (len(feeds), int(early.sum()), len(labels), cutoff) != (14423, 7290, 1389971, 1790107648):
        raise ValueError("Different competition validation split")

    completed_file = args.output / "completed.json"
    completed = set(read_json(completed_file)) if completed_file.is_file() else set()
    trials = [*CONTROL, *(trial for trial in QUEUE if trial.name in completed)]
    columns, predictions, metadata = [], [], {}
    for trial in trials:
        parent = args.prior / trial.model if trial in CONTROL else args.output / trial.name
        source = read_json(parent / "report.json")
        if (not source["complete"] or source["train_feeds"] != TRAIN_FEEDS or source["train_offset"] != trial.offset
                or source["tuning_cutoff_timestamp"] != cutoff or source["val_candidates"] != len(labels)):
            raise ValueError(f"Unexpected train sample or validation keys for {trial.name}")
        metadata[trial.name] = {"train_feeds": TRAIN_FEEDS, "offset": trial.offset, "history_window": trial.window,
                                "seconds": seconds(parent, trial.model)}
        columns.append(trial.name)
        predictions.append(pl.read_parquet(parent / trial.model / "val_predictions.parquet"))
    if args.lightgbm:
        lgbm = read_json(args.lightgbm / "metrics.json")
        if (lgbm["train_feeds"] != TRAIN_FEEDS or lgbm["tuning_cutoff_timestamp"] != cutoff
                or lgbm["val_feeds"] != len(feeds) or lgbm["options"]["train_offset"] != 0):
            raise ValueError("LightGBM uses a different sample")
        columns.append("lightgbm-b1")
        metadata["lightgbm-b1"] = {"train_feeds": TRAIN_FEEDS, "offset": 0, "seconds": lgbm["seconds"]}
        predictions.append(pl.read_parquet(args.lightgbm / "val_predictions.parquet"))
    frame = add_model_scores(labels.select(*KEYS, "target"), predictions, columns)
    slates = Slates(frame, "cuda")
    if not np.array_equal(slates.frame["feed_id"].to_numpy()[slates.rows[:, 0]], feeds["feed_id"].to_numpy()):
        raise ValueError("Validation feed order does not match")
    results, metrics = [], {}

    def record(name, values, **details):
        metrics[name] = per_feed = slates.per_feed_ndcg(values).cpu().numpy().astype(np.float64)
        results.append({"model": name, "tuning": {"ndcg": float(per_feed[early].mean())},
                        "holdout": {"ndcg": float(per_feed[~early].mean())}, "val": {"ndcg": float(per_feed.mean())},
                        **details})
        print(f"Finished {name}: holdout_ndcg={results[-1]['holdout']['ndcg']:.8f}", flush=True)

    for name in columns:
        record(name, slates.pack(name), **metadata[name])
    signals = Slates(frame.with_columns(zscore(name).alias(name) for name in columns), "cuda")
    groups = {"cat": [trial.name for trial in trials if trial.family == "cat"]}
    for family in ("gru", "din"):
        for window in (32, 64):
            group = [trial.name for trial in trials if (trial.family, trial.window) == (family, window)]
            if group:
                groups[f"{family}-{window}"] = group
    averages = {name: weighted([1 / len(group)] * len(group), [signals.pack(col) for col in group])
                for name, group in groups.items()}
    for name, group in groups.items():
        if len(group) > 1:
            record(f"{name}-bag", averages[name], components=group, train_feeds=TRAIN_FEEDS * len(group))
    baseline = weighted([.45, .30, .25], [signals.pack(trial.name) for trial in CONTROL])
    record("first-block-z-45-30-25", baseline, components=[trial.name for trial in CONTROL])
    variants = []
    for window in (32, 64):
        gru, din = f"gru-{window}", f"din-{window}"
        if gru in averages and din in averages:
            name = f"scaled-z-{window}"
            components = ["cat", gru, din]
            record(name, weighted([.45, .30, .25], [averages[item] for item in components]),
                   components={item: groups[item] for item in components})
            variants.append(name)
        if args.lightgbm and gru in averages:
            lgbm_scores = signals.pack("lightgbm-b1")
            record(f"scaled-lightgbm-z-{window}", weighted([.20, .50, .30], [averages["cat"], lgbm_scores, averages[gru]]),
                   components={"cat": groups["cat"], "lightgbm": ["lightgbm-b1"], "gru": groups[gru]})
    users = feeds["user_id"].to_numpy()[~early]
    intervals = {name: paired_interval((metrics[name] - metrics["first-block-z-45-30-25"])[~early], users)
                 for name in variants}
    report = {"complete": True, "gpu": vars(gpu), "models": sorted(results, key=lambda row: -row["holdout"]["ndcg"]),
              "val_feeds": len(feeds), "val_feed_limit": 0, "tuning_cutoff_timestamp": cutoff,
              "early_feeds": int(early.sum()), "holdout_feeds": int((~early).sum()), "iterations": ITERATIONS,
              "epochs": EPOCHS, "block_feeds": TRAIN_FEEDS, "completed_trials": [trial.name for trial in trials],
              "skipped_trials": read_json(args.output / "skipped.json"), "holdout_intervals_vs_first_block": intervals}
    folder = args.output / "comparison"
    folder.mkdir(exist_ok=True)
    atomic_text(folder / "report.json", json.dumps(report, indent=2) + "\n")


def publish_after_connecting(args):
    from team_reports import ROOT, git, publish

    reports = [("candidate-order-200k", args.prior / "comparison/report.json", args.prior_revision),
               ("evening-candidate-scaling", args.output / "comparison/report.json", "HEAD")]
    deadline = datetime.now().astimezone() + timedelta(hours=8)
    for name, report, revision in reports:
        while True:
            try:
                prefix = f"refs/heads/team-reports/{args.report_round}/{name}"
                if name != "candidate-order-200k" or not git(ROOT, "ls-remote", "--heads", "origin", prefix).stdout:
                    result = read_json(report)
                    extra = {"Fine-tune": "Нет", "Размер блока train": "200 000", "Разбиение val": "7 290 / 7 133"}
                    if name == "candidate-order-200k":
                        extra["Контроль GRU"] = "Переиспользован" if result["gru_control_reused"] else "Обучен заново"
                        for model, delta in result["holdout_paired_bootstrap"].items():
                            lo, hi = delta["ci95"]
                            extra[f"Δ NDCG {model}, 95% ДИ"] = f"{delta['delta']:+.6f} [{lo:+.6f}, {hi:+.6f}]"
                    else:
                        extra["Запуски, пропущенные по времени"] = ", ".join(result["skipped_trials"]) or "—"
                        extra["Число CatBoost-блоков по 200 000"] = sum(
                            model.startswith("cat-b") for model in result["completed_trials"])
                        for model, delta in result["holdout_intervals_vs_first_block"].items():
                            lo, hi = delta["ci95"]
                            extra[f"Δ NDCG {model} к первому блоку, 95% ДИ"] = (
                                f"{delta['delta']:+.6f} [{lo:+.6f}, {hi:+.6f}]")
                    publish(args.report_round, name, report, revision=revision, extra=extra)
                break
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                note(f"GitHub unavailable, will retry: {error}")
                if datetime.now().astimezone() >= deadline:
                    return
                time.sleep(120)


def launch(args):
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["AVITO_NOTIFY_TELEGRAM"] = "0"
    os.environ["GIT_TERMINAL_PROMPT"] = "0"
    note(f"Queued behind {args.prior.name}; budget ends {args.deadline.isoformat()}")
    while not (args.prior / "comparison/report.json").is_file():
        if any(path.read_text(encoding="utf-8").strip() not in ("", "0") for path in args.prior.glob("*.status")):
            raise RuntimeError("Candidate-order experiment failed; see its stage log")
        keep_awake()
        time.sleep(15)
    report = args.output / "comparison/report.json"
    if not report.is_file():
        train(args)
        names = [trial.name for trial in CONTROL] + read_json(args.output / "completed.json")
        run_stage(args, "comparison", report.parent, [*names, "Ансамбли"], "Вечерний прогон · результаты", [
            "evening_benchmark.py", "--compare", "--data", str(args.data), "--output", str(args.output),
            "--gpu", args.gpu, "--prior", str(args.prior), "--deadline", args.deadline.isoformat(),
            *(["--lightgbm", str(args.lightgbm)] if args.lightgbm else [])
        ], watch_root=args.prior)
    if args.report_round:
        publish_after_connecting(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("data", "output", "prior"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--deadline", type=datetime.fromisoformat, required=True)
    parser.add_argument("--report-round")
    parser.add_argument("--prior-revision", default="HEAD")
    parser.add_argument("--lightgbm", type=Path, help="Optional saved 200k LightGBM validation scores")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if args.deadline.tzinfo is None:
        parser.error("--deadline requires an explicit timezone offset")
    args.data, args.output, args.prior = args.data.resolve(), args.output.resolve(), args.prior.resolve()
    if args.lightgbm:
        args.lightgbm = args.lightgbm.resolve()
    (compare if args.compare else launch)(args)
