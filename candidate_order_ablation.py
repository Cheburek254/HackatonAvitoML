"""Compare candidate tie handling across CatBoost, DIN Plus and GRU Plus."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from training_queue import ROOT, atomic_text, complete, run_stage


PAIRS = (("din-events-v2", "din-events-v2-ties"),
         ("neural-din-plus", "neural-din-plus-ties"),
         ("gru-din-plus", "gru-din-plus-ties"))
MODELS = tuple(name for pair in PAIRS for name in pair)
TRAIN_FEEDS, ITERATIONS, EPOCHS, WINDOW, BATCH_SIZE, THREADS = 200_000, 600, 5, 32, 16, 6
FEATURE_WORKERS = 3


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_control(args):
    if args.reuse_gru is None:
        return
    control = read_json(args.reuse_gru / "report.json")
    expected = {"train_feeds": TRAIN_FEEDS, "train_offset": 0, "val_feed_limit": 0,
                "epochs": EPOCHS, "requested_models": ["gru-din-plus"], "complete": True}
    options = {"history_window": WINDOW, "batch_size": BATCH_SIZE, "threads": THREADS, "neural_device": "cuda"}
    if (any(control.get(key) != value for key, value in expected.items())
            or any(control["options"].get(key) != value for key, value in options.items())
            or Path(control["data"]).resolve() != args.data):
        raise ValueError("Saved GRU control has a different dataset or training configuration")


def compare(args):
    import numpy as np
    import polars as pl

    from accelerator import select_gpu
    from ensemble import Slates
    from evaluation import KEYS, attach_scores
    from experiments import read_frame, validation
    from pair_blend import paired_interval

    gpu = select_gpu(args.gpu)
    labels, cutoff = validation(args.data, 0)
    feeds = read_frame(args.data, "val", ["feed_id", "timestamp", "user_id"], 0).sort("feed_id")
    early = feeds["timestamp"].to_numpy() < cutoff
    if (len(feeds), int(early.sum()), len(labels), cutoff) != (14423, 7290, 1389971, 1790107648):
        raise ValueError("The dataset has a different competition validation split")
    users = feeds["user_id"].to_numpy()[~early]
    results, per_feed = [], {}
    for name in MODELS:
        folder = args.reuse_gru if name == "gru-din-plus" and args.reuse_gru else args.output / name
        report = read_json(folder / "report.json")
        if (not report["complete"] or report["train_feeds"] != TRAIN_FEEDS or report["train_offset"] != 0
                or report["tuning_cutoff_timestamp"] != cutoff or report["val_candidates"] != len(labels)):
            raise ValueError(f"{name} was evaluated on different training or validation data")
        print(f"Starting {name}", flush=True)
        predictions = pl.read_parquet(folder / name / "val_predictions.parquet")
        slates = Slates(attach_scores(labels.select(*KEYS, "target"), predictions), "cuda")
        if not np.array_equal(slates.frame["feed_id"].to_numpy()[slates.rows[:, 0]], feeds["feed_id"].to_numpy()):
            raise ValueError(f"Val feed IDs do not match for {name}")
        values = slates.per_feed_ndcg(slates.pack("score")).cpu().numpy().astype(np.float64)
        per_feed[name] = values
        row = next(model for model in report["models"] if model["model"] == name)
        if abs(row["holdout"]["ndcg"] - values[~early].mean()) > 1e-6:
            raise ValueError(f"Reported and recomputed holdout NDCG differ for {name}")
        results.append(row)
        print(f"Finished {name}: holdout_ndcg={row['holdout']['ndcg']:.8f}", flush=True)
    changes = {after: paired_interval((per_feed[after] - per_feed[before])[~early], users)
               for before, after in PAIRS}
    result = {"complete": True, "gpu": vars(gpu), "models": results, "holdout_paired_bootstrap": changes,
              "train_feeds": TRAIN_FEEDS, "train_feed_limit": TRAIN_FEEDS, "val_feed_limit": 0,
              "train_offset": 0, "val_feeds": len(feeds), "val_candidates": len(labels),
              "tuning_cutoff_timestamp": cutoff, "early_feeds": int(early.sum()), "holdout_feeds": int((~early).sum()),
              "iterations": ITERATIONS, "epochs": EPOCHS, "history_window": WINDOW,
              "gru_control_reused": args.reuse_gru is not None, "feature_workers": FEATURE_WORKERS,
              "neural_prefetch": True}
    folder = args.output / "comparison"
    folder.mkdir(exist_ok=True)
    atomic_text(folder / "report.json", json.dumps(result, indent=2) + "\n")


def launch(args):
    check_control(args)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"data": str(args.data), "gpu": args.gpu, "reuse_gru": str(args.reuse_gru)}
    job = args.output / "job.json"
    if job.is_file() and read_json(job) != manifest:
        raise ValueError("Output belongs to a different candidate-order experiment")
    atomic_text(job, json.dumps(manifest))
    os.environ["AVITO_NOTIFY_TELEGRAM"] = "0"
    if args.ui:
        subprocess.Popen([sys.executable, "-X", "utf8", str(ROOT / "dashboard.py"),
                          "--watch", str(args.output / "watch.json")], cwd=ROOT,
                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    for name in MODELS:
        if name == "gru-din-plus" and args.reuse_gru is not None:
            continue
        folder = args.output / name
        run_stage(args, name, folder, [name], f"Порядок кандидатов · {name}", [
            "experiments.py", "benchmark", "--data", str(args.data), "--output", str(folder),
            "--gpu", args.gpu, "--models", name, "--train-feeds", str(TRAIN_FEEDS), "--val-feeds", "0",
            "--iterations", str(ITERATIONS), "--epochs", str(EPOCHS), "--history-window", str(WINDOW),
            "--batch-size", str(BATCH_SIZE), "--threads", str(THREADS), "--no-rrf", "--prefetch-neural",
            *(["--feature-workers", str(FEATURE_WORKERS)] if name.startswith("din-events-v2") else []),
            *(["--resume"] if folder.exists() else [])
        ], watch_root=args.output)
    comparison = args.output / "comparison"
    run_stage(args, "comparison", comparison, list(MODELS), "Сравнение до и после", [
        "candidate_order_ablation.py", "--compare", "--data", str(args.data), "--output", str(args.output),
        "--gpu", args.gpu, *(["--reuse-gru", str(args.reuse_gru)] if args.reuse_gru else [])
    ], watch_root=args.output)
    if args.report_round:
        from team_reports import sync

        report_path = comparison / "report.json"
        changes = read_json(report_path)["holdout_paired_bootstrap"]
        extra = {"Контроль GRU": "Переиспользован" if args.reuse_gru else "Обучен заново",
                 "Fine-tune": "Нет", "Окно истории нейросетей": WINDOW, "Размер батча": BATCH_SIZE,
                 "CPU-процессы для признаков CatBoost": FEATURE_WORKERS, "Предзагрузка батчей": "Да"}
        for after, row in changes.items():
            lo, hi = row["ci95"]
            extra[f"Δ NDCG {after}, 95% ДИ"] = f"{row['delta']:+.6f} [{lo:+.6f}, {hi:+.6f}]"
        sync(args.report_round, "candidate-order-200k", report_path, extra=extra)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--reuse-gru", type=Path, help="Completed 200k baseline GRU benchmark")
    parser.add_argument("--ui", action="store_true")
    parser.add_argument("--report-round")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    args.data, args.output = args.data.resolve(), args.output.resolve()
    if args.reuse_gru:
        args.reuse_gru = args.reuse_gru.resolve()
    if args.compare:
        compare(args)
    else:
        launch(args)
