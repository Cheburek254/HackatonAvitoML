"""Run GRU and compare blends after a local sharded CatBoost run with its dashboard."""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def note(message):
    print(datetime.now().isoformat(timespec="seconds"), message, flush=True)


def keep_awake():
    # The first SYSTEM_POWER_STATUS byte is ACLineStatus. A one-shot request
    # resets the idle timer only on AC and never changes the Windows power plan.
    if os.name != "nt":
        return False
    status = ctypes.create_string_buffer(12)
    power = ctypes.windll.kernel32
    if power.GetSystemPowerStatus(ctypes.byref(status)) and status.raw[0] == 1:
        if not power.SetThreadExecutionState(1):
            raise ctypes.WinError()
        return True
    return False


def atomic_text(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def complete(report):
    return report.is_file() and json.loads(report.read_text(encoding="utf-8")).get("complete")


def run_stage(args, key, directory, models, label, command):
    if complete(directory / "report.json"):
        note(f"Reusing completed {key}")
        return
    log, status = args.output / f"{key}.log", args.output / f"{key}.status"
    status.unlink(missing_ok=True)
    config = dict(report=str(directory / "report.json"), log=str(log), status=str(status),
                  data=str(args.data), models=models, epochs=5, iterations=600, label=label)
    with log.open("wb") as out:
        atomic_text(args.catboost / "watch.json", json.dumps(config))
        note(f"Starting {key}")
        code = 1
        try:
            with subprocess.Popen([sys.executable, "-u", "-X", "utf8", *command], cwd=ROOT,
                                  stdout=out, stderr=subprocess.STDOUT) as proc:
                while proc.poll() is None:
                    keep_awake()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        pass
                code = proc.returncode
        finally:
            atomic_text(status, str(code))
    if code or not complete(directory / "report.json"):
        raise RuntimeError(f"{key} failed, see {log}")
    note(f"Completed {key}")


def main(args):
    fit, pair = args.output / "gru-fit", args.output / "comparison"
    source_status = args.catboost / "merged/training.status"
    job = {key: str(getattr(args, key)) for key in ("data", "catboost", "gpu")}
    manifest = args.output / "job.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != job:
        raise ValueError("This queue directory belongs to another run")
    atomic_text(manifest, json.dumps(job))
    note(f"Waiting for CatBoost; AC idle-sleep prevention active={keep_awake()}")
    while not source_status.exists():
        if any(p.read_text(encoding="utf-8").strip() not in ("", "0")
               for p in args.catboost.glob("parts/part-*/training.status")):
            raise RuntimeError("CatBoost shard failed; inspect its training.log")
        wrapper = args.catboost.with_suffix(".status")
        if wrapper.is_file() and wrapper.read_text(encoding="utf-8-sig").strip() not in ("", "0"):
            raise RuntimeError("CatBoost wrapper failed; inspect its log")
        keep_awake()
        time.sleep(15)
    if source_status.read_text(encoding="utf-8-sig").strip() != "0" or not complete(args.catboost / "merged/report.json"):
        raise RuntimeError("CatBoost did not finish successfully")
    if not (args.catboost / "merged/din-events-v2/val_predictions.parquet").is_file():
        raise FileNotFoundError("CatBoost merged val predictions for din-events-v2 are missing")
    os.environ.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", GIT_TERMINAL_PROMPT="0")
    run_stage(args, "gru", fit, ["gru-din-plus"], "GRU-DIN-Plus · 200 000 лент · 5 эпох",
              ["experiments.py", "benchmark", "--data", str(args.data), "--output", str(fit),
               "--gpu", args.gpu, "--models", "gru-din-plus", "--train-feeds", "200000",
               "--val-feeds", "0", "--epochs", "5", "--iterations", "600", "--batch-size", "16",
               "--threads", "6", "--history-window", "32", "--no-rrf", *(["--resume"] if fit.exists() else [])])
    run_stage(args, "blends", pair, ["rrf", "z-blend"], "CatBoost + GRU · RRF и Z-бленд",
              ["pair_blend.py", "--data", str(args.data), "--output", str(pair), "--gpu", args.gpu,
               "--left", str(args.catboost / "merged/din-events-v2"), "--right", str(fit / "gru-din-plus")])
    # All computation is complete and available in the UI before any network call.
    if not args.report_round:
        return
    from team_reports import sync
    report = json.loads((pair / "report.json").read_text(encoding="utf-8"))
    sync(args.report_round, "gru-din-plus-200k", fit / "report.json")
    sync(args.report_round, "cat-gru-blends", pair / "report.json", extra={
        "Train CatBoost / GRU": " / ".join(str(report["train_feeds_by_model"][name]) for name in report["members"]),
        "Вес GRU по ранней val": ", ".join(f"{k}: {v:.0%}" for k, v in report["right_weights"].items()),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("data", "catboost", "output"):
        parser.add_argument("--" + option, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--report-round")
    args = parser.parse_args()
    if not (args.catboost / "watch.json").is_file():
        parser.error("Start sharded_runs.py train with --person all --ui first")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "queue.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (args.output / "queue.status").unlink(missing_ok=True)
        try:
            main(args)
        except Exception:
            traceback.print_exc()
            atomic_text(args.output / "queue.status", "1")
            raise SystemExit(1)
        atomic_text(args.output / "queue.status", "0")
