"""Run V4 pilots and the assigned robust shards after the V4 reconstruction."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import sleep

from team_reports import sync
from training_queue import atomic_text, keep_awake, note


ROOT = Path(__file__).resolve().parent


def run(command, *, cwd=ROOT):
    with subprocess.Popen([sys.executable, "-u", "-X", "utf8", *map(str, command)], cwd=cwd) as process:
        while True:
            try:
                code = process.wait(timeout=15)
                break
            except subprocess.TimeoutExpired:
                keep_awake()
    if code:
        raise subprocess.CalledProcessError(code, command)


def pilot(args, stage, name, directory, required):
    if all((directory / name).is_file() for name in required):
        return
    log, status = args.pilots / f"{name}.log", args.pilots / f"{name}.status"
    status.unlink(missing_ok=True)
    atomic_text(args.pilots / "watch.json", json.dumps(dict(report=str(args.pilots / "report.json"),
                log=str(log), status=str(status), data=str(args.data), models=[name], epochs=5, iterations=200,
                label="V4 · пилоты на 200 000 лент")))
    command = [ROOT / "v4_extensions.py", stage, "--data", args.data, "--output", directory, "--gpu", args.gpu,
               "--v4-code", args.v4_code, "--retrain", args.retrain]
    code = 1
    try:
        with log.open("w", encoding="utf-8") as handle:
            print(f"Starting {name}", file=handle, flush=True)
            with subprocess.Popen([sys.executable, "-u", "-X", "utf8", *map(str, command)], cwd=ROOT,
                                  stdout=handle, stderr=subprocess.STDOUT) as child:
                while True:
                    try:
                        code = child.wait(timeout=15)
                        break
                    except subprocess.TimeoutExpired:
                        keep_awake()
            if code == 0 and not all((directory / artifact).is_file() for artifact in required):
                code = 1
            if code == 0:
                print(f"Finished {name}", file=handle, flush=True)
    finally:
        atomic_text(status, str(code))
    if code:
        raise RuntimeError(f"Pilot {name} failed. See {log}")


def publish(args):
    path = args.pilots / "report.json"
    if not path.is_file():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    weights = report["weights"]
    extra = {"Доля GRU с очисткой в слоте GRU v4": f"{weights['v4-end4rec']:.0%}",
             "Коэффициент EDG4Rec": f"{weights['v4-edg4rec']:.2f}",
             "Покрытие графом val": f"{report['graph_coverage']:.1%}",
             "NDCG поздней val контрольного V4 с fine-tune на ранней": report["fine_tune_reference_late"],
             "Подбор весов": "Ранняя val, timestamp < 1790107648; исходные веса V4 40/30/15/15",
             "GRU и граф": "GRU v4 с мягким весом событий; граф из прошлых дней train, 200 000 лент"}
    sync("avito-v4-extensions", "kirill-200k", path, extra=extra, revision=args.revision)


def run_full(args):
    stats = args.robust_code / "runs/global_stats"
    required = ("item_stats.parquet", "item_stats_with_val.parquet", "meta.json", "meta_with_val.json", "user_stats.parquet")
    if not all((stats / name).is_file() for name in required):
        note("Building V3 robust statistics")
        run(["build_global_stats.py", "--data", args.data, "--output", stats], cwd=args.robust_code)
    if not all((stats / name).is_file() for name in required):
        raise RuntimeError("Robust statistics files are incomplete")
    note(f"Starting robust shards for {args.person}")
    run(["sharded_runs.py", "train", "--data", args.data, "--output", args.full, "--models", "din-events-v3-robust",
         "--train-feeds", 0, "--val-feeds", 0, "--shards", 21, "--iterations", 800, "--person", args.person,
         "--predict-test", "--ui", "--gpu", args.gpu, "--threads", 8, "--report-round", "v3-robust-full"], cwd=args.robust_code)


def wait_for_reconstruction(args):
    note("Waiting for V4 reconstruction")
    reconstruction = args.retrain / "report.json"
    error_log = args.retrain / "queue.err.log"
    while not reconstruction.is_file():
        if error_log.is_file() and "Traceback (most recent call last)" in error_log.read_text(encoding="utf-8", errors="replace"):
            raise RuntimeError(f"V4 reconstruction stopped before the comparison. See {error_log}")
        keep_awake()
        sleep(15)
    if not json.loads(reconstruction.read_text(encoding="utf-8"))["complete"]:
        raise RuntimeError("V4 reconstruction report is incomplete")


def run_pilots(args):
    subprocess.Popen([sys.executable, "-u", str(ROOT / "dashboard.py"), "--watch", str(args.pilots / "watch.json")],
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), cwd=ROOT)
    pilot_failure = None
    for stage, name, folder in (("end", "gru-din-v4", "end4rec"), ("graph", "edg4rec", "edg4rec")):
        note(f"Starting {folder}")
        try:
            pilot(args, stage, name, args.pilots / folder, ("val_predictions.parquet", "metrics.json", "model.pt" if stage == "end" else "model.cbm"))
        except RuntimeError as error:
            note(error)
            pilot_failure = error
    if pilot_failure:
        raise pilot_failure
    note("Comparing V4 and the two extensions")
    pilot(args, "compare", "v4-comparison", args.pilots, ("report.json",))
    publish(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "retrain", "v4-code", "robust-code", "pilots", "full"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--person", choices=("kirill", "savva", "dimon"), default="kirill")
    parser.add_argument("--revision", default="HEAD")
    args = parser.parse_args()
    args.pilots.mkdir(parents=True, exist_ok=True)
    pilot_failure = None
    try:
        wait_for_reconstruction(args)
        run_pilots(args)
    except (RuntimeError, subprocess.SubprocessError) as error:
        note(error)
        pilot_failure = error
    try:
        run_full(args)
    finally:
        publish(args)
    if pilot_failure:
        raise pilot_failure
    note("Follow-up queue finished")


if __name__ == "__main__":
    main()
