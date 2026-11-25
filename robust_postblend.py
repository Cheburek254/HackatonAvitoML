"""Compare the completed 21-shard model with the fixed QSM/GRU/LGBM blend."""

import argparse
import json
from pathlib import Path
from shutil import copyfile
import subprocess
from tempfile import TemporaryDirectory
from time import sleep

import polars as pl

from evaluation import KEYS, attach_scores, ndcg, zscore
from team_reports import checksum, git, publish, ROOT


MODEL = "din-events-v3-robust"
CUTOFF = 1790107648
GRID = (0., .05, .10, .15, .20)


def add_scores(frame, path, column):
    return attach_scores(frame, pl.read_parquet(path, columns=[*KEYS, "score"])).rename({"score": column})


def mix(weight):
    if weight == 0:
        return pl.col("base")
    if weight == 1:
        return pl.col("robust")
    return ((1 - weight) * pl.col("z_base") + weight * pl.col("z_robust")).cast(pl.Float32)


def measure(frame, weight):
    return ndcg(frame.with_columns(mix(weight).alias("score")))


def load_validation(data, runs):
    repro = runs / "valishinqa-v4-repro"
    frame = (pl.scan_parquet(str(data / "val/impressions/part-*.parquet"))
             .select(*KEYS, "timestamp", "target").explode("item_id", "target").collect())
    frame = add_scores(frame, repro / "qsm_val_predictions.parquet", "qsm")
    frame = add_scores(frame, repro / "gru4-published_val_predictions.parquet", "gru")
    frame = add_scores(frame, runs / "lightgbm-200k-rtx/lightgbm/val_predictions.parquet", "lgbm")
    return frame.with_columns((.40 * zscore("qsm") + .45 * zscore("gru") + .15 * zscore("lgbm"))
                              .cast(pl.Float32).alias("base")).drop("qsm", "gru", "lgbm")


def compare(frame):
    frame = frame.with_columns(zscore("base").alias("z_base"), zscore("robust").alias("z_robust"))
    early = frame.filter(pl.col("timestamp") < CUTOFF)
    times = early.select("feed_id", "timestamp").unique().sort("timestamp")["timestamp"]
    boundary = times[len(times) // 2]
    folds = [early.filter(pl.col("timestamp") < boundary), early.filter(pl.col("timestamp") >= boundary)]
    if not all(f.height for f in folds):
        raise ValueError("Need two nonempty chronological folds")
    trials = []
    for weight in GRID:
        scores = [measure(fold, weight)["ndcg"] for fold in folds]
        gains = [s - ref for s, ref in zip(scores, trials[0]["scores"] if trials else scores, strict=True)]
        trials.append(dict(weight=weight, scores=scores, worst_gain=min(gains)))
    selected = max(trials, key=lambda trial: (trial["worst_gain"], -trial["weight"]))["weight"]
    late = frame.filter(pl.col("timestamp") >= CUTOFF)
    rows = []
    for name, weight in ((MODEL, 1.), ("qsm-gru4-lgbm2", 0.), ("postblend", selected)):
        rows.append(dict(model=name, weight=weight, tuning=measure(early, weight), holdout=measure(late, weight),
                         val=measure(frame, weight)))
    rows.sort(key=lambda row: row["holdout"]["ndcg"], reverse=True)
    return dict(models=rows, early_boundary=boundary, late_boundary=CUTOFF, trials=trials, selected=selected,
                fold_sizes=[fold["feed_id"].n_unique() for fold in (*folds, late)])


def finish(data, runs, merged, destination):
    source = json.loads((merged / "report.json").read_text(encoding="utf-8"))
    if (not source.get("complete") or sorted(p["index"] for p in source["parts"]) != list(range(1, 22))
            or source["settings"]["models"] != [MODEL] or source["tuning_cutoff_timestamp"] != CUTOFF):
        raise ValueError("Unexpected model, validation split or shard plan in the merged report")
    frame = add_scores(load_validation(data, runs), merged / MODEL / "val_predictions.parquet", "robust")
    if frame["feed_id"].n_unique() != source["val_feeds"]:
        raise ValueError("Validation feeds differ from the merged report")
    result = compare(frame)
    reference = source["models"][0]["holdout"]["ndcg"]
    actual = next(row["holdout"]["ndcg"] for row in result["models"] if row["model"] == MODEL)
    if abs(reference - actual) > 1e-5:
        raise ValueError(f"Merged late-val NDCG mismatch: {reference} vs {actual}")
    original = merged / MODEL / "submission.parquet"
    base = runs / "valishinqa-v4-repro/submission_qsm_gru4_base_lgbm2.parquet"
    sample = pl.read_parquet(data / "sample_submission.parquet")
    test = add_scores(sample.select(KEYS), base, "base")
    test = add_scores(test, original, "robust").with_columns(zscore("base").alias("z_base"), zscore("robust").alias("z_robust"))
    winner = result["models"][0]
    expected = test.select(*KEYS, mix(winner["weight"]).alias("score")).cast(sample.schema)
    if not expected.select(KEYS).equals(sample.select(KEYS)) or not expected["score"].is_finite().all():
        raise ValueError("Winning test predictions differ from the submission template")
    with TemporaryDirectory(dir=destination.parent) as temp:
        folder = Path(temp) / "result"
        folder.mkdir()
        saved = folder / "submission.parquet"
        if winner["weight"] in (0., 1.):
            copyfile(original if winner["weight"] else base, saved)
        else:
            expected.write_parquet(saved)
        if not pl.read_parquet(saved).equals(expected):
            raise ValueError("Saved predictions failed verification")
        result.update(complete=True, train_feeds=source["train_feeds"], val_feeds=source["val_feeds"],
                      winner=winner["model"], original_sha256=checksum(original), winner_sha256=checksum(saved))
        (folder / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        folder.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs/avito-real")
    args = parser.parse_args()
    revision = git(ROOT, "rev-parse", "HEAD").stdout.strip()
    merged = args.runs / "v3-robust-full/merged"
    destination = args.runs / "v3-robust-postblend/result"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_dir():
        print("Waiting for the completed 21-shard report and original submission", flush=True)
        while not (merged / "report.json").is_file() or not (merged / MODEL / "submission.parquet").is_file():
            sleep(30)
        finish(args.data, args.runs, merged, destination)
    report_path = destination / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if checksum(destination / "submission.parquet") != report["winner_sha256"]:
        raise ValueError("Saved winner does not match the report")
    extra = {"Победитель по поздней val": report["winner"], "Вес robust по двум ранним отрезкам": report["selected"],
             "Ленты в двух ранних отрезках и в поздней val": str(report["fold_sizes"]),
             "Границы по timestamp": f"{report['early_boundary']}, {CUTOFF}",
             "Подбор": "Максимум худшего прироста относительно 40/45/15 на двух ранних отрезках",
             "SHA-256 исходного сабмишена": report["original_sha256"], "SHA-256 победителя": report["winner_sha256"]}
    for trial in report["trials"]:
        extra[f"Вес {trial['weight']:.0%}, NDCG на двух ранних отрезках"] = " / ".join(f"{s:.8f}" for s in trial["scores"])
    while True:
        try:
            publish("v3-robust-full", "postblend", report_path, extra=extra, revision=revision)
            return
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            print(f"Report saved, retrying GitHub in two minutes: {error}", flush=True)
            sleep(120)


if __name__ == "__main__":
    main()
