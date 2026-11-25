"""Benchmark independent rankers and fit an RRF blend on the same Avito val."""

import argparse
import json
import os
from shutil import copytree
import subprocess
import sys
import zipimport
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import polars as pl

from evaluation import KEYS, attach_scores, blend, ndcg
from ranker import STACK_VARIANTS, VARIANTS, feed_batches

NEURAL = ("transformer", "transformer-listmle", "avito-joint", "neural-din", "bst", "gru-din", "sasrec",
          "set-transformer", "neural-pairwise", "neural-din-plus", "gru-din-plus", "gru-din-v3")
OFFICIAL = ("avito-catboost", "avito-transformer")
CUDA_NEURAL = ("dual-history",)
TREE_MODELS = tuple(name for name in VARIANTS if name not in STACK_VARIANTS)
GPU_ABLATION = ("history", "category-trend", "category-trend-wide", "category-model", "popularity-trend",
                "category-covis", "pixie-decays", "covis-pixie", "embedding-clusters", "history-distill", *CUDA_NEURAL)


def read_frame(data, split, columns, limit):
    frames = list(feed_batches(data / split, columns, limit))
    if not frames:
        raise ValueError(f"No feeds in {data / split}")
    return pl.concat(frames)


def validation(data, limit):
    columns = ["feed_id", "item_id", "category_id", "target", "timestamp", "history__item_id"]
    feeds = read_frame(data, "val", columns, limit)
    timestamps = feeds["timestamp"].unique().sort()
    if len(timestamps) < 2:
        raise ValueError("Benchmark needs at least two distinct validation timestamps")
    cutoff = int(timestamps[len(timestamps) // 2])
    labels = feeds.with_columns(pl.col("history__item_id").list.len().alias("history_length")).drop("history__item_id")
    return labels.explode("item_id", "category_id", "target"), cutoff


def organizer_module(archive, name):
    spec = zipimport.zipimporter(str(archive.resolve())).find_spec(name)
    if spec is None:
        raise ValueError(f"{archive} does not contain {name}.py at its root")
    from importlib.util import module_from_spec

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_catboost(args, tune_ids, output):
    baseline = organizer_module(args.baseline_zip, "catboost_baseline")
    training = baseline.read_pool(args.data / "train", args.train_feeds)
    candidates = baseline.candidate_dataset(args.data / "val", args.val_feeds).collect(streaming=True).rechunk()
    tuning = candidates.filter(pl.col("feed_id").is_in(tune_ids))
    pool = baseline.Pool(tuning.select(baseline.FEATURES), label=tuning["target"],
                         cat_features=baseline.CAT_FEATURES, group_id=tuning["feed_id"])
    params = {**baseline.MODEL_PARAMS, "num_trees": args.iterations, "thread_count": args.threads,
              "task_type": args.cat_device.upper()}
    if args.cat_device == "cpu":
        params.pop("devices", None)
        params.pop("gpu_ram_part", None)
    model = baseline.CatBoostRanker(**params)
    model.fit(training, eval_set=pool, **baseline.FIT_PARAMS)
    output.mkdir(parents=True)
    model.save_model(str(output / "model.cbm"))
    return baseline.predict_scores(candidates, model)


def official_transformer(args, tune_ids, output):
    import torch
    from torch.utils.data import Subset

    baseline = organizer_module(args.baseline_zip, "neural_baseline")
    training = baseline.CandidateData(args.data / "train", args.train_feeds)
    val = baseline.CandidateData(args.data / "val", args.val_feeds, training.categories_mapping)
    feed_ids = val.keys["feed_id"].to_numpy()[val.offsets[:-1]]
    tuning = Subset(val, np.flatnonzero(np.isin(feed_ids, tune_ids.to_numpy())).tolist())
    tuning.keys = val.keys.filter(val.keys["feed_id"].is_in(tune_ids))
    config = {**baseline.TRAIN_CONFIG, "device": args.neural_device, "epochs": args.epochs,
              "threads": args.threads, "batch_size": args.batch_size}
    baseline.fit(training, tuning, output, config)
    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    model = baseline.Ranker(len(training.categories_mapping), **checkpoint["model_params"]).to(args.neural_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    batches = baseline.DataLoader(val, batch_size=args.batch_size, collate_fn=baseline.collate_feeds)
    predictions = []
    with torch.inference_mode():
        for features, categories, users, padding, _ in batches:
            inputs = [tensor.to(args.neural_device) for tensor in (features, categories, users, padding)]
            predictions.append(torch.sigmoid(model(*inputs).cpu()[~padding]).numpy())
    return val.keys.with_columns(pl.Series("score", np.concatenate(predictions), dtype=pl.Float32))


def run_model(args, name, output, tune_ids, cutoff, extra=()):
    if name in OFFICIAL:
        runner = official_catboost if name == "avito-catboost" else official_transformer
        predictions = runner(args, tune_ids, output)
        predictions.write_parquet(output / "val_predictions.parquet")
        return predictions

    script = "ranker.py" if name in VARIANTS else "neural.py"
    command = [sys.executable, "-u", "-X", "utf8", str(Path(__file__).with_name(script)), "train", "--variant", name,
               "--data", str(args.data), "--output", str(output), "--train-feeds", str(args.train_feeds),
               "--val-feeds", str(args.val_feeds), "--tune-before", str(cutoff), "--threads", str(args.threads)]
    if name in CUDA_NEURAL:
        command = [sys.executable, "-u", "-X", "utf8", str(Path(__file__).with_name("dual_history.py")), "train",
                   "--data", str(args.data), "--output", str(output), "--train-feeds", str(args.train_feeds),
                   "--val-feeds", str(args.val_feeds), "--tune-before", str(cutoff),
                   "--epochs", str(args.epochs), "--batch-size", str(args.dual_batch_size)]
    elif name in VARIANTS:
        device = "cpu" if name == "lambda" else args.cat_device
        command.extend(["--iterations", str(args.iterations), "--device", device])
    else:
        command.extend(["--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                        "--device", args.neural_device, "--history-window", str(args.history_window)])
    if args.train_offset:
        command.extend(["--train-offset", str(args.train_offset)])
    subprocess.run([*command, *extra], check=True)
    return pl.read_parquet(output / "val_predictions.parquet")


def score_summary(joined, cutoff, metric=ndcg, groups=None):
    early = joined.filter(pl.col("timestamp") < cutoff)
    late = joined.filter(pl.col("timestamp") >= cutoff)
    by_history = {}
    for name, mask in (("empty", pl.col("history_length") == 0),
                       ("short", pl.col("history_length").is_between(1, 32)),
                       ("long", pl.col("history_length") > 32)):
        rows = late.filter(mask)
        if rows.height:
            by_history[name] = metric(rows)["ndcg"]
    result = {"val": metric(joined), "tuning": metric(early), "holdout": metric(late), "holdout_by_history": by_history}
    if groups:
        result["holdout_by_category"] = {}
        for category, ids in groups.items():
            result["holdout_by_category"][str(category)] = {
                "feeds": len(ids), "ndcg": metric(late.filter(pl.col("feed_id").is_in(ids)))["ndcg"]
            }
    return result


def fit_blend(labels, cutoff, results, files, metric=ndcg, groups=None):
    early = labels.filter(pl.col("timestamp") < cutoff)
    best = sorted(results, key=lambda result: result["tuning"]["ndcg"], reverse=True)
    left = best[0]["model"]
    families = {model: "neural" if model in (*NEURAL, *CUDA_NEURAL, "avito-transformer") else "tree" for model in files}
    right = next((result["model"] for result in best[1:] if families[result["model"]] != families[left]), best[1]["model"])
    left_scores, right_scores = (pl.read_parquet(files[name]) for name in (left, right))
    tuning_ids = early["feed_id"].unique()
    early_left, early_right = (scores.filter(pl.col("feed_id").is_in(tuning_ids)) for scores in (left_scores, right_scores))
    weights = np.linspace(0, 1, 11)
    weight = max(weights, key=lambda w: metric(attach_scores(early, blend(early_left, early_right, w)))["ndcg"])
    predictions = blend(left_scores, right_scores, weight)
    scores = score_summary(attach_scores(labels, predictions), cutoff, metric, groups)
    return {"model": "rrf", "left": left, "right": right, "left_weight": float(weight), **scores}, predictions


def category_groups(args, cutoff):
    feeds = read_frame(args.data, "val", ["feed_id", "timestamp", "category_id", "target"], args.val_feeds)
    positive = feeds.filter(pl.col("timestamp") >= cutoff).select("feed_id", "category_id", "target").explode("category_id", "target")
    dominant = (
        positive.filter((pl.col("target") > 0) & (pl.col("category_id") != 0)).group_by("feed_id", "category_id").len()
        .sort(["feed_id", "len", "category_id"], descending=[False, True, False]).unique("feed_id")
    )
    grouped = dominant.group_by("category_id").agg("feed_id").sort(pl.col("feed_id").list.len(), descending=True)
    return {category: ids for category, ids in grouped.head(8).iter_rows() if len(ids) >= 25}


def save_report(path, report, results):
    report["models"] = sorted(results, key=lambda row: row["holdout"]["ndcg"], reverse=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def pareto_models(results):
    measured = [row for row in results if row.get("peak_ram_mib") is not None]
    points = {row["model"]: (-row["holdout"]["ndcg"], row["seconds"], row["peak_ram_mib"]) for row in measured}
    return [
        name for name, values in points.items()
        if not any(other != values and all(a <= b for a, b in zip(other, values, strict=True)) for other in points.values())
    ]


def import_results(args, report, labels, results, files):
    for directory in args.reuse_models_from:
        previous = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        fields = ("val_feeds", "val_candidates", "tuning_cutoff_timestamp", "train_feed_limit", "iterations", "gpu")
        if (Path(previous["data"]).resolve() != args.data.resolve() or not previous["complete"]
                or any(previous.get(field) != report[field] for field in fields)
                or previous.get("train_offset", 0) != report["train_offset"]):
            raise ValueError(f"Incompatible source for reusing trained models: {directory}")
        for row in previous["models"]:
            name = row["model"]
            supported = (*TREE_MODELS, *CUDA_NEURAL, "avito-catboost")
            if name not in args.models or name in files or name not in supported:
                continue
            if name in CUDA_NEURAL and previous["epochs"] != args.epochs:
                raise ValueError(f"Incompatible neural epoch budget in {directory}")
            source = directory / name
            predictions = source / "val_predictions.parquet"
            attach_scores(labels, pl.read_parquet(predictions))
            destination = args.output / name
            copytree(source, destination)
            files[name] = destination / predictions.name
            results.append({**row, "reused_from": str(directory)})
            print(f"Reusing {name} from {directory}", flush=True)


def fit_stack_models(args, tune_ids, cutoff, files):
    oof_paths = []
    teachers = ("avito-joint", "gru-din")
    for teacher in teachers:
        path = args.output / "oof" / f"{teacher}.parquet"
        oof_paths.append(str(path))
        if not path.is_file():
            script = Path(__file__).with_name("neural.py")
            subprocess.run([
                sys.executable, "-u", "-X", "utf8", str(script), "oof", "--variant", teacher, "--data", str(args.data),
                "--output", str(path), "--train-feeds", str(args.train_feeds), "--train-offset", str(args.train_offset),
                "--batch-size", str(args.batch_size),
                "--epochs", str(args.epochs), "--threads", str(args.threads), "--device", args.neural_device,
                "--history-window", str(args.history_window),
            ], check=True)
    extra = ["--stack-train", *oof_paths, "--stack-val", *[str(files[teacher]) for teacher in teachers]]
    for name in STACK_VARIANTS:
        yield name, extra


def benchmark(args):
    metric = ndcg
    selected = None
    if args.gpu is not None:
        from accelerator import cuda_tensor, gpu_ndcg, has_torch_cuda, select_gpu

        selected = select_gpu(args.gpu)
        print(f"GPU: {selected.name} ({selected.memory})", flush=True)
        if not has_torch_cuda():
            cuda_tensor()
        metric = gpu_ndcg
    labels, cutoff = validation(args.data, args.val_feeds)
    tune_ids = labels.filter(pl.col("timestamp") < cutoff)["feed_id"].unique()
    if not args.resume:
        args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / ".benchmark.lock").open("a+b") as lock:
        try:
            if os.name == "nt":
                import msvcrt

                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"A benchmark is already writing to {args.output}") from error
        run_benchmark(args, labels, cutoff, tune_ids, metric, selected)


def run_benchmark(args, labels, cutoff, tune_ids, metric, selected):
    results, files = [], {}
    batches = feed_batches(args.data / "train", ["feed_id"], args.train_feeds, offset=args.train_offset)
    train_feeds = sum(batch.height for batch in batches)
    if not train_feeds:
        raise ValueError("No training feeds in the requested range")
    report = {"data": str(args.data), "val_feeds": labels["feed_id"].n_unique(), "val_candidates": labels.height,
              "tuning_cutoff_timestamp": cutoff, "train_feeds": train_feeds,
              "train_feed_limit": args.train_feeds, "val_feed_limit": args.val_feeds, "train_offset": args.train_offset,
              "iterations": args.iterations, "epochs": args.epochs, "requested_models": args.models,
              "gpu": vars(selected) if selected else None, "complete": False}
    options = {"by_category": args.by_category, "history_window": args.history_window, "threads": args.threads,
               "batch_size": args.batch_size, "dual_batch_size": args.dual_batch_size,
               "cat_device": args.cat_device, "neural_device": args.neural_device,
               "baseline_zip": str(args.baseline_zip) if args.baseline_zip else None,
               "family_blends": args.family_blends, "stack": args.stack, "optuna_trials": args.optuna_trials,
               "reuse_models_from": [str(path) for path in args.reuse_models_from], "no_rrf": args.no_rrf}
    report_path = args.output / "report.json"
    if args.resume and report_path.is_file():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        # Older reports recorded the train limit and actual val count only.
        previous.setdefault("train_feeds", train_feeds)
        previous.setdefault("val_feed_limit", args.val_feeds)
        previous.setdefault("train_offset", 0)
        if (any(previous.get(key) != value for key, value in report.items())
                or previous.get("options", options) != options):
            raise ValueError("Resume requires an incomplete report and the same data, model list, GPU and budgets")
        results = previous["models"]
        files = {row["model"]: args.output / row["model"] / "val_predictions.parquet" for row in results}
        for path in files.values():
            attach_scores(labels, pl.read_parquet(path))
        print(f"Resuming with {len(results)} completed models", flush=True)
    elif args.resume and any(path.name != ".benchmark.lock" for path in args.output.iterdir()):
        raise ValueError("Cannot resume: the benchmark has outputs but no report")
    report["options"] = options
    groups = category_groups(args, cutoff) if args.by_category else None
    import_results(args, report, labels, results, files)
    save_report(report_path, report, results)

    def train_model(name, extra=()):
        if name in files:
            return
        start = perf_counter()
        print(f"Starting {name}", flush=True)
        output = args.output / name
        if args.resume and output.exists():
            output.rename(output.with_name(f".incomplete-{name}-{uuid4().hex[:8]}"))
        predictions = run_model(args, name, output, tune_ids, cutoff, extra)
        scores = score_summary(attach_scores(labels, predictions), cutoff, metric, groups)
        metrics_path = output / "metrics.json"
        stats = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        results.append({"model": name, **scores, "seconds": round(perf_counter() - start, 2),
                        **{key: stats[key] for key in ("train_feeds", "peak_ram_mib", "peak_private_mib",
                                                      "tuning_curve", "epoch_seconds")
                           if key in stats}})
        files[name] = output / "val_predictions.parquet"
        save_report(report_path, report, results)
        print(f"Finished {name}: holdout_ndcg={scores['holdout']['ndcg']:.8f}", flush=True)

    for name in args.models:
        train_model(name)
    if args.stack:
        print("Preparing out-of-fold neural predictions for stacking", flush=True)
        for name, extra in fit_stack_models(args, tune_ids, cutoff, files):
            train_model(name, extra)
    if len(results) >= 2 and not args.no_rrf:
        print("Fitting RRF on early validation", flush=True)
        result, predictions = fit_blend(labels, cutoff, results, files, metric, groups)
        results.append(result)
        predictions.write_parquet(args.output / "rrf_val_predictions.parquet")
    if args.family_blends:
        from ensemble import fit_families

        for name, predictions, metadata in fit_families(labels, cutoff, results, files, args.neural_device, args.optuna_trials):
            scores = score_summary(attach_scores(labels, predictions), cutoff, metric, groups)
            results.append({"model": name, **metadata, **scores})
            predictions.write_parquet(args.output / f"{name}_val_predictions.parquet")
            save_report(report_path, report, results)
    report["complete"] = True
    report["pareto"] = pareto_models(results)
    save_report(report_path, report, results)
    print("Model                  holdout NDCG    full val NDCG")
    for result in report["models"]:
        print(f"{result['model']:<22} {result['holdout']['ndcg']:>12.6f} {result['val']['ndcg']:>15.6f}")
    try:
        import notify_tg
        notify_tg.send(notify_tg.format_benchmark(report_path))
    except Exception:
        pass


def diagnose(args):
    train = read_frame(args.data, "train", ["feed_id", "user_id", "item_id", "pos", "target"], args.train_feeds)
    val = read_frame(args.data, "val", ["user_id", "item_id", "history__item_id"], args.val_feeds)
    known_items = train["item_id"].explode().unique()
    known_users = train["user_id"].filter(train["user_id"] > 0).unique()
    val_items = val["item_id"].explode()
    position = train.select("pos", "target").explode("pos", "target").group_by((pl.col("pos") // 10 * 10).alias("pos_start"))
    report = {"known_user_share": val["user_id"].is_in(known_users).mean(),
              "known_item_share": val_items.is_in(known_items.filter(known_items > 0)).mean(),
              "empty_history_share": val["history__item_id"].list.len().eq(0).mean(),
              "train_positive_rate_by_position": position.agg(pl.len().alias("candidates"), pl.col("target").mean().alias("positive_rate")).sort("pos_start").to_dicts()}
    print(json.dumps(report, indent=2))


def blend_submission(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    template = pl.read_parquet(args.data / "sample_submission.parquet", columns=KEYS)
    left = attach_scores(template, pl.read_parquet(args.left))
    right = attach_scores(template, pl.read_parquet(args.right))
    submission = attach_scores(template, blend(left, right, args.weight))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.write_parquet(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    run = commands.add_parser("benchmark")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--models", choices=(*OFFICIAL, *TREE_MODELS, *NEURAL, *CUDA_NEURAL), nargs="+")
    run.add_argument("--baseline-zip", type=Path)
    run.add_argument("--iterations", type=int, default=500)
    run.add_argument("--epochs", type=int, default=5)
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--train-offset", type=int, default=0, help="Skip the first N training feeds")
    run.add_argument("--dual-batch-size", type=int, default=4096)
    run.add_argument("--threads", type=int, default=4)
    run.add_argument("--history-window", type=int, default=32)
    run.add_argument("--cat-device", choices=("cpu", "gpu"), default="cpu")
    run.add_argument("--neural-device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--gpu", help="Use the selected NVIDIA GPU for training and NDCG (index or UUID)")
    run.add_argument("--by-category", action="store_true")
    run.add_argument("--resume", action="store_true", help="Reuse completed models in a stopped benchmark")
    run.add_argument("--reuse-models-from", nargs="+", type=Path, default=[], metavar="RUN")
    run.add_argument("--family-blends", action="store_true")
    run.add_argument("--stack", action="store_true")
    run.add_argument("--optuna-trials", type=int, default=48)
    run.add_argument("--no-rrf", action="store_true")
    run.set_defaults(action=benchmark)
    diagnostic = commands.add_parser("diagnose")
    diagnostic.set_defaults(action=diagnose)
    for command in (run, diagnostic):
        command.add_argument("--train-feeds", type=int, default=50_000)
        command.add_argument("--val-feeds", type=int, default=0)
    ensemble = commands.add_parser("blend")
    ensemble.add_argument("--left", type=Path, required=True)
    ensemble.add_argument("--right", type=Path, required=True)
    ensemble.add_argument("--weight", type=float, required=True)
    ensemble.add_argument("--output", type=Path, required=True)
    ensemble.set_defaults(action=blend_submission)
    for command in (run, diagnostic, ensemble):
        command.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    if min(getattr(args, name, 0) for name in ("train_feeds", "val_feeds", "train_offset")) < 0:
        parser.error("Feed limits and offset must be nonnegative")
    if args.action == benchmark:
        args.models = args.models or (list(GPU_ABLATION) if args.gpu is not None else
                                      [*OFFICIAL, *TREE_MODELS, *NEURAL] if args.baseline_zip else list(TREE_MODELS))
        if args.train_offset and set(args.models).intersection((*OFFICIAL, *CUDA_NEURAL)):
            parser.error("Training offsets are unavailable for the official baselines and dual-history")
        if args.gpu is not None:
            if "lambda" in args.models or "als" in args.models:
                parser.error("This GPU benchmark excludes CPU-only lambda and als")
            args.cat_device, args.neural_device = "gpu", "cuda"
        if set(CUDA_NEURAL) & set(args.models) and args.gpu is None:
            parser.error("dual-history requires --gpu")
        if args.stack and not {"avito-joint", "gru-din"}.issubset(args.models):
            parser.error("--stack requires avito-joint and gru-din")
        if args.family_blends and ("strong-gbdt" not in args.models or not {"avito-joint", "set-transformer", "neural-pairwise"}.intersection(args.models)
                                   or not {"gru-din", "sasrec"}.intersection(args.models)):
            parser.error("--family-blends needs strong-gbdt, a listwise model, and a sequence model")
        if len(set(args.models)) != len(args.models) or set(OFFICIAL) & set(args.models) and not args.baseline_zip:
            parser.error("Models must be unique and official models require --baseline-zip")
        if min(args.batch_size, args.dual_batch_size, args.epochs, args.iterations, args.threads) < 1 or not 1 <= args.history_window <= 256:
            parser.error("Iterations, epochs, batch size and threads must be positive and history window must be 1–256")
    if args.action == blend_submission and not 0 <= args.weight <= 1:
        parser.error("Weight must be between 0 and 1")
    args.action(args)


if __name__ == "__main__":
    main()
