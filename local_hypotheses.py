"""Local validation queue for GRU epochs, time periods, shard grouping and QSM."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter, sleep

import numpy as np
import polars as pl

from evaluation import KEYS, attach_scores, ndcg, zscore
from training_queue import ROOT, atomic_text, keep_awake, note
from v4_followup import (EARLY_SPLIT, LATE_SPLIT, SEED, SIZE, VARIANT, adaptation_data,
                         init_neural, labels_and_blend, scores_into, write_scores)


ROBUST = "din-events-v3-robust"
STAGES = ("epochs", "period", "groups", "qsm")
EVENING = ("third-period", "adapt-v3", "period-blend")
EARLY = pl.col("timestamp") < EARLY_SPLIT
CONFIRM = pl.col("timestamp").is_between(EARLY_SPLIT, LATE_SPLIT, closed="left")


def signals(args):
    frame = labels_and_blend(args.data, args.reference)
    frame = scores_into(frame, args.previous / "val_predictions.parquet", "first")
    return scores_into(frame, args.runs / "v3-robust-full/merged" / ROBUST / "val_predictions.parquet", "rob")


def blend(frame, gru=pl.col("first"), qsm=pl.col("qsm"), robust=pl.col("rob"), gru3=None):
    v4 = pl.col("base") + .30 * (gru - pl.col("gru4")) + .40 * (qsm - pl.col("qsm"))
    if gru3 is not None:
        v4 += .15 * (gru3 - pl.col("gru3"))
    frame = frame.with_columns(v4.cast(pl.Float32).alias("v4"))
    return frame.with_columns((.85 * zscore("v4") + .15 * robust).cast(pl.Float32).alias("score"))


def metrics(frame, name):
    from experiments import score_summary

    return dict(model=name, **score_summary(frame, LATE_SPLIT))


def report(args, rows, **settings):
    result = dict(complete=True, val_feeds=14423, tuning_cutoff_timestamp=LATE_SPLIT,
                  models=sorted(rows, key=lambda row: -row["holdout"]["ndcg"]), **settings)
    atomic_text(args.folder / "report.json", json.dumps(result, indent=2))


def grouping(args):
    from scipy.cluster.hierarchy import cut_tree, linkage
    from scipy.spatial.distance import squareform

    frame = signals(args)
    cols = []
    for i in range(1, 22):
        path = args.runs / f"v3-robust-full/parts/part-{i:03d}/fit" / ROBUST / "val_predictions.parquet"
        cols.append(f"part{i:02d}")
        frame = scores_into(frame, path, cols[-1])
    early = frame.filter(EARLY)
    ranks = early.select([pl.col(name).rank("average").over("feed_id").alias(name) for name in cols])
    ranks = ranks.with_columns(early["feed_id"]).select([zscore(name).alias(name) for name in cols])
    distances = np.clip(1 - np.corrcoef(ranks.to_numpy(), rowvar=False), 0, 2)
    tree = linkage(squareform(distances, checks=False), method="average")
    assignments = cut_tree(tree, n_clusters=5).ravel()
    groups = [[name for name, cluster in zip(cols, assignments, strict=True) if cluster == i] for i in range(5)]
    group_scores = [sum(pl.col(name) for name in group) / len(group) for group in groups]
    frame = frame.with_columns((sum(group_scores) / len(groups)).cast(pl.Float32).alias("grouped"))
    rows = [metrics(blend(frame), "equal-parts"), metrics(blend(frame, robust=zscore("grouped")), "equal-groups"),
            metrics(frame.with_columns(pl.col("grouped").alias("score")), "grouped-robust")]
    report(args, rows, groups=groups, cluster_fit_feeds=early["feed_id"].n_unique(), train_feeds=3151208)


def gru_run(args):
    neural, torch = init_neural(args)
    from torch.utils.data import Subset

    frame = signals(args)
    v3 = args.stage == "adapt-v3"
    variant, window = ("gru-din-v3", 32) if v3 else (VARIANT, 64)
    validation = neural.Feeds(args.data / "val", 0, True, window, True, rich_features=True,
                              v3_features=True, v4_features=not v3)
    model = neural.Ranker(variant).to("cuda")
    def load_state(path):
        return torch.load(path, map_location="cpu", weights_only=True)

    def predict(path, mask=None):
        if mask is None:
            subset = validation
        else:
            ids = validation.feeds.with_row_index("_index").filter(mask)["_index"].to_list()
            subset = Subset(validation, ids)
        model.load_state_dict(load_state(path)["state_dict"])
        return neural.scores(model, neural.loader(subset, 16, prefetch=True), "cuda")

    extending = args.stage == "epochs"
    source_selection = args.previous / "train-feeds.parquet"
    if args.stage == "third-period":
        source_selection = args.prior / "period/train-feeds.parquet"
    source = pl.read_parquet(source_selection)
    initial = (args.previous / "epoch-6.pt" if extending else
               args.reference / "valishinqa-gru-din-v4-300k/gru-din-v4/model.pt")
    if v3:
        initial = args.reference / "retrain/gru-din-v3/model.pt"
    early = frame.filter(EARLY)
    curve = {}
    confirmation = {}
    if extending:
        for epoch in range(1, 7):
            scored = scores_into(early, args.previous / f"epoch-{epoch}-early.parquet", "candidate")
            curve[epoch] = ndcg(blend(scored, pl.col("candidate")))["ndcg"]
        confirm = frame.filter(CONFIRM)
        fifth = predict(args.previous / "epoch-5.pt", CONFIRM)
        fifth = attach_scores(confirm, fifth).with_columns(zscore("score").alias("candidate"))
        confirmation = {"5": ndcg(blend(fifth, pl.col("candidate")))["ndcg"], "6": ndcg(blend(confirm))["ndcg"]}
        atomic_text(args.folder / "epoch-5-vs-6.json", json.dumps(confirmation))
        note(f"Epoch 5 vs 6 on confirmation feeds: {confirmation}")

    same_period = extending or v3
    fit, selection, stats_feeds = adaptation_data(args, neural, source_selection if same_period else None,
                                                  None if same_period else source["timestamp"].min(),
                                                  window=window, v4_features=not v3)
    if not same_period and selection["feed_id"].is_in(source["feed_id"]).any():
        raise ValueError("The two periods overlap")
    if args.stage == "third-period":
        first = pl.read_parquet(args.previous / "train-feeds.parquet")
        if selection["timestamp"].max() >= first["timestamp"].min() or selection["feed_id"].is_in(first["feed_id"]).any():
            raise ValueError("Third period overlaps first")
    selection.write_parquet(args.folder / "train-feeds.parquet")
    batches = neural.loader(fit, 16, shuffle=True, prefetch=True)
    state = load_state(initial)
    if (state["variant"], state["history_window"]) != (variant, window):
        raise ValueError(f"Unexpected checkpoint for {variant}")
    model.load_state_dict(state["state_dict"])
    lr = 2e-5 if extending else 1e-4
    start, stop = (7, 10) if extending else (1, 6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if extending:
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:
            group.update(lr=lr, initial_lr=lr)
        torch.set_rng_state(state["rng_cpu"])
        torch.cuda.set_rng_state(state["rng_cuda"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(stop-start+1)*len(batches))
    for epoch in range(start, stop+1):
        path = args.folder / f"epoch-{epoch}.pt"
        if path.exists():
            state = load_state(path)
            model.load_state_dict(state["state_dict"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["schedule"])
            torch.set_rng_state(state["rng_cpu"])
            torch.cuda.set_rng_state(state["rng_cuda"])
        else:
            fit.feeds.rng = np.random.default_rng(SEED + epoch)
            neural.train_epoch(model, optimizer, batches, "cuda", variant, epoch, scheduler)
            tmp = path.with_suffix(".tmp")
            torch.save(dict(state_dict=model.state_dict(), variant=variant, history_window=window,
                            optimizer=optimizer.state_dict(), schedule=scheduler.state_dict(),
                            rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state()), tmp)
            tmp.replace(path)
        if extending:
            pred = predict(path, EARLY)
            scored = attach_scores(early, pred).with_columns(zscore("score").alias("candidate"))
            curve[epoch] = ndcg(blend(scored, pl.col("candidate")))["ndcg"]
            print(f"{VARIANT} epoch={epoch} tuning_ndcg={curve[epoch]:.8f}", flush=True)
    best = max(curve, key=curve.get) if extending else stop
    folder = args.previous if extending and best <= 6 else args.folder
    path = args.folder / "val_predictions.parquet"
    write_scores(predict(folder / f"epoch-{best}.pt"), path)
    frame = scores_into(frame, path, "candidate")
    scored = blend(frame, gru3=pl.col("candidate")) if v3 else blend(frame, pl.col("candidate"))
    rows = [metrics(blend(frame), "first-period-epoch6"), metrics(scored, f"{args.stage}-epoch{best}")]
    if args.stage == "period":
        rows.append(metrics(blend(frame, .5*(pl.col("candidate")+pl.col("first"))), "two-periods-50-50"))
    if args.stage in EVENING:
        tmp = args.folder / "model.tmp"
        torch.save(dict(state_dict=model.state_dict(), variant=variant, history_window=window), tmp)
        tmp.replace(args.folder / "model.pt")
    report(args, rows, train_feeds=SIZE, epochs=stop, best_epoch=best, first_fold_curve=curve,
           confirmation_ndcg=ndcg(scored.filter(CONFIRM))["ndcg"], early_pair=confirmation,
           stats_feeds=stats_feeds, learning_rate=lr, timestamps=[selection["timestamp"].min(), selection["timestamp"].max()])


def qsm(args):
    frame = signals(args)
    rows = [metrics(blend(frame), "published-qsm-control")]
    for name, variant, dropout in (("base", "din-events-v3-qsm", 0), ("recent", ROBUST, 0),
                                   ("recent-drop", ROBUST, .35)):
        dest = args.folder / name
        if not (dest / "metrics.json").exists():
            subprocess.run([sys.executable, "-u", "ranker.py", "train", "--data", str(args.data),
                            "--output", str(dest), "--gpu", args.gpu, "--threads", "8", "--train-feeds", str(SIZE),
                            "--val-feeds", "0", "--iterations", "800", "--variant", variant, "--cold-drop",
                            str(dropout), "--tune-before", str(EARLY_SPLIT)], cwd=args.runs/"v3-robust-worktree", check=True)
        scored = scores_into(frame, dest/"val_predictions.parquet", "candidate")
        rows.append(metrics(blend(scored, qsm=pl.col("candidate")), name))
    report(args, rows, train_feeds=SIZE, iterations=800, depth=7, cold_drop=.35)


def period_blend(args):
    frame = signals(args)
    for name, folder in (("second", args.prior/"period"), ("third", args.output/"third-period"),
                         ("new3", args.output/"adapt-v3")):
        frame = scores_into(frame, folder/"val_predictions.parquet", name)
    early = frame.filter(EARLY)
    confirm = frame.filter(CONFIRM)
    # Keep the original as the first candidate so an exact tie preserves it.
    control = (1., 0., 0., 0.)
    def score(data, weights):
        first, second, third, alpha = weights
        gru = first*pl.col("first") + second*pl.col("second") + third*pl.col("third")
        gru3 = (1-alpha)*pl.col("gru3") + alpha*pl.col("new3")
        return blend(data, gru, gru3=gru3)

    reference = [ndcg(score(fold, control))["ndcg"] for fold in (early, confirm)]
    trials = []
    for periods in ((1., 0., 0.), (.5, .5, 0.), (.5, .3, .2), (1/3, 1/3, 1/3)):
        for alpha in (0., .5, 1.):
            weights = (*periods, alpha)
            values = [ndcg(score(fold, weights))["ndcg"] for fold in (early, confirm)]
            delta = min(x-y for x, y in zip(values, reference, strict=True))
            trials.append(dict(weights=weights, early_ndcg=values, min_gain=delta))
    winner = max(trials, key=lambda trial: trial["min_gain"])
    atomic_text(args.folder/"selection.json", json.dumps(winner, indent=2))
    selected = score(frame, winner["weights"])
    write_scores(selected.select(*KEYS, "score"), args.folder/"val_predictions.parquet")
    rows = [metrics(score(frame, control), "one-period-control"),
            metrics(score(frame, (.5, .5, 0., 0.)), "two-periods-control"),
            metrics(selected, "early-selected")]
    report(args, rows, selection=winner, early_trials=trials, split1_end=EARLY_SPLIT,
           weight_order=["first-v4", "second-v4", "third-v4", "adapted-share-of-v3"],
           early_fold_feeds=[early["feed_id"].n_unique(), confirm["feed_id"].n_unique()])


def cat_periods(args):
    frame = signals(args)
    evening = args.runs/"local-evening-200k"
    for name, folder in (("second", args.prior/"period"), ("third", evening/"third-period")):
        frame = scores_into(frame, folder/"val_predictions.parquet", name)
    cats = ["qsm"]
    for name in ("base", "recent", "recent-drop"):
        path = args.prior/"qsm"/name/"val_predictions.parquet"
        if path.is_file():
            frame = scores_into(frame, path, name)
            cats.append(name)
    for name, weights in (("two", (.5, .5, 0.)), ("three", (1/3, 1/3, 1/3))):
        mixture = sum(w*pl.col(c) for w,c in zip(weights, ("first","second","third"), strict=True))
        frame = frame.with_columns(mixture.alias(name)).with_columns(zscore(name).alias(name))

    def score(data, family, period, cat, weight):
        pair = weight*pl.col(cat) + (1-weight)*pl.col(period)
        if family == "full":
            v4 = .70*pair + .15*pl.col("gru3") + .15*pl.col("lgbm")
            data = data.with_columns(v4.cast(pl.Float32).alias("v4"))
            pair = .85*zscore("v4") + .15*pl.col("rob")
        return data.with_columns(pair.cast(pl.Float32).alias("score"))

    folds = [frame.filter(EARLY), frame.filter(CONFIRM)]
    selected, curves = {}, {}
    for family in ("pair", "full"):
        # The two-family reference is a single-period GRU + published QSM.
        # The full reference is exactly the existing .40/.30/.15/.15 V4 + robust.
        base = [ndcg(score(f, family, "first", "qsm", 4/7))["ndcg"] for f in folds]
        trials = []
        for period in ("two", "three"):
            for cat in cats:
                for weight in (.35, .50, 4/7, .70):
                    values = [ndcg(score(f, family, period, cat, weight))["ndcg"] for f in folds]
                    trials.append(dict(period=period, cat=cat, weight=weight, early_ndcg=values,
                                       min_gain=min(x-y for x,y in zip(values,base,strict=True))))
        curves[family] = trials
        selected[family] = max(trials, key=lambda row: row["min_gain"])
    atomic_text(args.folder/"selection.json", json.dumps(selected, indent=2))
    rows = [metrics(blend(frame), "full-control")]
    for family, winner in selected.items():
        if family == "pair":
            rows.append(metrics(score(frame, family, "first","qsm",4/7), "pair-control"))
        chosen = score(frame, family, winner["period"], winner["cat"], winner["weight"])
        write_scores(chosen.select(*KEYS,"score"), args.folder/f"{family}-val.parquet")
        rows.append(metrics(chosen, family+"-period-catboost"))
    report(args, rows, selection=selected, early_trials=curves, available_catboost=cats,
           selected_by="largest minimum gain over the two early folds within each family",
           period_weights={"two":[.5,.5,0], "three":[1/3]*3}, cat_weight_grid=[.35,.50,4/7,.70])


def queue(args):
    from team_reports import sync

    cat_queue = args.stage == "cat-queue"
    evening = args.stage == "evening-queue"
    watch = args.output/"watch.json"
    if evening:
        prior_watch = json.loads((args.prior/"watch.json").read_text(encoding="utf-8"))
        if Path(prior_watch["status"]).resolve() != (args.prior/"qsm/run.status").resolve():
            raise ValueError("Expected the preceding queue to be at its last QSM stage")
        atomic_text(watch, json.dumps(prior_watch))
    if cat_queue:
        previous = args.runs/"local-evening-200k"
        note("Waiting for the evening period-blend to finish")
        while not (previous/"period-blend/run.status").exists():
            keep_awake()
            sleep(15)
    subprocess.Popen([sys.executable, str(ROOT/"dashboard.py"), "--watch", str(watch)], cwd=ROOT,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if evening:
        note("Waiting for preceding QSM process to exit")
        while not (args.prior/"qsm/run.status").exists():
            keep_awake()
            sleep(15)
        note("Preceding QSM exited; starting evening experiments")
    failures = []
    stages = ("cat-periods",) if cat_queue else EVENING if evening else ("groups", "epochs", "period", "qsm")
    for name in stages:
        folder = args.output/name
        folder.mkdir(exist_ok=True)
        result, log, status = folder/"report.json", folder/"run.log", folder/"run.status"
        if not result.exists():
            status.unlink(missing_ok=True)
            atomic_text(watch, json.dumps(dict(report=str(result), log=str(log), status=str(status), data=str(args.data),
                       models=[VARIANT if name in ("period","epochs","third-period") else
                               "gru-din-v3" if name=="adapt-v3" else name], label=name, iterations=800,
                       epochs=10 if name=="epochs" else 6)))
            note(f"Starting {name}")
            started = perf_counter()
            with log.open("ab") as stream, subprocess.Popen(
                [sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve()), name, "--data", str(args.data),
                 "--runs", str(args.runs), "--output", str(args.output), "--gpu", args.gpu],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT) as child:
                while child.poll() is None:
                    keep_awake()
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        pass
                code = child.returncode or int(not result.exists())
            atomic_text(status, str(code))
            note(f"{name} finished in {perf_counter()-started:.0f}s, exit={code}")
            if code:
                failures.append(name)
                continue
        if cat_queue:
            instructions = folder/"REPRODUCE.md"
            atomic_text(instructions, "After local-evening-200k and local-hypotheses-200k, run\n\n"
                        "`python local_hypotheses.py cat-periods --data competition --runs runs/avito-real "
                        "--output runs/avito-real/local-cat-periods --gpu 0`\n\n"
                        "Selected early-fold recipes (weight is the CatBoost share)\n\n```json\n"
                        + (folder/"selection.json").read_text(encoding="utf-8") + "\n```\n")
            sync("avito-local-evening", name, result, revision=args.revision, instructions=instructions)
        elif evening:
            instructions = folder/"REPRODUCE.md"
            recipe = ("Use the Avito validation and pretrained artifacts from v4-next-200k/adapt, "
                      "valishinqa-v4-repro, valishinqa-v4-worktree, v3-robust-full and local-hypotheses-200k/period. "
                      "Run stages third-period, adapt-v3, period-blend in this order.\n\n"
                      f"`python local_hypotheses.py {name} --data competition --runs runs/avito-real "
                      "--output runs/avito-real/local-evening-200k --gpu 0`\n")
            if name == "period-blend":
                recipe += "\nFixed selection and fold results\n\n```json\n" + result.read_text(encoding="utf-8") + "\n```\n"
            atomic_text(instructions, recipe)
            sync("avito-local-evening", name, result, revision=args.revision, instructions=instructions,
                 checkpoints=[] if name=="period-blend" else ["model.pt"])
        else:
            sync("avito-local-hypotheses", name, result, revision=args.revision)
    if failures:
        raise RuntimeError(f"Failed stages: {failures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(*STAGES, *EVENING, "cat-periods", "queue", "evening-queue", "cat-queue"))
    for key in ("data", "runs", "output"):
        parser.add_argument("--"+key, required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--revision", default="HEAD")
    args = parser.parse_args()
    args.previous = args.runs / "v4-next-200k/adapt"
    args.prior = args.runs / "local-hypotheses-200k"
    args.reference = args.runs / "valishinqa-v4-repro"
    args.v4_code = args.runs / "valishinqa-v4-worktree"
    args.folder = args.output / args.stage
    destination = args.output if args.stage.endswith("queue") else args.folder
    destination.mkdir(parents=True, exist_ok=True)
    {"epochs": gru_run, "period": gru_run, "third-period": gru_run, "adapt-v3": gru_run,
     "period-blend": period_blend, "cat-periods": cat_periods, "groups": grouping, "qsm": qsm,
     "queue": queue, "evening-queue": queue, "cat-queue": queue}[args.stage](args)
