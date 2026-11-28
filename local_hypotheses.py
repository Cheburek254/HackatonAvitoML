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


def blend(frame, gru=pl.col("first"), qsm=pl.col("qsm"), robust=pl.col("rob"), gru3=None, gru_delta=None):
    v4 = pl.col("base") + .30 * (gru - pl.col("gru4")) + .40 * (qsm - pl.col("qsm"))
    if gru3 is not None:
        v4 += .15 * (gru3 - pl.col("gru3"))
    if gru_delta is not None:
        v4 += gru_delta * (gru - qsm)
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


def epochs_trees(args):
    from catboost import CatBoost
    from ensemble import Slates

    # Use the original feature implementation and global statistics from the published QSM.
    sys.path.insert(0, str(args.v4_code))
    try:
        import ranker
    finally:
        sys.path.pop(0)
    from pair_blend import paired_interval

    frame = signals(args)
    epoch_report = json.loads((args.prior/"epochs/report.json").read_text(encoding="utf-8"))
    if epoch_report["best_epoch"] != 5:
        raise ValueError("Expected saved predictions from the fifth GRU epoch")
    frame = scores_into(frame, args.prior/"epochs/val_predictions.parquet", "fifth")
    frame = scores_into(frame, args.prior/"period/val_predictions.parquet", "second")

    checkpoint = args.reference/"valishinqa-din-events-v3-qsm-200k/din-events-v3-qsm/model.cbm"
    model = CatBoost().load_model(str(checkpoint))
    if model.tree_count_ != 800:
        raise ValueError(f"Expected 800 QSM trees, found {model.tree_count_}")
    parts = []
    for batch in ranker.candidate_batches(args.data/"val", features=model.feature_names_):
        pool = ranker.make_pool(batch, model.feature_names_, 4)
        parts.append(batch.select(KEYS).with_columns([
            pl.Series(f"score{trees}", model.predict(pool, ntree_end=trees, task_type="CPU", thread_count=4))
            for trees in (720, 800)
        ]))
    predictions = pl.concat(parts)
    for trees in (720, 800):
        scores = predictions.select(*KEYS, pl.col(f"score{trees}").alias("score"))
        frame = attach_scores(frame, scores).with_columns(zscore("score").alias(f"qsm{trees}")).drop("score")
    score_error = frame.select((pl.col("qsm800") - pl.col("qsm")).abs().max()).item()
    if score_error > .0005:
        raise ValueError(f"Recomputed 800-tree QSM differs from the published scores: {score_error}")

    late_mask = pl.col("timestamp") >= LATE_SPLIT
    masks = (EARLY, CONFIRM, late_mask)
    control_name = "2-periods-e6-qsm800"
    rows, split_scores, bootstrap_scores = [], {}, {}
    for periods in (1, 2):
        for epoch, gru in ((6, pl.col("first")), (5, pl.col("fifth"))):
            if periods == 2:
                gru = .5*(gru + pl.col("second"))
            for trees, qsm in ((800, "qsm"), (720, "qsm720")):
                name = f"{periods}-periods-e{epoch}-qsm{trees}"
                scored = blend(frame, gru, qsm=pl.col(qsm))
                rows.append(metrics(scored, name))
                split_scores[name] = [ndcg(scored.filter(mask))["ndcg"] for mask in masks]
                if name == control_name or (epoch, trees) == (5, 720):
                    slates = Slates(scored.filter(late_mask), "cpu")
                    bootstrap_scores[name] = slates.per_feed_ndcg(slates.pack("score")).numpy().astype(np.float64)
    if abs(split_scores[control_name][2] - .3764309233010262) > 1e-8:
        raise ValueError("Did not reproduce the 0.376431 reference")
    users = (pl.scan_parquet(str(args.data/"val/impressions/part-*.parquet"))
             .filter(late_mask).select("feed_id", "user_id").sort("feed_id").collect())
    if not np.array_equal(users["feed_id"].to_numpy(), slates.frame["feed_id"].unique(maintain_order=True).to_numpy()):
        raise ValueError("User rows do not match the validation feeds")
    intervals = {name: paired_interval(scores - bootstrap_scores[control_name], users["user_id"].to_numpy())
                 for name, scores in bootstrap_scores.items() if name != control_name}
    early_reference = split_scores[control_name][:2]
    early_winner = max(split_scores, key=lambda name: min(score - ref for score, ref in
                       zip(split_scores[name][:2], early_reference, strict=True)))
    report(args, rows, split_scores=split_scores, split_order=["early1", "early2", "late"],
           split1_end=EARLY_SPLIT, holdout_feeds=users.height, control=control_name,
           early_winner=early_winner, qsm800_max_z_error=score_error, holdout_paired_bootstrap=intervals)


def average_epochs(args):
    from ensemble import Slates
    from pair_blend import paired_interval

    frame = signals(args)
    epoch_report = json.loads((args.prior/"epochs/report.json").read_text(encoding="utf-8"))
    if epoch_report["best_epoch"] != 5:
        raise ValueError("Expected saved predictions from the fifth GRU epoch")
    frame = scores_into(frame, args.prior/"epochs/val_predictions.parquet", "fifth")
    frame = scores_into(frame, args.prior/"period/val_predictions.parquet", "second")
    late = pl.col("timestamp") >= LATE_SPLIT
    rows, split_scores, per_feed = [], {}, {}
    for name, first in (("epoch5-control", pl.col("fifth")), ("epoch6-reference", pl.col("first")),
                        ("epoch5-epoch6-average", .5*(pl.col("fifth") + pl.col("first")))):
        scored = blend(frame, .5*(first + pl.col("second")))
        rows.append(metrics(scored, name))
        split_scores[name] = [ndcg(scored.filter(mask))["ndcg"] for mask in (EARLY, CONFIRM, late)]
        slates = Slates(scored.filter(late), "cpu")
        per_feed[name] = slates.per_feed_ndcg(slates.pack("score")).numpy().astype(np.float64)
    if abs(split_scores["epoch5-control"][2] - .37657187723743313) > 1e-8:
        raise ValueError("Did not reproduce the 0.376572 reference")
    users = (pl.scan_parquet(str(args.data/"val/impressions/part-*.parquet"))
             .filter(late).select("feed_id", "user_id").sort("feed_id").collect())
    if not np.array_equal(users["feed_id"].to_numpy(), slates.frame["feed_id"].unique(maintain_order=True).to_numpy()):
        raise ValueError("User rows do not match the validation feeds")
    interval = paired_interval(per_feed["epoch5-epoch6-average"] - per_feed["epoch5-control"],
                               users["user_id"].to_numpy())
    report(args, rows, split_scores=split_scores, split_order=["early1", "early2", "late"],
           split1_end=EARLY_SPLIT, holdout_feeds=users.height, qsm_trees=800,
           gru_v4_weights={"first_epoch5": .25, "first_epoch6": .25, "second_epoch6": .5},
           holdout_paired_bootstrap=interval)


def epoch5_period_signals(args):
    frame = signals(args)
    epoch_report = json.loads((args.prior/"epochs/report.json").read_text(encoding="utf-8"))
    if epoch_report["best_epoch"] != 5:
        raise ValueError("Expected saved predictions from the fifth GRU epoch")
    frame = scores_into(frame, args.prior/"epochs/val_predictions.parquet", "fifth")
    return scores_into(frame, args.prior/"period/val_predictions.parquet", "second")


def history_weights(args):
    frame = epoch5_period_signals(args)
    relative_length = pl.col("history_length").clip(0, 64) / 32 - 1
    rules = {"fixed-control": None, "more-gru-for-long-history": .05*relative_length,
             "more-gru-for-short-history": -.05*relative_length}
    compare_gru_weights(args, frame, rules, max_weight_shift=.05, saturation_history_events=64)


def agreement_weights(args):
    frame = epoch5_period_signals(args)
    frame = scores_into(frame, args.runs/"local-evening-200k/third-period/val_predictions.parquet", "third")
    # Ordering by item ID makes membership deterministic when a model has tied scores.
    top = frame.select(*KEYS, "timestamp", "fifth", "second", "third").sort(KEYS).with_columns(
        (pl.col(name).rank("ordinal", descending=True).over("feed_id") <= 10).alias(name)
        for name in ("fifth", "second", "third")
    )
    pairs = (("fifth", "second"), ("fifth", "third"), ("second", "third"))
    agreement = top.group_by("feed_id").agg(
        (sum((pl.col(left) & pl.col(right)).sum() for left, right in pairs)
         / (3 * pl.len().clip(upper_bound=10))).alias("agreement"),
        pl.col("timestamp").first(),
    )
    center = agreement.filter(EARLY)["agreement"].mean()
    scale = max(center, 1 - center)
    centered = (pl.col("agreement") - center) / scale
    rules = {"fixed-control": None,
             "increase-with-agreement-2.5pp": .025*centered,
             "increase-with-agreement-5pp": .05*centered,
             "decrease-with-agreement-2.5pp": -.025*centered,
             "decrease-with-agreement-5pp": -.05*centered}
    frame = frame.join(agreement.select("feed_id", "agreement"), on="feed_id", validate="m:1")
    compare_gru_weights(args, frame, rules, top_k=10, max_weight_shift=.05, agreement_center=center,
                        agreement_scale=scale, agreement_means=[agreement.filter(mask)["agreement"].mean()
                        for mask in (EARLY, CONFIRM, pl.col("timestamp") >= LATE_SPLIT)])


def compare_gru_weights(args, frame, rules, **settings):
    from ensemble import Slates
    from pair_blend import paired_interval

    gru = .5*(pl.col("fifth") + pl.col("second"))
    folds = [frame.filter(mask) for mask in (EARLY, CONFIRM)]
    early_scores = {name: [ndcg(blend(fold, gru, gru_delta=delta))["ndcg"] for fold in folds]
                    for name, delta in rules.items()}
    reference = early_scores["fixed-control"]
    min_gain = {name: min(value - base for value, base in zip(values, reference, strict=True))
                for name, values in early_scores.items()}
    selected = max(min_gain, key=min_gain.get)
    challenger = max((name for name in rules if name != "fixed-control"), key=min_gain.get)
    atomic_text(args.folder/"selection.json", json.dumps(dict(selected=selected, challenger=challenger,
                                                             early_scores=early_scores, min_gain=min_gain), indent=2))

    late = pl.col("timestamp") >= LATE_SPLIT
    rows, per_feed = [], {}
    for name in dict.fromkeys(("fixed-control", challenger, selected)):
        scored = blend(frame, gru, gru_delta=rules[name])
        rows.append(metrics(scored, name))
        slates = Slates(scored.filter(late), "cpu")
        per_feed[name] = slates.per_feed_ndcg(slates.pack("score")).numpy().astype(np.float64)
    if abs(rows[0]["holdout"]["ndcg"] - .37657187723743313) > 1e-8:
        raise ValueError("Did not reproduce the 0.376572 reference")
    users = (pl.scan_parquet(str(args.data/"val/impressions/part-*.parquet"))
             .filter(late).select("feed_id", "user_id").sort("feed_id").collect())
    if not np.array_equal(users["feed_id"].to_numpy(), slates.frame["feed_id"].unique(maintain_order=True).to_numpy()):
        raise ValueError("User rows do not match the validation feeds")
    interval = paired_interval(per_feed[challenger] - per_feed["fixed-control"], users["user_id"].to_numpy())
    report(args, rows, selected=selected, challenger=challenger, early_scores=early_scores,
           min_early_gain=min_gain, split1_end=EARLY_SPLIT, holdout_feeds=users.height,
           qsm_trees=800, holdout_paired_bootstrap=interval, **settings)


def clip_blend_scores(args):
    from ensemble import Slates
    from pair_blend import paired_interval

    frame = epoch5_period_signals(args)
    for index in (2, 3):
        path = args.runs.parent/f"gru-din-xl-team-exchange/parts/part-{index:03d}/fit/gru-din-xl/val_predictions.parquet"
        frame = scores_into(frame, path, f"xl{index}")
    frame = frame.with_columns((.5*(pl.col("fifth") + pl.col("second"))).alias("gru_control"),
                               (.5*(pl.col("xl2") + pl.col("xl3"))).alias("xl_pair"))
    frame = frame.with_columns((.75*pl.col("gru_control") + .25*zscore("xl_pair")).alias("gru_with_xl"))
    components = ("qsm", "gru_control", "gru_with_xl")
    frame = frame.with_columns(zscore(name).alias(f"{name}_standardized") for name in components)
    frame = frame.with_columns(pl.col(f"{name}_standardized").clip(-3, 3).alias(f"{name}_bounded")
                               for name in components)
    for name in components:
        # Preserve the original weight scale, and leave feeds without extreme scores intact.
        value = pl.col(name)
        flagged = (pl.col(f"{name}_standardized").abs() > 3).any().over("feed_id")
        rescaled = zscore(f"{name}_bounded")*value.std(ddof=0).over("feed_id") + value.mean().over("feed_id")
        frame = frame.with_columns(flagged.alias(f"{name}_changed"),
                                   pl.when(flagged).then(rescaled).otherwise(value).alias(f"{name}_clipped"))

    def score(data, gru_name, rule):
        gru_column = f"{gru_name}_clipped" if rule == "clip-gru" else gru_name
        qsm_column = "qsm_clipped" if rule == "clip-qsm" else "qsm"
        return blend(data, pl.col(gru_column), qsm=pl.col(qsm_column))

    folds = [frame.filter(mask) for mask in (EARLY, CONFIRM)]
    rules = ("fixed", "clip-qsm", "clip-gru")
    trials, selections = {}, {}
    for gru_name in components[1:]:
        early = {rule: [ndcg(score(fold, gru_name, rule))["ndcg"] for fold in folds] for rule in rules}
        gains = {rule: min(new-base for new, base in zip(values, early["fixed"], strict=True))
                 for rule, values in early.items()}
        selections[gru_name] = dict(selected=max(gains, key=gains.get),
                                    challenger=max(rules[1:], key=gains.get))
        trials[gru_name] = early
    atomic_text(args.folder/"selection.json", json.dumps(dict(selections=selections, early=trials), indent=2))

    late_mask = pl.col("timestamp") >= LATE_SPLIT
    late = frame.filter(late_mask)
    users = (pl.scan_parquet(str(args.data/"val/impressions/part-*.parquet"))
             .filter(late_mask).select("feed_id", "user_id").sort("feed_id").collect())
    rows, effects = [], {}
    references = {"gru_control": .37657187723743313, "gru_with_xl": .3768931070503751}
    for gru_name, chosen in selections.items():
        rule = chosen["challenger"]
        fixed = score(frame, gru_name, "fixed")
        challenger = score(frame, gru_name, rule)
        baseline = metrics(fixed, f"{gru_name}-fixed")
        if abs(baseline["holdout"]["ndcg"] - references[gru_name]) > 1e-8:
            raise ValueError(f"Unexpected control for {gru_name}")
        rows.extend([baseline, metrics(challenger, f"{gru_name}-{rule}")])
        fixed_late, challenger_late = fixed.filter(late_mask), challenger.filter(late_mask)
        slates = [Slates(scored, "cpu") for scored in (fixed_late, challenger_late)]
        ids = slates[0].frame["feed_id"].unique(maintain_order=True).to_numpy()
        if not np.array_equal(ids, users["feed_id"].to_numpy()):
            raise ValueError("User rows do not match the validation feeds")
        base, trial = [s.per_feed_ndcg(s.pack("score")).numpy().astype(np.float64) for s in slates]
        flag = "qsm_changed" if rule == "clip-qsm" else f"{gru_name}_changed"
        affected = late.filter(pl.col(flag)).select("feed_id").unique().height
        effects[gru_name] = dict(affected_feeds=affected,
                                 bootstrap=paired_interval(trial-base, users["user_id"].to_numpy()))
        if affected:
            effects[gru_name]["affected_ndcg"] = [ndcg(scored.filter(pl.col(flag)))["ndcg"]
                                                  for scored in (fixed_late, challenger_late)]
    report(args, rows, selections=selections, early_trials=trials, split1_end=EARLY_SPLIT,
           holdout_feeds=users.height, clipping_z_threshold=3, holdout_effects=effects)


def adapted_period_blend(args):
    from ensemble import Slates
    from pair_blend import paired_interval

    frame = signals(args)
    frame = scores_into(frame, args.prior/"period/val_predictions.parquet", "second")
    frame = scores_into(frame, args.runs/"local-evening-200k/adapt-v3/val_predictions.parquet", "adapted_v3")

    def score(data, second_share, adapted_share):
        gru = (1 - second_share)*pl.col("first") + second_share*pl.col("second")
        gru3 = (1 - adapted_share)*pl.col("gru3") + adapted_share*pl.col("adapted_v3")
        return blend(data, gru, gru3=gru3)

    folds = [frame.filter(mask) for mask in (EARLY, CONFIRM)]
    reference = [ndcg(score(fold, .5, 0))["ndcg"] for fold in folds]
    trials = []
    for second_share in (.25, .375, .5, .625, .75):
        for adapted_share in (0., .25, .5, .75, 1.):
            values = [ndcg(score(fold, second_share, adapted_share))["ndcg"] for fold in folds]
            trials.append(dict(second_share=second_share, adapted_share=adapted_share, early_ndcg=values,
                               min_gain=min(value-base for value, base in zip(values, reference, strict=True))))
    joint = max((trial for trial in trials if trial["adapted_share"] > 0), key=lambda trial: trial["min_gain"])
    fixed_v4 = max((trial for trial in trials if trial["second_share"] == .5 and trial["adapted_share"] > 0),
                   key=lambda trial: trial["min_gain"])
    atomic_text(args.folder/"selection.json", json.dumps(dict(joint=joint, fixed_v4=fixed_v4), indent=2))

    rows, late_values = [], {}
    for name, second_share, adapted_share in (
        ("two-periods-control", .5, 0.),
        ("fixed-v4-with-adapted-v3", .5, fixed_v4["adapted_share"]),
        ("joint-early-selected", joint["second_share"], joint["adapted_share"]),
    ):
        scored = score(frame, second_share, adapted_share)
        rows.append(metrics(scored, name))
        late = Slates(scored.filter(pl.col("timestamp") >= LATE_SPLIT), "cpu")
        late_values[name] = late.per_feed_ndcg(late.pack("score")).numpy().astype(np.float64)
        if name == "joint-early-selected":
            write_scores(scored.select(*KEYS, "score"), args.folder/"val_predictions.parquet")
    control = rows[0]["holdout"]["ndcg"]
    if abs(control - .376430923) > 1e-8:
        raise ValueError(f"Unexpected two-period control: {control}")
    users = (pl.scan_parquet(str(args.data/"val/impressions/part-*.parquet"))
             .filter(pl.col("timestamp") >= LATE_SPLIT).select("feed_id", "user_id").sort("feed_id").collect())
    if not np.array_equal(users["feed_id"].to_numpy(), late.frame["feed_id"].unique(maintain_order=True).to_numpy()):
        raise ValueError("User rows do not match the validation feeds")
    intervals = {name: paired_interval(values - late_values["two-periods-control"], users["user_id"].to_numpy())
                 for name, values in late_values.items() if name != "two-periods-control"}
    report(args, rows, selection={"joint": joint, "fixed_v4": fixed_v4}, early_trials=trials,
           early_reference=reference, early_fold_feeds=[fold["feed_id"].n_unique() for fold in folds],
           holdout_feeds=users.height, split1_end=EARLY_SPLIT, holdout_paired_bootstrap=intervals)


def cat_periods(args):
    frame = signals(args)
    evening = args.runs/"local-evening-200k"
    for name, folder in (("second", args.prior/"period"), ("third", evening/"third-period")):
        frame = scores_into(frame, folder/"val_predictions.parquet", name)
    cats = ["qsm"]
    for name in ("base", "recent", "recent-drop"):
        path = args.prior/"qsm"/name/"val_predictions.parquet"
        if path.is_file():
            column = f"qsm-{name}"
            frame = scores_into(frame, path, column)
            cats.append(column)
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
    parser.add_argument("stage", choices=(*STAGES, *EVENING, "cat-periods", "adapted-period-blend", "epochs-trees", "average-epochs",
                                         "history-weights", "agreement-weights", "clip-blend-scores",
                                         "queue", "evening-queue", "cat-queue"))
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
     "period-blend": period_blend, "cat-periods": cat_periods, "adapted-period-blend": adapted_period_blend,
     "groups": grouping, "qsm": qsm, "epochs-trees": epochs_trees, "average-epochs": average_epochs,
     "history-weights": history_weights, "agreement-weights": agreement_weights, "clip-blend-scores": clip_blend_scores,
     "queue": queue, "evening-queue": queue, "cat-queue": queue}[args.stage](args)
