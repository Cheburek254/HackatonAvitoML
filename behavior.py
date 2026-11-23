"""Popularity and collaborative features fitted on earlier training feeds."""

from functools import lru_cache

import numpy as np
import polars as pl


FEATURES = {
    "popularity": ["item_positive_rate", "item_observations", "category_positive_rate", "category_idf_affinity"],
    "itemcf": ["itemcf_max", "itemcf_mean", "itemcf_supported"],
    "als": ["als_score", "als_known_pair"],
    "covis": ["covis_max", "covis_recency_weighted", "covis_supported"],
    "pixie": ["pixie_score", "pixie_neighbors"],
    "popularity-trend": ["item_rate_recent", "item_rate_earlier", "item_rate_change", "item_recent_count"],
    "category-model": ["category_logit", "category_logit_percentile"],
    "category-covis": ["category_covis_max", "category_covis_weighted", "category_covis_support"],
    "pixie-decays": ["pixie_fast", "pixie_slow", "pixie_fast_percentile", "pixie_slow_percentile"],
    "covis-pixie": ["joint_" + name for name in (
        "covis_max", "covis_recency_weighted", "covis_supported", "pixie_score", "pixie_neighbors"
    )],
}
HISTORY_WINDOW = 8
PIXIE_FRONTIER = 256
CATEGORY_ONE_HOT = np.eye(256, dtype=np.float32)


def reference_kind(features):
    return next((kind for kind, names in FEATURES.items() if names[0] in features), None)


def candidates(feeds):
    return feeds.select("user_id", "item_id", "category_id", "target").explode("item_id", "category_id", "target")


def interaction_pairs(feeds):
    history = feeds.select("user_id", pl.col("history__item_id").list.head(HISTORY_WINDOW).alias("item_id")).explode("item_id")
    positives = candidates(feeds).filter(pl.col("target") == 1).select("user_id", "item_id")
    return pl.concat([history, positives]).filter((pl.col("user_id") > 0) & (pl.col("item_id") > 0)).unique()


def transition_counts(feeds, by_category=False):
    history = pl.col("history__item_id").list.head(HISTORY_WINDOW)
    categories = pl.col("history__category_id").list.head(HISTORY_WINDOW)
    frames = []
    for lag in (1, 2, 3):
        count = (history.list.len().cast(pl.Int32) - lag).clip(lower_bound=0)
        columns = [history.list.slice(lag, count).alias("older"), history.list.slice(0, count).alias("newer")]
        if by_category:
            columns.extend([categories.list.slice(lag, count).alias("category"),
                            categories.list.slice(0, count).alias("next_category")])
        names = [column.meta.output_name() for column in columns]
        frames.append(feeds.select("user_id", *columns).explode(names))
    edges = pl.concat(frames).filter((pl.col("user_id") > 0) & (pl.col("older") > 0) &
                                     (pl.col("newer") > 0) & (pl.col("older") != pl.col("newer")))
    keys = ["older", "newer"]
    if by_category:
        edges = edges.filter((pl.col("category") != 0) & (pl.col("category") == pl.col("next_category")))
        keys.insert(0, "category")
    return edges.unique().group_by(keys).len().select(*keys, "len").to_numpy().astype(np.int64)


def category_inputs(categories, history):
    known = np.asarray(history[:64], dtype=np.int32)
    recent, earlier = known[:8], known[8:]
    recent, earlier = recent[recent != 0], earlier[earlier != 0]
    recent_share = (categories[:, None] == recent).sum(axis=1) / max(len(recent), 1)
    earlier_share = (categories[:, None] == earlier).sum(axis=1) / max(len(earlier), 1)
    membership = (categories[:, None] == known[known != 0]).sum(axis=1)
    return np.column_stack([np.ones(len(categories)), recent_share, earlier_share,
                            np.log1p(membership), categories != 0,
                            CATEGORY_ONE_HOT[categories.astype(np.uint8)]]).astype(np.float32)


def percentiles(values):
    return (pl.Series(values).rank("average") / len(values)).to_numpy()


def fit_category_model(feeds, device):
    inputs, targets = [], []
    for categories, history, items in feeds.select("category_id", "history__category_id", "target").iter_rows():
        unique, inverse = np.unique(categories, return_inverse=True)
        positive = np.zeros(len(unique), dtype=np.float32)
        np.maximum.at(positive, inverse, items)
        inputs.append(category_inputs(unique, history))
        targets.append(positive)
    design = np.concatenate(inputs)
    target = np.concatenate(targets)
    if device == "gpu":
        from accelerator import category_coefficients

        return category_coefficients(design, target)
    weights = np.zeros(design.shape[1], dtype=np.float32)
    for _ in range(80):
        probability = 1 / (1 + np.exp(-(design @ weights)))
        weights -= .6 * (design.T @ (probability - target) / len(target) + .002 * weights)
    return weights


def fit(kind, feeds, threads, device="cpu"):
    if kind in {"covis", "category-covis"}:
        return Reference(kind, {"edges": transition_counts(feeds, by_category=kind == "category-covis")})
    if kind == "covis-pixie":
        return Reference(kind, {"edges": transition_counts(feeds), "pairs": interaction_pairs(feeds).to_numpy()})
    if kind == "category-model":
        return Reference(kind, {"coefficients": fit_category_model(feeds, device)})
    if kind == "popularity-trend":
        cutoff = feeds["timestamp"].max()
        day = 86400
        arrays = {}
        for name, start, end in (("recent", cutoff - day, cutoff + 1),
                                 ("earlier", cutoff - 8 * day, cutoff - day)):
            rows = candidates(feeds.filter(pl.col("timestamp").is_between(start, end, closed="left")))
            rows = rows.filter(pl.col("item_id") > 0)
            arrays[name] = rows.group_by("item_id").agg(
                pl.len().alias("count"), pl.col("target").sum().alias("hits")
            ).select("item_id", "count", "hits").to_numpy().astype(np.int64)
        arrays["prior"] = np.array([feeds["target"].explode().mean() or 0])
        return Reference(kind, arrays)
    if kind == "popularity":
        rows = candidates(feeds).filter(pl.col("item_id") > 0)
        arrays = {"prior": np.array([rows["target"].mean() or 0])}
        for key in ("item_id", "category_id"):
            counts = rows.group_by(key).agg(pl.len().alias("n"), pl.col("target").sum().alias("positive"))
            arrays[key] = counts.select(key, "n", "positive").to_numpy().astype(np.int64)
        return Reference(kind, arrays)

    pairs = interaction_pairs(feeds)
    arrays = {"pairs": pairs.select("user_id", "item_id").to_numpy()}
    if kind == "als" and pairs.height:
        from implicit.cpu.als import AlternatingLeastSquares
        from scipy.sparse import csr_matrix
        from threadpoolctl import threadpool_limits

        users, user_index = np.unique(arrays["pairs"][:, 0], return_inverse=True)
        items, item_index = np.unique(arrays["pairs"][:, 1], return_inverse=True)
        interactions = csr_matrix((np.full(pairs.height, 10, dtype=np.float32), (user_index, item_index)),
                                  shape=(len(users), len(items)))
        with threadpool_limits(limits=1, user_api="blas"):
            model = AlternatingLeastSquares(factors=32, iterations=8, regularization=0.1,
                                            num_threads=threads, random_state=42)
            model.fit(interactions, show_progress=False)
        arrays = {"users": users, "items": items, "user_factors": model.user_factors, "item_factors": model.item_factors}
    return Reference(kind, arrays)


class PixieGraph:
    """Score existing candidates by weighted paths through a bounded user frontier."""

    def __init__(self, pairs):
        self.items, item_indices = np.unique(pairs[:, 1], return_inverse=True)
        self.users, user_indices = np.unique(pairs[:, 0], return_inverse=True)
        order = np.lexsort((user_indices, item_indices))
        self.postings = user_indices[order]
        self.item_degree = np.bincount(item_indices)
        self.user_degree = np.bincount(user_indices)
        self.offsets = np.r_[0, self.item_degree.cumsum()]

    def item_users(self, index):
        return self.postings[self.offsets[index]:self.offsets[index + 1]]

    def frontier(self, seeds, user_id, half_life):
        current = np.searchsorted(self.users, user_id)
        current_is_known = current < len(self.users) and self.users[current] == user_id
        user_blocks = []
        weight_blocks = []
        for item, position in seeds.items():
            neighbors = self.item_users(item)
            if current_is_known:
                neighbors = neighbors[neighbors != current]
            if neighbors.size:
                user_blocks.append(neighbors)
                weight_blocks.append(np.full(neighbors.size, 2 ** (-position / half_life) / neighbors.size))
        if not user_blocks:
            return np.empty(0, dtype=np.intp), np.empty(0)

        users, inverse = np.unique(np.concatenate(user_blocks), return_inverse=True)
        weights = np.bincount(inverse, weights=np.concatenate(weight_blocks)) / self.user_degree[users]
        if users.size > PIXIE_FRONTIER:
            # Prefer stronger paths and break exact ties by user ID.
            best = np.lexsort((users, -weights))[:PIXIE_FRONTIER]
            best.sort()
            users, weights = users[best], weights[best]
        return users, weights

    def score(self, feed, half_life=8):
        result = np.zeros((len(FEATURES["pixie"]), len(feed["item_id"])), dtype=np.float32)
        if not self.items.size:
            return result
        history = np.asarray(feed["history__item_id"][:HISTORY_WINDOW])
        seed_indices = np.searchsorted(self.items, history)
        seeds = {}
        for position, index in enumerate(seed_indices):
            if index < len(self.items) and self.items[index] == history[position]:
                seeds.setdefault(index, position)
        users, weights = self.frontier(seeds, feed["user_id"], half_life)
        if not users.size:
            return result

        candidates = np.asarray(feed["item_id"])
        indices = np.searchsorted(self.items, candidates)
        for position, index in enumerate(indices):
            if index == len(self.items) or self.items[index] != candidates[position] or index in seeds:
                continue
            members = self.item_users(index)
            matches = np.minimum(np.searchsorted(members, users), len(members) - 1)
            supported = members[matches] == users
            result[0, position] = weights[supported].sum() / np.sqrt(self.item_degree[index])
            result[1, position] = supported.sum()
        return result


class Reference:
    def __init__(self, kind, arrays=None):
        self.kind = kind
        self.arrays = arrays if arrays is not None else {}
        if kind in {"pixie", "pixie-decays", "covis-pixie"} and self.arrays:
            self.graph = PixieGraph(self.arrays["pairs"])
        if kind == "popularity-trend" and self.arrays:
            self.rates = [{int(item): (count, hits) for item, count, hits in self.arrays[name]}
                          for name in ("recent", "earlier")]
        if kind == "popularity" and self.arrays:
            self.items = {int(key): (count, positive) for key, count, positive in self.arrays["item_id"]}
            self.categories = {int(key): (count, positive) for key, count, positive in self.arrays["category_id"]}
            self.size = float(self.arrays["category_id"][:, 1].sum())
        if kind == "itemcf":
            item_users = {}
            for user, item in self.arrays.get("pairs", []):
                item_users.setdefault(int(item), set()).add(int(user))

            @lru_cache(maxsize=100_000)
            def similarity(left, right):
                users_a, users_b = item_users[left], item_users[right]
                overlap = len(users_a & users_b)
                return overlap / np.sqrt(len(users_a) * len(users_b)) * overlap / (overlap + 5)

            self.item_users = item_users
            self.similarity = similarity
        if kind in {"covis", "category-covis", "covis-pixie"}:
            edges = self.arrays.get("edges", [])
            totals = {}
            for *keys, _, count in edges:
                key = tuple(keys)
                totals[key] = totals.get(key, 0) + int(count)
            self.transitions = {tuple(keys): count / (totals[tuple(keys[:-1])] + 5)
                                for *keys, count in edges}

    def score(self, feed):
        if not self.arrays:
            return np.zeros((len(FEATURES[self.kind]), len(feed["item_id"])), dtype=np.float32)
        if self.kind == "pixie":
            return self.graph.score(feed)
        if self.kind == "pixie-decays":
            fast, slow = (self.graph.score(feed, half_life)[0] for half_life in (2, 16))
            return np.stack([fast, slow, percentiles(fast), percentiles(slow)])
        if self.kind == "covis-pixie":
            return np.concatenate([self.covis(feed), self.graph.score(feed)])
        if self.kind == "category-model":
            categories, inverse = np.unique(feed["category_id"], return_inverse=True)
            logits = category_inputs(categories, feed["history__category_id"]) @ self.arrays["coefficients"]
            return np.stack([logits[inverse], percentiles(logits)[inverse]])
        if self.kind == "popularity-trend":
            prior = float(self.arrays["prior"][0])
            rows = []
            for item in feed["item_id"]:
                recent, earlier = (table.get(item, (0, 0)) for table in self.rates)
                short, long = ((hits + 20 * prior) / (count + 20) for count, hits in (recent, earlier))
                rows.append((short, long, short - long, np.log1p(recent[0])))
            return np.asarray(rows, dtype=np.float32).T
        scorer = {"popularity": self.popularity, "itemcf": self.itemcf, "als": self.als,
                  "covis": self.covis, "category-covis": self.covis}[self.kind]
        return scorer(feed)

    def popularity(self, feed):
        matches = {}
        for position, category in enumerate(feed["history__category_id"][:32]):
            if category:
                matches[category] = matches.get(category, 0) + 2 ** (-position / 8)
        rows = []
        for item, category in zip(feed["item_id"], feed["category_id"], strict=True):
            total, hits = self.items.get(item, (0, 0))
            category_total, category_hits = self.categories.get(category, (0, 0))
            category_rate = (category_hits + 20 * self.arrays["prior"][0]) / (category_total + 20)
            item_rate = (hits + 20 * category_rate) / (total + 20)
            idf = np.log((self.size + 1) / (category_total + 1)) if category_total else 0
            rows.append((item_rate, np.log1p(total), category_rate, matches.get(category, 0) * idf))
        return np.asarray(rows, dtype=np.float32).T

    def itemcf(self, feed):
        recent = {item for item in feed["history__item_id"][:HISTORY_WINDOW] if item in self.item_users}
        rows = []
        for item in feed["item_id"]:
            scores = [self.similarity(*sorted((item, other))) for other in recent if other != item] if item in self.item_users else []
            rows.append((max(scores), sum(scores) / len(scores), sum(score > 0 for score in scores)) if scores else (0, 0, 0))
        return np.asarray(rows, dtype=np.float32).T

    def covis(self, feed):
        recent = {item: position for position, item in reversed(list(enumerate(feed["history__item_id"][:HISTORY_WINDOW]))) if item > 0}
        rows = []
        for item, category in zip(feed["item_id"], feed["category_id"], strict=True):
            prefix = (category,) if self.kind == "category-covis" else ()
            scores = [(score, 2 ** (-position / 8)) for previous, position in recent.items()
                      if (not prefix or feed["history__category_id"][position] == category)
                      and (score := self.transitions.get((*prefix, previous, item))) is not None]
            rows.append((max(score for score, _ in scores), max(score * weight for score, weight in scores), len(scores))
                        if scores else (0, 0, 0))
        return np.asarray(rows, dtype=np.float32).T

    def als(self, feed):
        items, users = self.arrays.get("items", np.empty(0)), self.arrays.get("users", np.empty(0))
        scores = np.zeros((2, len(feed["item_id"])), dtype=np.float32)
        user = int(np.searchsorted(users, feed["user_id"]))
        if user == len(users) or users[user] != feed["user_id"]:
            return scores
        item_ids = np.asarray(feed["item_id"])
        indices = np.minimum(np.searchsorted(items, item_ids), len(items) - 1)
        known = items[indices] == item_ids
        scores[0, known] = np.einsum("ij,j->i", self.arrays["item_factors"][indices[known]], self.arrays["user_factors"][user])
        scores[1, known] = 1
        return scores

    def save(self, path):
        np.savez(path, **self.arrays)

    @classmethod
    def load(cls, kind, path):
        with np.load(path, allow_pickle=False) as arrays:
            return cls(kind, dict(arrays.items()))
