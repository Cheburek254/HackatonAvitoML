"""Spherical clusters fitted on train embeddings and reused on val and test."""

import numpy as np


def normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-6)


def fit(embeddings, indices, device):
    generator = np.random.default_rng(42)
    if len(indices) > 20_000:
        indices = generator.choice(indices, 20_000, replace=False)
    vectors = np.asarray(embeddings[indices], dtype=np.float32)
    vectors = vectors[np.any(vectors != 0, axis=1)]
    count = min(len(vectors), 32)
    if not count:
        return np.empty((0, embeddings.shape[1]), dtype=np.float32)
    choice = generator.choice(len(vectors), count, replace=False)
    if device == "gpu":
        from accelerator import cuda_tensor

        tensor = cuda_tensor()
        x = tensor(vectors, device="CUDA")
        x = (x / x.square().sum(axis=1, keepdim=True).sqrt().maximum(1e-6)).realize()
        centers = x[tensor(choice.astype(np.int32), device="CUDA")].realize()
        for _ in range(12):
            members = ((x @ centers.T).argmax(axis=1).unsqueeze(1)
                       == tensor.arange(count, device="CUDA")).cast("float")
            totals = members.T @ x
            norm = totals.square().sum(axis=1, keepdim=True).sqrt()
            centers = (norm > 0).where(totals / norm.maximum(1e-6), centers).realize()
        return centers.numpy()
    x = normalize(vectors)
    centers = x[choice]
    for _ in range(12):
        members = np.argmax(x @ centers.T, axis=1)[:, None] == np.arange(count)
        totals = members.astype(np.float32).T @ x
        populated = members.any(axis=0)
        centers[populated] = normalize(totals[populated])
    return centers


def assign(embeddings, centers, device, indices):
    ids = np.full(len(indices), -1, dtype=np.int16)
    similarity = np.zeros(len(indices), dtype=np.float32)
    if not len(centers):
        return indices, ids, similarity
    if device == "gpu":
        from accelerator import cuda_tensor

        tensor = cuda_tensor()
        prototypes = tensor(centers.T.copy(), device="CUDA").realize()
    for start in range(0, len(ids), 8192):
        values = np.asarray(embeddings[indices[start:start + 8192]], dtype=np.float32)
        if device == "gpu":
            x = tensor(values, device="CUDA")
            x = x / x.square().sum(axis=1, keepdim=True).sqrt().maximum(1e-6)
            dots = (x @ prototypes).realize()
            closest, scores = dots.argmax(axis=1).numpy(), dots.max(axis=1).numpy()
        else:
            dots = normalize(values) @ centers.T
            closest, scores = np.argmax(dots, axis=1), dots.max(axis=1)
        batch = slice(start, start + len(values))
        ids[batch] = np.where(np.any(values != 0, axis=1), closest, -1)
        similarity[batch] = scores
    return indices, ids, similarity


def history_match(indices, history, assignment):
    embedded_ids, clusters, similarity = assignment
    item_ids = np.full(len(indices), -1, dtype=np.int16)
    valid = indices >= 0
    item_positions = np.searchsorted(embedded_ids, indices[valid])
    item_ids[valid] = clusters[item_positions]
    history = history[history >= 0]
    available = clusters[np.searchsorted(embedded_ids, history)]
    available = available[available >= 0]
    shares = ((item_ids[:, None] == available).sum(axis=1) / max(len(available), 1)).astype(np.float32)
    cosine = np.zeros(len(indices), dtype=np.float32)
    cosine[valid] = similarity[item_positions]
    return item_ids, shares, cosine
