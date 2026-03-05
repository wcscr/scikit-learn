"""Ray-parallelised MNIST benchmark: tree-merge vs batch fit.

Splits MNIST training data into 10 chunks, fits each via partial_fit
as a Ray remote task, then merges with a binary tree reduction.
Compares wall-clock time and accuracy to a single batch fit.
"""

import sys
from pathlib import Path

# Ensure sklearn_ext is importable from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

import numpy as np
import ray
from sklearn.datasets import fetch_openml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn_ext.distributed_lda import merge_lda_models


# ── Ray remote functions ────────────────────────────────────────────────


@ray.remote
def fit_worker(X_chunk, y_chunk, classes, lda_kwargs):
    """Fit a single LDA worker via partial_fit."""
    m = LinearDiscriminantAnalysis(solver="svd", **lda_kwargs)
    m.partial_fit(X_chunk, y_chunk, classes=classes)
    return m


@ray.remote
def merge_pair(est_a, est_b):
    """Merge two estimators (one level of the tree)."""
    return merge_lda_models([est_a, est_b])


def tree_merge(futures):
    """Binary tree reduction: merge pairs until one model remains."""
    while len(futures) > 1:
        next_level = []
        for i in range(0, len(futures) - 1, 2):
            next_level.append(merge_pair.remote(futures[i], futures[i + 1]))
        # If odd number, carry the last one forward
        if len(futures) % 2 == 1:
            next_level.append(futures[-1])
        futures = next_level
    return ray.get(futures[0])


# ── Main ────────────────────────────────────────────────────────────────


def main():
    # Load MNIST
    print("Loading MNIST …")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = mnist.data, mnist.target.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    classes = np.unique(y)
    print(
        f"  {X_train.shape[0]} train, {X_test.shape[0]} test, "
        f"{X_train.shape[1]} features, {len(classes)} classes"
    )

    # ── Batch fit ───────────────────────────────────────────────────────
    print("\n── Batch fit ──")
    t0 = time.perf_counter()
    batch = LinearDiscriminantAnalysis(solver="svd").fit(X_train, y_train)
    batch_time = time.perf_counter() - t0
    batch_acc = accuracy_score(y_test, batch.predict(X_test))
    print(f"  Time:     {batch_time:.3f}s")
    print(f"  Accuracy: {batch_acc:.4f}")

    # ── Ray tree merge ──────────────────────────────────────────────────
    n_chunks = 10
    print(f"\n── Ray tree merge ({n_chunks} chunks) ──")

    ray.init(ignore_reinit_error=True, log_to_driver=False)

    # Put training data in object store once
    X_ref = ray.put(X_train)
    y_ref = ray.put(y_train)

    chunks_X = np.array_split(X_train, n_chunks)
    chunks_y = np.array_split(y_train, n_chunks)

    t0 = time.perf_counter()

    # Launch all workers in parallel
    worker_futures = [
        fit_worker.remote(cx, cy, classes, {})
        for cx, cy in zip(chunks_X, chunks_y)
    ]

    # Tree-reduce merge
    merged = tree_merge(worker_futures)

    ray_time = time.perf_counter() - t0

    merged_acc = accuracy_score(y_test, merged.predict(X_test))
    agree = np.mean(batch.predict(X_test) == merged.predict(X_test))

    print(f"  Time:     {ray_time:.3f}s")
    print(f"  Accuracy: {merged_acc:.4f}")
    print(f"  Agreement with batch: {agree:.4f}")

    # ── Summary ─────────────────────────────────────────────────────────
    speedup = batch_time / ray_time if ray_time > 0 else float("inf")
    print(f"\n── Summary ──")
    print(f"  Batch time:     {batch_time:.3f}s")
    print(f"  Ray merge time: {ray_time:.3f}s")
    print(f"  Speedup:        {speedup:.2f}x")
    print(f"  Accuracy diff:  {abs(batch_acc - merged_acc):.4f}")

    ray.shutdown()


if __name__ == "__main__":
    main()
