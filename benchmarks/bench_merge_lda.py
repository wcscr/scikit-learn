"""Benchmark: merge_lda_models vs sequential partial_fit."""

import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn_ext.distributed_lda import merge_lda_models


def make_data(n_samples, n_features, n_classes, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(n_features, max(n_classes, 2)),
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_state,
    )
    return X, y


def bench_sequential(X, y, n_workers, classes):
    """Time sequential partial_fit on all chunks (single model)."""
    chunks_X = np.array_split(X, n_workers)
    chunks_y = np.array_split(y, n_workers)
    m = LinearDiscriminantAnalysis(solver="svd")
    t0 = time.perf_counter()
    for cx, cy in zip(chunks_X, chunks_y):
        m.partial_fit(cx, cy, classes=classes)
    # Force lazy reconstruction
    _ = m.coef_
    elapsed = time.perf_counter() - t0
    return elapsed


def bench_merge(X, y, n_workers, classes):
    """Time independent partial_fit + merge (simulates parallel workers)."""
    chunks_X = np.array_split(X, n_workers)
    chunks_y = np.array_split(y, n_workers)

    # Phase 1: independent fits (in real usage these run in parallel)
    t0 = time.perf_counter()
    models = []
    for cx, cy in zip(chunks_X, chunks_y):
        m = LinearDiscriminantAnalysis(solver="svd")
        m.partial_fit(cx, cy, classes=classes)
        models.append(m)
    t_fit = time.perf_counter() - t0

    # Phase 2: merge
    t1 = time.perf_counter()
    merged = merge_lda_models(models)
    _ = merged.coef_  # force lazy reconstruction
    t_merge = time.perf_counter() - t1

    return t_fit, t_merge, t_fit + t_merge


def run_suite():
    print("=" * 78)
    print("Benchmark: merge_lda_models vs sequential partial_fit")
    print("=" * 78)

    # --- Vary number of workers ---
    print("\n--- Vary n_workers (N=100k, D=50, C=5) ---")
    print(f"{'workers':>8} {'sequential':>12} {'fit(par)':>12} {'merge':>12} {'total':>12} {'speedup':>8}")
    X, y = make_data(100_000, 50, 5)
    classes = np.unique(y)
    for n_workers in [2, 4, 8, 16, 32]:
        t_seq = bench_sequential(X, y, n_workers, classes)
        t_fit, t_merge, t_total = bench_merge(X, y, n_workers, classes)
        # Parallel speedup: if fits ran in parallel, wall time = max(fit) + merge
        t_parallel_wall = t_fit / n_workers + t_merge
        speedup = t_seq / t_parallel_wall
        print(
            f"{n_workers:>8} {t_seq:>11.4f}s {t_fit:>11.4f}s {t_merge:>11.4f}s "
            f"{t_total:>11.4f}s {speedup:>7.1f}x"
        )

    # --- Vary data size ---
    print("\n--- Vary N (workers=4, D=50, C=5) ---")
    print(f"{'N':>10} {'sequential':>12} {'fit(par)':>12} {'merge':>12} {'total':>12} {'speedup':>8}")
    for n_samples in [10_000, 50_000, 100_000, 500_000]:
        X, y = make_data(n_samples, 50, 5)
        classes = np.unique(y)
        t_seq = bench_sequential(X, y, 4, classes)
        t_fit, t_merge, t_total = bench_merge(X, y, 4, classes)
        t_parallel_wall = t_fit / 4 + t_merge
        speedup = t_seq / t_parallel_wall
        print(
            f"{n_samples:>10} {t_seq:>11.4f}s {t_fit:>11.4f}s {t_merge:>11.4f}s "
            f"{t_total:>11.4f}s {speedup:>7.1f}x"
        )

    # --- Vary features ---
    print("\n--- Vary D (workers=4, N=50k, C=5) ---")
    print(f"{'D':>8} {'sequential':>12} {'fit(par)':>12} {'merge':>12} {'total':>12} {'speedup':>8}")
    for n_features in [10, 50, 200, 500]:
        X, y = make_data(50_000, n_features, 5)
        classes = np.unique(y)
        t_seq = bench_sequential(X, y, 4, classes)
        t_fit, t_merge, t_total = bench_merge(X, y, 4, classes)
        t_parallel_wall = t_fit / 4 + t_merge
        speedup = t_seq / t_parallel_wall
        print(
            f"{n_features:>8} {t_seq:>11.4f}s {t_fit:>11.4f}s {t_merge:>11.4f}s "
            f"{t_total:>11.4f}s {speedup:>7.1f}x"
        )

    # --- Vary classes ---
    print("\n--- Vary C (workers=4, N=50k, D=50) ---")
    print(f"{'C':>8} {'sequential':>12} {'fit(par)':>12} {'merge':>12} {'total':>12} {'speedup':>8}")
    for n_classes in [2, 5, 10, 20]:
        X, y = make_data(50_000, 50, n_classes)
        classes = np.unique(y)
        t_seq = bench_sequential(X, y, 4, classes)
        t_fit, t_merge, t_total = bench_merge(X, y, 4, classes)
        t_parallel_wall = t_fit / 4 + t_merge
        speedup = t_seq / t_parallel_wall
        print(
            f"{n_classes:>8} {t_seq:>11.4f}s {t_fit:>11.4f}s {t_merge:>11.4f}s "
            f"{t_total:>11.4f}s {speedup:>7.1f}x"
        )

    # --- Merge-only cost (no fit) ---
    print("\n--- Merge-only cost (N=100k, D=50, C=5) ---")
    print(f"{'workers':>8} {'merge_only':>12}")
    X, y = make_data(100_000, 50, 5)
    classes = np.unique(y)
    for n_workers in [2, 4, 8, 16, 32]:
        chunks_X = np.array_split(X, n_workers)
        chunks_y = np.array_split(y, n_workers)
        models = []
        for cx, cy in zip(chunks_X, chunks_y):
            m = LinearDiscriminantAnalysis(solver="svd")
            m.partial_fit(cx, cy, classes=classes)
            models.append(m)
        t0 = time.perf_counter()
        merged = merge_lda_models(models)
        _ = merged.coef_
        t_merge = time.perf_counter() - t0
        print(f"{n_workers:>8} {t_merge:>11.4f}s")

    print()


if __name__ == "__main__":
    run_suite()
