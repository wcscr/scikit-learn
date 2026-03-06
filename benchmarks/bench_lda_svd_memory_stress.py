"""Memory-stress benchmark: batch vs streaming SVD partial_fit on large data.

Generates a synthetic dataset sized to stress available RAM during batch SVD,
then compares batch fit vs streaming partial_fit with 10%-sized chunks.

Measures wall-clock time, peak RSS, stored SVD rank, and prediction agreement.
"""

import gc
import os
import sys
import tracemalloc
from time import perf_counter

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def peak_rss_gb():
    """Current peak RSS in GB (Linux)."""
    # /proc/self/status gives VmHWM in kB
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1e6  # kB -> GB
    except Exception:
        pass
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def make_low_rank_data(n_samples, n_features, n_informative, n_classes, seed=42):
    """Synthetic data with controlled intrinsic rank."""
    rng = np.random.RandomState(seed)
    # Low-rank signal: n_informative latent dims projected to n_features
    X_core = rng.randn(n_samples, n_informative)
    proj = rng.randn(n_informative, n_features) / np.sqrt(n_informative)
    X = X_core @ proj
    # Small additive noise (keeps near-low-rank structure)
    X += rng.randn(n_samples, n_features) * 1e-4
    # Labels from first few latent dims
    boundaries = np.linspace(
        X_core[:, 0].min(), X_core[:, 0].max(), n_classes + 1
    )
    y = np.digitize(X_core[:, 0], boundaries[1:-1])
    return X, y


def make_full_rank_data(n_samples, n_features, n_classes, seed=42):
    """Full-rank random data (all features independent)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = rng.randint(0, n_classes, size=n_samples)
    return X, y


def run_benchmark(X, y, chunk_frac=0.10, label=""):
    n_samples, n_features = X.shape
    classes = np.unique(y)
    n_classes = len(classes)
    chunk_size = max(1, int(n_samples * chunk_frac))
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    dataset_gb = X.nbytes / 1e9

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  N={n_samples:,}  D={n_features:,}  K={n_classes}")
    print(f"  Dataset: {dataset_gb:.2f} GB   Chunk: {chunk_size:,} ({n_chunks} chunks)")
    print(f"{'='*70}")

    # --- Batch fit ---
    gc.collect()
    rss_before = peak_rss_gb()
    tracemalloc.start()
    t0 = perf_counter()
    clf_batch = LinearDiscriminantAnalysis(solver="svd")
    clf_batch.fit(X, y)
    batch_time = perf_counter() - t0
    batch_peak_traced = tracemalloc.get_traced_memory()[1] / 1e9
    tracemalloc.stop()
    batch_rss = peak_rss_gb()
    batch_acc = clf_batch.score(X, y)
    preds_batch = clf_batch.predict(X)
    print(f"\n  BATCH FIT")
    print(f"    Time:     {batch_time:8.1f}s")
    print(f"    Accuracy: {batch_acc:.4f}")
    print(f"    Peak RSS: {batch_rss:.2f} GB  (traced alloc: {batch_peak_traced:.2f} GB)")

    # Free batch model internals to reduce memory pressure for streaming
    del clf_batch
    gc.collect()

    # --- Streaming partial_fit ---
    gc.collect()
    tracemalloc.start()
    t0 = perf_counter()
    clf_stream = LinearDiscriminantAnalysis(solver="svd")
    chunk_times = []
    for i, start in enumerate(range(0, n_samples, chunk_size)):
        end = min(start + chunk_size, n_samples)
        tc = perf_counter()
        clf_stream.partial_fit(
            X[start:end], y[start:end],
            classes=classes if start == 0 else None,
        )
        chunk_times.append(perf_counter() - tc)
    stream_time = perf_counter() - t0
    stream_peak_traced = tracemalloc.get_traced_memory()[1] / 1e9
    tracemalloc.stop()
    stream_rss = peak_rss_gb()
    stream_acc = clf_stream.score(X, y)
    preds_stream = clf_stream.predict(X)
    agreement = np.mean(preds_batch == preds_stream)
    stored_rank = clf_stream._unscaled_S.shape[0]

    print(f"\n  STREAMING partial_fit  (chunk_size={chunk_size:,})")
    print(f"    Time:     {stream_time:8.1f}s")
    print(f"    Accuracy: {stream_acc:.4f}")
    print(f"    Peak RSS: {stream_rss:.2f} GB  (traced alloc: {stream_peak_traced:.2f} GB)")
    print(f"    Stored SVD rank: {stored_rank}")
    print(f"    Agreement with batch: {agreement:.4f}")

    print(f"\n  Per-chunk times:")
    for i, ct in enumerate(chunk_times):
        rank_after = "?" # can't easily get intermediate rank
        print(f"    chunk {i:2d}: {ct:7.2f}s")

    ratio = stream_time / batch_time if batch_time > 0 else float("inf")
    print(f"\n  SUMMARY:  stream/batch = {ratio:.2f}x", end="")
    if ratio < 1:
        print(f"  ({1/ratio:.1f}x FASTER)")
    else:
        print(f"  ({ratio:.1f}x slower)")

    return {
        "batch_time": batch_time,
        "stream_time": stream_time,
        "batch_acc": batch_acc,
        "stream_acc": stream_acc,
        "agreement": agreement,
        "stored_rank": stored_rank,
        "ratio": ratio,
    }


def main():
    N = 20_000
    D = 5_000
    K = 10

    print(f"System: {os.cpu_count()} CPUs")
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"RAM: {mem.total/1e9:.1f} GB total, {mem.available/1e9:.1f} GB available")
    except ImportError:
        pass

    # --- Scenario 1: Low intrinsic rank (50 informative dims in 8000-d) ---
    print("\nGenerating low-rank dataset...")
    X_lr, y_lr = make_low_rank_data(N, D, n_informative=50, n_classes=K)
    results_lr = run_benchmark(
        X_lr, y_lr, chunk_frac=0.10,
        label="LOW RANK: 50 informative dims in 8000-d space",
    )

    del X_lr, y_lr
    gc.collect()

    # --- Scenario 2: Full rank (all features independent) ---
    print("\n\nGenerating full-rank dataset...")
    X_fr, y_fr = make_full_rank_data(N, D, n_classes=K)
    results_fr = run_benchmark(
        X_fr, y_fr, chunk_frac=0.10,
        label="FULL RANK: all 8000 features independent",
    )

    del X_fr, y_fr
    gc.collect()

    # --- Final comparison ---
    print(f"\n\n{'='*70}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*70}")
    for name, r in [("Low-rank", results_lr), ("Full-rank", results_fr)]:
        print(f"\n  {name}:")
        print(f"    Batch:  {r['batch_time']:6.1f}s  acc={r['batch_acc']:.4f}")
        print(f"    Stream: {r['stream_time']:6.1f}s  acc={r['stream_acc']:.4f}"
              f"  rank={r['stored_rank']}")
        print(f"    Ratio:  {r['ratio']:.2f}x  agreement={r['agreement']:.4f}")


if __name__ == "__main__":
    main()
