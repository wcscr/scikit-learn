"""
Benchmark: Large-scale correctness & performance tests for LDA partial_fit.

Validates that partial_fit and fit produce identical results at scale,
measures performance overhead, memory efficiency, and numerical stability.

Usage:
    python benchmarks/bench_lda_partial_fit.py --quick   # smoke test (~30s)
    python benchmarks/bench_lda_partial_fit.py            # full run (~5-10 min)
"""

import argparse
import gc
import sys
import time
import tracemalloc

import numpy as np
from scipy.stats import ortho_group

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_multivariate_data(
    n_samples_per_class, n_features, n_classes, rng, cov=None
):
    """Generate data from multivariate normals with shared covariance.

    Returns X, y, means, cov used for generation.
    """
    if cov is None:
        # Random positive-definite covariance
        A = rng.standard_normal((n_features, n_features))
        cov = A @ A.T / n_features + np.eye(n_features)

    # Spread class means apart so they're well separated
    means = np.zeros((n_classes, n_features))
    for k in range(n_classes):
        means[k] = rng.standard_normal(n_features) * 3.0

    chunks = []
    labels = []
    for k in range(n_classes):
        X_k = rng.multivariate_normal(means[k], cov, size=n_samples_per_class)
        chunks.append(X_k)
        labels.append(np.full(n_samples_per_class, k))

    X = np.vstack(chunks)
    y = np.concatenate(labels)

    # Shuffle
    perm = rng.permutation(len(y))
    X = X[perm]
    y = y[perm]
    return X, y, means, cov


def compare_models(ref, test, X_test, y_test, tolerances=None):
    """Compare two fitted LDA models on attributes and predictions.

    Returns a list of (attr_name, passed, max_abs_diff, max_rel_diff).
    """
    if tolerances is None:
        tolerances = {}

    default_tols = {
        "means_": {"rtol": 1e-10, "atol": 1e-12},
        "priors_": {"rtol": 1e-10, "atol": 1e-12},
        "covariance_": {"rtol": 1e-8, "atol": 1e-10},
        "coef_": {"rtol": 1e-6, "atol": 1e-8},
        "intercept_": {"rtol": 1e-6, "atol": 1e-8},
        "predict": {"rtol": 0, "atol": 0},
        "predict_proba": {"rtol": 1e-6, "atol": 1e-8},
    }
    default_tols.update(tolerances)

    results = []

    # Compare array attributes
    for attr in ["means_", "priors_", "covariance_", "coef_", "intercept_"]:
        ref_val = getattr(ref, attr)
        test_val = getattr(test, attr)
        tol = default_tols.get(attr, {"rtol": 1e-8, "atol": 1e-10})

        abs_diff = np.max(np.abs(ref_val - test_val))
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = np.maximum(np.abs(ref_val), 1e-30)
            rel_diff = np.max(np.abs(ref_val - test_val) / denom)

        passed = np.allclose(ref_val, test_val, rtol=tol["rtol"], atol=tol["atol"])
        results.append((attr, passed, abs_diff, rel_diff))

    # Compare predictions
    ref_pred = ref.predict(X_test)
    test_pred = test.predict(X_test)
    pred_match = np.all(ref_pred == test_pred)
    mismatch_frac = np.mean(ref_pred != test_pred)
    results.append(("predict", pred_match, mismatch_frac, mismatch_frac))

    # Compare predict_proba
    ref_proba = ref.predict_proba(X_test)
    test_proba = test.predict_proba(X_test)
    tol = default_tols["predict_proba"]
    abs_diff = np.max(np.abs(ref_proba - test_proba))
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.maximum(np.abs(ref_proba), 1e-30)
        rel_diff = np.max(np.abs(ref_proba - test_proba) / denom)
    passed = np.allclose(ref_proba, test_proba, rtol=tol["rtol"], atol=tol["atol"])
    results.append(("predict_proba", passed, abs_diff, rel_diff))

    return results


def print_comparison(results, label):
    """Print comparison results in a formatted table."""
    all_pass = True
    print(f"\n  {label}")
    print(f"  {'Attribute':<18} {'Status':<8} {'Max Abs Diff':<16} {'Max Rel Diff':<16}")
    print(f"  {'-'*58}")
    for attr, passed, abs_diff, rel_diff in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {attr:<18} {status:<8} {abs_diff:<16.2e} {rel_diff:<16.2e}")
        if not passed:
            all_pass = False
    return all_pass


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_correctness(quick=False):
    """Test 1: partial_fit matches fit across configurations (n_features=20)."""
    print("\n" + "=" * 70)
    print("TEST 1: Correctness — partial_fit matches fit (n_features=20)")
    print("=" * 70)

    n_features = 20
    n_chunks = 3
    rng = np.random.default_rng(42)

    if quick:
        # ~50MB per chunk: 50e6 / (20 * 8) ≈ 312,500 samples per chunk
        chunk_size = 312_500
    else:
        # ~1.3GB per chunk: 1.3e9 / (20 * 8) ≈ 8,125,000 samples per chunk
        chunk_size = 8_125_000

    configs = []
    for n_classes in [2, 5, 10]:
        for solver in ["eigen", "lsqr"]:
            for shrinkage in [None, 0.5]:
                configs.append(
                    {
                        "n_classes": n_classes,
                        "solver": solver,
                        "shrinkage": shrinkage,
                    }
                )

    n_test = 1000
    all_pass = True

    for cfg in configs:
        n_classes = cfg["n_classes"]
        solver = cfg["solver"]
        shrinkage = cfg["shrinkage"]
        label = f"n_classes={n_classes}, solver={solver}, shrinkage={shrinkage}"
        print(f"\n  Config: {label}")

        samples_per_class = chunk_size * n_chunks // n_classes
        total = samples_per_class * n_classes

        print(f"  Generating {total:,} samples ({n_chunks} chunks)...")
        t0 = time.perf_counter()
        X, y, _, _ = generate_multivariate_data(
            samples_per_class, n_features, n_classes, rng
        )
        print(f"  Data generated in {time.perf_counter() - t0:.1f}s")

        # Test set
        X_test, y_test, _, _ = generate_multivariate_data(
            n_test // n_classes, n_features, n_classes, rng
        )

        # fit()
        lda_fit = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
        t0 = time.perf_counter()
        lda_fit.fit(X, y)
        t_fit = time.perf_counter() - t0
        print(f"  fit() took {t_fit:.2f}s")

        # partial_fit()
        lda_pf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
        chunk_sz = total // n_chunks
        t0 = time.perf_counter()
        for i in range(n_chunks):
            start = i * chunk_sz
            end = start + chunk_sz
            lda_pf.partial_fit(
                X[start:end],
                y[start:end],
                classes=np.arange(n_classes) if i == 0 else None,
            )
        t_pf = time.perf_counter() - t0
        print(f"  partial_fit() took {t_pf:.2f}s")

        results = compare_models(lda_fit, lda_pf, X_test, y_test)
        cfg_pass = print_comparison(results, label)
        if not cfg_pass:
            all_pass = False

        # Memory cleanup
        del X, y, X_test, y_test, lda_fit, lda_pf
        gc.collect()

    status = "PASSED" if all_pass else "FAILED"
    print(f"\n  Test 1 {status}")
    return all_pass


def test_correctness_high_dim(quick=False):
    """Test 2: Correctness at higher dimensionality (n_features=50)."""
    print("\n" + "=" * 70)
    print("TEST 2: Correctness — higher dimensionality (n_features=50)")
    print("=" * 70)

    n_features = 50
    n_chunks = 3
    rng = np.random.default_rng(123)

    if quick:
        # ~50MB per chunk: 50e6 / (50 * 8) ≈ 125,000 samples per chunk
        chunk_size = 125_000
    else:
        # ~1.3GB per chunk: 1.3e9 / (50 * 8) ≈ 3,250,000 samples per chunk
        chunk_size = 3_250_000

    configs = []
    for n_classes in [2, 5]:
        for solver in ["eigen", "lsqr"]:
            configs.append({"n_classes": n_classes, "solver": solver})

    n_test = 1000
    all_pass = True

    for cfg in configs:
        n_classes = cfg["n_classes"]
        solver = cfg["solver"]
        label = f"n_classes={n_classes}, solver={solver}, n_features={n_features}"
        print(f"\n  Config: {label}")

        samples_per_class = chunk_size * n_chunks // n_classes
        total = samples_per_class * n_classes

        print(f"  Generating {total:,} samples ({n_chunks} chunks)...")
        t0 = time.perf_counter()
        X, y, _, _ = generate_multivariate_data(
            samples_per_class, n_features, n_classes, rng
        )
        print(f"  Data generated in {time.perf_counter() - t0:.1f}s")

        X_test, y_test, _, _ = generate_multivariate_data(
            n_test // n_classes, n_features, n_classes, rng
        )

        # fit()
        lda_fit = LinearDiscriminantAnalysis(solver=solver)
        t0 = time.perf_counter()
        lda_fit.fit(X, y)
        t_fit = time.perf_counter() - t0
        print(f"  fit() took {t_fit:.2f}s")

        # partial_fit()
        lda_pf = LinearDiscriminantAnalysis(solver=solver)
        chunk_sz = total // n_chunks
        t0 = time.perf_counter()
        for i in range(n_chunks):
            start = i * chunk_sz
            end = start + chunk_sz
            lda_pf.partial_fit(
                X[start:end],
                y[start:end],
                classes=np.arange(n_classes) if i == 0 else None,
            )
        t_pf = time.perf_counter() - t0
        print(f"  partial_fit() took {t_pf:.2f}s")

        results = compare_models(lda_fit, lda_pf, X_test, y_test)
        cfg_pass = print_comparison(results, label)
        if not cfg_pass:
            all_pass = False

        del X, y, X_test, y_test, lda_fit, lda_pf
        gc.collect()

    status = "PASSED" if all_pass else "FAILED"
    print(f"\n  Test 2 {status}")
    return all_pass


def test_chunk_size_invariance(quick=False):
    """Test 3: Results are invariant to chunk size."""
    print("\n" + "=" * 70)
    print("TEST 3: Chunk-size invariance")
    print("=" * 70)

    n_features = 20
    n_classes = 3
    solver = "lsqr"
    rng = np.random.default_rng(77)

    if quick:
        total_samples = 300_000
        chunk_sizes = [total_samples, 100_000, 30_000, 10_000]
    else:
        total_samples = 9_000_000
        chunk_sizes = [total_samples, 3_000_000, 1_000_000, 100_000]

    samples_per_class = total_samples // n_classes

    print(f"  Generating {total_samples:,} samples...")
    t0 = time.perf_counter()
    X, y, _, _ = generate_multivariate_data(
        samples_per_class, n_features, n_classes, rng
    )
    print(f"  Data generated in {time.perf_counter() - t0:.1f}s")

    n_test = 1000
    X_test, y_test, _, _ = generate_multivariate_data(
        n_test // n_classes, n_features, n_classes, rng
    )

    # Reference: single-chunk (equivalent to fit)
    ref = LinearDiscriminantAnalysis(solver=solver)
    ref.partial_fit(X, y, classes=np.arange(n_classes))

    all_pass = True

    for cs in chunk_sizes[1:]:
        n_chunks = total_samples // cs
        label = f"chunk_size={cs:,} ({n_chunks} chunks)"
        print(f"\n  Config: {label}")

        lda = LinearDiscriminantAnalysis(solver=solver)
        t0 = time.perf_counter()
        for i in range(n_chunks):
            start = i * cs
            end = start + cs
            lda.partial_fit(
                X[start:end],
                y[start:end],
                classes=np.arange(n_classes) if i == 0 else None,
            )
        elapsed = time.perf_counter() - t0
        print(f"  partial_fit() took {elapsed:.2f}s")

        results = compare_models(ref, lda, X_test, y_test)
        cfg_pass = print_comparison(results, label)
        if not cfg_pass:
            all_pass = False

        del lda
        gc.collect()

    del X, y, X_test, y_test, ref
    gc.collect()

    status = "PASSED" if all_pass else "FAILED"
    print(f"\n  Test 3 {status}")
    return all_pass


def test_performance(quick=False):
    """Test 4: Performance benchmarking — fit() vs partial_fit()."""
    print("\n" + "=" * 70)
    print("TEST 4: Performance benchmarking")
    print("=" * 70)

    n_features = 20
    n_classes = 3
    n_chunks = 3
    rng = np.random.default_rng(99)

    if quick:
        total_samples = 900_000
    else:
        total_samples = 24_000_000

    samples_per_class = total_samples // n_classes

    all_pass = True

    for solver in ["eigen", "lsqr"]:
        print(f"\n  Solver: {solver}")
        print(f"  Generating {total_samples:,} samples...")
        t0 = time.perf_counter()
        X, y, _, _ = generate_multivariate_data(
            samples_per_class, n_features, n_classes, rng
        )
        print(f"  Data generated in {time.perf_counter() - t0:.1f}s")

        # Timing fit()
        lda_fit = LinearDiscriminantAnalysis(solver=solver)
        t0 = time.perf_counter()
        lda_fit.fit(X, y)
        t_fit = time.perf_counter() - t0

        # Timing partial_fit()
        lda_pf = LinearDiscriminantAnalysis(solver=solver)
        chunk_sz = total_samples // n_chunks
        t0 = time.perf_counter()
        for i in range(n_chunks):
            start = i * chunk_sz
            end = start + chunk_sz
            lda_pf.partial_fit(
                X[start:end],
                y[start:end],
                classes=np.arange(n_classes) if i == 0 else None,
            )
        t_pf = time.perf_counter() - t0

        overhead = t_pf / t_fit if t_fit > 0 else float("inf")
        fit_rate = total_samples / t_fit if t_fit > 0 else 0
        pf_rate = total_samples / t_pf if t_pf > 0 else 0

        print(f"  fit():        {t_fit:.2f}s  ({fit_rate:,.0f} samples/sec)")
        print(f"  partial_fit(): {t_pf:.2f}s  ({pf_rate:,.0f} samples/sec)")
        print(f"  Overhead ratio: {overhead:.2f}x")

        passed = overhead < 2.0
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: overhead {'<' if passed else '>='} 2x")
        if not passed:
            all_pass = False

        del X, y, lda_fit, lda_pf
        gc.collect()

    status = "PASSED" if all_pass else "FAILED"
    print(f"\n  Test 4 {status}")
    return all_pass


def test_memory_efficiency(quick=False):
    """Test 5: Memory efficiency — fit(full) vs chunked partial_fit."""
    print("\n" + "=" * 70)
    print("TEST 5: Memory efficiency")
    print("=" * 70)

    n_features = 20
    n_classes = 3
    n_chunks = 3
    solver = "lsqr"
    rng_seed = 55

    if quick:
        chunk_size = 312_500
    else:
        chunk_size = 8_125_000

    samples_per_class_per_chunk = chunk_size // n_classes
    total_per_class = samples_per_class_per_chunk * n_chunks
    total = total_per_class * n_classes

    # --- Measure fit() peak memory (includes data allocation) ---
    print(f"\n  Measuring fit() memory on {total:,} samples...")
    gc.collect()
    tracemalloc.start()
    rng = np.random.default_rng(rng_seed)
    X_full, y_full, _, _ = generate_multivariate_data(
        total_per_class, n_features, n_classes, rng
    )
    lda_fit = LinearDiscriminantAnalysis(solver=solver)
    lda_fit.fit(X_full, y_full)
    _, fit_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del X_full, y_full, lda_fit
    gc.collect()

    # --- Measure partial_fit() peak memory (generate-and-discard per chunk) ---
    print(f"  Measuring partial_fit() memory ({n_chunks} chunks)...")
    gc.collect()
    tracemalloc.start()
    lda_pf = LinearDiscriminantAnalysis(solver=solver)
    rng = np.random.default_rng(rng_seed)
    for i in range(n_chunks):
        # Generate one chunk at a time using per-chunk RNG state
        X_chunk, y_chunk, _, _ = generate_multivariate_data(
            samples_per_class_per_chunk, n_features, n_classes, rng
        )
        lda_pf.partial_fit(
            X_chunk,
            y_chunk,
            classes=np.arange(n_classes) if i == 0 else None,
        )
        del X_chunk, y_chunk
        gc.collect()
    _, pf_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del lda_pf
    gc.collect()

    fit_mb = fit_peak / (1024 * 1024)
    pf_mb = pf_peak / (1024 * 1024)
    ratio = fit_mb / pf_mb if pf_mb > 0 else float("inf")

    print(f"\n  fit() peak memory:         {fit_mb:,.1f} MB")
    print(f"  partial_fit() peak memory: {pf_mb:,.1f} MB")
    print(f"  Reduction ratio:           {ratio:.1f}x")

    # partial_fit should use less memory
    passed = pf_mb < fit_mb
    status = "PASS" if passed else "FAIL"
    print(f"  {status}: partial_fit {'<' if passed else '>='} fit memory")

    status_str = "PASSED" if passed else "FAILED"
    print(f"\n  Test 5 {status_str}")
    return passed


def test_ill_conditioned(quick=False):
    """Test 6: Numerical stability with ill-conditioned covariance."""
    print("\n" + "=" * 70)
    print("TEST 6: Numerical stability — ill-conditioned covariance")
    print("=" * 70)

    n_features = 20
    n_classes = 3
    solver = "lsqr"
    rng = np.random.default_rng(2024)
    n_chunks = 3

    if quick:
        total_samples = 300_000
    else:
        total_samples = 3_000_000

    samples_per_class = total_samples // n_classes

    # Construct covariance with condition number ~1e6
    Q = ortho_group.rvs(n_features, random_state=42)
    eigenvalues = np.logspace(0, -6, n_features)  # 1 to 1e-6
    cov = Q @ np.diag(eigenvalues) @ Q.T
    cov = (cov + cov.T) / 2  # Ensure exact symmetry
    actual_cond = np.linalg.cond(cov)
    print(f"\n  Covariance condition number: {actual_cond:.2e}")

    print(f"  Generating {total_samples:,} samples...")
    t0 = time.perf_counter()
    X, y, _, _ = generate_multivariate_data(
        samples_per_class, n_features, n_classes, rng, cov=cov
    )
    print(f"  Data generated in {time.perf_counter() - t0:.1f}s")

    n_test = 1000
    X_test, y_test, _, _ = generate_multivariate_data(
        n_test // n_classes, n_features, n_classes, rng, cov=cov
    )

    # fit()
    lda_fit = LinearDiscriminantAnalysis(solver=solver)
    lda_fit.fit(X, y)

    # partial_fit()
    lda_pf = LinearDiscriminantAnalysis(solver=solver)
    chunk_sz = total_samples // n_chunks
    for i in range(n_chunks):
        start = i * chunk_sz
        end = start + chunk_sz
        lda_pf.partial_fit(
            X[start:end],
            y[start:end],
            classes=np.arange(n_classes) if i == 0 else None,
        )

    # Relaxed tolerances for ill-conditioned case
    tolerances = {
        "means_": {"rtol": 1e-8, "atol": 1e-10},
        "priors_": {"rtol": 1e-10, "atol": 1e-12},
        "covariance_": {"rtol": 1e-6, "atol": 1e-8},
        "coef_": {"rtol": 1e-3, "atol": 1e-3},
        "intercept_": {"rtol": 1e-3, "atol": 1e-3},
        "predict_proba": {"rtol": 1e-3, "atol": 1e-3},
    }

    results = compare_models(lda_fit, lda_pf, X_test, y_test, tolerances)
    all_pass = print_comparison(
        results, f"ill-conditioned (cond={actual_cond:.2e})"
    )

    del X, y, X_test, y_test, lda_fit, lda_pf
    gc.collect()

    status = "PASSED" if all_pass else "FAILED"
    print(f"\n  Test 6 {status}")
    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LDA partial_fit correctness and performance"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run with smaller data sizes for smoke testing (~30s)",
    )
    args = parser.parse_args()

    mode = "QUICK" if args.quick else "FULL"
    print(f"\n{'#' * 70}")
    print(f"# LDA partial_fit Benchmark ({mode} mode)")
    print(f"{'#' * 70}")

    tests = [
        ("Test 1: Correctness (n_features=20)", test_correctness),
        ("Test 2: Correctness high-dim (n_features=50)", test_correctness_high_dim),
        ("Test 3: Chunk-size invariance", test_chunk_size_invariance),
        ("Test 4: Performance", test_performance),
        ("Test 5: Memory efficiency", test_memory_efficiency),
        ("Test 6: Numerical stability", test_ill_conditioned),
    ]

    results = []
    for name, func in tests:
        try:
            passed = func(quick=args.quick)
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")
            traceback.print_exc()
            passed = False
        results.append((name, passed))
        gc.collect()

    # Final summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    n_passed = 0
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {status}: {name}")
        if passed:
            n_passed += 1

    total = len(results)
    print(f"\n  {n_passed}/{total} tests passed")
    print(f"{'=' * 70}\n")

    sys.exit(0 if n_passed == total else 1)


if __name__ == "__main__":
    main()
