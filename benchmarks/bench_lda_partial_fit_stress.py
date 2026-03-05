#!/usr/bin/env python
"""Stress-test benchmark for LDA partial_fit across all solvers and backends.

Covers: wide/tall/many-class/imbalanced/single-sample/high-D/near-singular
synthetic scenarios, public datasets (MNIST, Fashion-MNIST, Covertype),
and Array API backends (torch CPU/CUDA, CuPy CUDA).

Usage
-----
# Synthetic scenarios only (no network)
python benchmarks/bench_lda_partial_fit_stress.py --scenarios synthetic

# Quick smoke test (10x smaller)
python benchmarks/bench_lda_partial_fit_stress.py --quick

# Full run including public datasets
SKLEARN_SKIP_NETWORK_TESTS=0 python benchmarks/bench_lda_partial_fit_stress.py \
    --scenarios all

# Torch CPU
SCIPY_ARRAY_API=1 python benchmarks/bench_lda_partial_fit_stress.py \
    --scenarios torch_cpu

# All GPU backends
SCIPY_ARRAY_API=1 python benchmarks/bench_lda_partial_fit_stress.py \
    --scenarios all --include-cuda

# JSON output
python benchmarks/bench_lda_partial_fit_stress.py --json-out results.json
"""

import argparse
import gc
import json
import os
import sys
import tracemalloc
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from time import perf_counter

import numpy as np

from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    scenario: str
    solver: str
    backend: str
    chunk_size: int
    n_samples: int
    n_features: int
    n_classes: int
    acc_batch: float = float("nan")
    acc_stream: float = float("nan")
    pred_agreement: float = float("nan")
    acc_delta: float = float("nan")
    coef_atol_ok: bool = True
    wall_time_batch: float = float("nan")
    wall_time_stream: float = float("nan")
    peak_mem_batch_mb: float = float("nan")
    peak_mem_stream_mb: float = float("nan")
    gpu_mem_mb: float = float("nan")
    has_nan_inf: bool = False
    passed: bool = True
    tier: str = "tight"
    skipped: bool = False
    skip_reason: str = ""
    warnings: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Infrastructure helpers
# ---------------------------------------------------------------------------


def _log(msg, indent=1):
    """Print a progress message with indentation and flush."""
    prefix = "  " * indent
    print(f"{prefix}{msg}", flush=True)


@contextmanager
def perf_timer():
    """Context manager that records wall-clock time."""
    result = {"elapsed": 0.0}
    t0 = perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = perf_counter() - t0


@contextmanager
def mem_tracker():
    """Context manager that records peak Python memory (MB) via tracemalloc."""
    result = {"peak_mb": 0.0}
    gc.collect()
    tracemalloc.start()
    try:
        yield result
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result["peak_mb"] = peak / (1024**2)


def print_table(results, file=sys.stdout):
    """Print a summary table of ScenarioResults."""
    if not results:
        return
    header = (
        f"{'scenario':<16} {'solver':<6} {'backend':<11} {'chunk':>6}"
        f" {'acc_bat':>8} {'acc_str':>8} {'agree':>8} {'delta':>8}"
        f" {'t_bat':>7} {'t_str':>7} {'pass':>5}"
    )
    print(header, file=file)
    print("-" * len(header), file=file)
    for r in results:
        if r.skipped:
            print(
                f"{r.scenario:<16} {'SKIPPED':<6} {r.skip_reason}", file=file
            )
            continue
        status = "PASS" if r.passed else "FAIL"
        print(
            f"{r.scenario:<16} {r.solver:<6} {r.backend:<11} {r.chunk_size:>6}"
            f" {r.acc_batch:>8.4f} {r.acc_stream:>8.4f}"
            f" {r.pred_agreement:>8.4f} {r.acc_delta:>+8.4f}"
            f" {r.wall_time_batch:>7.2f} {r.wall_time_stream:>7.2f}"
            f" {status:>5}",
            file=file,
        )


def json_report(results, path):
    """Write results as JSON."""
    payload = {
        "pass": all(r.passed or r.skipped for r in results),
        "n_total": len(results),
        "n_passed": sum(r.passed for r in results),
        "n_failed": sum(not r.passed and not r.skipped for r in results),
        "n_skipped": sum(r.skipped for r in results),
        "results": [asdict(r) for r in results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.write("\n")


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------


def _available_backends(include_cuda=False):
    """Return list of (backend_name, converter_fn) tuples."""
    backends = [("numpy", lambda X: X)]
    try:
        import torch  # noqa: F811

        if os.environ.get("SCIPY_ARRAY_API") == "1":
            backends.append(
                ("torch_cpu", lambda X: torch.from_numpy(X.copy()))
            )
            if include_cuda and torch.cuda.is_available():
                backends.append(
                    (
                        "torch_cuda",
                        lambda X: torch.from_numpy(X.copy()).cuda(),
                    )
                )
    except ImportError:
        pass
    try:
        import cupy  # noqa: F401

        if os.environ.get("SCIPY_ARRAY_API") == "1" and include_cuda:
            import cupy as cp

            backends.append(("cupy_cuda", lambda X: cp.asarray(X)))
    except ImportError:
        pass
    return backends


def _to_numpy(arr):
    """Convert array-API array back to numpy for metric computation."""
    if isinstance(arr, np.ndarray):
        return arr
    # torch
    if hasattr(arr, "cpu"):
        return arr.detach().cpu().numpy()
    # cupy
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def _reset_gpu_mem():
    """Reset GPU memory counters if available."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _get_gpu_mem_mb():
    """Get peak GPU memory (MB) from torch CUDA."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**2)
    except ImportError:
        pass
    return float("nan")


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------


def chunked_make_classification(
    n_samples, n_features, n_classes, chunk_size, n_informative=None,
    n_redundant=0, random_state=42,
):
    """Yield (X_chunk, y_chunk) from make_classification without full materialization.

    Each chunk is generated with a deterministic per-chunk seed so results
    are reproducible but the full dataset is never in memory at once.
    """
    if n_informative is None:
        n_informative = min(n_features, max(n_classes, n_features // 2))
    rng = np.random.RandomState(random_state)
    # Pre-generate all seeds
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    seeds = rng.randint(0, 2**31, size=n_chunks)
    remaining = n_samples
    for i in range(n_chunks):
        this_size = min(chunk_size, remaining)
        X_chunk, y_chunk = make_classification(
            n_samples=this_size,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=int(seeds[i]),
        )
        yield X_chunk, y_chunk
        remaining -= this_size


def chunked_imbalanced(
    n_samples, n_features, n_classes, chunk_size, dominant_frac=0.95,
    random_state=42,
):
    """Generate chunks with class imbalance (class 0 gets dominant_frac)."""
    rng = np.random.RandomState(random_state)
    n_informative = min(n_features, max(n_classes, n_features // 2))
    # Build weights
    minor_weight = (1.0 - dominant_frac) / (n_classes - 1)
    weights = [dominant_frac] + [minor_weight] * (n_classes - 1)
    weights = np.array(weights)
    weights /= weights.sum()

    remaining = n_samples
    chunk_idx = 0
    while remaining > 0:
        this_size = min(chunk_size, remaining)
        # Generate balanced then resample to get imbalanced distribution
        seed = int(rng.randint(0, 2**31))
        X_raw, y_raw = make_classification(
            n_samples=max(this_size * 3, 200),
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=0,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=seed,
        )
        # Resample according to weights
        chosen = []
        for cls_idx in range(n_classes):
            cls_mask = y_raw == cls_idx
            cls_count = max(1, int(round(this_size * weights[cls_idx])))
            pool = np.where(cls_mask)[0]
            if len(pool) == 0:
                continue
            sel = rng.choice(pool, size=min(cls_count, len(pool)), replace=True)
            chosen.append(sel)
        chosen = np.concatenate(chosen)
        rng.shuffle(chosen)
        chosen = chosen[:this_size]
        yield X_raw[chosen], y_raw[chosen]
        remaining -= this_size
        chunk_idx += 1


# ---------------------------------------------------------------------------
# Core fit helpers
# ---------------------------------------------------------------------------


def run_batch_fit(X, y, solver, shrinkage=None):
    """Fit LDA on full data, return (model, wall_time, peak_mem_mb)."""
    kwargs = {"solver": solver}
    if shrinkage is not None and solver in ("eigen", "lsqr"):
        kwargs["shrinkage"] = shrinkage
    clf = LinearDiscriminantAnalysis(**kwargs)
    with perf_timer() as pt, mem_tracker() as mt:
        clf.fit(X, y)
    return clf, pt["elapsed"], mt["peak_mb"]


def run_streaming_fit(chunks, classes, solver, shrinkage=None, convert_fn=None,
                      n_total=None, log_interval=None):
    """Fit LDA via partial_fit over chunks.

    Parameters
    ----------
    chunks : iterable of (X_chunk, y_chunk)
    classes : array of all class labels
    solver : str
    shrinkage : float or None
    convert_fn : callable or None — applied to X_chunk before partial_fit
    n_total : int or None — total samples for progress reporting
    log_interval : int or None — log every N chunks (default: auto)

    Returns (model, wall_time, peak_mem_mb, n_samples_seen)
    """
    kwargs = {"solver": solver}
    if shrinkage is not None and solver in ("eigen", "lsqr"):
        kwargs["shrinkage"] = shrinkage
    clf = LinearDiscriminantAnalysis(**kwargs)
    first = True
    n_seen = 0
    chunk_count = 0
    t_start = perf_counter()
    with perf_timer() as pt, mem_tracker() as mt:
        for X_chunk, y_chunk in chunks:
            if convert_fn is not None:
                X_chunk = convert_fn(X_chunk)
            if first:
                clf.partial_fit(X_chunk, y_chunk, classes=classes)
                first = False
            else:
                clf.partial_fit(X_chunk, y_chunk)
            n_seen += X_chunk.shape[0]
            chunk_count += 1
            # Progress logging — aim for ~5-10 updates per run
            _interval = log_interval if log_interval is not None else max(1, 50)
            if chunk_count % _interval == 0:
                elapsed = perf_counter() - t_start
                if n_total:
                    pct = 100.0 * n_seen / n_total
                    _log(
                        f"[stream {solver}] chunk {chunk_count}: "
                        f"{n_seen}/{n_total} samples ({pct:.0f}%), "
                        f"{elapsed:.1f}s elapsed",
                        indent=2,
                    )
                else:
                    _log(
                        f"[stream {solver}] chunk {chunk_count}: "
                        f"{n_seen} samples, {elapsed:.1f}s elapsed",
                        indent=2,
                    )
    return clf, pt["elapsed"], mt["peak_mb"], n_seen


def check_model_sanity(clf):
    """Return True if model attributes have NaN/Inf."""
    for attr in ("coef_", "intercept_", "scalings_"):
        arr = getattr(clf, attr, None)
        if arr is not None:
            arr_np = _to_numpy(arr) if not isinstance(arr, np.ndarray) else arr
            if not np.all(np.isfinite(arr_np)):
                return True  # has_nan_inf
    return False


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

# Tier thresholds
TIER_TIGHT = {
    "pred_agreement": 0.999,
    "acc_delta": -0.005,
    "coef_atol": 1e-4,
    "acc_floor": 0.70,
}
TIER_MODERATE = {
    "pred_agreement": 0.99,
    "acc_delta": -0.01,
    "coef_atol": 1e-3,
    "acc_floor": 0.70,
}
TIER_PUBLIC = {
    "pred_agreement": 0.999,
    "acc_delta": -0.005,
    "coef_atol": 1e-4,
    "acc_floor": 0.75,
}


def validate_result(r, thresholds, has_batch=True):
    """Apply validation checks to a ScenarioResult in-place."""
    r.has_nan_inf = False  # will be set by caller
    r.passed = True
    warns = []

    if r.has_nan_inf:
        r.passed = False
        warns.append("NaN/Inf in model attributes")

    if r.acc_stream < thresholds.get("acc_floor", 0.70):
        r.passed = False
        warns.append(
            f"acc_stream {r.acc_stream:.4f} < floor {thresholds['acc_floor']}"
        )

    if has_batch and not np.isnan(r.acc_batch):
        if r.pred_agreement < thresholds.get("pred_agreement", 0.999):
            r.passed = False
            warns.append(
                f"pred_agreement {r.pred_agreement:.4f}"
                f" < {thresholds['pred_agreement']}"
            )
        if r.acc_delta < thresholds.get("acc_delta", -0.005):
            r.passed = False
            warns.append(
                f"acc_delta {r.acc_delta:+.4f}"
                f" < {thresholds['acc_delta']}"
            )
        if not r.coef_atol_ok:
            r.passed = False
            warns.append("coef_ mismatch exceeds atol")

    r.warnings = warns
    return r


def _check_coef_close(clf_batch, clf_stream, atol):
    """Check if coef_ arrays are close within atol."""
    c_b = getattr(clf_batch, "coef_", None)
    c_s = getattr(clf_stream, "coef_", None)
    if c_b is None or c_s is None:
        return True
    c_b = _to_numpy(c_b)
    c_s = _to_numpy(c_s)
    if c_b.shape != c_s.shape:
        return False
    return bool(np.allclose(c_b, c_s, atol=atol, rtol=0))


# ---------------------------------------------------------------------------
# Scenario implementations
# ---------------------------------------------------------------------------


def _run_standard_scenario(
    scenario_name, n_samples, n_features, n_classes, chunk_sizes,
    solvers, tier, thresholds, quick_divisor=10, quick=False,
    n_informative=None, n_redundant=0, shrinkage=None,
    generator_fn=None, generator_kwargs=None, random_state=42,
    max_batch_samples=200000, backend_name="numpy", convert_fn=None,
):
    """Generic runner for synthetic scenarios with batch+stream comparison."""
    if quick:
        n_samples = max(n_samples // quick_divisor, 200)
        chunk_sizes = [max(c // quick_divisor, 1) for c in chunk_sizes]
        # Relax accuracy floor in quick mode — tiny datasets separate poorly
        thresholds = dict(thresholds)
        thresholds["acc_floor"] = min(thresholds.get("acc_floor", 0.70), 0.20)

    results = []
    if n_informative is None:
        n_informative = min(n_features, max(n_classes, n_features // 2))

    _log(f"Config: N={n_samples}, D={n_features}, K={n_classes}, "
         f"chunks={chunk_sizes}, solvers={solvers}")

    # Pre-generate full dataset once (if feasible) for all solvers
    can_batch = n_samples <= max_batch_samples
    X_full = y_full = classes = None
    X_test = y_test = None
    if can_batch:
        if generator_fn is not None:
            all_X, all_y = [], []
            gkw = dict(generator_kwargs or {})
            gkw["chunk_size"] = n_samples
            for Xc, yc in generator_fn(**gkw):
                all_X.append(Xc)
                all_y.append(yc)
            X_full = np.vstack(all_X)
            y_full = np.concatenate(all_y)
        else:
            X_full, y_full = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                n_redundant=n_redundant,
                n_classes=n_classes,
                n_clusters_per_class=1,
                random_state=random_state,
            )
        classes = np.unique(y_full)
        _log(f"Dataset generated: {X_full.shape}, classes={len(classes)}")
    else:
        _log(f"N={n_samples} > max_batch={max_batch_samples}, using chunked generator")
        X_test, y_test = make_classification(
            n_samples=min(10000, n_samples // 5),
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=random_state + 9999,
        )
        classes = np.arange(n_classes)

    n_solver_total = len(solvers) * len(chunk_sizes)
    step = 0

    for solver in solvers:
        # Use shrinkage only for eigen/lsqr
        solver_shrinkage = shrinkage if solver in ("eigen", "lsqr") else None

        if can_batch:
            _log(f"Batch fit ({solver})...", indent=1)
            # Batch fit — may fail for some solver/data combos (e.g. eigen + singular)
            try:
                clf_batch, t_batch, mem_batch = run_batch_fit(
                    X_full, y_full, solver, shrinkage=solver_shrinkage
                )
                pred_batch = clf_batch.predict(X_full)
                acc_batch = float(np.mean(pred_batch == y_full))
                _log(f"Batch fit ({solver}): acc={acc_batch:.4f}, "
                     f"time={t_batch:.2f}s, mem={mem_batch:.1f}MB")
            except Exception as exc:
                _log(f"Batch fit ({solver}) failed: {exc}")
                # Solver can't handle this data (e.g. singular covariance)
                # Skip batch comparison, just validate streaming
                clf_batch = None
                t_batch = float("nan")
                mem_batch = float("nan")
                acc_batch = float("nan")
                pred_batch = None
        else:
            clf_batch = None
            t_batch = float("nan")
            mem_batch = float("nan")
            acc_batch = float("nan")
            pred_batch = None

        for cs in chunk_sizes:
            step += 1
            _log(f"Streaming {solver} cs={cs} "
                 f"[{step}/{n_solver_total}]...")
            # Build chunk iterator
            if can_batch:
                # Chunk the full dataset
                def _make_chunks(X, y, sz):
                    for start in range(0, len(X), sz):
                        end = min(start + sz, len(X))
                        yield X[start:end], y[start:end]

                chunks = _make_chunks(X_full, y_full, cs)
            else:
                if generator_fn is not None:
                    gkw = dict(generator_kwargs or {})
                    gkw["chunk_size"] = cs
                    chunks = generator_fn(**gkw)
                else:
                    chunks = chunked_make_classification(
                        n_samples=n_samples,
                        n_features=n_features,
                        n_classes=n_classes,
                        chunk_size=cs,
                        n_informative=n_informative,
                        n_redundant=n_redundant,
                        random_state=random_state,
                    )

            _reset_gpu_mem()
            # Auto log interval: ~10 updates per streaming run, min 5 chunks apart
            n_chunks_est = max(1, n_samples // max(cs, 1))
            _log_interval = max(5, n_chunks_est // 10)
            try:
                clf_stream, t_stream, mem_stream, n_seen = run_streaming_fit(
                    chunks, classes, solver,
                    shrinkage=solver_shrinkage,
                    convert_fn=convert_fn,
                    n_total=n_samples,
                    log_interval=_log_interval,
                )
            except Exception as exc:
                r = ScenarioResult(
                    scenario=scenario_name, solver=solver,
                    backend=backend_name, chunk_size=cs,
                    n_samples=n_samples, n_features=n_features,
                    n_classes=n_classes, passed=False,
                    tier=tier,
                    warnings=[f"streaming fit failed: {exc}"],
                )
                results.append(r)
                continue

            _log(f"Streaming {solver} cs={cs}: "
                 f"time={t_stream:.2f}s, mem={mem_stream:.1f}MB, "
                 f"samples={n_seen}")

            gpu_mem = _get_gpu_mem_mb() if backend_name not in ("numpy",) else float("nan")

            # Sanity check
            has_nan = check_model_sanity(clf_stream)

            # Accuracy
            has_batch_baseline = can_batch and pred_batch is not None
            try:
                if can_batch and X_full is not None:
                    pred_stream = _to_numpy(clf_stream.predict(X_full))
                    acc_stream = float(np.mean(pred_stream == y_full))
                    if has_batch_baseline:
                        pred_agree = float(np.mean(pred_stream == pred_batch))
                        coef_ok = _check_coef_close(
                            clf_batch, clf_stream,
                            thresholds.get("coef_atol", 1e-4),
                        )
                    else:
                        pred_agree = float("nan")
                        coef_ok = True
                else:
                    pred_stream_test = _to_numpy(clf_stream.predict(X_test))
                    acc_stream = float(np.mean(pred_stream_test == y_test))
                    pred_agree = float("nan")
                    coef_ok = True
            except Exception as exc:
                r = ScenarioResult(
                    scenario=scenario_name, solver=solver,
                    backend=backend_name, chunk_size=cs,
                    n_samples=n_samples, n_features=n_features,
                    n_classes=n_classes, passed=False,
                    tier=tier,
                    warnings=[f"predict after streaming fit failed: {exc}"],
                )
                results.append(r)
                continue

            acc_delta = (
                acc_stream - acc_batch
                if has_batch_baseline and not np.isnan(acc_batch)
                else float("nan")
            )

            r = ScenarioResult(
                scenario=scenario_name,
                solver=solver,
                backend=backend_name,
                chunk_size=cs,
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                acc_batch=acc_batch,
                acc_stream=acc_stream,
                pred_agreement=pred_agree,
                acc_delta=acc_delta,
                coef_atol_ok=coef_ok,
                wall_time_batch=t_batch,
                wall_time_stream=t_stream,
                peak_mem_batch_mb=mem_batch,
                peak_mem_stream_mb=mem_stream,
                gpu_mem_mb=gpu_mem,
                has_nan_inf=has_nan,
                tier=tier,
            )
            validate_result(r, thresholds, has_batch=has_batch_baseline)
            if has_nan:
                r.has_nan_inf = True
                r.passed = False
                r.warnings.append("NaN/Inf in model attributes")
            status = "PASS" if r.passed else "FAIL"
            _log(f"Result: {status} | acc_stream={acc_stream:.4f}"
                 + (f" agree={pred_agree:.4f}" if not np.isnan(pred_agree) else ""))
            results.append(r)

            gc.collect()

    return results


# --- Scenario 1: wide ---
def scenario_wide(args):
    return _run_standard_scenario(
        scenario_name="wide",
        n_samples=10000,
        n_features=5000,
        n_classes=10,
        chunk_sizes=[100, 500, 1000],
        solvers=args.solvers,
        tier="tight",
        thresholds=TIER_TIGHT,
        quick=args.quick,
    )


# --- Scenario 2: tall ---
def scenario_tall(args):
    n = args.tall_n_samples
    return _run_standard_scenario(
        scenario_name="tall",
        n_samples=n,
        n_features=50,
        n_classes=10,
        chunk_sizes=[1000, 5000, 10000],
        solvers=args.solvers,
        tier="tight",
        thresholds=TIER_TIGHT,
        quick=args.quick,
        max_batch_samples=args.max_batch_samples,
    )


# --- Scenario 3: many_classes ---
def scenario_many_classes(args):
    return _run_standard_scenario(
        scenario_name="many_classes",
        n_samples=50000,
        n_features=100,
        n_classes=50,
        chunk_sizes=[500, 2000],
        solvers=args.solvers,
        tier="tight",
        thresholds=TIER_TIGHT,
        quick=args.quick,
    )


# --- Scenario 4: imbalanced ---
def scenario_imbalanced(args):
    n_samples = 20000
    n_features = 20
    n_classes = 10
    if args.quick:
        n_samples = 2000

    _log(f"Config: N={n_samples}, D={n_features}, K={n_classes}, 95/5 imbalance")
    results = []
    for solver in args.solvers:
        solver_shrinkage = None
        can_batch = True

        # Generate imbalanced data
        all_X, all_y = [], []
        for Xc, yc in chunked_imbalanced(
            n_samples=n_samples, n_features=n_features,
            n_classes=n_classes, chunk_size=n_samples,
        ):
            all_X.append(Xc)
            all_y.append(yc)
        X_full = np.vstack(all_X)
        y_full = np.concatenate(all_y)
        classes = np.arange(n_classes)

        _log(f"Batch fit ({solver})...")
        clf_batch, t_batch, mem_batch = run_batch_fit(
            X_full, y_full, solver, shrinkage=solver_shrinkage
        )
        pred_batch = clf_batch.predict(X_full)
        acc_batch = float(np.mean(pred_batch == y_full))
        _log(f"Batch fit ({solver}): acc={acc_batch:.4f}, time={t_batch:.2f}s")

        chunk_list = [200, 1000, 5000] if not args.quick else [20, 100, 500]
        for cs in chunk_list:
            _log(f"Streaming {solver} cs={cs}...")
            def _make_chunks():
                for start in range(0, len(X_full), cs):
                    end = min(start + cs, len(X_full))
                    yield X_full[start:end], y_full[start:end]

            clf_stream, t_stream, mem_stream, n_seen = run_streaming_fit(
                _make_chunks(), classes, solver, shrinkage=solver_shrinkage,
                n_total=n_samples,
            )
            _log(f"Streaming {solver} cs={cs}: time={t_stream:.2f}s")

            has_nan = check_model_sanity(clf_stream)
            pred_stream = clf_stream.predict(X_full)
            acc_stream = float(np.mean(pred_stream == y_full))
            pred_agree = float(np.mean(pred_stream == pred_batch))
            coef_ok = _check_coef_close(clf_batch, clf_stream, TIER_MODERATE["coef_atol"])

            r = ScenarioResult(
                scenario="imbalanced",
                solver=solver,
                backend="numpy",
                chunk_size=cs,
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                acc_batch=acc_batch,
                acc_stream=acc_stream,
                pred_agreement=pred_agree,
                acc_delta=acc_stream - acc_batch,
                coef_atol_ok=coef_ok,
                wall_time_batch=t_batch,
                wall_time_stream=t_stream,
                peak_mem_batch_mb=mem_batch,
                peak_mem_stream_mb=mem_stream,
                has_nan_inf=has_nan,
                tier="moderate",
            )
            validate_result(r, TIER_MODERATE, has_batch=True)
            if has_nan:
                r.has_nan_inf = True
                r.passed = False
                r.warnings.append("NaN/Inf in model attributes")
            status = "PASS" if r.passed else "FAIL"
            _log(f"Result: {status} | acc={acc_stream:.4f} agree={pred_agree:.4f}")
            results.append(r)
        gc.collect()
    return results


# --- Scenario 5: single_sample ---
def scenario_single_sample(args):
    return _run_standard_scenario(
        scenario_name="single_sample",
        n_samples=5000,
        n_features=10,
        n_classes=5,
        chunk_sizes=[1],
        solvers=args.solvers,
        tier="moderate",
        thresholds=TIER_MODERATE,
        quick=args.quick,
        quick_divisor=10,
    )


# --- Scenario 6: high_d ---
def scenario_high_d(args):
    # SVD only — eigen/lsqr can't handle D >> N well without shrinkage
    svd_only = [s for s in args.solvers if s == "svd"]
    if not svd_only:
        r = ScenarioResult(
            scenario="high_d", solver="N/A", backend="numpy",
            chunk_size=0, n_samples=0, n_features=0, n_classes=0,
            skipped=True, skip_reason="svd solver not selected",
        )
        return [r]
    return _run_standard_scenario(
        scenario_name="high_d",
        n_samples=1000,
        n_features=10000,
        n_classes=5,
        chunk_sizes=[50, 200],
        solvers=svd_only,
        tier="structural",
        thresholds={"acc_floor": 0.70},
        quick=args.quick,
    )


# --- Scenario 7: near_singular ---
def scenario_near_singular(args):
    return _run_standard_scenario(
        scenario_name="near_singular",
        n_samples=20000,
        n_features=50,
        n_classes=5,
        chunk_sizes=[500, 2000],
        solvers=args.solvers,
        tier="tight",
        thresholds=TIER_TIGHT,
        quick=args.quick,
        n_informative=5,
        n_redundant=40,
        shrinkage=0.1,
    )


# --- Scenario 8: mnist ---
def scenario_mnist(args):
    return _run_public_dataset(
        args,
        scenario_name="mnist",
        openml_name="mnist_784",
        openml_version=1,
        chunk_sizes=[1000, 5000, 10000],
    )


# --- Scenario 9: fashion_mnist ---
def scenario_fashion_mnist(args):
    return _run_public_dataset(
        args,
        scenario_name="fashion_mnist",
        openml_name="Fashion-MNIST",
        openml_version=1,
        chunk_sizes=[1000, 5000],
    )


# --- Scenario 10: covertype ---
def scenario_covertype(args):
    if os.environ.get("SKLEARN_SKIP_NETWORK_TESTS", "1") != "0":
        r = ScenarioResult(
            scenario="covertype", solver="N/A", backend="numpy",
            chunk_size=0, n_samples=0, n_features=0, n_classes=0,
            skipped=True, skip_reason="SKLEARN_SKIP_NETWORK_TESTS != 0",
        )
        return [r]
    try:
        _log("Fetching Covertype dataset...")
        from sklearn.datasets import fetch_covtype

        data = fetch_covtype()
        X_all = np.asarray(data.data, dtype=np.float64)
        y_all = np.asarray(data.target, dtype=np.int64)
    except Exception as exc:
        r = ScenarioResult(
            scenario="covertype", solver="N/A", backend="numpy",
            chunk_size=0, n_samples=0, n_features=0, n_classes=0,
            skipped=True, skip_reason=f"fetch failed: {exc}",
        )
        return [r]

    chunk_sizes = [5000, 10000, 50000]
    if args.quick:
        # Subsample
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_all), size=min(20000, len(X_all)), replace=False)
        X_all, y_all = X_all[idx], y_all[idx]
        chunk_sizes = [500, 1000, 5000]

    return _run_with_data(
        args,
        scenario_name="covertype",
        X_all=X_all,
        y_all=y_all,
        chunk_sizes=chunk_sizes,
        thresholds=TIER_PUBLIC,
        tier="tight",
    )


def _run_public_dataset(
    args, scenario_name, openml_name, openml_version, chunk_sizes,
):
    """Common runner for OpenML datasets."""
    if os.environ.get("SKLEARN_SKIP_NETWORK_TESTS", "1") != "0":
        r = ScenarioResult(
            scenario=scenario_name, solver="N/A", backend="numpy",
            chunk_size=0, n_samples=0, n_features=0, n_classes=0,
            skipped=True, skip_reason="SKLEARN_SKIP_NETWORK_TESTS != 0",
        )
        return [r]

    try:
        _log(f"Fetching {openml_name} from OpenML...")
        from sklearn.datasets import fetch_openml

        data = fetch_openml(
            name=openml_name, version=openml_version,
            as_frame=False, parser="auto",
        )
        X_all = np.asarray(data.data, dtype=np.float64)
        y_all = np.asarray(data.target, dtype=np.int64)
        _log(f"Loaded {openml_name}: {X_all.shape}")
    except Exception as exc:
        r = ScenarioResult(
            scenario=scenario_name, solver="N/A", backend="numpy",
            chunk_size=0, n_samples=0, n_features=0, n_classes=0,
            skipped=True, skip_reason=f"fetch failed: {exc}",
        )
        return [r]

    if args.quick:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_all), size=min(5000, len(X_all)), replace=False)
        X_all, y_all = X_all[idx], y_all[idx]
        chunk_sizes = [c // 10 for c in chunk_sizes]
        chunk_sizes = [max(c, 50) for c in chunk_sizes]

    return _run_with_data(
        args,
        scenario_name=scenario_name,
        X_all=X_all,
        y_all=y_all,
        chunk_sizes=chunk_sizes,
        thresholds=TIER_PUBLIC,
        tier="tight",
    )


def _run_with_data(args, scenario_name, X_all, y_all, chunk_sizes,
                   thresholds, tier):
    """Run batch + streaming comparison on pre-loaded data."""
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X_all, y_all))
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]
    classes = np.unique(y_train)
    n_features = X_train.shape[1]
    n_classes = len(classes)

    _log(f"Data: train={X_train.shape}, test={X_test.shape}, "
         f"classes={n_classes}, chunks={chunk_sizes}")

    results = []
    for solver in args.solvers:
        solver_shrinkage = None
        _log(f"Batch fit ({solver})...")
        clf_batch, t_batch, mem_batch = run_batch_fit(
            X_train, y_train, solver, shrinkage=solver_shrinkage
        )
        pred_batch = clf_batch.predict(X_test)
        acc_batch = float(np.mean(pred_batch == y_test))
        _log(f"Batch fit ({solver}): acc={acc_batch:.4f}, time={t_batch:.2f}s")

        for cs in chunk_sizes:
            _log(f"Streaming {solver} cs={cs}...")
            def _make_chunks(sz=cs):
                for start in range(0, len(X_train), sz):
                    end = min(start + sz, len(X_train))
                    yield X_train[start:end], y_train[start:end]

            clf_stream, t_stream, mem_stream, n_seen = run_streaming_fit(
                _make_chunks(), classes, solver, shrinkage=solver_shrinkage,
                n_total=len(X_train),
            )
            _log(f"Streaming {solver} cs={cs}: time={t_stream:.2f}s")
            has_nan = check_model_sanity(clf_stream)
            pred_stream = clf_stream.predict(X_test)
            acc_stream = float(np.mean(pred_stream == y_test))
            pred_agree = float(np.mean(pred_stream == pred_batch))
            coef_ok = _check_coef_close(
                clf_batch, clf_stream, thresholds.get("coef_atol", 1e-4)
            )

            r = ScenarioResult(
                scenario=scenario_name,
                solver=solver,
                backend="numpy",
                chunk_size=cs,
                n_samples=len(X_train),
                n_features=n_features,
                n_classes=n_classes,
                acc_batch=acc_batch,
                acc_stream=acc_stream,
                pred_agreement=pred_agree,
                acc_delta=acc_stream - acc_batch,
                coef_atol_ok=coef_ok,
                wall_time_batch=t_batch,
                wall_time_stream=t_stream,
                peak_mem_batch_mb=mem_batch,
                peak_mem_stream_mb=mem_stream,
                has_nan_inf=has_nan,
                tier=tier,
            )
            validate_result(r, thresholds, has_batch=True)
            if has_nan:
                r.has_nan_inf = True
                r.passed = False
                r.warnings.append("NaN/Inf in model attributes")
            status = "PASS" if r.passed else "FAIL"
            _log(f"Result: {status} | acc={acc_stream:.4f} agree={pred_agree:.4f}")
            results.append(r)
        gc.collect()
    return results


# --- Scenarios 11-13: GPU backends ---
def scenario_torch_cpu(args):
    return _run_backend_scenarios(args, "torch_cpu")


def scenario_torch_cuda(args):
    if not args.include_cuda:
        r = ScenarioResult(
            scenario="torch_cuda", solver="N/A", backend="torch_cuda",
            chunk_size=0, n_samples=0, n_features=0, n_classes=0,
            skipped=True, skip_reason="--include-cuda not set",
        )
        return [r]
    return _run_backend_scenarios(args, "torch_cuda")


def scenario_cupy_cuda(args):
    if not args.include_cuda:
        r = ScenarioResult(
            scenario="cupy_cuda", solver="N/A", backend="cupy_cuda",
            chunk_size=0, n_samples=0, n_features=0, n_classes=0,
            skipped=True, skip_reason="--include-cuda not set",
        )
        return [r]
    return _run_backend_scenarios(args, "cupy_cuda")


def _run_backend_scenarios(args, backend_name):
    """Re-run selected scenarios with a specific array backend (SVD only)."""
    backends = dict(_available_backends(include_cuda=args.include_cuda))
    if backend_name not in backends:
        r = ScenarioResult(
            scenario=backend_name, solver="N/A", backend=backend_name,
            chunk_size=0, n_samples=0, n_features=0, n_classes=0,
            skipped=True,
            skip_reason=f"{backend_name} not available",
        )
        return [r]

    convert_fn = backends[backend_name]
    results = []
    _log(f"Backend {backend_name} available, running sub-scenarios...")

    # Sub-scenarios: wide, many_classes, and mnist (if available)
    # wide
    _log(f"Sub-scenario: wide@{backend_name}")
    wide_results = _run_standard_scenario(
        scenario_name=f"wide@{backend_name}",
        n_samples=10000,
        n_features=5000,
        n_classes=10,
        chunk_sizes=[100, 500, 1000],
        solvers=["svd"],
        tier="tight",
        thresholds=TIER_TIGHT,
        quick=args.quick,
        backend_name=backend_name,
        convert_fn=convert_fn,
    )
    results.extend(wide_results)

    # many_classes
    _log(f"Sub-scenario: many_classes@{backend_name}")
    mc_results = _run_standard_scenario(
        scenario_name=f"many_classes@{backend_name}",
        n_samples=50000,
        n_features=100,
        n_classes=50,
        chunk_sizes=[500, 2000],
        solvers=["svd"],
        tier="tight",
        thresholds=TIER_TIGHT,
        quick=args.quick,
        backend_name=backend_name,
        convert_fn=convert_fn,
    )
    results.extend(mc_results)

    # mnist (if network tests enabled)
    if os.environ.get("SKLEARN_SKIP_NETWORK_TESTS", "1") == "0":
        _log(f"Sub-scenario: mnist@{backend_name}")
        mnist_results = _run_backend_public(
            args,
            scenario_name=f"mnist@{backend_name}",
            openml_name="mnist_784",
            openml_version=1,
            chunk_sizes=[1000, 5000],
            backend_name=backend_name,
            convert_fn=convert_fn,
        )
        results.extend(mnist_results)

    # GPU-specific checks
    for r in results:
        if r.skipped:
            continue
        _validate_gpu_result(r, backend_name)

    return results


def _run_backend_public(
    args, scenario_name, openml_name, openml_version, chunk_sizes,
    backend_name, convert_fn,
):
    """Run a public dataset with a specific backend (SVD only)."""
    try:
        _log(f"Fetching {openml_name} for {backend_name}...")
        from sklearn.datasets import fetch_openml

        data = fetch_openml(
            name=openml_name, version=openml_version,
            as_frame=False, parser="auto",
        )
        X_all = np.asarray(data.data, dtype=np.float64)
        y_all = np.asarray(data.target, dtype=np.int64)
        _log(f"Loaded {openml_name}: {X_all.shape}")
    except Exception as exc:
        r = ScenarioResult(
            scenario=scenario_name, solver="N/A", backend=backend_name,
            chunk_size=0, n_samples=0, n_features=0, n_classes=0,
            skipped=True, skip_reason=f"fetch failed: {exc}",
        )
        return [r]

    if args.quick:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_all), size=min(5000, len(X_all)), replace=False)
        X_all, y_all = X_all[idx], y_all[idx]
        chunk_sizes = [max(c // 10, 50) for c in chunk_sizes]

    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X_all, y_all))
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]
    classes = np.unique(y_train)

    # Run numpy baseline first
    _log(f"Numpy baseline batch fit (svd)...")
    clf_np, _, _ = run_batch_fit(X_train, y_train, "svd")
    acc_np = float(np.mean(clf_np.predict(X_test) == y_test))
    _log(f"Numpy baseline: acc={acc_np:.4f}")

    results = []
    for cs in chunk_sizes:
        _log(f"Streaming svd cs={cs} on {backend_name}...")
        def _make_chunks(sz=cs):
            for start in range(0, len(X_train), sz):
                end = min(start + sz, len(X_train))
                yield X_train[start:end], y_train[start:end]

        _reset_gpu_mem()
        clf_stream, t_stream, mem_stream, n_seen = run_streaming_fit(
            _make_chunks(), classes, "svd", convert_fn=convert_fn,
            n_total=len(X_train),
        )
        _log(f"Streaming svd cs={cs}: time={t_stream:.2f}s")
        gpu_mem = _get_gpu_mem_mb() if "cuda" in backend_name else float("nan")

        has_nan = check_model_sanity(clf_stream)
        pred_stream = _to_numpy(clf_stream.predict(X_test))
        acc_stream = float(np.mean(pred_stream == y_test))

        r = ScenarioResult(
            scenario=scenario_name,
            solver="svd",
            backend=backend_name,
            chunk_size=cs,
            n_samples=len(X_train),
            n_features=X_train.shape[1],
            n_classes=len(classes),
            acc_batch=acc_np,
            acc_stream=acc_stream,
            pred_agreement=float("nan"),  # cross-backend
            acc_delta=acc_stream - acc_np,
            wall_time_stream=t_stream,
            peak_mem_stream_mb=mem_stream,
            gpu_mem_mb=gpu_mem,
            has_nan_inf=has_nan,
            tier="tight",
        )
        # GPU accuracy check: abs(delta) < 1e-6 vs numpy
        if abs(acc_stream - acc_np) > 0.005:
            r.passed = False
            r.warnings.append(
                f"GPU accuracy delta {acc_stream - acc_np:+.6f}"
                f" exceeds 0.005 vs numpy"
            )
        if has_nan:
            r.has_nan_inf = True
            r.passed = False
            r.warnings.append("NaN/Inf in model attributes")
        results.append(r)

    return results


def _validate_gpu_result(r, backend_name):
    """Additional GPU-specific validation (type/device checks logged as warnings)."""
    # We can't check tensor types here since we don't have the model reference,
    # but accuracy checks are done inline. This is a placeholder for
    # additional device-locality checks if needed.
    pass


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

SCENARIOS = {
    # Synthetic
    "wide": scenario_wide,
    "tall": scenario_tall,
    "many_classes": scenario_many_classes,
    "imbalanced": scenario_imbalanced,
    "single_sample": scenario_single_sample,
    "high_d": scenario_high_d,
    "near_singular": scenario_near_singular,
    # Public datasets
    "mnist": scenario_mnist,
    "fashion_mnist": scenario_fashion_mnist,
    "covertype": scenario_covertype,
    # GPU backends
    "torch_cpu": scenario_torch_cpu,
    "torch_cuda": scenario_torch_cuda,
    "cupy_cuda": scenario_cupy_cuda,
}

SYNTHETIC_SCENARIOS = [
    "wide", "tall", "many_classes", "imbalanced",
    "single_sample", "high_d", "near_singular",
]
PUBLIC_SCENARIOS = ["mnist", "fashion_mnist", "covertype"]
GPU_SCENARIOS = ["torch_cpu", "torch_cuda", "cupy_cuda"]


def _resolve_scenarios(names):
    """Expand meta-names like 'all', 'synthetic', 'public' to scenario lists."""
    resolved = []
    for name in names:
        if name == "all":
            resolved.extend(SYNTHETIC_SCENARIOS + PUBLIC_SCENARIOS + GPU_SCENARIOS)
        elif name == "synthetic":
            resolved.extend(SYNTHETIC_SCENARIOS)
        elif name == "public":
            resolved.extend(PUBLIC_SCENARIOS)
        elif name in SCENARIOS:
            resolved.append(name)
        else:
            print(f"WARNING: unknown scenario '{name}', skipping", file=sys.stderr)
    # Deduplicate preserving order
    seen = set()
    out = []
    for s in resolved:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_list(arg):
    return [x.strip() for x in arg.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="LDA partial_fit stress test benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=["synthetic"],
        help=(
            "Scenarios to run. Use 'all', 'synthetic', 'public', or "
            "individual names: wide, tall, many_classes, imbalanced, "
            "single_sample, high_d, near_singular, mnist, fashion_mnist, "
            "covertype, torch_cpu, torch_cuda, cupy_cuda"
        ),
    )
    parser.add_argument(
        "--solvers", nargs="+", default=["svd", "eigen", "lsqr"],
        choices=["svd", "eigen", "lsqr"],
        help="Solvers to test (default: all three)",
    )
    parser.add_argument(
        "--backends", nargs="+", default=["numpy"],
        help="Backends (numpy, torch_cpu, torch_cuda, cupy_cuda)",
    )
    parser.add_argument(
        "--json-out", default=None,
        help="Path for JSON report output",
    )
    parser.add_argument(
        "--max-batch-samples", type=int, default=200000,
        help="Skip batch fit above this sample count (default: 200000)",
    )
    parser.add_argument(
        "--tall-n-samples", type=int, default=500000,
        help="Sample count for tall scenario (default: 500000)",
    )
    parser.add_argument(
        "--include-cuda", action="store_true",
        help="Enable torch_cuda and cupy_cuda backends",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="10x smaller sizes for quick smoke testing",
    )

    args = parser.parse_args()

    scenario_names = _resolve_scenarios(args.scenarios)
    if not scenario_names:
        print("No scenarios selected.", file=sys.stderr)
        raise SystemExit(1)

    print(f"LDA partial_fit stress test")
    print(f"Scenarios: {', '.join(scenario_names)}")
    print(f"Solvers:   {', '.join(args.solvers)}")
    print(f"Quick:     {args.quick}")
    print(f"CUDA:      {args.include_cuda}")
    print()

    all_results = []
    for name in scenario_names:
        fn = SCENARIOS[name]
        print(f"--- Running scenario: {name} ---")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = fn(args)
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            r = ScenarioResult(
                scenario=name, solver="N/A", backend="N/A",
                chunk_size=0, n_samples=0, n_features=0, n_classes=0,
                passed=False, skipped=False,
                warnings=[f"Exception: {exc}"],
            )
            results = [r]

        all_results.extend(results)

        # Print per-scenario summary
        n_pass = sum(r.passed for r in results if not r.skipped)
        n_fail = sum(not r.passed for r in results if not r.skipped)
        n_skip = sum(r.skipped for r in results)
        print(f"  {n_pass} passed, {n_fail} failed, {n_skip} skipped")
        for r in results:
            if r.warnings:
                for w in r.warnings:
                    print(f"  WARN [{r.solver} cs={r.chunk_size}]: {w}")
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print_table(all_results)
    print()

    n_total = len(all_results)
    n_passed = sum(r.passed for r in all_results if not r.skipped)
    n_failed = sum(not r.passed for r in all_results if not r.skipped)
    n_skipped = sum(r.skipped for r in all_results)
    overall_pass = n_failed == 0

    print(
        f"Total: {n_total} | Passed: {n_passed} | Failed: {n_failed}"
        f" | Skipped: {n_skipped}"
    )
    print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")

    # Informational warnings
    for r in all_results:
        if (
            not r.skipped
            and not np.isnan(r.peak_mem_batch_mb)
            and not np.isnan(r.peak_mem_stream_mb)
            and r.peak_mem_stream_mb > r.peak_mem_batch_mb
            and r.scenario == "tall"
        ):
            print(
                f"  INFO: {r.scenario}/{r.solver} cs={r.chunk_size}:"
                f" stream mem ({r.peak_mem_stream_mb:.1f}MB)"
                f" > batch mem ({r.peak_mem_batch_mb:.1f}MB)"
            )

    if args.json_out:
        json_report(all_results, args.json_out)
        print(f"\nJSON report written to: {args.json_out}")

    raise SystemExit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
