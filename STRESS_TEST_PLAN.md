# Plan: LDA partial_fit Stress Test Suite

## Context

We have `partial_fit` for `LinearDiscriminantAnalysis` across all 3 solvers (eigen, lsqr, svd). Existing unit tests (29 tests) cover correctness and edge cases at small scale. Existing benchmarks cover parity validation (synthetic/digits/MNIST) and numpy-vs-torch comparison on Olivetti Faces. **What's missing**: large-scale stress testing, larger-than-memory simulation, performance profiling across the full solver/backend matrix, and edge cases at scale (extreme imbalance, many classes, high dimensionality).

## Deliverable

**New file: `benchmarks/bench_lda_partial_fit_stress.py`** (~850-900 lines)

A standalone benchmark script following existing sklearn conventions (argparse, JSON output, exit code 0/1).

---

## Architecture

```
Infrastructure:  PerfTimer, MemTracker, ScenarioResult dataclass, print_table(), json_report()
Generators:      chunked_make_classification(), chunked_imbalanced(), load_openml_safe()
Core helpers:    run_batch_fit(), run_streaming_fit(), run_streaming_fit_gpu(), check_model_sanity()
Scenarios:       13 scenario functions, each returning list[ScenarioResult]
Harness:         main() with argparse, scenario dispatch, aggregation
```

## Scenarios

### Synthetic (generated in real-time)

| # | Name | n_samples | n_features | n_classes | chunk_sizes | Solvers | Purpose |
|---|------|-----------|------------|-----------|-------------|---------|---------|
| 1 | `wide` | 10,000 | 5,000 | 10 | [100, 500, 1000] | svd, eigen, lsqr | D >> N/chunk, rank-deficiency per chunk |
| 2 | `tall` | 500,000 | 50 | 10 | [1000, 5000, 10000] | svd, eigen, lsqr | Larger-than-memory sim (chunked generator) |
| 3 | `many_classes` | 50,000 | 100 | 50 | [500, 2000] | svd, eigen, lsqr | Between-class scatter at scale |
| 4 | `imbalanced` | 20,000 | 20 | 10 | [200, 1000, 5000] | svd, eigen, lsqr | 95%/5% class 0 dominance |
| 5 | `single_sample` | 5,000 | 10 | 5 | [1] | svd, eigen, lsqr | Worst-case incremental stress |
| 6 | `high_d` | 1,000 | 10,000 | 5 | [50, 200] | svd only | N << D face-like data |
| 7 | `near_singular` | 20,000 | 50 | 5 | [500, 2000] | all (shrinkage=0.1 for eigen/lsqr) | 5 informative / 40 redundant features |

### Public Datasets (gated by `SKLEARN_SKIP_NETWORK_TESTS=0`)

| # | Name | Source | n_samples | n_features | n_classes | chunk_sizes |
|---|------|--------|-----------|------------|-----------|-------------|
| 8 | `mnist` | `fetch_openml("mnist_784")` | 70k | 784 | 10 | [1000, 5000, 10000] |
| 9 | `fashion_mnist` | `fetch_openml("Fashion-MNIST")` | 70k | 784 | 10 | [1000, 5000] |
| 10 | `covertype` | `fetch_covtype()` | 581k | 54 | 7 | [5000, 10000, 50000] |

### Array API / GPU Backends

| # | Name | Description |
|---|------|-------------|
| 11 | `torch_cpu` | Re-run scenarios 1 (wide), 3 (many_classes), 8 (mnist) with torch CPU tensors, SVD only |
| 12 | `torch_cuda` | Same scenarios on torch CUDA (x86_64 + NVIDIA GPU), SVD only |
| 13 | `cupy_cuda` | Same scenarios using CuPy arrays on CUDA, SVD only |

All GPU scenarios gated by hardware availability checks at runtime.

---

## Key Design Decisions

### Larger-than-memory simulation (Scenario 2: `tall`)
- Python generator yields `(X_chunk, y_chunk)` tuples — full dataset never materialized
- Each chunk generated via `make_classification` with deterministic per-chunk seed
- For batch baseline: only materialize if `n_samples <= --max-batch-samples` (default 200k)
- For larger sizes: generate a separate held-out test set (10k), validate absolute accuracy only

### Performance capture
- **Timing**: `time.perf_counter()` context manager
- **CPU Memory**: `tracemalloc` peak tracking (Python-level; documented limitation for C allocations)
- **GPU Memory (torch CUDA)**: `torch.cuda.max_memory_allocated()` — reset before each run, read after
- **GPU Memory (CuPy)**: `cupy.get_default_memory_pool().used_bytes()` — snapshot before/after
- Each scenario captures: `wall_time_batch`, `wall_time_stream`, `peak_mem_batch_mb`, `peak_mem_stream_mb`, `gpu_mem_mb` (when applicable)

### GPU / Array API integration

Three GPU backends supported (all SVD solver only — eigen/lsqr convert to numpy internally):

**Torch CPU/CUDA:**
- Gated by `torch` importability + `SCIPY_ARRAY_API=1`
- CUDA gated by `torch.cuda.is_available()` + `--include-cuda`
- Data converted via `torch.from_numpy(X).to(device)` where device is `"cpu"` or `"cuda"`
- Validates: results are `torch.Tensor` on correct device, accuracy matches numpy within 0.5%

**CuPy CUDA:**
- Gated by `cupy` importability + `SCIPY_ARRAY_API=1` + `--include-cuda`
- CuPy detected via sklearn's existing `yield_namespaces()` which includes `"cupy"`
- Data converted via `cupy.asarray(X)` (always float64)
- Validates: results are `cupy.ndarray`, accuracy matches numpy within 0.5%
- CuPy uses its own CUDA runtime — no torch dependency needed

**Backend availability helper:**
```python
def _available_backends(include_cuda=False):
    """Return list of (backend_name, converter_fn) tuples for available backends."""
    backends = [("numpy", lambda X: X)]
    try:
        import torch
        if os.environ.get("SCIPY_ARRAY_API") == "1":
            backends.append(("torch_cpu", lambda X: torch.from_numpy(X.copy())))
            if include_cuda and torch.cuda.is_available():
                backends.append(("torch_cuda", lambda X: torch.from_numpy(X.copy()).cuda()))
    except ImportError:
        pass
    try:
        import cupy
        if os.environ.get("SCIPY_ARRAY_API") == "1" and include_cuda:
            backends.append(("cupy_cuda", lambda X: cupy.asarray(X)))
    except ImportError:
        pass
    return backends
```

### CLI interface
```
--scenarios all|synthetic|public|wide|tall|...   (select scenarios)
--solvers svd eigen lsqr                         (select solvers)
--backends numpy|torch_cpu|torch_cuda|cupy_cuda  (select backends, default: numpy)
--json-out path                                  (JSON report file)
--max-batch-samples 200000                       (skip batch above this)
--tall-n-samples 500000                          (configurable tall size)
--include-cuda                                   (enable torch_cuda + cupy_cuda)
--quick                                          (10x smaller sizes for CI)
```

Note: `--include-cuda` is a convenience flag that adds `torch_cuda` and `cupy_cuda` to the backend list. Backends can also be selected individually via `--backends`.

## Validation Criteria

Thresholds aligned with existing unit tests (`atol=1e-5` on coef_/intercept_) and existing benchmarks (0.999 agreement for digits/MNIST).

### Hard checks (all scenarios)

| Check | Threshold | Notes |
|-------|-----------|-------|
| No NaN/Inf in `coef_`, `intercept_`, `scalings_` | Hard fail | |
| Streaming accuracy floor | `> 0.70` for well-separated synthetic, `> 0.75` for public datasets | Existing benchmarks use 0.75 floor |

### Batch-vs-stream parity (when batch baseline exists)

Tiered by scenario difficulty:

| Tier | Scenarios | Prediction agreement | Accuracy delta | `coef_` atol | Rationale |
|------|-----------|---------------------|----------------|-------------|-----------|
| **Tight** | well-conditioned synthetic (wide, tall, many_classes, near_singular), public datasets (mnist, fashion, covertype) | `>= 0.999` | `>= -0.005` | `1e-4` | Same data, same algorithm — streaming should nearly match batch. Existing benchmarks require 0.999 for digits/MNIST. |
| **Moderate** | imbalanced, single_sample | `>= 0.99` | `>= -0.01` | `1e-3` | Extreme chunking (chunk_size=1) or severe imbalance can amplify floating-point drift. Existing synthetic benchmark uses -0.01 delta. |
| **Structural** | high_d (no batch baseline for eigen/lsqr) | N/A | N/A | N/A | Only SVD runs; validate accuracy > 0.70 and no NaN/Inf. |

### GPU-specific checks

| Check | Threshold | When |
|-------|-----------|------|
| State tensors are `torch.Tensor` on correct device | Type + device check | Torch scenarios |
| State arrays are `cupy.ndarray` | Type check | CuPy scenarios |
| GPU accuracy matches numpy baseline | `abs(delta) < 1e-6` | All GPU scenarios — same solver, same precision, should be near-identical |
| GPU tensors remain on-device (no silent CPU fallback) | Device check | All GPU scenarios |

### Informational (warn, don't fail)

| Check | Notes |
|-------|-------|
| Peak memory stream < batch | Tall streaming scenario — directional check, tracemalloc has limitations |
| Wall-clock timing regression | Captured for tracking, no hard threshold |

Exit code: `0` if all non-skipped pass, `1` otherwise.

## Files to Create/Modify

- **Create**: `benchmarks/bench_lda_partial_fit_stress.py`
- **No modifications** to existing files

## Reference Files
- `benchmarks/bench_lda_svd_partial_fit_parity.py` — pattern for argparse, JSON output, exit code
- `output/benchmark_numpy_vs_torch_cpu.py` — pattern for torch + tracemalloc benchmarking
- `sklearn/discriminant_analysis.py` — implementation under test
- `sklearn/tests/test_discriminant_analysis.py` — existing test patterns to avoid duplication

## Verification

```bash
# Run all synthetic scenarios (no network needed)
python benchmarks/bench_lda_partial_fit_stress.py --scenarios synthetic

# Quick smoke test (reduced sizes)
python benchmarks/bench_lda_partial_fit_stress.py --quick

# Full run with public datasets
SKLEARN_SKIP_NETWORK_TESTS=0 python benchmarks/bench_lda_partial_fit_stress.py --scenarios all

# Torch CPU backend
SCIPY_ARRAY_API=1 python benchmarks/bench_lda_partial_fit_stress.py --scenarios torch_cpu

# Torch CUDA + CuPy CUDA (x86_64 with NVIDIA GPU)
SCIPY_ARRAY_API=1 python benchmarks/bench_lda_partial_fit_stress.py --scenarios torch_cuda cupy_cuda --include-cuda

# All backends including GPU
SCIPY_ARRAY_API=1 python benchmarks/bench_lda_partial_fit_stress.py --scenarios all --include-cuda

# JSON output for CI
python benchmarks/bench_lda_partial_fit_stress.py --json-out results.json --scenarios synthetic
```
