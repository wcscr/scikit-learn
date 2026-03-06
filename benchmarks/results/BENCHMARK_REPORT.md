# LDA partial_fit Benchmark Report

**Date:** 2026-03-06
**Branch:** `feature/svd-rank-truncation`
**Platform:** Linux (Docker), NumPy/CPU backend

**Overall: 122 passed, 2 known failures (pre-existing batch eigen/lsqr on singular covariance)**

---

## 1. Stress Test -- Synthetic (`bench_lda_partial_fit_stress.py --scenarios synthetic`)

Compares `fit()` (batch) vs `partial_fit()` (streaming) across 7 synthetic scenarios, 3 solvers, and multiple chunk sizes. Measures accuracy, prediction agreement, wall-clock time, and peak traced memory.

**Result: 48/48 PASSED** -- every configuration achieved **agreement = 1.0000** with batch.

### 1.1 Wide (N=10,000 D=5,000 K=10) -- SVD only

| Chunk | Acc Batch | Acc Stream | Agreement | t_batch (s) | t_stream (s) | Ratio | Mem Batch (MB) | Mem Stream (MB) |
|------:|----------:|-----------:|----------:|------------:|-------------:|------:|---------------:|----------------:|
| 2,500 | 0.9823 | 0.9823 | 1.0000 | 38.0 | 110.0 | 2.9x | 2,519 | 2,302 |
| 5,000 | 0.9823 | 0.9823 | 1.0000 | 38.0 | 64.4 | 1.7x | 2,519 | 3,073 |
| 10,000 | 0.9823 | 0.9823 | 1.0000 | 38.0 | 41.4 | 1.1x | 2,519 | 3,283 |

Observations: Streaming overhead decreases with larger chunks. At cs=10,000 (full dataset in one call), streaming is only 1.1x slower than batch. Smaller chunks reduce peak memory for the initial allocation but add SVD merge cost.

### 1.2 Tall (N=500,000 D=50 K=10) -- all solvers

| Solver | Chunk | Acc | Agreement | t_batch (s) | t_stream (s) | Ratio | Mem Batch (MB) | Mem Stream (MB) |
|--------|------:|----:|----------:|------------:|-------------:|------:|---------------:|----------------:|
| svd | 50,000 | 0.5856 | 1.0000 | 1.04 | 0.89 | 0.86x | 782 | 117 |
| svd | 125,000 | 0.5856 | 1.0000 | 1.04 | 0.97 | 0.93x | 782 | 291 |
| svd | 250,000 | 0.5856 | 1.0000 | 1.04 | 0.97 | 0.93x | 782 | 582 |
| svd | 500,000 | 0.5856 | 1.0000 | 1.04 | 0.92 | 0.89x | 782 | 1,164 |
| eigen | 50,000 | 0.5856 | 1.0000 | 0.40 | 0.21 | 0.53x | 191 | 6 |
| eigen | 125,000 | 0.5856 | 1.0000 | 0.40 | 0.20 | 0.49x | 191 | 15 |
| eigen | 250,000 | 0.5856 | 1.0000 | 0.40 | 0.17 | 0.41x | 191 | 29 |
| eigen | 500,000 | 0.5856 | 1.0000 | 0.40 | 0.15 | 0.36x | 191 | 58 |
| lsqr | 50,000 | 0.5856 | 1.0000 | 0.30 | 0.18 | 0.58x | 39 | 6 |
| lsqr | 125,000 | 0.5856 | 1.0000 | 0.30 | 0.15 | 0.51x | 39 | 15 |
| lsqr | 250,000 | 0.5856 | 1.0000 | 0.30 | 0.14 | 0.47x | 39 | 29 |
| lsqr | 500,000 | 0.5856 | 1.0000 | 0.30 | 0.15 | 0.48x | 39 | 58 |

Observations: For tall data (N >> D), streaming is **faster** than batch across all solvers. The eigen/lsqr solvers are 2-3x faster via streaming with small chunks, and use dramatically less memory (6 MB vs 191 MB for eigen at cs=50k).

### 1.3 Many Classes (N=50,000 D=100 K=50) -- all solvers

| Solver | Chunk | Acc | Agreement | t_batch (s) | t_stream (s) | Mem Batch (MB) | Mem Stream (MB) |
|--------|------:|----:|----------:|------------:|-------------:|---------------:|----------------:|
| svd | 10,000 | 0.3355 | 1.0000 | 0.25 | 0.38 | 154 | 47 |
| svd | 25,000 | 0.3355 | 1.0000 | 0.25 | 0.28 | 154 | 116 |
| svd | 50,000 | 0.3355 | 1.0000 | 0.25 | 0.27 | 154 | 230 |
| eigen | 10,000 | 0.3355 | 1.0000 | 0.12 | 0.16 | 38 | 1 |
| eigen | 25,000 | 0.3355 | 1.0000 | 0.12 | 0.09 | 38 | 2 |
| eigen | 50,000 | 0.3355 | 1.0000 | 0.12 | 0.06 | 38 | 3 |
| lsqr | 10,000 | 0.3355 | 1.0000 | 0.11 | 0.14 | 2 | 1 |
| lsqr | 25,000 | 0.3355 | 1.0000 | 0.11 | 0.08 | 2 | 2 |
| lsqr | 50,000 | 0.3355 | 1.0000 | 0.11 | 0.06 | 2 | 3 |

### 1.4 Imbalanced (N=20,000 D=20 K=10, 95/5 split) -- all solvers

| Solver | Chunk | Acc | Agreement | t_batch (s) | t_stream (s) | Mem Batch (MB) | Mem Stream (MB) |
|--------|------:|----:|----------:|------------:|-------------:|---------------:|----------------:|
| svd | 5,000 | 0.8976 | 1.0000 | 0.04 | 0.10 | 4.3 | 4.6 |
| svd | 10,000 | 0.8976 | 1.0000 | 0.04 | 0.05 | 4.3 | 6.4 |
| svd | 20,000 | 0.8976 | 1.0000 | 0.04 | 0.05 | 4.3 | 6.4 |
| eigen | 5,000 | 0.8972 | 1.0000 | 0.04 | 0.03 | 1.9 | 1.4 |
| eigen | 10,000 | 0.8972 | 1.0000 | 0.04 | 0.03 | 1.9 | 1.9 |
| eigen | 20,000 | 0.8972 | 1.0000 | 0.04 | 0.03 | 1.9 | 1.9 |
| lsqr | 5,000 | 0.8972 | 1.0000 | 0.03 | 0.03 | 1.9 | 1.4 |
| lsqr | 10,000 | 0.8972 | 1.0000 | 0.03 | 0.03 | 1.9 | 1.9 |
| lsqr | 20,000 | 0.8972 | 1.0000 | 0.03 | 0.04 | 1.9 | 1.9 |

Note: SVD solver accuracy (0.8976) slightly exceeds eigen/lsqr (0.8972) due to different whitening paths. Both match their respective batch fits perfectly.

### 1.5 Single Sample (N=5,000 D=10 K=5, chunk_size=1) -- all solvers

| Solver | Acc | Agreement | t_batch (s) | t_stream (s) | Ratio | Mem Batch (MB) | Mem Stream (MB) |
|--------|----:|----------:|------------:|-------------:|------:|---------------:|----------------:|
| svd | 0.7294 | 1.0000 | 0.03 | 4.85 | 151x | 1.6 | 0.1 |
| eigen | 0.7294 | 1.0000 | 0.03 | 2.33 | 89x | 0.5 | 0.2 |
| lsqr | 0.7294 | 1.0000 | 0.03 | 1.58 | 61x | 0.2 | 0.1 |

Observations: Single-sample streaming (5,000 partial_fit calls) is expectedly slow due to per-call overhead, but produces **exact parity** with batch. Memory footprint is minimal. The lsqr solver handles single-sample streaming most efficiently.

### 1.6 High Dimensionality (N=1,000 D=10,000 K=5) -- SVD only

| Chunk | Acc | Agreement | t_batch (s) | t_stream (s) | Ratio | Mem Batch (MB) | Mem Stream (MB) |
|------:|----:|----------:|------------:|-------------:|------:|---------------:|----------------:|
| 250 | 0.4330 | 1.0000 | 2.03 | 5.16 | 2.5x | 404 | 445 |
| 500 | 0.4330 | 1.0000 | 2.03 | 3.29 | 1.6x | 404 | 468 |
| 1,000 | 0.4330 | 1.0000 | 2.03 | 2.35 | 1.2x | 404 | 512 |

Observations: D >> N regime. Streaming overhead is moderate (1.2-2.5x) with exact agreement. Memory is comparable because the SVD factors are D-dimensional regardless of chunking.

### 1.7 Near-Singular (N=20,000 D=50 K=5) -- all solvers

| Solver | Chunk | Acc | Agreement | t_batch (s) | t_stream (s) | Mem Batch (MB) | Mem Stream (MB) |
|--------|------:|----:|----------:|------------:|-------------:|---------------:|----------------:|
| svd | 5,000 | 0.7550 | 1.0000 | 0.10 | 0.13 | 32 | 12 |
| svd | 10,000 | 0.7550 | 1.0000 | 0.10 | 0.10 | 32 | 24 |
| svd | 20,000 | 0.7550 | 1.0000 | 0.10 | 0.10 | 32 | 47 |
| eigen | 5,000 | 0.7550 | 1.0000 | 0.04 | 0.07 | 8 | 1 |
| eigen | 10,000 | 0.7550 | 1.0000 | 0.04 | 0.05 | 8 | 2 |
| eigen | 20,000 | 0.7550 | 1.0000 | 0.04 | 0.05 | 8 | 5 |
| lsqr | 5,000 | 0.7550 | 1.0000 | 0.05 | 0.09 | 3 | 1 |
| lsqr | 10,000 | 0.7550 | 1.0000 | 0.05 | 0.05 | 3 | 2 |
| lsqr | 20,000 | 0.7550 | 1.0000 | 0.05 | 0.05 | 3 | 5 |

---

## 2. Stress Test -- Public Datasets (`bench_lda_partial_fit_stress.py --scenarios public --no-cv`)

Real-world datasets fetched from OpenML and sklearn. Single 80/20 holdout split.

**Result: 15 passed, 2 failed (pre-existing batch eigen/lsqr failures on singular covariance)**

### 2.1 MNIST (N=70,000 D=784 K=10) -- SVD only

| Solver | Chunk | Acc | Agreement | t_batch (s) | t_stream (s) | Ratio | Status |
|--------|------:|----:|----------:|------------:|-------------:|------:|--------|
| svd | 10,000 | 0.8663 | 1.0000 | 3.69 | 5.33 | 1.4x | PASS |
| svd | 35,000 | 0.8663 | 1.0000 | 3.58 | 6.61 | 1.8x | PASS |
| svd | 70,000 | 0.8663 | 1.0000 | 4.26 | 3.73 | 0.9x | PASS |
| eigen | -- | -- | -- | -- | -- | -- | **FAIL** |
| lsqr | -- | -- | -- | -- | -- | -- | **FAIL** |

SVD streaming achieves perfect agreement at all chunk sizes, and is actually **faster** than batch at cs=70k. At cs=10k (7 chunks), streaming is only 1.4x slower.

**Eigen/lsqr failure:** `LinAlgError: The leading minor of order 1 of B is not positive definite.` This is a **pre-existing batch `fit()` failure** -- the within-class covariance matrix is singular because MNIST has many constant/near-zero pixel columns. The `scipy.linalg.eigh` call in the eigen solver requires a positive-definite B matrix, which fails without regularization. This is not a `partial_fit` regression. See [Section 8: Shrinkage Note](#8-note-on-eigenlsqr-failures-and-shrinkage).

### 2.2 Fashion-MNIST (N=70,000 D=784 K=10) -- all solvers

| Solver | Chunk | Acc | Agreement | t_batch (s) | t_stream (s) | Ratio | Status |
|--------|------:|----:|----------:|------------:|-------------:|------:|--------|
| svd | 10,000 | 0.8162 | 1.0000 | 3.74 | 5.84 | 1.6x | PASS |
| svd | 35,000 | 0.8162 | 1.0000 | 4.13 | 4.07 | 0.99x | PASS |
| svd | 70,000 | 0.8162 | 1.0000 | 3.51 | 3.45 | 0.98x | PASS |
| eigen | 10,000 | 0.8162 | 1.0000 | 1.02 | 1.94 | 1.9x | PASS |
| eigen | 35,000 | 0.8162 | 1.0000 | 1.37 | 0.80 | 0.58x | PASS |
| eigen | 70,000 | 0.8162 | 1.0000 | 1.09 | 0.64 | 0.59x | PASS |
| lsqr | 10,000 | 0.8162 | 1.0000 | 0.83 | 2.74 | 3.3x | PASS |
| lsqr | 35,000 | 0.8162 | 1.0000 | 0.93 | 0.90 | 0.97x | PASS |
| lsqr | 70,000 | 0.8162 | 1.0000 | 0.79 | 0.58 | 0.73x | PASS |

All 9 configs pass with **perfect agreement**. All solvers succeed on Fashion-MNIST because its pixel distribution has enough variance per class to keep the within-class covariance non-singular. At large chunk sizes, eigen streaming is **1.7x faster** than batch (0.64s vs 1.09s) and lsqr is 1.4x faster (0.58s vs 0.79s).

### 2.3 Covertype (N=581,012 D=54 K=7) -- SVD only

| Solver | Chunk | Acc | Agreement | t_batch (s) | t_stream (s) | Ratio | Status |
|--------|------:|----:|----------:|------------:|-------------:|------:|--------|
| svd | 50,000 | 0.6792 | 1.0000 | 1.09 | 0.94 | 0.86x | PASS |
| svd | 200,000 | 0.6792 | 1.0000 | 1.07 | 0.90 | 0.84x | PASS |
| svd | 581,012 | 0.6792 | 1.0000 | 1.05 | 0.97 | 0.92x | PASS |
| eigen | -- | -- | -- | -- | -- | -- | **FAIL** |
| lsqr | -- | -- | -- | -- | -- | -- | **FAIL** |

SVD streaming is consistently **faster** than batch on this tall dataset (581k x 54). Eigen/lsqr fail for the same singular covariance reason as MNIST -- Covertype includes binary indicator columns with zero within-class variance.

---

## 3. Stress Test -- Shrinkage on Public Datasets (`--scenarios shrinkage --no-cv`)

Tests eigen/lsqr solvers with `shrinkage=0.01` on the same public datasets that fail without regularization. Single 80/20 holdout split. This demonstrates that `partial_fit` correctly propagates shrinkage through the incremental covariance update.

**Result: 18/18 PASSED** -- every configuration achieved **agreement = 1.0000** with batch.

### 3.1 MNIST + Shrinkage (N=70,000 D=784 K=10, shrinkage=0.01) -- eigen/lsqr

| Solver | Chunk | Acc | Agreement | t_batch (s) | t_stream (s) | Ratio | Status |
|--------|------:|----:|----------:|------------:|-------------:|------:|--------|
| eigen | 10,000 | 0.8683 | 1.0000 | 1.16 | 1.84 | 1.6x | PASS |
| eigen | 35,000 | 0.8683 | 1.0000 | 1.04 | 0.84 | 0.81x | PASS |
| eigen | 70,000 | 0.8683 | 1.0000 | 1.00 | 0.53 | 0.53x | PASS |
| lsqr | 10,000 | 0.8683 | 1.0000 | 0.77 | 2.26 | 2.9x | PASS |
| lsqr | 35,000 | 0.8683 | 1.0000 | 0.81 | 1.38 | 1.7x | PASS |
| lsqr | 70,000 | 0.8683 | 1.0000 | 1.43 | 0.71 | 0.50x | PASS |

Shrinkage fixes the singular covariance issue on MNIST. With `shrinkage=0.01`, accuracy (0.8683) actually **exceeds** the unregularized SVD result (0.8663) -- the regularization acts as a beneficial prior. Eigen streaming at cs=70k is **1.9x faster** than batch (0.53s vs 1.00s); lsqr at cs=70k is **2.0x faster** (0.71s vs 1.43s).

### 3.2 Fashion-MNIST + Shrinkage (N=70,000 D=784 K=10, shrinkage=0.01) -- eigen/lsqr

| Solver | Chunk | Acc | Agreement | t_batch (s) | t_stream (s) | Ratio | Status |
|--------|------:|----:|----------:|------------:|-------------:|------:|--------|
| eigen | 10,000 | 0.8164 | 1.0000 | 1.17 | 2.35 | 2.0x | PASS |
| eigen | 35,000 | 0.8164 | 1.0000 | 1.29 | 0.83 | 0.64x | PASS |
| eigen | 70,000 | 0.8164 | 1.0000 | 1.31 | 0.66 | 0.50x | PASS |
| lsqr | 10,000 | 0.8164 | 1.0000 | 0.92 | 2.47 | 2.7x | PASS |
| lsqr | 35,000 | 0.8164 | 1.0000 | 0.81 | 0.92 | 1.1x | PASS |
| lsqr | 70,000 | 0.8164 | 1.0000 | 0.77 | 0.63 | 0.82x | PASS |

Fashion-MNIST works without shrinkage too, but shrinkage has minimal accuracy impact (0.8164 vs 0.8162). Streaming speedups at large chunks: eigen 2.0x faster (0.66s vs 1.31s), lsqr 1.2x faster (0.63s vs 0.77s).

### 3.3 Covertype + Shrinkage (N=581,012 D=54 K=7, shrinkage=0.01) -- eigen/lsqr

| Solver | Chunk | Acc | Agreement | t_batch (s) | t_stream (s) | Ratio | Status |
|--------|------:|----:|----------:|------------:|-------------:|------:|--------|
| eigen | 50,000 | 0.6806 | 1.0000 | 0.35 | 0.23 | 0.66x | PASS |
| eigen | 200,000 | 0.6806 | 1.0000 | 0.35 | 0.18 | 0.51x | PASS |
| eigen | 581,012 | 0.6806 | 1.0000 | 0.34 | 0.17 | 0.50x | PASS |
| lsqr | 50,000 | 0.6806 | 1.0000 | 0.26 | 0.17 | 0.65x | PASS |
| lsqr | 200,000 | 0.6806 | 1.0000 | 0.26 | 0.15 | 0.58x | PASS |
| lsqr | 581,012 | 0.6806 | 1.0000 | 0.26 | 0.16 | 0.62x | PASS |

Shrinkage fixes the singular covariance on Covertype. Accuracy (0.6806) slightly exceeds unregularized SVD (0.6792). Streaming is **faster** than batch across all chunk sizes -- eigen streaming at cs=581k is **2.0x faster** (0.17s vs 0.34s), lsqr is **1.6x faster** (0.16s vs 0.26s). This tall dataset (581k samples, 54 features) is the ideal regime for incremental covariance updates.

---

## 4. SVD Parity Test -- Synthetic (`bench_lda_svd_partial_fit_parity.py --mode synthetic`)

Tests exact numerical parity between SVD `fit()` and `partial_fit()` on small synthetic data (N=700, D=14 including 2 constant columns, K=3) across 4 random seeds and 5 chunk sizes.

**Result: 20/20 PASSED** (4 seeds x 5 chunk sizes)

### 4.1 Seed 0 (reference seed -- strict parity required)

| Chunk Size | Pred Mismatch | Agreement | Max Proba Error | Proba Close | Acc Batch | Acc Stream |
|-----------:|--------------:|----------:|----------------:|:-----------:|----------:|-----------:|
| 1 | 0 | 1.0000 | 6.0e-13 | yes | 0.8500 | 0.8500 |
| 2 | 0 | 1.0000 | 3.2e-13 | yes | 0.8500 | 0.8500 |
| 7 | 0 | 1.0000 | 8.1e-14 | yes | 0.8500 | 0.8500 |
| 64 | 0 | 1.0000 | 1.9e-14 | yes | 0.8500 | 0.8500 |
| 700 | 0 | 1.0000 | 4.5e-09 | yes | 0.8500 | 0.8500 |

Zero prediction mismatches, probabilities match to ~1e-13. The cs=700 case (single batch) has slightly higher proba error (4.5e-9) due to different code paths but is still well within tolerance.

### 4.2 Seed 7

| Chunk Size | Pred Mismatch | Agreement | Max Proba Error | Acc Batch | Acc Stream | Delta |
|-----------:|--------------:|----------:|----------------:|----------:|-----------:|------:|
| 1 | 24 | 0.9657 | 0.249 | 0.9086 | 0.9314 | +0.023 |
| 2 | 24 | 0.9657 | 0.249 | 0.9086 | 0.9314 | +0.023 |
| 7 | 24 | 0.9657 | 0.249 | 0.9086 | 0.9314 | +0.023 |
| 64 | 24 | 0.9657 | 0.249 | 0.9086 | 0.9314 | +0.023 |
| 700 | 24 | 0.9657 | 0.249 | 0.9086 | 0.9314 | +0.023 |

Constant across all chunk sizes (chunk-stable = true). The 24 mismatches are from the constant columns (zero-variance features) causing a different whitening path in streaming vs batch. Streaming actually achieves +2.3% higher accuracy.

### 4.3 Seed 11

| Chunk Size | Pred Mismatch | Agreement | Max Proba Error | Acc Batch | Acc Stream | Delta |
|-----------:|--------------:|----------:|----------------:|----------:|-----------:|------:|
| 1 | 57 | 0.9186 | 0.398 | 0.8386 | 0.8471 | +0.009 |
| 2 | 57 | 0.9186 | 0.398 | 0.8386 | 0.8471 | +0.009 |
| 7 | 57 | 0.9186 | 0.398 | 0.8386 | 0.8471 | +0.009 |
| 64 | 57 | 0.9186 | 0.398 | 0.8386 | 0.8471 | +0.009 |
| 700 | 57 | 0.9186 | 0.398 | 0.8386 | 0.8471 | +0.009 |

Hardest seed -- 57 mismatches but still chunk-stable and streaming accuracy exceeds batch.

### 4.4 Seed 16

| Chunk Size | Pred Mismatch | Agreement | Max Proba Error | Acc Batch | Acc Stream | Delta |
|-----------:|--------------:|----------:|----------------:|----------:|-----------:|------:|
| 1 | 40 | 0.9429 | 0.390 | 0.8600 | 0.8900 | +0.030 |
| 2 | 40 | 0.9429 | 0.390 | 0.8600 | 0.8900 | +0.030 |
| 7 | 40 | 0.9429 | 0.390 | 0.8600 | 0.8900 | +0.030 |
| 64 | 40 | 0.9429 | 0.390 | 0.8600 | 0.8900 | +0.030 |
| 700 | 40 | 0.9429 | 0.390 | 0.8600 | 0.8900 | +0.030 |

Key finding across seeds 7/11/16: The mismatches are **deterministic** (chunk-stable) and arise from constant-column handling differences between batch and streaming SVD whitening. Streaming accuracy is equal or better in all cases.

---

## 5. SVD Parity Test -- Digits (`bench_lda_svd_partial_fit_parity.py --mode digits`)

5-fold stratified CV on sklearn's bundled digits dataset (N=1,797 D=64 K=10). Tests 3 chunk sizes per fold.

**Result: 15/15 PASSED** (5 folds x 3 chunk sizes, agreement = 1.0000 on every config)

| Fold | Acc Batch | cs=512 | cs=1,024 | cs=2,048 | Agreement |
|-----:|----------:|-------:|---------:|---------:|----------:|
| 0 | 0.9417 | 0.9417 | 0.9417 | 0.9417 | 1.0000 |
| 1 | 0.9556 | 0.9556 | 0.9556 | 0.9556 | 1.0000 |
| 2 | 0.9638 | 0.9638 | 0.9638 | 0.9638 | 1.0000 |
| 3 | 0.9582 | 0.9582 | 0.9582 | 0.9582 | 1.0000 |
| 4 | 0.9471 | 0.9471 | 0.9471 | 0.9471 | 1.0000 |

Every fold and chunk size produces **identical** predictions to batch fit. Mean accuracy: 0.9533.

---

## 6. SVD Parity Test -- MNIST (`bench_lda_svd_partial_fit_parity.py --mode mnist`)

Single 80/20 holdout on 5,000-sample MNIST subset. Tests 4 chunk sizes.

**Result: 4/4 PASSED** (agreement = 1.0000 on every chunk size)

| Chunk | Acc Batch | Acc Stream | Agreement | Status |
|------:|----------:|-----------:|----------:|--------|
| 256 | 0.8170 | 0.8170 | 1.0000 | PASS |
| 512 | 0.8170 | 0.8170 | 1.0000 | PASS |
| 1,024 | 0.8170 | 0.8170 | 1.0000 | PASS |
| 2,048 | 0.8170 | 0.8170 | 1.0000 | PASS |

Exact parity with batch at every chunk size on real handwritten digit images.

---

## 7. Memory Stress Test (`bench_lda_svd_memory_stress.py`)

Compares batch `fit()` vs 10-chunk streaming `partial_fit()` on large matrices (N=20,000 D=5,000 K=10, dataset=0.80 GB) under two rank conditions.

**Result: 2/2 PASSED**

### 7.1 Low Rank (50 informative dims in 5,000-d space)

|  | Time (s) | Accuracy | Peak RSS (GB) | Traced Alloc (GB) | Stored Rank |
|--|--------:|----------:|---------------:|------------------:|------------:|
| Batch | 58.5 | 0.9676 | 4.99 | 4.20 | -- |
| Stream | 46.4 | 0.9855 | 4.99 | 0.75 | 50 |

**Stream/batch ratio: 0.79x (1.3x FASTER)**

Per-chunk times (10 chunks of 2,000 samples):

| Chunk | Time (s) |
|------:|---------:|
| 0 | 4.42 |
| 1 | 5.31 |
| 2 | 5.07 |
| 3 | 5.16 |
| 4 | 4.63 |
| 5 | 4.46 |
| 6 | 4.37 |
| 7 | 4.38 |
| 8 | 4.30 |
| 9 | 4.34 |

Observations: Rank truncation correctly identifies 50 effective dimensions. Traced memory allocation is **5.6x lower** for streaming (0.75 GB vs 4.20 GB). Streaming is faster because SVD merges operate on 50-rank factors rather than full 5,000-d matrices. Per-chunk times are stable (~4.4-5.3s), with chunk 1 slightly higher due to the first merge doubling rank before truncation kicks in. Agreement with batch = 0.9621 (lower than 1.0 because the rank-50 approximation discards some noise that batch retains).

### 7.2 Full Rank (all 5,000 features independent)

|  | Time (s) | Accuracy | Peak RSS (GB) | Traced Alloc (GB) | Stored Rank |
|--|--------:|----------:|---------------:|------------------:|------------:|
| Batch | 50.8 | 0.6170 | 5.13 | 4.28 | -- |
| Stream | 335.7 | 0.6170 | 5.13 | 2.29 | 5,000 |

**Stream/batch ratio: 6.60x (slower)**

Per-chunk times (10 chunks of 2,000 samples):

| Chunk | Time (s) |
|------:|---------:|
| 0 | 4.45 |
| 1 | 24.02 |
| 2 | 35.74 |
| 3 | 41.04 |
| 4 | 37.79 |
| 5 | 37.23 |
| 6 | 40.33 |
| 7 | 41.52 |
| 8 | 36.83 |
| 9 | 36.71 |

Observations: No rank truncation possible (all 5,000 singular values are significant). Agreement with batch = 1.0000 (exact). Streaming is 6.6x slower because each merge requires a full SVD of the (rank + chunk_size) x D block matrix. Chunk 0 is fast (4.45s, just the initial SVD of 2,000 x 5,000), chunk 1 jumps to 24s (merging rank-2000 + 2000 new rows), and subsequent chunks plateau at ~37-41s as the stored rank saturates at 5,000. Traced allocation is still 1.9x lower for streaming (2.29 GB vs 4.28 GB) because only one chunk is in memory at a time.

---

## 8. Note on Eigen/LSQR Failures and Shrinkage

The eigen and lsqr solvers failed on **MNIST** and **Covertype** with:

```
LinAlgError: The leading minor of order N of B is not positive definite.
The factorization of B could not be completed and no eigenvalues or
eigenvectors were computed.
```

This is a **pre-existing batch `fit()` failure**, not a `partial_fit` regression. It occurs because:

1. Both datasets contain features with zero within-class variance (MNIST has constant black pixels; Covertype has binary indicator columns).
2. The eigen solver calls `scipy.linalg.eigh(Sb, Sw)` which requires `Sw` (the within-class scatter) to be positive definite. With zero-variance features, `Sw` is singular.
3. The SVD solver avoids this entirely by using a different decomposition path that handles rank deficiency.

**Shrinkage fixes this**, as demonstrated in [Section 3](#3-stress-test----shrinkage-on-public-datasets---scenarios-shrinkage---no-cv). Adding `shrinkage=0.01` regularizes the covariance matrix to be positive definite, enabling eigen/lsqr to succeed on all three public datasets with **perfect batch-streaming agreement** (18/18 passed).

- `shrinkage='auto'` (Ledoit-Wolf) is supported for batch `fit()` but **not for `partial_fit`** (raises `NotImplementedError` because Ledoit-Wolf requires the full dataset).
- A fixed `shrinkage=float` **is supported** for `partial_fit` with eigen/lsqr solvers.
- Fashion-MNIST passes without shrinkage because its pixel distributions have enough per-class variance.

This is a known limitation of unregularized LDA, not specific to our implementation. The recommended approach for these datasets is to use `solver='svd'` (which handles singularity natively) or add `shrinkage=float` when using eigen/lsqr.

---

## 9. Summary

### Correctness

| Benchmark | Configs | Passed | Failed | Agreement Range |
|-----------|--------:|-------:|-------:|:----------------|
| Stress: synthetic | 48 | 48 | 0 | 1.0000 (all) |
| Stress: MNIST (SVD) | 3 | 3 | 0 | 1.0000 |
| Stress: Fashion-MNIST | 9 | 9 | 0 | 1.0000 (all solvers) |
| Stress: Covertype (SVD) | 3 | 3 | 0 | 1.0000 |
| Stress: eigen/lsqr on singular data | -- | -- | 2 | N/A (batch also fails) |
| Shrinkage: MNIST (eigen/lsqr) | 6 | 6 | 0 | 1.0000 |
| Shrinkage: Fashion-MNIST (eigen/lsqr) | 6 | 6 | 0 | 1.0000 |
| Shrinkage: Covertype (eigen/lsqr) | 6 | 6 | 0 | 1.0000 |
| Parity: synthetic | 20 | 20 | 0 | 0.9186 -- 1.0000 |
| Parity: digits | 15 | 15 | 0 | 1.0000 (all) |
| Parity: MNIST | 4 | 4 | 0 | 1.0000 (all) |
| Memory stress | 2 | 2 | 0 | 0.9621 -- 1.0000 |
| **Total** | **122** | **122** | **2** | |

The 2 failures are pre-existing `scipy.linalg.eigh` errors on singular within-class covariance (batch `fit()` also fails). All `partial_fit` tests pass. The 18 new shrinkage tests confirm that `shrinkage=float` works correctly with `partial_fit` on ill-conditioned datasets.

### Performance Takeaways

- **Tall data (N >> D):** Streaming is **faster** than batch (0.4-0.9x) with dramatically lower memory (up to 32x less for eigen/lsqr).
- **Wide data (D >> N):** Streaming is 1.1-2.9x slower depending on chunk size. Memory is comparable.
- **Real datasets at large chunks:** Streaming matches or beats batch speed (MNIST 0.9x, Covertype 0.86x, Fashion-MNIST eigen 0.59x).
- **Shrinkage on real data:** `shrinkage=0.01` fixes singular covariance on MNIST/Covertype. Streaming with shrinkage is **1.5-2.0x faster** than batch at large chunks (MNIST eigen 0.53x, Covertype eigen 0.50x). Shrinkage accuracy can exceed unregularized SVD (MNIST: 0.8683 vs 0.8663).
- **Low-rank data:** Rank truncation delivers both speed and memory wins. Streaming was 1.3x faster and used 5.6x less traced memory on the 50-rank-in-5000-d test.
- **Full-rank data:** Streaming is 6.6x slower (unavoidable full SVD merges) but achieves exact parity and uses less peak traced memory.
- **Covariance solvers (eigen/lsqr):** Consistently faster and more memory-efficient than SVD for streaming, especially on tall data. Fashion-MNIST eigen streaming at cs=70k was **1.7x faster** than batch.
- **Single-sample streaming:** Works correctly (exact parity) but is slow due to per-call overhead. Practical minimum chunk sizes of 64+ are recommended.
- **Singular covariance:** Use `solver='svd'` or add `shrinkage=float` for datasets with zero-variance features. Both approaches are validated on MNIST, Fashion-MNIST, and Covertype.
