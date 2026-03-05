@glemaitre Following up on my earlier comment — I've extended the implementation to cover **all three solvers** (`eigen`, `lsqr`, and `svd`), including an incremental SVD path as you asked about.

Updated feature branch:
- https://github.com/wcscr/scikit-learn/tree/feature/lda-partial-fit-clean-withsvd

---

## What's new since the last update

### SVD solver support

The SVD path uses an incremental rank-revealing SVD update. Each `partial_fit` call:

1. Computes per-class chunk means, counts, and within-class sum-of-squares.
2. Updates running class means and counts via Chan, Golub, and LeVeque's pairwise merge formula.
3. Builds a block matrix `Z` by stacking the previous compact SVD state (`diag(S) @ Vt`), the class-centered chunk data, and mean-shift correction rows (parallel axis theorem).
4. Computes `svd(Z, full_matrices=False)` and truncates negligible singular values using a relative threshold (`eps * max(Z.shape) * S[0]`).
5. Stores only the truncated `(S, Vt)` — the rank stays bounded by `min(N_total, n_features)` regardless of how many chunks are processed.

Prediction attributes (`coef_`, `intercept_`, `scalings_`, `explained_variance_ratio_`) are **lazily reconstructed** from the compact `(S, Vt)` factorization on first access (via `__getattr__`), avoiding redundant work during streaming.

The within-class standard deviation (needed by the SVD solver for `predict_proba`) is tracked incrementally using Chan's parallel sum-of-squares formula, matching the approach used by the eigen/lsqr paths.

### Array API compatibility

The SVD `partial_fit` path is fully Array API compatible. All accumulation and SVD computation stays in the caller's array namespace (`xp`) and on-device (`dev`). When numpy arrays are passed, `scipy.linalg.svd` is used; for other backends (e.g. PyTorch), `xp.linalg.svd` is dispatched.

### Known limitation: `torch.linalg.svd` on Apple Silicon

When using PyTorch CPU tensors via Array API dispatch on **ARM64 / Apple Silicon**, `torch.linalg.svd` can fail to converge on the ill-conditioned block matrices that arise in the N < D regime (e.g. Olivetti Faces: 400 samples, 4096 features). The error is:

```
torch._C._LinAlgError: linalg.svd: The algorithm failed to converge because
the input matrix is ill-conditioned or has too many repeated singular values
```

This is a known PyTorch platform issue — the same inputs succeed with NumPy/SciPy, and with PyTorch on x86_64 (MKL) and NVIDIA GPU (cuSOLVER). Relevant upstream issues:
- [pytorch/pytorch#64237](https://github.com/pytorch/pytorch/issues/64237)
- [pytorch/pytorch#149591](https://github.com/pytorch/pytorch/issues/149591)
- [numpy/numpy#21756](https://github.com/numpy/numpy/issues/21756) (same LAPACK root cause)

The implementation is left as-is (dispatching to `xp.linalg.svd`) rather than forcing a scipy fallback, since the Array API spec does not define portable exception types and sklearn does not import torch directly. This is consistent with how `_gaussian_mixture.py` handles similar linalg exception portability limitations.

---

## Updated scope (all solvers)

- `partial_fit(X, y, classes=None)` for `solver="eigen"`, `solver="lsqr"`, and `solver="svd"`.
- **Eigen/LSQR**: incremental sufficient statistics (class counts, means, pooled covariance) via Chan et al. pairwise merge.
- **SVD**: incremental rank-revealing SVD with deferred attribute reconstruction.
- Explicit guards for unsupported configurations: `shrinkage="auto"` and custom `covariance_estimator` raise `NotImplementedError`.
- Fixed-value shrinkage is supported for eigen/lsqr.

## References

- Chan, T. F., Golub, G. H., LeVeque, R. J. (1979). *Updating formulae and a pairwise algorithm for computing sample variances*. Technical Report STAN-CS-79-773.
- Chan, T. F., Golub, G. H., LeVeque, R. J. (1983). *Algorithms for computing the sample variance: Analysis and recommendations*. https://doi.org/10.1080/00031305.1983.10483115
- Brand, M. (2006). *Fast low-rank modifications of the thin singular value decomposition*. Linear Algebra and its Applications, 415(1), 20–30. (incremental SVD update structure)

## Tests added/updated

**Eigen/LSQR tests** (unchanged from previous comment):
- `test_lda_partial_fit_batch_equivalence`
- `test_lda_partial_fit_missing_classes_in_chunk`
- `test_lda_partial_fit_single_sample_chunks`
- `test_lda_partial_fit_collinear_features`
- `test_lda_partial_fit_raises_auto_shrinkage`
- `test_lda_partial_fit_raises_covariance_estimator`
- `test_lda_partial_fit_raises_missing_classes`
- `test_lda_partial_fit_binary`
- `test_lda_partial_fit_predict`
- `test_lda_partial_fit_after_fit`
- `test_lda_partial_fit_raises_unknown_labels`
- `test_lda_partial_fit_honors_priors`
- `test_lda_partial_fit_early_return_not_fitted`
- `test_lda_partial_fit_lsqr_low_samples`
- `test_lda_partial_fit_after_fit_one_class_chunk`
- `test_lda_partial_fit_multiclass_not_fitted_until_all_seen`

**SVD-specific tests** (new):
- `test_lda_svd_partial_fit_transform_equivalence` — batch vs streaming parity for predict, transform, predict_proba, decision_function
- `test_lda_svd_partial_fit_collinear` — collinear features with rank deficiency
- `test_lda_svd_partial_fit_store_covariance` — `store_covariance=True` parity
- `test_lda_svd_partial_fit_scale_invariance` — accuracy invariant to feature scaling
- `test_lda_svd_partial_fit_near_zero_variance` — near-constant features
- `test_lda_svd_partial_fit_tiny_nonzero_noise` — features with tiny but nonzero variance
- `test_lda_svd_partial_fit_multiclass_constant_columns` — 10-class with constant columns
- `test_lda_svd_partial_fit_multiclass_constant_columns_make_classification` — same with `make_classification`
- `test_lda_svd_partial_fit_constant_columns_seed_sweep_stability` — stability across 20 random seeds
- `test_lda_svd_partial_fit_within_std_matches_batch_to_roundoff` — within-class std matches batch to machine precision
- `test_lda_svd_partial_fit_array_api_torch_cpu` — PyTorch CPU Array API roundtrip
- `test_lda_svd_partial_fit_array_api_torch_cuda` — PyTorch CUDA Array API roundtrip
- `test_lda_svd_partial_fit_array_api_covariance_path_converts` — covariance solver falls back to numpy when Array API is active
- `test_lda_svd_partial_fit_mnist_accuracy_parity` — MNIST accuracy parity (network test)

## Local verification

All 109 tests pass (4 skipped: 3 require PyTorch + `SCIPY_ARRAY_API=1`, 1 requires network). The 3 PyTorch tests pass when run with the required env vars. Style checks pass.

---

## Draft PR Description

### Title
ENH Add `partial_fit` to `LinearDiscriminantAnalysis` (eigen, lsqr, svd)

### Summary
Adds incremental learning support to `LinearDiscriminantAnalysis` through `partial_fit` for all three solvers.

- **Eigen/LSQR**: incremental sufficient statistics (class counts, means, pooled covariance) via Chan et al. pairwise merge formula.
- **SVD**: incremental rank-revealing SVD with compact `(S, Vt)` storage and deferred attribute reconstruction. Handles the N < D regime (singular covariance) without shrinkage, matching batch `fit` behavior.
- Array API compatible across all paths.

Closes #30042.

### Why this change
`LinearDiscriminantAnalysis` currently requires full-batch `fit`, which limits use on datasets that do not fit in memory and in streaming/online workflows. Adding `partial_fit` aligns LDA with scikit-learn estimators that already support incremental updates (e.g. `MultinomialNB`, `SGDClassifier`).

### What changed
- `sklearn/discriminant_analysis.py`:
  - Added `partial_fit(X, y, classes=None)`.
  - Added `_partial_fit_svd()` — incremental SVD with block-matrix update and rank truncation.
  - Added `_partial_fit_covariance()` — incremental stats for eigen/lsqr (unchanged from previous).
  - Added `_reconstruct_svd_attrs()` with lazy `__getattr__` dispatch for deferred attribute computation.
  - Added `_invalidate_svd_attrs()` / `_clear_prediction_attrs()` for streaming state management.
- `sklearn/tests/test_discriminant_analysis.py`: 30 new tests covering all solvers.
- `doc/whats_new/upcoming_changes/sklearn.discriminant_analysis/99999.feature.rst`: changelog entry.

### Behavioral notes
- For `solver="svd"`, prediction attributes are reconstructed lazily from `(S, Vt)` on first access after `partial_fit`, avoiding redundant computation during streaming.
- For `solver="eigen"`, parameter solving is deferred until within-class degrees of freedom are sufficient.
- For `solver="lsqr"`, solving is allowed in rank-deficient settings via least squares.
- `shrinkage="auto"` (Ledoit-Wolf) and custom `covariance_estimator` raise `NotImplementedError` for `partial_fit` since they require the full dataset.
- Known limitation: `torch.linalg.svd` may fail on Apple Silicon with severely ill-conditioned matrices in the N < D regime (upstream PyTorch issue, not specific to this implementation).

### Risks / areas for careful review
- Numerical parity between chunked and batch modes, especially for SVD in high-dimensional settings.
- Lazy attribute reconstruction via `__getattr__` — interaction with `check_is_fitted`, pickling, cloning.
- Array API namespace handling in `_reconstruct_svd_attrs`.

This pull request includes code written with the assistance of AI. The code has been reviewed and tested by a human.
