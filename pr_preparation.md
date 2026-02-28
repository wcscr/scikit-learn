# PR Preparation: `LinearDiscriminantAnalysis.partial_fit`

## Upstream Issue Status

- Existing issue found: `scikit-learn/scikit-learn#30042`
  - Title: "Add partial_fit Functionality to LinearDiscriminantAnalysis Classifier"
  - Opened: 2024-10-10
  - Current status observed: **Open**
  - Development section observed: **No branches or pull requests**

Link: https://github.com/scikit-learn/scikit-learn/issues/30042

---

## Draft Issue Comment (for #30042)

I’ve prepared an implementation of `LinearDiscriminantAnalysis.partial_fit` and would like to open a PR linked to this issue.

Current scope:
- Adds `partial_fit(X, y, classes=None)` to `LinearDiscriminantAnalysis`.
- Supports `solver="lsqr"` and `solver="eigen"`.
- Uses incremental sufficient statistics updates (class counts, class means, pooled covariance accumulator), so repeated chunked updates are consistent with batch fitting in supported configurations.
- The incremental covariance/statistics updates use the pairwise merge recurrence
  (`S_new = S_old + S_chunk + n_old*n_chunk/n_new * delta*delta^T`) from
  Chan, Golub, and LeVeque (1979), with the 1983 paper used as the analysis/
  recommendations reference.
- Keeps unsupported configurations explicit for now (`solver="svd"`, `shrinkage="auto"`, and custom `covariance_estimator`) with clear `NotImplementedError`.

Validation and test coverage added:
- Batch-vs-online equivalence checks.
- Missing-class chunks and single-sample chunk handling.
- Unknown-label rejection.
- `fit()` then `partial_fit()` behavior.
- Low-sample/high-dimensional `lsqr` path.
- Explicit-priors behavior parity with `fit`.

I also added a changelog fragment and updated `versionadded` to the current dev cycle.

If this scope looks acceptable, I’ll open the PR and link it here.

This pull request includes code written with the assistance of AI.
The code has **not yet been reviewed** by a human.

---

## Draft PR Description

### Title
ENH Add `partial_fit` to `LinearDiscriminantAnalysis` (`lsqr` and `eigen`)

### Summary
This PR adds incremental learning support to `LinearDiscriminantAnalysis` through a new `partial_fit` method for `solver="lsqr"` and `solver="eigen"`.

The implementation updates class-level sufficient statistics across chunks and derives model parameters from those statistics, enabling out-of-core / streaming workflows while preserving expected estimator behavior.

Closes #30042.

### Why this change
`LinearDiscriminantAnalysis` currently requires full-batch `fit`, which limits use on datasets that do not fit in memory and in online-training workflows. Adding `partial_fit` aligns LDA with scikit-learn estimators that already support incremental updates.

The online statistics update rule is the pairwise merge formula from Chan,
Golub, and LeVeque (1979), with Chan et al. (1983) cited for analysis and
recommendations:

- Chan, T. F., Golub, G. H., LeVeque, R. J. (1979). *Updating formulae and a
  pairwise algorithm for computing sample variances*. Technical Report
  STAN-CS-79-773.
- Chan, T. F., Golub, G. H., LeVeque, R. J. (1983). *Algorithms for computing
  the sample variance: Analysis and recommendations*.
  https://doi.org/10.1080/00031305.1983.10483115

For reviewer context: this second-moment merge equation is algebraically the
same as the variance-only combine formula later written in Pébay (2008), i.e.,
same recurrence with different notation.

### What changed
- Added `LinearDiscriminantAnalysis.partial_fit(X, y, classes=None)`.
- Refactored solver internals to allow reuse from both `fit` and `partial_fit`.
- Added explicit guards for unsupported `partial_fit` configurations:
  - `solver="svd"`
  - `shrinkage="auto"`
  - custom `covariance_estimator`
- Added class-label validation for incremental calls (reject labels not in initial `classes`).
- Ensured explicit `priors` are honored consistently with `fit`.
- Added formal references in the `partial_fit` docstring to:
  - Chan et al. (1979) for the selected pairwise merge recurrence.
  - Chan et al. (1983) for the numerical analysis/recommendations.
- Added changelog entry:
  - `doc/whats_new/upcoming_changes/sklearn.discriminant_analysis/99999.feature.rst`

### Behavioral notes
- For `solver="eigen"`, parameter solving is deferred until enough within-class degrees of freedom are available.
- For `solver="lsqr"`, solving is allowed in rank-deficient settings via least squares.

### Tests added/updated
- `test_lda_partial_fit_batch_equivalence`
- `test_lda_partial_fit_missing_classes_in_chunk`
- `test_lda_partial_fit_single_sample_chunks`
- `test_lda_partial_fit_collinear_features`
- `test_lda_partial_fit_raises_svd`
- `test_lda_partial_fit_raises_auto_shrinkage`
- `test_lda_partial_fit_raises_covariance_estimator`
- `test_lda_partial_fit_raises_missing_classes`
- `test_lda_partial_fit_binary`
- `test_lda_partial_fit_predict`
- `test_lda_partial_fit_after_fit`
- `test_lda_partial_fit_raises_unknown_labels`
- `test_lda_partial_fit_honors_priors`
- `test_lda_partial_fit_lsqr_low_samples`

### Risks / areas for careful review
- Numerical parity between chunked and batch modes for `eigen` with shrinkage.
- Fitted-state behavior prior to solver readiness in early incremental calls.
- Consistency with estimator checks and error-message conventions.

### Local verification
- Targeted discriminant analysis test module run locally.
- Style checks for modified files (line length / lint formatting).

This pull request includes code written with the assistance of AI.
The code has **not yet been reviewed** by a human.

---

## Notes Before Posting Upstream

- scikit-learn contribution policy asks contributors to manually review and understand changes before submission.
- scikit-learn also asks contributors not to paste AI-generated PR/issue text directly; adapt this draft into your own wording before posting.
