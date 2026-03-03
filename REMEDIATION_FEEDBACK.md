# Remediation Feedback (Latest Commit Review)

## Verdict
Most fixes are in place, but not all deficiencies are fully resolved yet.

## Remaining Findings

### [P1] SVD `partial_fit` is still not fully batch-equivalent for tiny nonzero-variance features
`partial_fit` reconstruction now applies a relative `std` floor, while batch `fit` still only clamps exact zeros.  
This leaves a parity gap for near-zero (but nonzero) variance features.

- Streaming path floor: `std <= sqrt(eps) * std_max` in reconstruction.
- Batch path floor: `std == 0`.

Observed repro (added noise columns at ~`1e-8` and `1e-9`):
- `max_proba_err ~= 0.0246`
- Prediction mismatch remains (not just numerical precision noise)

### [P2] New regression tests are too permissive for strict parity
New tests only require `>95%` prediction agreement for hard scale/near-zero scenarios, so meaningful drift can still pass:

- `test_lda_svd_partial_fit_scale_invariance`
- `test_lda_svd_partial_fit_near_zero_variance`

## Confirmed Fixes

- `fit()->partial_fit()` one-class chunk crash fixed for `lsqr`, `eigen`, and `svd`.
- "All declared classes must be seen before fitted" gating works.
- `store_covariance=True` for SVD `partial_fit` now exposes `covariance_` and matches batch covariance closely.
- `pytest -q sklearn/tests/test_discriminant_analysis.py` passes.
- `check_estimator(LinearDiscriminantAnalysis())` passes in this local environment.

## Suggested Next Remediation

1. Align batch and streaming near-zero handling so both use the same `std` floor policy (or both use exact-zero handling, but consistently).
2. Strengthen tests to assert near-exact parity (probabilities/coefficients/transform) in adversarial cases, not only `>95%` agreement.
3. Add one explicit regression test for the tiny-nonzero-noise-column repro that still fails strict parity.

This pull request includes code written with the assistance of AI.
The code has **not yet been reviewed** by a human.
