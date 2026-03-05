#!/bin/sh
source .venv-docker/bin/activate
SCIPY_ARRAY_API=1 SKLEARN_SKIP_NETWORK_TESTS=0 \
  python benchmarks/bench_lda_partial_fit_stress.py --scenarios all --include-cuda --no-cv --quick
