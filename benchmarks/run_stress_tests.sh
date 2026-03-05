#!/bin/sh
#
# Run all LDA partial_fit benchmark scripts with progress updates.
#
# Usage:
#   benchmarks/run_stress_tests.sh              # defaults (synthetic only, no network)
#   benchmarks/run_stress_tests.sh --all        # include network datasets + memory stress
#   benchmarks/run_stress_tests.sh --cuda       # include CUDA/GPU backends
#   benchmarks/run_stress_tests.sh --quick      # fast smoke-test mode
#   benchmarks/run_stress_tests.sh --all --cuda # everything
#
# Environment variables respected:
#   PYTHON          - python interpreter (default: python)
#   VENV            - virtualenv to activate (auto-detected if unset)
#   RESULTS_DIR     - directory for JSON output (default: benchmarks/results)

set -e

# ── Defaults ──────────────────────────────────────────────────────────────────
INCLUDE_NETWORK=0
INCLUDE_CUDA=0
QUICK=0

for arg in "$@"; do
    case "$arg" in
        --all)    INCLUDE_NETWORK=1 ;;
        --cuda)   INCLUDE_CUDA=1 ;;
        --quick)  QUICK=1 ;;
        --help|-h)
            sed -n '2,/^$/s/^# \?//p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 1
            ;;
    esac
done

# ── Resolve paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-python}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"

# Auto-detect virtualenv
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -n "$VENV" ] && [ -f "$REPO_ROOT/$VENV/bin/activate" ]; then
        . "$REPO_ROOT/$VENV/bin/activate"
    elif [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
        . "$REPO_ROOT/.venv/bin/activate"
    elif [ -f "$REPO_ROOT/.venv-local/bin/activate" ]; then
        . "$REPO_ROOT/.venv-local/bin/activate"
    fi
fi

mkdir -p "$RESULTS_DIR"

# ── Helpers ───────────────────────────────────────────────────────────────────
PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
TOTAL=0

timestamp() {
    date "+%H:%M:%S"
}

banner() {
    echo ""
    echo "=================================================================="
    echo "  [$TOTAL] $(timestamp)  $1"
    echo "=================================================================="
}

run_bench() {
    # run_bench <label> <command...>
    label="$1"; shift
    TOTAL=$((TOTAL + 1))
    banner "$label"
    echo "  -> $*"
    echo ""
    if "$@"; then
        PASS_COUNT=$((PASS_COUNT + 1))
        echo ""
        echo "  => PASSED"
    else
        rc=$?
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo ""
        echo "  => FAILED (exit $rc)"
    fi
}

skip_bench() {
    TOTAL=$((TOTAL + 1))
    SKIP_COUNT=$((SKIP_COUNT + 1))
    banner "$1 [SKIPPED]"
    echo "  Reason: $2"
}

# ── Build extra flags ─────────────────────────────────────────────────────────
STRESS_SCENARIOS="synthetic"
STRESS_EXTRA=""
if [ "$INCLUDE_NETWORK" -eq 1 ]; then
    STRESS_SCENARIOS="all"
    export SKLEARN_SKIP_NETWORK_TESTS=0
fi
if [ "$INCLUDE_CUDA" -eq 1 ]; then
    STRESS_EXTRA="--include-cuda"
    export SCIPY_ARRAY_API=1
fi
if [ "$QUICK" -eq 1 ]; then
    STRESS_EXTRA="$STRESS_EXTRA --quick"
fi

# ── 1. Partial-fit stress test ────────────────────────────────────────────────
run_bench \
    "LDA partial_fit stress test (scenarios=$STRESS_SCENARIOS)" \
    $PYTHON "$SCRIPT_DIR/bench_lda_partial_fit_stress.py" \
        --scenarios "$STRESS_SCENARIOS" \
        --json-out "$RESULTS_DIR/stress_test.json" \
        $STRESS_EXTRA

# ── 2. SVD fit-vs-partial_fit parity ─────────────────────────────────────────
PARITY_MODE="synthetic"
if [ "$INCLUDE_NETWORK" -eq 1 ]; then
    PARITY_MODE="all"
fi

run_bench \
    "SVD fit vs partial_fit parity (mode=$PARITY_MODE)" \
    $PYTHON "$SCRIPT_DIR/bench_lda_svd_partial_fit_parity.py" \
        --mode "$PARITY_MODE" \
        --json-out "$RESULTS_DIR/parity_test.json"

# ── 3. SVD memory stress ─────────────────────────────────────────────────────
if [ "$QUICK" -eq 1 ]; then
    skip_bench "SVD memory stress (batch vs streaming)" \
        "Skipped in --quick mode (allocates large matrices)"
else
    run_bench \
        "SVD memory stress (batch vs streaming)" \
        $PYTHON "$SCRIPT_DIR/bench_lda_svd_memory_stress.py"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "  SUMMARY  $(timestamp)"
echo "=================================================================="
echo "  Total:   $TOTAL"
echo "  Passed:  $PASS_COUNT"
echo "  Failed:  $FAIL_COUNT"
echo "  Skipped: $SKIP_COUNT"
echo "  Results: $RESULTS_DIR/"
echo "=================================================================="

if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi
