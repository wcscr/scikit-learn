#!/bin/sh
set -e

# ---------------------------------------------------------------------------
# Bootstrap venv if it doesn't exist yet (first run with a fresh mount).
# If the venv already exists, just ensure the editable install is current.
# ---------------------------------------------------------------------------
if [ ! -f "/workspace/.venv/pyvenv.cfg" ]; then
  echo "[entrypoint] Creating venv at /workspace/.venv ..."
  python3.12 -m venv --clear /workspace/.venv
  . /workspace/.venv/bin/activate

  # 1) Build deps for scikit-learn (from pyproject.toml [build-system].requires)
  pip install --upgrade pip setuptools wheel \
      "meson-python>=0.17.1" meson ninja "Cython>=3.1.2" "numpy>=2" "scipy>=1.10.0"

  # 2) Clean stale host meson artifacts, then editable-install sklearn only
  rm -rf /workspace/build/
  pip install --no-build-isolation --no-deps -e .

  # 3) Install all runtime + test deps WITH build isolation (handles pyamg/pybind11 etc.)
  pip install joblib threadpoolctl scipy numpy \
      matplotlib pandas "pytest>=7.1.2" pytest-cov "pyamg>=5.0.0" \
      polars pyarrow numpydoc pooch "ruff>=0.12.2" "mypy>=1.15"

elif [ -f "/workspace/pyproject.toml" ]; then
  . /workspace/.venv/bin/activate
  echo "[entrypoint] Venv activated, refreshing editable install ..."
  rm -rf /workspace/build/
  pip install --no-build-isolation --no-deps -e .
else
  . /workspace/.venv/bin/activate
fi

echo "[entrypoint] Venv ready: $(python --version), $(which python)"

# ---------------------------------------------------------------------------
# Ensure Claude Code settings survive the bind-mount that overlays the
# build-time /home/agent/.claude directory.
# ---------------------------------------------------------------------------
mkdir -p "$HOME/.claude"
if [ ! -f "$HOME/.claude/settings.json" ]; then
  cat > "$HOME/.claude/settings.json" <<'SETTINGS'
{"skipDangerousModePermissionPrompt":true,"env":{"CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS":"1"},"teammateMode":"tmux"}
SETTINGS
  echo "[entrypoint] Wrote $HOME/.claude/settings.json"
fi

# Verify claude is available before handing off to tmux.
if ! command -v claude >/dev/null 2>&1; then
  echo "[entrypoint] ERROR: 'claude' not found on PATH" >&2
  echo "[entrypoint] PATH=$PATH" >&2
  echo "[entrypoint] Dropping to shell for debugging." >&2
  exec bash
fi
echo "[entrypoint] claude CLI: $(claude --version 2>&1 || true)"

# ---------------------------------------------------------------------------
# Launch Claude inside tmux.
# Wrap in a shell so that if claude exits (error, auth failure, etc.) we
# land in bash instead of losing the tmux session (and with it, the error).
# ---------------------------------------------------------------------------
CLAUDE_CMD='claude --dangerously-skip-permissions'

launch_in_tmux() {
  exec tmux new-session -s claude \
    "echo '[tmux] Starting claude ...'; ${CLAUDE_CMD} $*; echo; echo '[tmux] claude exited (\$?)  —  dropping to bash'; exec bash"
}

if [ "$#" -eq 0 ]; then
  launch_in_tmux
fi

case "$1" in
  bash|sh)
    exec "$@"
    ;;
  *)
    launch_in_tmux "$@"
    ;;
esac
