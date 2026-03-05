#!/bin/sh
set -e

# ---------------------------------------------------------------------------
# Bootstrap environment if it doesn't exist yet (first run with a fresh mount).
# If the environment already exists, just ensure dependencies are current.
# ---------------------------------------------------------------------------
if [ ! -f "/workspace/.venv-docker/pyvenv.cfg" ]; then
  echo "[entrypoint] Creating venv at /workspace/.venv-docker ..."
  python3.12 -m venv --clear /workspace/.venv-docker
  . /workspace/.venv-docker/bin/activate
  pip install --upgrade pip setuptools wheel
  pip install "meson-python>=0.17.1" "cython>=3.1.2" "numpy>=2" "scipy>=1.10.0"
  # CPU-only for aarch64 (Apple Silicon Docker), CUDA for x86_64
  if [ "$(uname -m)" = "x86_64" ]; then
    pip install torch --index-url https://download.pytorch.org/whl/cu128
  else
    pip install torch --index-url https://download.pytorch.org/whl/cpu
  fi
  pip install jupyterlab notebook ipywidgets ipykernel matplotlib
  pip install --verbose --no-build-isolation --editable .
elif [ -f "/workspace/pyproject.toml" ]; then
  . /workspace/.venv-docker/bin/activate
  echo "[entrypoint] Venv activated, refreshing editable install ..."
  # Ensure Jupyter is available in existing venvs
  if ! command -v jupyter >/dev/null 2>&1; then
    echo "[entrypoint] Installing Jupyter into existing venv ..."
    pip install jupyterlab notebook ipywidgets ipykernel matplotlib
  fi
  pip install --no-build-isolation -e . 2>/dev/null || true
else
  . /workspace/.venv-docker/bin/activate
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
# Start Jupyter Lab in the background (accessible from host on port 8888).
# ---------------------------------------------------------------------------
if command -v jupyter >/dev/null 2>&1; then
  jupyter lab \
    --ip=0.0.0.0 --port=8888 \
    --no-browser \
    --notebook-dir=/workspace \
    > /tmp/jupyter.log 2>&1 &
  echo "[entrypoint] Jupyter Lab started on port 8888 (log: /tmp/jupyter.log)"
  echo "[entrypoint] Jupyter token: run 'jupyter server list' to see the URL with token"
fi

# ---------------------------------------------------------------------------
# Launch Claude inside tmux.
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
