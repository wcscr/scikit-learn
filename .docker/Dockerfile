# Claude Code sandbox image — scikit-learn
# Base: NVIDIA CUDA 12.8 runtime on Ubuntu 24.04
# Provides libcuda, libcudart, and GPU device access for PyTorch CUDA.
# Falls back to CPU-only if the container is launched without --gpus.
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# ---------- System packages ----------
# build-essential, gfortran: compile Cython extensions and BLAS/LAPACK
# python3.12-*: Python runtime, venv, and C API headers
# pkg-config: locate system libraries for Meson build
# libopenblas-dev: BLAS/LAPACK backend for NumPy/SciPy
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        build-essential \
        gfortran \
        pkg-config \
        libopenblas-dev \
        git \
        curl \
        ca-certificates \
        tmux \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default python3 / python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1

# Verify Python version
RUN python3 --version | grep "3\.12\."

# ---------- Working directory ----------
WORKDIR /workspace

ENV PATH="/workspace/.venv-docker/bin:$PATH"

# ---------------------------------------------------------------------------
# Non-root user — UID/GID must match the host so bind-mounted files are
# writable.  Pass --build-arg HOST_UID=<your uid> at build time;
# build_sandbox.sh does this automatically.
# ---------------------------------------------------------------------------
ARG HOST_UID=1000
ARG HOST_GID=1001
RUN (groupadd -g ${HOST_GID} agent 2>/dev/null || groupadd agent 2>/dev/null || true) \
    && existing_user=$(getent passwd ${HOST_UID} | cut -d: -f1) \
    && if [ -n "$existing_user" ]; then \
         usermod -l agent -d /home/agent -m -g agent "$existing_user"; \
       else \
         useradd --uid ${HOST_UID} --gid agent --create-home agent; \
       fi \
    && chown -R agent:agent /workspace

USER agent

# ---------- Claude Code CLI ----------
RUN curl -fsSL https://claude.ai/install.sh | bash \
    && ~/.local/bin/claude --version

ENV PATH="/home/agent/.local/bin:${PATH}"

ENV COLORTERM=truecolor
ENV ANTHROPIC_MODEL=claude-opus-4-6

# Claude Code configuration.
RUN echo '{"hasCompletedOnboarding": true}' > ~/.claude.json \
    && mkdir -p /home/agent/.claude \
    && echo '{"env":{"CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS":"1"},"teammateMode":"tmux"}' > /home/agent/.claude/settings.json \
    && echo 'set -g default-terminal "screen-256color"' > /home/agent/.tmux.conf

# Trust /workspace for git
RUN git config --global --add safe.directory /workspace

# entrypoint.sh is bind-mounted at runtime (see start_claude scripts)
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
