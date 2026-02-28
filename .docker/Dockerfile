# Claude Code sandbox image for scikit-learn LDA partial_fit development
# Base: Ubuntu 24.04
# Python 3.12 pinned explicitly via deadsnakes PPA (base image may default to newer)
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# ---------- System packages ----------
# python3.12*:        Pin Python 3.12 for consistency with local dev env
# libopenblas-dev:    numpy/scipy BLAS/LAPACK backend
# libfreetype-dev,    matplotlib rendering (used in sklearn plot utilities)
#   libpng-dev,
#   libjpeg-dev:
# pkg-config:         lets pip find BLAS/FreeType during wheel builds
# build-essential:    gcc/g++/make for Cython compilation (required by sklearn)
# git:                version control
# curl, ca-certs:     pip index access / general fetch
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        build-essential \
        pkg-config \
        libopenblas-dev \
        libfreetype-dev \
        libpng-dev \
        libjpeg-dev \
        git \
        curl \
        ca-certificates \
        tmux \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default python3 / python, install pip via get-pip.py
# (ensurepip is disabled for system Python on Debian/Ubuntu)
# Remove the PEP 668 EXTERNALLY-MANAGED marker so get-pip.py works inside Docker.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1 \
    && rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3 1 \
    && update-alternatives --install /usr/bin/pip  pip  /usr/local/bin/pip3 1

# Verify Python 3.12 and pip are available
RUN python3 --version | grep "3\.12\." && pip --version

# ---------- Working directory ----------
WORKDIR /workspace

# ---------- Python venv ----------
# The venv lives in the repo at .venv and is brought in via the /workspace
# bind-mount — no venv creation or pip install happens at build time.
# Set PATH so the mounted venv's binaries are found automatically.
ENV PATH="/workspace/.venv/bin:$PATH"

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
# Install via native installer (no Node.js / npm required).
# The installer places the binary under ~/.local/bin/.
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

# Trust /workspace for git — the bind-mount may be owned by a different UID
RUN git config --global --add safe.directory /workspace

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
COPY --chmod=0755 .docker/entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
