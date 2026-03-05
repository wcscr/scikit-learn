#!/bin/sh
set -e

cd "$(dirname "$0")/.."

docker build -f .docker/Dockerfile \
    --build-arg HOST_UID="$(id -u)" \
    --build-arg HOST_GID="$(id -g)" \
    -t scikit-learn-dev-claude .
