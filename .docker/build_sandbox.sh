docker build --no-cache -f .docker/Dockerfile \
    --build-arg HOST_UID="$(id -u)" \
    --build-arg HOST_GID="$(id -g)" \
    -t sklearn-lda-claude .
