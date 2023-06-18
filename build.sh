#!/usr/bin/env bash

version() {
    grep version pyproject.toml
    NEW_VERSION=$(sed -n 's/__version__ = "\(.*\)"/\1/p' polyai/__init__.py)
    sed -i '' "s/\(version = \"\)[^\"]*\"/\1$NEW_VERSION\"/" pyproject.toml
    grep version pyproject.toml
}

tag() {
    # create a new git tag using the pyproject.toml version
    # and push the tag to origin
    version=$(sed -n 's/version = "\(.*\)"/\1/p' pyproject.toml)
    git tag v$version && git push origin v$version
}

venv() {
    if conda info --envs | grep -q $(basename $PWD); then 
        conda activate $(basename $PWD)
    else
        conda create -n $(basename $PWD) python=3.10 -c conda-forge
        conda activate $(basename $PWD)
    fi
}

container() {
    source .env || echo ".env file read error, continuing anyway ..."
    export BUILDKIT_PROGRESS=plain && docker build -t polyai .

    mkdir -p "$POLYAI_SERV_CACHE" "$POLYAI_MODELS"
    docker run -it --gpus all \
        -v "$POLYAI_SERV_CACHE:/home/user/.cache/" \
        -v "$POLYAI_MODELS:/home/user/models" \
        -e "POLYAI_MODEL_PATH=$POLYAI_MODEL_PATH" \
        -p $POLYAI_SERV_PORT:8080 \
        polyai
}

"$@"

