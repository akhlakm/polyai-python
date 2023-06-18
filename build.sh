#!/usr/bin/env bash

version() {
    # print current version
    grep version pyproject.toml
    read -p "new version string? " NEW_VERSION
    sed -i "s/\(version = \"\)[^\"]*\"/\1$NEW_VERSION\"/" pyproject.toml
    sed -i "s/\(__version__ = \"\)[^\"]*\"/\1$NEW_VERSION\"/" polyai/__init__.py
    # confirm
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
    docker run -it --gpus all \
        -v $POLYAI_SERV_CACHE:/home/user/.cache/ \
        -p $POLYAI_SERV_PORT:8080 \
        polyai
}

"$@"

