#!/usr/bin/env bash

bump() {
    ## Bump the version number
    grep version pyproject.toml
    VERSION=$(sed -n 's/version = "\(.*\)"/\1/p' pyproject.toml)
    VERSION=$(python -c "v='$VERSION'.split('.');print('%s.%s.%d' %(v[0], v[1], int(v[2])+1))")
    echo "   >>>"
    sed -i "s/\(version = \"\)[^\"]*\"/\1$VERSION\"/" pyproject.toml
    sed -i "s/\(__version__ = \"\)[^\"]*\"/\1$VERSION\"/" polyai/__init__.py
    grep version pyproject.toml
    git add pyproject.toml polyai/__init__.py
    git commit -m "Bump to version $VERSION"
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


docker-entry() {
    ## Function to be called from inside a docker container.
    ## --------------------------------------------------------------------------------------
    if [ ! -f .docker_env/bin/activate ]; then
        echo "Setting up VENV ..."
        python3 -m venv .docker_env      || exit 101
        source .docker_env/bin/activate  || exit 102
        # Install libraries
        pip install -vv torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu118
        pip install transformers datasets evaluate peft safetensors
    fi
}

docker-server-entry() {
    ## Function to be called from inside a docker container.
    ## --------------------------------------------------------------------------------------
    docker-entry
    source .docker_env/bin/activate  || exit 102

    # Uncomment to reinstall
    # pip install -e .
    # pip install -vv -e .[server]
    # spacy download en_core_web_sm

    export POLYAI_REQUEST_LENGTH=4096

    python -m polyai server --listen --debug
    # /bin/bash --init-file <(echo ". .bashrc; . .docker_env/bin/activate")
}

docker-server() {
    source .env || echo ".env file read error, continuing anyway ..."
    export BUILDKIT_PROGRESS=plain && docker build -t polyai . || exit 10

    mkdir -p "$POLYAI_SERV_CACHE" "$POLYAI_MODELS"

    mkdir -p .docker_env
    docker rm polyai > /dev/null
    docker run -it --gpus all --network $POLYAI_NETWORK \
        -v "$POLYAI_SERV_CACHE:/home/user/.cache/" \
        -v "$POLYAI_MODELS:/home/user/models" \
        -v "./.docker_env:/home/user/.docker_env" \
        -v "./polyai:/home/user/polyai" \
        -v "./.env:/home/user/.env" \
        -v "./pyproject.toml:/home/user/pyproject.toml" \
        -p $POLYAI_SERV_PORT:8001 \
        -p 8002:8002 \
        --name polyai polyai \
        docker-server-entry
}

docker-test-entry() {
    ## Function to be called from inside a docker container.
    ## --------------------------------------------------------------------------------------
    docker-entry
    source .docker_env/bin/activate  || exit 102

    mkdir -p .docker_env/app
    ln -s .docker_env/app app

    ## Start terminal
    /bin/bash --init-file <(echo ". .bashrc; . .docker_env/bin/activate")
}

docker-test() {
    ## Build and run the test docker container.
    ## --------------------------------------------------------------------------------------
    source .env || exit 10
    export BUILDKIT_PROGRESS=plain && docker build -t polyai-test . || exit 20

    mkdir -p .docker_env
    docker rm polyai-test > /dev/null
    docker run -it --gpus all --network $POLYAI_NETWORK \
        -v "$POLYAI_SERV_CACHE:/home/user/.cache/" \
        -v "$POLYAI_MODELS:/home/user/models" \
        -v "./.docker_env:/home/user/.docker_env" \
        -v "./polyai:/home/user/polyai" \
        -p 8081:8080 \
        --name polyai-test polyai-test \
        docker-test-entry
}

docker-shell() {
    docker exec -it polyai /bin/bash
}

test() {
    curl    -v --data @tests/request.json \
            --header "Content-Type: application/json" \
            http://localhost:8080/api/chat/completions
    echo 
}

"$@"
