#!/bin/bash
#           ---   Makefile like bash script for development    ---
#  --- Source this script in your terminal to set up the dev environment ---
#      --- Execute the script to see a list of available commands ---
# ------------------------------------------------------------------------------

[[ -n $MKWD ]] || export MKWD=$(echo $(command cd $(dirname '$0'); pwd))
CONDAENV=polyai

## -----------------------------------------------------------------------------

if [[ $(basename "${0}") != "make.sh" ]]; then
    # Script sourced, load or create condaenv
    if ! conda activate $CONDAENV; then
        # Create a conda environment.
        echo "Setting up $CONDAENV conda environment."
        conda create -n $CONDAENV python=3.11 -c conda-forge || return 10
        conda activate $CONDAENV || return 11

        # Install packages.
        conda install -c conda-forge cxx-compiler==1.5.2 cmake ffmpeg \
                            openmpi-mpicxx fftw || return 12
        conda install -c conda-forge cudatoolkit cudatoolkit-dev || return 13

        if [[ -f requirements.txt ]]; then pip -v install -r requirements.txt; fi
    fi

    alias cdmk="cd $MKWD/"
    alias mk="$MKWD/make.sh"
    echo "Environment set up. You can now use 'mk' to execute this script."

    return 0
fi

install() {
    cd $MKWD
    pip install .
}

run() {
    
}

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
    curl    -v --data @examples/request.json \
            --header "Content-Type: application/json" \
            http://localhost:8080/api/chat/completions
    echo 
}

## EXECUTE OR SHOW USAGE.
## -----------------------------------------------------------------------------
if [[ "$#" -lt 1 ]]; then
    echo -e "\nUSAGE:  mk <command> [options ...]"
    echo -e "\tSource this script to setup the terminal environment."
    echo -e "\nAvailable commands:"
    echo -e "------------------------------------------------------------------"
    echo -e "    install      Install the polyai package."
    echo
    echo -e "    run          Run the server."
    echo
    echo -e "    bump         Bump the current package version."
    echo
    echo -e "    tag          Create a git tag and push to origin."
    echo
else
    cd $MKWD && pwd
    "$@"
    cd $MKWD
fi
