#!/usr/bin/env bash

bump() {
    grep version pyproject.toml
    VERSION=$(sed -n 's/version = "\(.*\)"/\1/p' pyproject.toml)
    VERSION=$(python -c "v='$VERSION'.split('.');print('%s.%s.%02d' %(v[0], v[1], int(v[2])+1))")
    echo "   >>>"
    sed -i "s/\(version = \"\)[^\"]*\"/\1$VERSION\"/" pyproject.toml
    sed -i "s/\(__version__ = \"\)[^\"]*\"/\1$VERSION\"/" polyai/__init__.py
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

    docker rm polyai > /dev/null
    docker run -it --gpus all --network $POLYAI_NETWORK \
        -v "$POLYAI_SERV_CACHE:/home/user/.cache/" \
        -v "$POLYAI_MODELS:/home/user/models" \
        -p $POLYAI_SERV_PORT:8080 \
        --name polyai polyai
}

apitest() {
    curl    -v --data @request.json \
            --header "Content-Type: application/json" \
            http://localhost:8080/api/chat/completions
    echo 
}

"$@"

