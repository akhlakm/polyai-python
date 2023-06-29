#!/usr/bin/env bash
# This script will set up a conda environment and 
# install the GCC and CUDA tools to run the polyai server.
# Note!! This script must be sourced.

if [ $(basename $0) = "conda_env.sh" ]; then
    echo "Please source this script."
    exit 1  # not sourced
fi

if [[ ! -f conda_env/bin/pip ]]; then
    conda create --prefix conda_env python=3.10 -c conda-forge
    conda activate $(realpath conda_env)

    # Needed for exllama
    conda install -c conda-forge cxx-compiler
    # For NVCC and cuda
    conda install -c conda-forge cudatoolkit cudatoolkit-dev
    # Torch and cuda
    pip install -vv torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu118
fi

conda activate $(realpath conda_env)
export LD_LIBRARY_PATH=$(realpath conda_env/lib64):$LD_LIBRARY_PATH

#export PYTHONPATH=$(realpath .)
