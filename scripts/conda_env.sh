#!/usr/bin/env bash

env-install() {
    conda create --prefix conda_env python=3.11
	conda activate $(realpath conda_env)

	# For running pytorch with cuda
	conda install -c pytorch pytorch==2.3 -c nvidia cuda==12.1 pytorch-cuda==12.1

	# For compiling CPP programs
	conda install -c conda-forge cxx-compiler make cmake openmpi-mpicxx fftw openssh

	# For compiling with ninja and building wheels
    pip install --upgrade setuptools build wheel safetensors sentencepiece ninja
}

if [[ ! -f conda_env/bin/pip ]]; then
    env-install
	pip install exllamav2 tokenizers transformers
fi

conda activate $(realpath conda_env)
export LD_LIBRARY_PATH=$(realpath conda_env/lib):$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=/data/akhlak/_hf_home

#export PYTHONPATH=$(realpath .)

