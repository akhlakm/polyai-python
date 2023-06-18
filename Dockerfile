# While running, make sure to
#   - Install nvidia-docker using apt/dnf

# Available versions: https://hub.docker.com/r/nvidia/cuda/tags
# Make sure to choose the devel version for nvcc and other tools.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         gfortran \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
         wget \
         google-perftools \
         python3 \
         python3-pip \
         python3-venv

RUN apt-get install -y pkg-config python3-dev
RUN apt-get install -y ffmpeg libsm6 libxext6
RUN apt-get install -y libcairo2-dev libjpeg-dev libgif-dev

# Setup system
RUN ln -s $(which python3) /bin/python

# Setup user
RUN useradd -m -s /bin/bash -u 1000 user
ENV PATH="/home/user/.local/bin:${PATH}"
ARG PATH="/home/user/.local/bin:${PATH}"
SHELL [ "/bin/bash", "-c" ]

# Setup app
USER user
WORKDIR /home/user/

# Install cuda
RUN pip install -vv torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu118

## --------------------------------------------------------------------------------------
# Install application
RUN python -m pip install pip -U
COPY polyai polyai
COPY requirements.txt polyai/requirements.txt
RUN  pip install -r polyai/requirements.txt

ENV PYTHONPATH=/home/user

ENTRYPOINT ["/bin/python", "polyai/__main__.py", "server"]
# ENTRYPOINT ["/bin/bash", "-i"]


