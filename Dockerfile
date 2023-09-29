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
RUN python -m pip install pip -U

# Setup user
RUN useradd -m -s /bin/bash -u 1000 user
ENV PATH="/home/user/.local/bin:${PATH}"
ARG PATH="/home/user/.local/bin:${PATH}"
SHELL [ "/bin/bash", "-c" ]

# Install certificate authority (needed for clients)
# RUN apt-get install -y ca-certificates
# COPY ./my_ssl_CA.crt /usr/local/share/ca-certificates/my_ssl_CA.crt
# ENV NODE_EXTRA_CA_CERTS=/usr/local/share/ca-certificates/my_ssl_CA.crt
# RUN update-ca-certificates

# Copy the SSL certificate (for servers)
COPY keys/ /home/user/keys/
RUN chown -R user /home/user/keys/

# Setup app
USER user
WORKDIR /home/user/


# Setup entry
COPY        build.sh      build.sh
ENTRYPOINT ["/bin/bash", "/home/user/build.sh"]
