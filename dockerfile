FROM ubuntu:22.04 as base
LABEL version="0.0.2"

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /3L-Cache

# Install basic development packages and libraries
RUN apt update && apt-get install -yqq \
    cmake \
    git \
    sudo \
    wget \
    build-essential \
    curl \
    pkg-config \
    python3 \
    python3-pip \
    libglib2.0-dev \
    libgoogle-perftools-dev

# Copy the local repository files into the container
COPY . /3L-Cache

# Build and install dependencies (LightGBM, XGBoost, Zstd, etc.)
WORKDIR /3L-Cache/scripts
RUN bash ./install_dependency.sh

# Build and install 3L-Cache/libCacheSim C++ code
RUN bash ./install_libcachesim.sh

# Set working directory to built binaries by default
WORKDIR /3L-Cache/_build/bin

# Start interactive shell by default
CMD ["/bin/bash"]

