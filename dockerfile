
FROM ubuntu:22.04 as base
LABEL version="0.0.2"

WORKDIR /

RUN apt update && apt-get install -yqq \
    cmake ninja-build git wget \
    gcc g++ build-essential \
    libglib2.0-dev \
    libzstd-dev \
    python3 python3-pip

RUN pip3 install pandas matplotlib numpy

# build and install LightGBM (required by 3LCache CMakeLists.txt)
COPY scripts/LightGBM/ /tmp/LightGBM/
RUN rm -rf /tmp/LightGBM/build && \
    mkdir -p /tmp/LightGBM/build && \
    cd /tmp/LightGBM/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# copy project source into container
COPY libCacheSim/ /build/libCacheSim/
COPY 3LCache/     /build/3LCache/
COPY 3LCache+/    /build/3LCache+/
COPY 3LCacheQL/   /build/3LCacheQL/
COPY CMakeLists.txt        /build/
COPY cmake/                /build/cmake/
COPY libCacheSim.cmake.in  /build/
COPY libCacheSim.pc.in     /build/

WORKDIR /build
RUN mkdir -p _build && cd _build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_GLCACHE=OFF -DENABLE_LRB=ON -DENABLE_TESTS=OFF && \
    make -j$(nproc)

WORKDIR /build/_build/bin/
