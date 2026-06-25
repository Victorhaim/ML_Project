#!/bin/bash
# Script to run 3L-Cache docker container with appropriate mounts

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run -v "$PROJECT_ROOT/data:/3L-Cache/data" \
           -v "$PROJECT_ROOT/3LCache:/3L-Cache/3LCache" \
           -v "$PROJECT_ROOT/3LCache+:/3L-Cache/3LCache+" \
           -v "$PROJECT_ROOT/libCacheSim:/3L-Cache/libCacheSim" \
           -v "$PROJECT_ROOT/CMakeLists.txt:/3L-Cache/CMakeLists.txt" \
           -v "$PROJECT_ROOT/cmake:/3L-Cache/cmake" \
           -v "$PROJECT_ROOT/scripts:/3L-Cache/scripts" \
           -v "$PROJECT_ROOT/run_experiments.sh:/3L-Cache/run_experiments.sh" \
           -it 3lcache bash
