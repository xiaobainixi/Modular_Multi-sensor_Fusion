#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  curl \
  unzip \
  python3 \
  python3-yaml \
  python3-numpy \
  python3-matplotlib \
  libopencv-dev \
  libceres-dev \
  libgoogle-glog-dev \
  libeigen3-dev \
  libsuitesparse-dev \
  libglew-dev \
  libblas-dev \
  liblapack-dev

cmake -S "${REPO_ROOT}" -B "${REPO_ROOT}/build"
cmake --build "${REPO_ROOT}/build" -j"$(nproc)"
