#!/usr/bin/env bash
# One-time setup on Ubuntu / WSL2: system packages, Python environment, every
# backend that the machine supports, test data, and the test suite.
#
# CUDA backend:    built when nvcc is found (CUDA toolkit for WSL-Ubuntu).
# OpenACC backend: built when nvc++ is found (NVIDIA HPC SDK); set
#                  ACC_COMPILER=g++ to build a GCC -fopenacc version instead.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if command -v apt-get >/dev/null; then
  sudo apt-get update
  sudo apt-get install -y build-essential cmake git openmpi-bin libopenmpi-dev \
    python3-venv python3-dev libgl1 libglib2.0-0
fi

for d in /usr/local/cuda/bin /opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/bin; do
  [ -d "$d" ] && export PATH="$d:$PATH"
done

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements-el.txt

# --no-build-isolation: the build must see the same cmake/pybind11 as the venv
pip install --no-build-isolation .

if command -v nvc++ >/dev/null || [ "${ACC_COMPILER:-}" = "g++" ]; then
  bash fpie/core/openacc/build.sh
else
  echo "nvc++ not found: OpenACC backends skipped"
fi

python bench/prepare_data.py
echo "backends: $(fpie --check-backend 2>/dev/null | tail -n 1)"
python -m pytest -q tests
