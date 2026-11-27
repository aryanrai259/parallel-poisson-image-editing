#!/usr/bin/env bash
# Build the OpenACC backends into the installed fpie package.
#
#   ACC_COMPILER=nvc++ (default)  NVIDIA HPC SDK:
#       fpie.core_openacc     -acc=gpu        (backend "openacc")
#       fpie.core_openacc_mc  -acc=multicore  (backend "openacc-mc")
#   ACC_COMPILER=g++              GCC -fopenacc; set ACC_GCC_OFFLOAD=-foffload=nvptx-none
#                                 for GPU offload (needs gcc-<ver>-offload-nvptx),
#                                 otherwise regions run on the host.
#   ACC_GPU=cc86                  compute capability for nvc++ (RTX 30xx = cc86).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python3}"
# -I (isolated mode) keeps the repo's cwd off sys.path, so this resolves the
# venv's installed fpie package instead of the source tree's fpie/ directory
# (which would otherwise shadow it when this script runs from the repo root).
pkg_dir="$("$python_bin" -I -c 'import os, fpie; print(os.path.dirname(fpie.__file__))')"
ext_suffix="$("$python_bin" -I -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
includes="$("$python_bin" -m pybind11 --includes) -I${here}/.."
sources="${here}/equ.cc ${here}/grid.cc ${here}/solver.cc"
compiler="${ACC_COMPILER:-nvc++}"

case "${compiler}" in
  nvc++*)
    common="-O3 -std=c++17 -fPIC -shared --diag_suppress=code_is_unreachable"
    echo "[openacc] ${compiler} -acc=gpu -gpu=${ACC_GPU:-cc86} -> core_openacc${ext_suffix}"
    "${compiler}" ${common} -acc=gpu -gpu="${ACC_GPU:-cc86}" -Minfo=accel ${includes} \
      -DFPIE_ACC_MODULE=core_openacc ${sources} -o "${pkg_dir}/core_openacc${ext_suffix}"
    echo "[openacc] ${compiler} -acc=multicore -> core_openacc_mc${ext_suffix}"
    "${compiler}" ${common} -acc=multicore -Minfo=accel ${includes} \
      -DFPIE_ACC_MODULE=core_openacc_mc ${sources} -o "${pkg_dir}/core_openacc_mc${ext_suffix}"
    ;;
  g++*|*gcc*)
    echo "[openacc] ${compiler} -fopenacc ${ACC_GCC_OFFLOAD:-} -> core_openacc${ext_suffix}"
    "${compiler}" -O3 -std=c++17 -fPIC -shared -fopenacc ${ACC_GCC_OFFLOAD:-} ${includes} \
      -DFPIE_ACC_MODULE=core_openacc ${sources} -o "${pkg_dir}/core_openacc${ext_suffix}"
    ;;
  *)
    echo "unsupported ACC_COMPILER=${compiler}" >&2
    exit 1
    ;;
esac
echo "[openacc] installed into ${pkg_dir}"
