#!/bin/bash
set -e
export PLANTGL_ROOT="${CONDA_PREFIX}"
if [ ! -d lpy ]; then
    git clone git@github.com:openalea/lpy.git
fi
cd lpy
if [ ! -d build ]; then
    mkdir build
fi
cd build
conda run -n canopy \
      cmake -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
      -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} \
      -DCMAKE_BUILD_TYPE=Release \
      -DPython3_ROOT_DIR=${CONDA_PREFIX} \
      -DPython3_FIND_STRATEGY=LOCATION \
      ..
conda run -n canopy make -j 8
conda run -n canopy make install
cd ..
conda run -n canopy pip install .
