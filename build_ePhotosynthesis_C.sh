#!/bin/bash
set -e
if [ ! -d lpy ]; then
    git clone --branch param https://github.com/cropsinsilico/ePhotosynthesis_C.git
fi
cd ePhotosynthesis_C
if [ ! -d build ]; then
    mkdir build
fi
cd build
# -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
conda run -n canopy \
      cmake \
      -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
      -DWITH_YGGDRASIL:BOOL=ON \
      ..
cmake --build . --config Release
cmake --install .
cd ..
