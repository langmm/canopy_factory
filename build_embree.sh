#!/bin/bash
set -e
if [ ! -d embree ]; then
    git clone --branch v2.17.7 git@github.com:RenderKit/embree.git
fi
cd embree
if [ ! -d build ]; then
    mkdir build
fi
cd build
touch ${CONDA_PREFIX}/include/tbb/task_scheduler_init.h
mamba run -n canopy \
      cmake -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
      -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} \
      -DCMAKE_BUILD_TYPE=Release \
      -DEMBREE_ISPC_SUPPORT=OFF \
      -DCMAKE_POLICY_DEFAULT_CMP0074=NEW \
      -DTBB_ROOT=${CONDA_PREFIX} \
      ..
mamba run -n canopy make -j 8
mamba run -n canopy make install
cd ..
