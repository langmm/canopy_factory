#!/bin/bash
set -e
if [ ! -d pyembree ]; then
    git clone --branch callbacks https://github.com/langmm/pyembree.git
fi
cd pyembree
python -m ensurepip --upgrade
python -m pip install --upgrade setuptools
pip install . --no-build-isolation
