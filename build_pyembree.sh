#!/bin/bash
set -e
conda update -n lpy --file environment_raytrace.yml
if [ ! -d pyembree ]; then
    git clone git@github.com:langmm/pyembree.git
fi
cd pyembree
python -m ensurepip --upgrade
python -m pip install --upgrade setuptools
pip install . --no-build-isolation
