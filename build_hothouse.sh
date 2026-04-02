#!/bin/bash
set -e
if [ ! -d hothouse ]; then
    git clone --branch maize https://github.com/langmm/pyembree.git
fi
cd hothouse
pip install . --no-build-isolation
