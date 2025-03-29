#!/bin/bash
set -e
if [ ! -d hothouse ]; then
    git clone --branch maize git@github.com:langmm/hothouse.git
fi
cd hothouse
pip install . --no-build-isolation
