#!/bin/bash
set -e
if [ ! -d yggdrasil ]; then
    git clone --branch topic/cache https://github.com/cropsinsilico/yggdrasil.git
fi
cd yggdrasil
pip install .  # --no-build-isolation
