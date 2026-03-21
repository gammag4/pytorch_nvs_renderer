#!/usr/bin/bash

cwd=$(pwd)

conda activate $1 \
&& mkdir -p build && cd build \
&& cmake -DPython3_EXECUTABLE=$(which python) .. \
&& make \
&& cd .. \
&& ./build/tensor_renderer $2 $3

cd "$cwd"

