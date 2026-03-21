#!/usr/bin/bash

cwd=$(pwd)

mkdir -p build && cd build \
&& cmake -DPython3_EXECUTABLE=$(which python) .. \
&& make \
&& cd .. \
&& ./build/tensor_renderer $1 $2

cd "$cwd"
