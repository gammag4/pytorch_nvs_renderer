conda activate nvs_renderer
mkdir -p build && cd build
cmake -DPython3_EXECUTABLE=$(which python) ..
make
cd ..
./tensor_renderer
