#python3 -m venv venv && source venv/bin/activate && pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
#conda deactivate
source venv/bin/activate
mkdir -p build && cd build
cmake ..
make
cd ..
./tensor_renderer
