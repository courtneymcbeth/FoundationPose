DIR=$(pwd)

cd $DIR/mycpp/ && mkdir -p build && cd build && cmake .. -DPYTHON_EXECUTABLE=$(which python) && make -j$(nproc)
cd $DIR/bundlesdf/mycuda && rm -rf build *egg* && pip install --no-build-isolation -e .

cd ${DIR}
