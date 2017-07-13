# SSD - TensorRT

Implementation of the SSD network using TensorRT. Hopefully, should be very fast and optimized for inference. In addition, the overhead of porting new TensorFlow models is minimal.

## Building from source - Ubuntu 16.04

Check the following packages are installed, in addition to CUDA and TensorRT 1.0.
```bash
sudo apt-get install -y libqt4-dev qt4-dev-tools libglew-dev glew-utils libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libglib2.0-dev libgflags-dev libgoogle-glog-dev
```

Then, use `cmake` & `make` to build the binaries.
```bash
mkdir build
cd build
cmake ../
make
```
Fully optimized binaries can be generated using:
```bash
cmake -DCMAKE_BUILD_TYPE=Release ../
```

Note, some libraries such as Glib and Gstreamer sometimes install development headers in some weird locations. You may need to modify the `CPLUS_INCLUDE_PATH` global variable to help the compiler finding them. For instance, on `x86-64`:
```bash
export CPLUS_INCLUDE_PATH=/usr/lib/x86_64-linux-gnu/glib-2.0/include:$CPLUS_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/usr/lib/x86_64-linux-gnu/gstreamer-1.0/include:$CPLUS_INCLUDE_PATH
```

## Python converting script TF -> TF-RT protobufs

One may first need to generate the protobuf python sources:
```bash
protoc  --python_out=../python network.proto
```
The convertion script then works as following:
```bash
python python/export_tfrt_network_weights.py \
    --checkpoint_path=./data/networks/inception_v2_fused.ckpt \
    --input_name=Input \
    --input_height=224 \
    --input_width=224 \
    --input_shift=-127.5 \
    --input_scale=0.00784313725	\
    --outputs_name=Softmax \
    --fp16=0
```

## Running some tests...

### Benchmark a network

```bask
./tfrt_giexec \
    --modelName=inception2 \
    --modelFile=../data/networks/inception_v2_fused.tfrt16 \
    --output=InceptionV2/Softmax/output \
    --batch=2 \
    --iterations=10 \
    --avgRuns=10 \
    --half2=true \
    --verbose=true
```

## blablabla
