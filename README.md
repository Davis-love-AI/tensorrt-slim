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
python python/tfrt_export_weights.py \
    --checkpoint_path=../../data/tfrt/inception_v2_fused.ckpt \
    --fp16=0
```

## Running some tests...

## blablabla
