# TensorFlowRT

This library is wrapper on top of TensorRt, easing the port of TensorFlow Neural Nets to the efficient TensorRT inference engine. It includes in particular:
* TF layers-like API, for quickly defining a network in C++;
* Import / export weights of TF models into protobuf binary files;
* Segmentation and SSD models implemented;
* CUDA tensor-like API
* Additional modules built on top of Visionworks: stabilization, ...


# Building from source - Ubuntu 16.04

## Dependencies

Check the following packages are installed on Ubuntu:
```bash
sudo apt-get install -y cmake libqt4-dev qt4-dev-tools libglew-dev glew-utils libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libglib2.0-dev libgflags-dev libgoogle-glog-dev protobuf-compiler libprotobuf-dev libfreetype6-dev
```

In addition, one needs to install a few (!) NVIDIA libraries for developping on the Jetson platform. More specifically, install the latest [JetPack 3.1](https://developer.nvidia.com/embedded/jetpack), which includes:
* CUDA (version 8)
* VisionWorks 1.6
* TensorRT 2.1 (can be downloaded directly here: https://developer.nvidia.com/tensorrt)

Finally, there are a few additional manual tasks:
* In the directory `ssd-tensorrt/3rdparty/glfw3/lib`, rename/copy `libglfw3_x64.a` into `libglfw3.a`
* Download the latest stable release of [EIGEN](http://eigen.tuxfamily.org/) and extract it in the directory `3rdparty/eigen`.

Hopefully, you should be done with dependencies after that.!

## Building

Then, use `cmake` & `make` to build the binaries.
```bash
export ROBIK_INSTALL=$HOME/local
mkdir $ROBIK_INSTALL
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=$ROBIK_INSTALL ../
make -j2
make install
```

In order to optimize the binaries, you can use:
```bash
export ROBIK_INSTALL=$HOME/local
cmake -DCMAKE_INSTALL_PREFIX:PATH=$ROBIK_INSTALL -DCMAKE_BUILD_TYPE=Release ../
```
And if you need to specify the CUDA install directory:
```bash
cmake -DCMAKE_INSTALL_PREFIX:PATH=$ROBIK_INSTALL -DCMAKE_BUILD_TYPE=Release -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 ../
```

Note, some libraries such as Glib and Gstreamer sometimes install development headers in some weird locations. You may need to modify the `CPLUS_INCLUDE_PATH` global variable to help the compiler finding them. For instance, on `x86-64`:
```bash
export CPLUS_INCLUDE_PATH=/usr/lib/x86_64-linux-gnu/glib-2.0/include:$CPLUS_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/usr/lib/x86_64-linux-gnu/gstreamer-1.0/include:$CPLUS_INCLUDE_PATH
```

Finally, on the Jetson TX2, you may face a weird OpenGL linking problem which can be solved this way:
```bash
sudo rm /usr/lib/aarch64-linux-gnu/libGL.so
sudo ln -s /usr/lib/aarch64-linux-gnu/tegra/libGL.so /usr/lib/aarch64-linux-gnu/libGL.so
```
Note, this trick is coming from the `jetson-inference` Github examples.

# Python converting script TF => TF-RT protobufs

In order to use a TensorFlow model in TF-RT, you first need to export the weights in some protobuf binary file.
To start with, generate the protobuf python sources:
```bash
protoc  --python_out=../python network.proto
```

A common imagenet classification network can be exported as following:
```bash
python python/export_tfrt_network_weights.py \
    --checkpoint_path=./data/networks/inception_v2_fused.ckpt \
    --fix_scopes=Mixed_5b/Branch_2/Conv2d_0a_3x3:Mixed_5b/Branch_2/Conv2d_0b_3x3 \
    --input_name=Input \
    --input_height=224 \
    --input_width=224 \
    --input_shift=-1 \
    --input_scale=0.00784313725490196 \
    --outputs_name=Softmax \
    --fp16=1
```

In the case of SSD networks, we use the specialized script:
```bash
python export_tfrt_ssd_inception2_v0.py \
    --checkpoint_path=../checkpoints/ssd_inception2_v0_orig.ckpt \
    --input_name=Input \
    --input_height=300 \
    --input_width=300 \
    --input_shift=-1 \
    --input_scale=0.00784313725490196 \
    --num_classes_2d=91 \
    --fp16=0
```

# Running some basic tests

### Benchmark a network

We can benchmark quite precisely a network using TensorRT, getting profiling time
for every layer. It gives a good overview on the bottlenecks in the network and which parts to improve.
```bask
GLOG_logtostderr=1 ./tfrt_giexec \
    --modelName=inception2 \
    --modelFile=../data/networks/inception_v2_fused.tfrt16 \
    --batch=2 \
    --iterations=10 \
    --avgRuns=10 \
    --workspace=16 \
    --height=300 \
    --width=300 \
    --verbose=true
```

```bask
GLOG_logtostderr=1 ./tfrt_benchmark \
    --network=inception2 \
    --network_pb=../data/networks/inception_v2_fused.tfrt16 \
    --batch_size=2 \
    --workspace=32 \
    --height=256 \
    --width=256
```

### Classification on image and video inputs

```bask
./imagenet_console --alsologtostderr \
    --network=inception2 \
    --network_pb=../data/networks/inception_v2_fused.tfrt16 \
    --imagenet_info=../data/networks/ilsvrc12_synset_words.txt \
    --image=../data/images/orange_0.jpg \
    --image_save=0
```

```bask
./imagenet_camera --alsologtostderr \
    --network=inception2 \
    --network_pb=../data/networks/inception_v2_fused.tfrt16 \
    --imagenet_info=../data/networks/ilsvrc12_synset_words.txt
```

### SSD detection on image and video inputs

```bask
./ssdnet_console --alsologtostderr \
    --network=ssd_inception2_v0 \
    --network_pb=../data/networks/ssd_inception2_v0_orig.tfrt32 \
    --image=../data/images/peds-001.jpg \
    --max_detections=200 \
    --threshold=0.5 \
    --image_save=0
```

## Video encoding

Install `ffmpeg` and additional libraries:
```bash
sudo apt-get install yasm libvpx. libx264. ffmpeg
```
Convert the video to `h264` format:
```bash
ffmpeg -i camera1-7JUN17.avi -acodec aac -vcodec libx264  camera1-7JUN17-enc.avi
```

### Camera output format

How to get the list of output formats supported by a camera:
```bash
sudo apt install v4l-utils
v4l2-ctl --list-formats-ext -d 1
```

## Robik AI demos!

```bash
./aarch64/bin/demo_single_input_stabilizer --alsologtostderr \
    --stab_crop_margin=-1 \
    --stab_num_smoothing_frames=5 \
    --source "device:///v4l2?index=1" \
    --source_width=1280 \
    --source_height=720 \
    --source_fps=60 \
    --net_width=400 \
    --net_height=225 \
    --display_fullscreen=false
```
