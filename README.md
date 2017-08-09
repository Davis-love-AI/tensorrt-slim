# SSD - TensorRT

Implementation of the SSD network using TensorRT. Hopefully, should be very fast and optimized for inference. In addition, the overhead of porting new TensorFlow models is minimal.

## Building from source - Ubuntu 16.04

Check the following packages are installed, in addition to CUDA and TensorRT 1.0.
```bash
sudo apt-get install -y cmake libqt4-dev qt4-dev-tools libglew-dev glew-utils libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libglib2.0-dev libgflags-dev libgoogle-glog-dev protobuf-compiler libprotobuf-dev libfreetype6-dev
```

Then, use `cmake` & `make` to build the binaries.
```bash
mkdir build
cd build
cmake ../
make
```

In order to optimize the binaries and set the install directory, you can use:
```bash
export ROBIK_INSTALL=/home/ubuntu/local
cmake -DCMAKE_INSTALL_PREFIX:PATH=$ROBIK_INSTALL -DCMAKE_BUILD_TYPE=Release ../
```

Note, some libraries such as Glib and Gstreamer sometimes install development headers in some weird locations. You may need to modify the `CPLUS_INCLUDE_PATH` global variable to help the compiler finding them. For instance, on `x86-64`:
```bash
export CPLUS_INCLUDE_PATH=/usr/lib/x86_64-linux-gnu/glib-2.0/include:$CPLUS_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/usr/lib/x86_64-linux-gnu/gstreamer-1.0/include:$CPLUS_INCLUDE_PATH
```

Fix a weird OpenGL linking problem:
```bash
sudo rm /usr/lib/aarch64-linux-gnu/libGL.so
sudo ln -s /usr/lib/aarch64-linux-gnu/tegra/libGL.so /usr/lib/aarch64-linux-gnu/libGL.so
```
Trick coming from the `jetson-inference` examples.

## Python converting script TF -> TF-RT protobufs

One may first need to generate the protobuf python sources:
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

## Running some basic tests

### Benchmark a network

We can benchmark quite precisely a network using TensorRt, getting profiling time
for every layer. It gives a good overview on the bottlenecks and which part to improve.
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
    --workspace=16 \
    --height=300 \
    --width=300
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
