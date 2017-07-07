# Dependencies

Need to install the following packages:
```bash
sudo apt install libprotobuf-dev libgoogle-glog-dev libgflags-dev
```

# Generating Protobuf source files

Protobuf is a mess on Ubuntu + Anaconda setup! The version coming with Ubuntu 16.04 is 2.6, whereas TensorFlow installs version 3 in the Anaconda environnment. Moreover, since no ppa is available for the Jetson TX2 arm64, we prefer to rely on version 2.6 for the C++ codebase, but use version 3 in the Python scripts.

Hence, we end up the following command for C++ generation:
```bash
/usr/bin/protoc  --cpp_out=./ tensorflowrt.proto
```
and for the Python generation:
```bash
protoc  --python_out=./python tensorflowrt.proto
```

# Convert from TF checkpoint to TF-RT protobuf format

Bash command:
```bash
python python/tfrt_export_weights.py \
    --checkpoint_path=../../data/tfrt/inception_v2_fused.ckpt \
    --fp16=0
```

# Testing and benchmarking the basic Inception2 network

```
./sample_tensorflowrt \
    --logtostderr=1 \
    --input_height=224 \
    --input_width=224 \
    --batch_size=2 \
    --checkpoint_path=../data/tfrt/inception_v2_fused_fp16.tfrt
```

