# ============================================================================
# [2017] - Robik AI Ltd - Paul Balanca
# All Rights Reserved.

# NOTICE: All information contained herein is, and remains
# the property of Robik AI Ltd, and its suppliers
# if any.  The intellectual and technical concepts contained
# herein are proprietary to Robik AI Ltd
# and its suppliers and may be covered by U.S., European and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
# Dissemination of this information or reproduction of this material
# is strictly forbidden unless prior written permission is obtained
# from Robik AI Ltd.
# ============================================================================
"""Script exporting the weights from a standard TensorFlow checkpoint
to a TensorFlowRT protobuf file.

Note:
 - TF weights format:
    [filter_height, filter_width, input_depth, output_depth]
    [filter_height, filter_width, in_channels, channel_multiplier]
 - TensorRT weights format: GKCRS order, where G is the number of groups,
    K the number of output feature maps,
    C the number of input channels, and
    R and S are the height and width of the filter.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import sys
# import argparse

import numpy as np
import tensorflow as tf
import network_pb2

slim = tf.contrib.slim


# =========================================================================== #
# Main flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'checkpoint_path', None, 'Path of the checkpoint files.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None, 'Moving average decay value. None if not used.')
tf.app.flags.DEFINE_boolean(
    'fp16', False, 'Export weights in half-precision fp16 format.')

tf.app.flags.DEFINE_string(
    'input_name', 'Input', 'Name of the input tensor.')
tf.app.flags.DEFINE_integer(
    'input_height', 224, 'Input height.')
tf.app.flags.DEFINE_integer(
    'input_width', 224, 'Input width.')
tf.app.flags.DEFINE_float(
    'input_shift', 0.0, 'Input preprocessing shift.')
tf.app.flags.DEFINE_float(
    'input_scale', 1.0, 'Input preprocessing scale.')

tf.app.flags.DEFINE_string(
    'outputs_name', 'Sotfmax', 'Name of the output tensors.')

FLAGS = tf.app.flags.FLAGS


def restore_checkpoint(sess, ckpt_filename, moving_average_decay=None):
    """Restore variables from a checkpoint file. Using either normal values
    or moving averaged values.
    """
    tf_global_step = slim.get_or_create_global_step()
    # Restore moving average variables or classic stuff!
    if moving_average_decay:
        variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, tf_global_step)
        variables_to_restore = variable_averages.variables_to_restore(
            tf.contrib.framework.get_model_variables())
        variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
        variables_to_restore = tf.contrib.framework.get_variables_to_restore()
    # Restore method.
    fn_restore = slim.assign_from_checkpoint_fn(ckpt_filename,
                                                variables_to_restore,
                                                ignore_missing_vars=True)
    fn_restore(sess)


def tensor_np_to_tfrt(name, np_tensor, pb_tensor):
    """Convert a numpy Tensor + name to TF-RT tensor format.
    """
    a = np_tensor
    # Permutation of axes.
    if a.ndim == 4:
        a = np.transpose(a, axes=[3, 2, 0, 1])
    # Modify 'depthwise weights'
    if 'depthwise_weights' in name:
        # GKCRS order. G == nb inputs, K == depth multiplier.
        a = np.expand_dims(a, axis=0)
        a = np.transpose(a, axes=[2, 1, 0, 3, 4])
    # Batch norm moving variables: transform into scaling parameters.
    if 'BatchNorm/moving_mean' in name:
        a = -a
    if 'BatchNorm/moving_variance' in name:
        a = 1. / np.sqrt(a)
    # TODO: fuse with scaling parameters beta and gamma...

    # Convert to half-precision if necessary.
    if FLAGS.fp16:
        minval = np.finfo(np.float16).min
        maxval = np.finfo(np.float16).max
        mask = np.logical_or(a < minval, a > maxval)
        if np.sum(mask):
            print('WARNING: FP16 infinite values in the tensor', name)
        a = a.astype(np.float16)

    # Fill protobuf tensor fields.
    pb_tensor.name = name
    pb_tensor.data = a.tobytes()
    pb_tensor.datatype = network_pb2.HALF if FLAGS.fp16 else network_pb2.FLOAT
    pb_tensor.shape[:] = a.shape
    pb_tensor.size = a.size


def tensor_tf_to_tfrt(sess, tf_tensor, pb_tensor):
    """Convert a TF tensor to the TFRT tensor format.
    """
    name = tf_tensor.op.name
    print('Proceeding with tensor:', tf_tensor.op.name, '...')
    # Get numpy array from tensor and convert it.
    a = sess.run(tf_tensor)
    tensor_np_to_tfrt(name, a, pb_tensor)


def network_name(pb_network):
    """Try to deduce the network name from the TF variables scope.
    """
    vname = tf.model_variables()[0].op.name
    pb_network.name = vname.split('/')[0]
    print('Network name:', pb_network.name)
    return pb_network.name


def network_input(pb_network):
    """Set the input parameters of the network.
    """
    pb_network.input.name = pb_network.name + "/" + FLAGS.input_name
    pb_network.input.h = FLAGS.input_height
    pb_network.input.w = FLAGS.input_width
    pb_network.input.c = 3
    # Only UNIFORM scaling mode available for now...
    pb_network.input.scalemode = network_pb2.UNIFORM
    # Create shift and scale weights if necessary.
    if FLAGS.input_shift != 0.0:
        a = np.array([FLAGS.input_shift])
        name = pb_network.name + '/' + FLAGS.input_name + '/shift'
        tensor_np_to_tfrt(name, a, pb_network.weights.add())
    if FLAGS.input_scale != 1.0:
        a = np.array([FLAGS.input_scale])
        name = pb_network.name + '/' + FLAGS.input_name + '/scale'
        tensor_np_to_tfrt(name, a, pb_network.weights.add())

    print('Input name: ', pb_network.input.name)
    print('Input shape: [%i, %i, %i]' % (pb_network.input.h, pb_network.input.w, pb_network.input.c))
    print('Input shift and scale: ', [FLAGS.input_shift, FLAGS.input_scale])


def network_outputs(pb_network):
    """Set the outputs collection of the network.
    """
    l_outputs = FLAGS.outputs_name.split(',')
    for i, o in enumerate(l_outputs):
        net_output = pb_network.outputs.add()
        net_output.name = pb_network.name + "/" + o
        # TODO: fix this crap for SSD networks.
        net_output.h = 1
        net_output.w = 1
        net_output.c = 1
        print('Output #%i name: %s' % (i, net_output.name))


def network_tf_to_tfrt(sess):
    """Convert the TF graph/network to a TensorFlowRT model.
    """
    pb_network = network_pb2.network()
    # Convert all model weights.
    model_variables = tf.model_variables()
    for v in model_variables:
        tensor_tf_to_tfrt(sess, v, pb_network.weights.add())
    # Network parameters.
    pb_network.datatype = network_pb2.HALF if FLAGS.fp16 else network_pb2.FLOAT
    network_name(pb_network)
    network_input(pb_network)
    network_outputs(pb_network)
    return pb_network


def network_pb_filename(checkpoint_path, fp16):
    """Generate the output filename based on the checkpoint path.
    """
    tfrt_filename = checkpoint_path
    tfrt_filename = tfrt_filename.replace('.ckpt', '')
    if fp16:
        tfrt_filename += '.tfrt16'
    else:
        tfrt_filename += '.tfrt32'
    return tfrt_filename


# =========================================================================== #
# Main converting routine.
# =========================================================================== #
def main(_):
    """Main convertion routine!
    """
    with tf.Session() as sess:
        # Restore graph + checkpoint.
        print('Restoring TF graph and weights...')
        metafile = FLAGS.checkpoint_path + '.meta'
        saver = tf.train.import_meta_graph(metafile)
        saver.restore(sess, FLAGS.checkpoint_path)
        # Convert model variables and add them to protobuf network.
        print('Converting weights to TensorFlowRT format.')
        pb_network = network_tf_to_tfrt(sess)
        # Saving protobuf TFRT model...
        tfrt_filename = network_pb_filename(FLAGS.checkpoint_path, FLAGS.fp16)
        with open(tfrt_filename, 'wb') as f:
            f.write(pb_network.SerializeToString())

if __name__ == '__main__':
    tf.app.run()