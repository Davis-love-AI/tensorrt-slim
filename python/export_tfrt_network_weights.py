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
import tensorflowrt_pb2 as tfrt_pb

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


def tensor_tf_to_tfrt(sess, tf_tensor, pb_tensor):
    """Convert a TF tensor to the TFRT format.
    """
    name = tf_tensor.op.name
    print('Proceeding with tensor:', tf_tensor.op.name, '...')
    # Get numpy array from tensor.
    a = sess.run(tf_tensor)
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
    pb_tensor.datatype = tfrt_pb.tensor.HALF if FLAGS.fp16 else tfrt_pb.tensor.FLOAT
    pb_tensor.shape[:] = a.shape
    pb_tensor.size = a.size


def network_tf_to_tfrt(sess):
    """Convert the TF graph/network to a TensorFlowRT model.
    """
    pb_network = tfrt_pb.network()
    # Convert all model variables.
    model_variables = tf.model_variables()
    for v in model_variables:
        tensor_tf_to_tfrt(sess, v, pb_network.tensors.add())
    return pb_network


def tfrt_pb_filename(checkpoint_path, fp16):
    """Generate the output filename based on the checkpoint path.
    """
    tfrt_filename = checkpoint_path
    tfrt_filename = tfrt_filename.replace('.ckpt', '')
    if fp16:
        tfrt_filename += '_fp16'
    else:
        tfrt_filename += '_fp32'
    tfrt_filename += '.tfrt'
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
        tfrt_filename = tfrt_pb_filename(FLAGS.checkpoint_path, FLAGS.fp16)
        with open(tfrt_filename, 'wb') as f:
            f.write(pb_network.SerializeToString())

if __name__ == '__main__':
    tf.app.run()
