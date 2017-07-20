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

import numpy as np
import tensorflow as tf

import network_pb2
import ssd_network_pb2
import export_tfrt_network_weights as tfrt_export

import sys
sys.path.append('../')

from preprocessing import ssd_vgg_preprocessing
from nets import ssd_common, np_methods
from nets import ssd_inception2_v0, ssd_inception2_v1

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
    'input_height', 300, 'Input height.')
tf.app.flags.DEFINE_integer(
    'input_width', 300, 'Input width.')
tf.app.flags.DEFINE_float(
    'input_shift', -1.0, 'Input preprocessing shift.')
tf.app.flags.DEFINE_float(
    'input_scale', 0.00784313725490196, 'Input preprocessing scale.')

tf.app.flags.DEFINE_integer(
    'num_classes', 81, 'Number of classes.')

# tf.app.flags.DEFINE_string(
#     'outputs_name', 'Sotfmax', 'Name of the output tensors.')
# tf.app.flags.DEFINE_string(
#     'fix_scopes', '', 'Scopes to be modify. Format: old0:new0,old1:new1')

FLAGS = tf.app.flags.FLAGS


def build_ssd_network():
    """Build the SSD network.
    """
    # Input placeholder.
    data_format = 'NHWC'
    input_shape = (FLAGS.input_height, FLAGS.input_width, 3)
    img_input = tf.placeholder(tf.uint8, shape=input_shape)
    img_scaled = tf.expand_dims(img_input, 0)

    # Define the SSD model.
    ssd_net = ssd_inception2_v0.SSDInceptionNet()
    ssd_net.params.num_classes = FLAGS.num_classes
    with slim.arg_scope(ssd_net.arg_scope(data_format=data_format, is_training=False)):
        r = ssd_net.net(img_scaled, is_training=False, prediction_fn=tf.nn.sigmoid)
        # predictions, localisations, _, _ = r
    # SSD dynamic anchor boxes.
    ssd_anchors = ssd_net.anchors(input_shape[0:2], dynamic=True)
    return ssd_net, ssd_anchors


# =========================================================================== #
# Main converting routine.
# =========================================================================== #
def main(_):
    """Main convertion routine!
    """
    # Build the SSD network manually.
    print('Build the SSD network...')
    ssd_net, ssd_anchors = build_ssd_network()
    with tf.Session() as sess:
        # Restore SSD weights.
        print('Restoring TF graph and weights...')
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.checkpoint_path)

        # # Convert model variables and add them to protobuf network.
        # print('Converting weights to TensorFlowRT format.')
        # pb_network = network_tf_to_tfrt(sess)
        # # Saving protobuf TFRT model...
        # tfrt_filename = network_pb_filename(FLAGS.checkpoint_path, FLAGS.fp16)
        # print('Export weights to: ', tfrt_filename)
        # with open(tfrt_filename, 'wb') as f:
        #     f.write(pb_network.SerializeToString())

if __name__ == '__main__':
    tf.app.run()
