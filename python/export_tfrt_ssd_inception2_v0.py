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

# import network_pb2
import ssd_network_pb2
import export_tfrt_network_weights as tfrt_export

import sys
sys.path.append('../')

from preprocessing import ssd_vgg_preprocessing
from nets import ssd_common, np_methods
from nets import ssd_inception2_v0, ssd_inception2_v1

slim = tf.contrib.slim


# =========================================================================== #
# Main SSD flags.
# =========================================================================== #
tf.app.flags.DEFINE_integer('num_classes_2d', 91, 'Number of 2D classes.')
tf.app.flags.DEFINE_integer('num_classes_3d', 0, 'Number of 3D classes.')
tf.app.flags.DEFINE_integer('num_classes_seg', 0, 'Number of segmentation classes.')

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

    # Define the SSD model and parameters.
    ssd_net = ssd_inception2_v0.SSDInceptionNet()
    ssd_net.params.num_classes = FLAGS.num_classes_2d
    # Build the graph.
    with slim.arg_scope(ssd_net.arg_scope(data_format=data_format, is_training=False)):
        r = ssd_net.net(img_scaled, is_training=False, prediction_fn=tf.nn.sigmoid)
        # predictions, localisations, _, _ = r
    # SSD dynamic anchor boxes.
    ssd_anchors = ssd_net.anchors(input_shape[0:2], dynamic=True)
    return ssd_net, ssd_anchors


def ssd_network_parameters(pb_ssd_network):
    """Set the main parameters from the network.
    """
    pb_ssd_network.num_classes_2d = FLAGS.num_classes_2d
    pb_ssd_network.num_classes_3d = FLAGS.num_classes_3d
    pb_ssd_network.num_classes_seg = FLAGS.num_classes_seg
    return pb_ssd_network


def ssd_network_anchor2d(pb_ssd_network, pb_anchor, asize, ascales, ssd_anchor):
    """Convert an anchor to TF-RT protobuf format.
    """
    pass


def ssd_network_anchors2d_weight(fidx, pb_ssd_network, ssd_net, ssd_anchors):
    """Generate the scaling weights necessary to 2D anchors decoding.
    """
    N = 256
    nanchors = np.sum([len(a[1]) for a in ssd_net.params.anchor_sizes[fidx]])
    # Channel scaling.
    adrift = np.zeros([nanchors, 4], np.float32)
    ascale = np.ones([nanchors, 4], np.float32)
    apower = np.ones([nanchors, 4], np.float32)
    for i in range(nanchors):
        # Order on 2nd axis: y, x, h, w
        # y, x coordinates.
        ascale[i, 0] = ssd_net.params.prior_scaling[0]
        ascale[i, 1] = ssd_net.params.prior_scaling[1]
        # height, width.
        adrift[i, 2] = 1.0
        adrift[i, 3] = 1.0
        ascale[i, 2] = ssd_net.params.prior_scaling[2] / N
        ascale[i, 3] = ssd_net.params.prior_scaling[3] / N
        apower[i, 2] = N
        apower[i, 3] = N

    # Elementwise scaling.
    anchor_y = ssd_anchors[fidx][0]
    anchor_x = ssd_anchors[fidx][1]
    ashape = anchor_y.shape
    adrift2 = np.zeros([nanchors, 4, ashape[0], ashape[1]], np.float32)
    ascale2 = np.ones([nanchors, 4, ashape[0], ashape[1]], np.float32)
    for i in range(nanchors):
        # Order on 2nd axis: y, x, h, w
        anchor_h = ssd_anchors[fidx][2][i]
        anchor_w = ssd_anchors[fidx][3][i]
        # y, x coordinates.
        adrift2[i, 0] = anchor_y
        adrift2[i, 1] = anchor_x
        ascale2[i, 0] = anchor_h
        ascale2[i, 1] = anchor_w
        # height, width.
        ascale2[i, 0] = anchor_h
        ascale2[i, 1] = anchor_w



def ssd_network_feature(idx, pb_ssd_network, ssd_net, ssd_anchors):
    """Convert the SSD features to the appropriate format.
    """
    f = ssd_net.params.feat_layers[idx]
    pb_feature = pb_ssd_network.features.add()
    pb_feature.name = f[0]
    pb_feature.fullname = f[0]
    # Feature shape.
    # pb_feature.shape.c = ssd_anchors[i][2].shape[0]
    pb_feature.shape.c = f[1]
    pb_feature.shape.h = ssd_anchors[idx][0].shape[0]
    pb_feature.shape.w = ssd_anchors[idx][0].shape[1]
    # Prior scaling.
    pb_feature.prior_scaling.y = ssd_net.params.prior_scaling[0]
    pb_feature.prior_scaling.x = ssd_net.params.prior_scaling[1]
    pb_feature.prior_scaling.h = ssd_net.params.prior_scaling[2]
    pb_feature.prior_scaling.w = ssd_net.params.prior_scaling[3]

    # Anchors associated with the feature.
    for asize, scales in ssd_net.params.anchor_sizes[idx]:
        pb_anchor = pb_feature.anchors2d.add()
        pb_anchor.size = asize
        pb_anchor.scales[:] = scales


def ssd_network_tf_to_tfrt(sess, ssd_net, ssd_anchors):
    """Convert SSD network to TF-RT protobuf format.
    """
    pb_ssd_network = ssd_network_pb2.ssd_network()
    # Convert basic network structure.
    tfrt_export.network_tf_to_tfrt(sess, pb_ssd_network.network)
    # SSD components.
    ssd_network_parameters(pb_ssd_network)
    for idx, _ in enumerate(ssd_net.params.feat_layers):
        ssd_network_feature(idx, pb_ssd_network, ssd_net, ssd_anchors)
    return pb_ssd_network


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

        # Convert model variables and add them to protobuf network.
        print('Converting SSD weights to TensorFlowRT format.')
        pb_ssd_network = ssd_network_tf_to_tfrt(sess, ssd_net, ssd_anchors)
        # Saving protobuf TFRT model...
        tfrt_filename = tfrt_export.network_pb_filename(FLAGS.checkpoint_path, FLAGS.fp16)
        print('Export weights to: ', tfrt_filename)
        with open(tfrt_filename, 'wb') as f:
            f.write(pb_ssd_network.SerializeToString())


if __name__ == '__main__':
    tf.app.run()
