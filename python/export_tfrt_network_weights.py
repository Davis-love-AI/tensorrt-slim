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
    'outputs_name', '', 'Name of the output tensors.')

tf.app.flags.DEFINE_string(
    'fix_scopes', '', 'Scopes to be modify. Format: old0:new0,old1:new1')
tf.app.flags.DEFINE_boolean(
    'fix_1x1_transpose', False, 
    'Fix 1x1 transpose convolution by replacing with 2x2 faster ones.')

FLAGS = tf.app.flags.FLAGS


def restore_checkpoint(sess, ckpt_filename, moving_average_decay=None):
    """Restore variables from a checkpoint file. Using either normal values
    or moving averaged values.
    """
    tf_global_step = slim.get_or_create_global_step()
    # Restore moving average variables or classic stuff!
    if moving_average_decay:
        print('Restoring moving average variables.')
        variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, tf_global_step)
        variables_to_restore = variable_averages.variables_to_restore(
            tf.contrib.framework.get_model_variables())
        variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
        print('Restoring last batch variables.')
        variables_to_restore = tf.contrib.framework.get_variables_to_restore()
    # Restore method.
    fn_restore = slim.assign_from_checkpoint_fn(ckpt_filename,
                                                variables_to_restore,
                                                ignore_missing_vars=True)
    fn_restore(sess)


def parse_fix_scopes():
    """Parse the fix_scopes FLAG into a list of pairs.
    """
    l_fix_scopes = FLAGS.fix_scopes.split(',')
    l_fix_scopes = [a.split(':') for a in l_fix_scopes if len(a)]
    l_fix_scopes = [(a[0], a[1]) for a in l_fix_scopes]
    return l_fix_scopes


def fix_scope_name(name, fix_scopes):
    """Fix a scope name.
    """
    for a in fix_scopes:
        if a[0] in name:
            new = name.replace(a[0], a[1])
            print('SCOPE fixing name from \'%s\' to \'%s\'' % (name, new))
            name = new
    return name


def tf_get_model_variable(sess, name):
    """Get the value of a model variable.
    """
    model_variables = tf.model_variables()
    v = next((a for a in model_variables if a.op.name == name), None)
    if v is None:
        print('WARNING: Could not find variable', name)
        return np.array([], np.float32)
    a = sess.run(v)
    return a


def tensor_to_float(a, name=''):
    """Convert a tensor to float16 or float32, depending on the type FLAGS.
    """
    if FLAGS.fp16:
        minval = np.finfo(np.float16).min
        maxval = np.finfo(np.float16).max
        mask = np.logical_or(a < minval, a > maxval)
        if np.sum(mask):
            print('WARNING: FP16 infinite values in the tensor', name)
        a = a.astype(np.float16)
    else:
        a = a.astype(np.float32)
    return a


def tensor_np_to_tfrt(sess, name, np_tensor, pb_tensor, permutation=[3, 2, 0, 1]):
    """Convert a numpy Tensor + name to TF-RT tensor format.
    """
    a = np_tensor
    # Permutation of axes.
    if permutation and a.ndim == len(permutation):
        a = np.transpose(a, axes=permutation)
    
    # Modify 'depthwise weights'
    if 'depthwise_weights' in name:
        # GKCRS order. G == nb groups, K: nb outputs, C: nb inputs.
        # a = np.expand_dims(a, axis=0)
        # a = np.transpose(a, axes=[2, 1, 0, 3, 4])

        # BUG: problem with group convolution in TensorRT 2.1?
        # Switch to normal one: reshape weights accordingly.
        # output_depth = multiplier * input.
        print('Fixing TensorRT bug in group convolution by switching to classic convolution.')
        shape = a.shape
        shape_conv = [shape[0]*shape[1], shape[1], shape[2], shape[3]]
        a_conv = np.zeros(shape_conv, a.dtype)
        # Reconstruct classic convolution weights.
        for i in range(shape[1]):
            for j in range(shape[0]):
                a_conv[i*shape[0] + j, i] = a[j, i]
        # for i in range(shape[1]):
        #     a_conv[i*shape[0]:(i+1)*shape[0], i] = a[:, i]
        a = a_conv

    # Fixing 1x1 transpose convolution by replacing with 2x2 kernel.
    if FLAGS.fix_1x1_transpose and a.ndim == 4 and a.shape[2] == 1 and a.shape[3] == 1 and 'tconv_weights' in name:
        print('Fixing 1x1 transpose convolution by replacing with 2x2 kernel in:', name)
        shape = list(a.shape)
        shape[2] = shape[3] = 2
        a_rt = np.zeros(shape, a.dtype)
        a_rt[:, :, 0, 0] = np.squeeze(a)
        a =  a_rt

    # Batch norm moving variables: transform into scaling parameters.
    # Must satisfy the equation y = s*x + b
    if 'BatchNorm/moving_mean' in name:
        # Need the variance to fix the coefficient...
        v = tf_get_model_variable(sess, str.replace(name, 'moving_mean', 'moving_variance'))
        a = -a / np.sqrt(v)
        # Incorporate gamma and beta coef.
        b = tf_get_model_variable(sess, str.replace(name, 'moving_mean', 'beta'))
        g = tf_get_model_variable(sess, str.replace(name, 'moving_mean', 'gamma'))
        a = a * g + b
    if 'BatchNorm/moving_variance' in name:
        # Change to scale convention.
        a = 1. / np.sqrt(a)
        # Incorporate gamma coef.
        g = tf_get_model_variable(sess, str.replace(name, 'moving_variance', 'gamma'))
        a = a * g
    if 'BatchNorm/beta' in name:
        # Not needed anymore after fusion.
        a = np.array([], np.float32)
    if 'BatchNorm/gamma' in name:
        # Not needed anymore after fusion.
        a = np.array([], np.float32)

    # Convert to half-precision if necessary.
    a = tensor_to_float(a, name=name)
    # Fill protobuf tensor fields.
    fix_scopes = parse_fix_scopes()
    pb_tensor.name = fix_scope_name(name, fix_scopes)
    pb_tensor.data = a.tobytes()
    pb_tensor.datatype = network_pb2.HALF if FLAGS.fp16 else network_pb2.FLOAT
    pb_tensor.shape[:] = a.shape
    pb_tensor.size = a.size


def tensor_tf_to_tfrt(sess, tf_tensor, pb_tensor):
    """Convert a TF tensor to the TFRT tensor format.
    """
    name = tf_tensor.op.name
    # Get numpy array from tensor and convert it.
    a = sess.run(tf_tensor)
    print('Proceeding with tensor: \'' + name + '\' with shape', a.shape)
    tensor_np_to_tfrt(sess, name, a, pb_tensor)


def network_name(pb_network):
    """Try to deduce the network name from the TF variables scope.
    """
    vname = tf.model_variables()[0].op.name
    pb_network.name = vname.split('/')[0]
    print('Network name:', pb_network.name)
    return pb_network.name


def network_input(sess, pb_network):
    """Set the input parameters of the network.
    """
    pb_network.input.name = FLAGS.input_name
    pb_network.input.h = FLAGS.input_height
    pb_network.input.w = FLAGS.input_width
    pb_network.input.c = 3
    # Only UNIFORM scaling mode available for now...
    pb_network.input.scalemode = network_pb2.UNIFORM
    # Create shift and scale weights if necessary.
    if FLAGS.input_shift != 0.0:
        a = np.array([FLAGS.input_shift], np.float32)
        name = pb_network.name + '/' + FLAGS.input_name + '/shift'
        tensor_np_to_tfrt(sess, name, a, pb_network.weights.add())
    if FLAGS.input_scale != 1.0:
        a = np.array([FLAGS.input_scale], np.float32)
        name = pb_network.name + '/' + FLAGS.input_name + '/scale'
        tensor_np_to_tfrt(sess, name, a, pb_network.weights.add())

    print('Input name: ', pb_network.input.name)
    print('Input CHW shape: ', [pb_network.input.c, pb_network.input.h, pb_network.input.w])
    print('Input shift and scale: ', [FLAGS.input_shift, FLAGS.input_scale])


def network_outputs(sess, pb_network):
    """Set the outputs collection of the network.
    """
    l_outputs = FLAGS.outputs_name.split(',')
    for i, o in enumerate(l_outputs):
        if len(o):
            net_output = pb_network.outputs.add()
            net_output.name = o
            net_output.suffix = 'output'
            # TODO: fix this crap for SSD networks.
            net_output.h = 1
            net_output.w = 1
            net_output.c = 1
            print('Output #%i name: %s' % (i, net_output.name))


def network_tf_to_tfrt(sess, pb_network=network_pb2.network()):
    """Convert the TF graph/network to a TensorFlowRT model.
    """
    # Convert all model weights.
    model_variables = tf.model_variables()
    for v in model_variables:
        tensor_tf_to_tfrt(sess, v, pb_network.weights.add())
    # Network parameters.
    pb_network.datatype = network_pb2.HALF if FLAGS.fp16 else network_pb2.FLOAT
    network_name(pb_network)
    network_input(sess, pb_network)
    network_outputs(sess, pb_network)
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
        print('Export weights to: ', tfrt_filename)
        with open(tfrt_filename, 'wb') as f:
            f.write(pb_network.SerializeToString())

if __name__ == '__main__':
    tf.app.run()
