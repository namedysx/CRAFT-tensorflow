import tensorflow as tf
import tensorflow.contrib.slim as slim
from upconvBlock import upconvBlock, upsample, conv2d_transpose_strided
from vgg import vgg_16, vgg_arg_scope


def arous_conv(x, filter_height, filter_width, num_filters, rate, name):
        # Get number of input channels
        input_channels = int(x.get_shape()[-1])
        with tf.variable_scope(name) as scope:
            # Create tf variables for the weights and biases of the arous_conv layer
            weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters])
            biases = tf.get_variable('biases', shape=[num_filters])
    
            arousconv = tf.nn.atrous_conv2d(x, weights, rate=rate, padding='SAME')
    
            bias = tf.nn.bias_add(arousconv, biases)
            return bias

def CRAFT_net(inputs,
              is_trianing=True,
              reuse=None,
              weight_decay=0.9):
    with slim.arg_scope(vgg_arg_scope()):
        vgg_res, end_points = vgg_16(inputs)
    with tf.variable_scope('vgg_16', [end_points.values]):
        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_trianing
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            f = [end_points['vgg_16/conv2/conv2_2'], end_points['vgg_16/conv3/conv3_3'],
                 end_points['vgg_16/conv4/conv4_3'], end_points['vgg_16/conv5/conv5_3']]
            net = f[3]
            # VGG end
            net = slim.max_pool2d(net, [3, 3], stride=1, padding='SAME', scope='pool5')  # w/16 512
            net = arous_conv(net, 3, 3, 1024, 6, name='arous_conv')  # w/16 1024
            net = slim.conv2d(net, 1024, [1, 1], padding='SAME', scope='conv6')     # w/16 1024

            # U-net start
            net = tf.concat([net, f[3]], axis=3)       # w/16 1024 + 512
            net = upconvBlock(net, 512, 256)        # w/16 256
            net = upsample(net, (64, 64))
            net = tf.concat([net, f[2]], axis=3)  # w/8 256 + 512
            net = upconvBlock(net, 256, 128)  # w/8 128
            net = upsample(net, (128, 128))
            net = tf.concat([net, f[1]], axis=3)    # w/4 128 + 256
            net = upconvBlock(net, 128, 64)  # w/4 64
            net = upsample(net, (256, 256))
            net = tf.concat([net, f[0]], axis=3)  # w/2 64 + 128
            net = upconvBlock(net, 64, 32)      # w/2 32
            # U-net end

            net = slim.repeat(net, 2, slim.conv2d, 32, [3, 3])  # w/2 32
            net = slim.conv2d(net, 16, [3, 3], padding='SAME')      # w/2 16
            net = slim.conv2d(net, 16, [1, 1], padding='SAME')      # w/2 16
            net = slim.conv2d(net, 2, [1, 1], padding='SAME')       # w/2 2
            return net, end_points