import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def upconvBlock(inputs, midchannels, outchannels):
    net = slim.conv2d(inputs, midchannels, [1, 1])
    net = slim.batch_norm(net)
    net = slim.conv2d(net, outchannels, [3, 3], padding='SAME')
    net = slim.batch_norm(net)
    return net

def upsampling_bilinear(channels):
    #确定卷积核大小
    def get_kernel_size(factor):
        return 2*factor-factor%2
    # 创建相关矩阵
    def upsample_filt(size):
        factor=(size+1)//2
        if size%2==1:
            center=factor-1
        else:
            center=factor-0.5
        og=np.ogrid[:size,:size]

        return (1-abs(og[0]-center)/factor)*(1-abs(og[1]-center)/factor)
    #进行上采样卷积核
    def bilinear_upsample_weights(factor,number_of_classes):
        filter_size=get_kernel_size(factor)
        weights=np.zeros((filter_size,filter_size,
                  number_of_classes,number_of_classes),dtype=np.float32)
        upsample_kernel=upsample_filt(filter_size)
        # print(upsample_kernel)
        for i in range(number_of_classes):
            weights[:,:,i,i]=upsample_kernel
            # print(weights[:,:,i,i])
        # print(weights)
        # print(weights.shape)
        return weights
    weights = bilinear_upsample_weights(3, channels)
    return weights

def upsample(inputs, new_size):
    net = tf.image.resize_bilinear(inputs, new_size)
    # net = slim.conv2d_transpose(inputs, k_channel, kernel_size=[2, 2], stride=2, padding='VALID')
    return net

def conv2d_transpose_strided(x, outch, output_shape=None, stride = 2):
    # print x.get_shape()
    # print W.get_shape()
    kernel = tf.get_variable('kernel', [3, 3, outch, outch])
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, kernel, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return conv