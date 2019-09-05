import tensorflow as tf
import cv2
import matplotlib.image as Image
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import numpy as np

from OHEM import MSE_OHEM_Loss
from net import CRAFT_net
from text_utils import get_result_img
from datagen import procces_function, generator, normalizeMeanVariance

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def test(ckpt_path, img_path):
    x = tf.placeholder(shape=[None, 512, 512, 3], dtype=tf.float32)
    # y = tf.placeholder(shape=[None, 256, 256, 2], dtype=tf.float32)
    y_pre, end_point = CRAFT_net(x)
    src_img = cv2.resize(Image.imread(img_path), (512, 512))
    textimg = normalizeMeanVariance(src_img)
    textimg = np.reshape(textimg, (1, 512, 512, 3))
    restore = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print('------loading weight------')
        restore.restore(sess, ckpt_path)
        print('------complete------')
        res = sess.run(y_pre, feed_dict={x: textimg})
        
        res = np.reshape(res, (256, 256, 2))
        get_result_img(src_img, res[:,:,0], res[:,:,1])
        res = cv2.resize(res, (512, 512))
        score_txt = res[:,:,0]
        score_link = res[:,:,1]
        plt.imsave('/home/user4/ysx/CRAFT/result/weight.jpg', score_txt)
        plt.imsave('/home/user4/ysx/CRAFT/result/weight_aff.jpg', score_link)


def train(train=True):
    x = tf.placeholder(shape=[None, 512, 512, 3], dtype=tf.float32, name='x')
    y = tf.placeholder(shape=[None, 256, 256, 2], dtype=tf.float32, name='y')
    y_pre, end_point = CRAFT_net(x)
    modelpath = '/home/user4/ysx/CRAFT/model'
    loss = MSE_OHEM_Loss(y_pre, y)
    # char_loss, aff_loss, loss_f = loss(y_pre, y)
    end_point['loss'] = loss
    textimg = Image.imread('/home/user4/ysx/CRAFT/te.jpg')
    textimg1 = np.reshape(textimg, (1, 512, 512, 3))
    textimg = normalizeMeanVariance(textimg1)

    exclude = ['vgg_16/fc6', 'vgg_16/fc7', 'vgg_16/mean_rgb', 'vgg_16/fc8']
    include = ['vgg_16/conv1/conv1_1', 'vgg_16/conv1/conv1_2', 'vgg_16/conv2/conv2_1', 'vgg_16/conv2/conv2_2'
               'vgg_16/conv3/conv3_1', 'vgg_16/conv3/conv3_2', 'vgg_16/conv3/conv3_3',
               'vgg_16/conv4/conv4_1', 'vgg_16/conv4/conv4_2', 'vgg_16/conv4/conv4_3',
               'vgg_16/conv5/conv5_1', 'vgg_16/conv5/conv5_2', 'vgg_16/conv5/conv5_3']
    variables_to_restore = slim.get_variables_to_restore(include=include)

    global_step = tf.Variable(0)
    boundaries = [50000, 200000]
    learning_rate = [0.001, 0.0001, 0.00001]
    learning_rate = tf.train.piecewise_constant(global_step, boundaries=boundaries, values=learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = optimizer.minimize(loss, global_step=global_step)
    if train:
        restorer = tf.train.Saver(variables_to_restore)
    else:
        restorer = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.35)
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        if train:
            print('-----load vgg-----')
            # ckpt = tf.train.get_checkpoint_state(modelpath)
            restorer.restore(sess, '/home/user4/ysx/CRAFT/model/vgg16.ckpt')
            print('-----load vgg complete-----')
            print('-----training-----')
        else:
            print('-----load ckpt-----')
            restorer.restore(sess, '/home/user4/ysx/demo/CRAFT_214000.ckpt')
            print('-----load ckpt complete')
            print('-----training------')
        batch_size = 2
        epoch = 5
        data_len = 858750
        char_loss_t = 0
        aff_loss_t = 0
        loss_t = 0
        for e in range(epoch):
            gen = generator(shuffle=True, batch_size=batch_size)
            for i in range(data_len//batch_size):
                image, label = next(gen)
                _, loss_f0, learning_rate0, global_step0 = sess.run([train_step, loss, learning_rate, global_step], feed_dict={x: image, y: label})
                print('\rstep: %2d   learning_rate: %4g   total_loss: %4g' 
                            % (global_step0, learning_rate0, loss_f0), end='')
                loss_t += loss_f0
                if global_step0%100==0:
                    avg_loss = loss_t/100
                    res = sess.run(y_pre, feed_dict={x: textimg})
                    get_result_img(textimg1, res[0,:,:,0], res[0,:,:,1])
                    # res = np.clip(res, 0, 1)
                    #res_0, res_1 = text_utils.get_res_hmp(res)
                    plt.imsave('result_c.jpg', cv2.resize(res[0,:,:,0], (512, 512)))
                    plt.imsave('result_a.jpg', cv2.resize(res[0,:,:,1], (512, 512)))
                    print('\nstep: %2d   learning_rate: %4g   avg_total_loss: %4g' 
                            % (global_step0, learning_rate0, avg_loss))
                    char_loss_t = 0
                    aff_loss_t = 0
                    loss_t = 0
                    if global_step0%1000==0:
                        saver.save(sess, "/home/user4/ysx/demo/CRAFT_%d.ckpt" %(global_step0))

if __name__ == '__main__':
    train(False)
    # test('/home/user4/ysx/demo/CRAFT_214000.ckpt', '/home/user4/ysx/CRAFT/802.jpg')