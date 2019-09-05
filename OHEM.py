import scipy.io as sio
import tensorflow as tf
from net import CRAFT_net
import tensorflow.contrib.slim as slim
import cv2


def MSE_OHEM_Loss(output_imgs, target_imgs):
    loss_every_sample = []
    batch_size = 2
    #batch_size = output_imgs.get_shape().as_list()[0]
    for i in range(batch_size):
        output_img = tf.reshape(output_imgs[i], [-1])
        target_img = tf.reshape(target_imgs[i], [-1])
        positive_mask = tf.cast(tf.greater(target_img, 0), dtype=tf.float32)
        sample_loss = tf.square(tf.subtract(output_img, target_img))
        
        num_all = output_img.get_shape().as_list()[0]
        num_positive = tf.cast(tf.reduce_sum(positive_mask), dtype=tf.int32)
        
        positive_loss = tf.multiply(sample_loss, positive_mask)
        positive_loss_m = tf.reduce_sum(positive_loss)/tf.cast(num_positive, dtype=tf.float32)
        nagative_loss = tf.multiply(sample_loss, (1-positive_mask))
        # nagative_loss_m = tf.reduce_sum(nagative_loss)/(num_all - num_positive)

        k = num_positive * 3        
        #nagative_loss_topk, _ = tf.nn.top_k(nagative_loss, k)
        # tensorflow 1.13存在bug，不能使用以下语句 Orz。。。
        k = tf.cond((k + num_positive) > num_all, lambda: tf.cast((num_all - num_positive), dtype=tf.int32), lambda: k)
        k = tf.cond(k>0, lambda: k, lambda: k+1)   
        nagative_loss_topk, _ = tf.nn.top_k(nagative_loss, k)
        res = tf.cond(k < 10, lambda: tf.reduce_mean(sample_loss),
                              lambda: positive_loss_m + tf.reduce_sum(nagative_loss_topk)/tf.cast(k, dtype=tf.float32))
        loss_every_sample.append(res)
    return tf.reduce_mean(tf.convert_to_tensor(loss_every_sample))

if __name__ == '__main__':
    output_imgs = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    output_imgs = tf.reshape(output_imgs, (1, 1, 2, 2))
    target_imgs = tf.constant([[1.1, 2.1], [3., 4.1]], dtype=tf.float32)
    target_imgs = tf.reshape(target_imgs, (1, 1, 2, 2))
    loss = MSE_OHEM_Loss(output_imgs, target_imgs)
    with tf.Session() as sess:
        print(sess.run(loss))