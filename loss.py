from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def loss(logits, labels):
    """Calculate the loss from the logits and the labels.
    Args:
      logits: tensor, float - [batch_size, width, height, 2].
          Use vgg_fcn.upscore as logits.
      labels: Labels tensor, float32 - [batch_size, width, height, 2].
          The ground truth of your data.
      head: numpy array - [num_classes]
          Weighting the loss of each class
          Optional: Prioritize some classes
    Returns:
      loss: Loss tensor of type float.
    """
    with tf.name_scope('loss'):
        char_pre = tf.reshape(logits[:, :, :, 0], -1, name='char_pre')
        aff_pre = tf.reshape(logits[:, :, :, 1], -1, name='aff_pre')
        char_gt = tf.reshape(labels[:, :, :, 0], -1, name='char_gt')
        aff_gt = tf.reshape(labels[:, :, :, 1], -1, name='aff_gt')
        char_loss = tf.norm(tf.subtract(char_pre, char_gt))
        aff_loss = tf.norm(tf.subtract(aff_pre, aff_gt))
        loss = tf.reduce_mean(-tf.reduce_sum(tf.add(char_loss, aff_loss)))
    return loss