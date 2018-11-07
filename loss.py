import numpy as np
from keras import backend as K
import tensorflow as tf


def cross_entropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)


def boot_soft(y_true, y_pred):
    """
    2015 - iclrws - Training deep neural networks on noisy labels with bootstrapping.
    https://arxiv.org/abs/1412.6596

    :param y_true: 
    :param y_pred: 
    :return: 
    """
    beta = 0.95

    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return -K.sum((beta * y_true + (1. - beta) * y_pred) *
                  K.log(y_pred), axis=-1)


def boot_hard(y_true, y_pred):
    """
    2015 - iclrws - Training deep neural networks on noisy labels with bootstrapping.
    https://arxiv.org/abs/1412.6596

    :param y_true: 
    :param y_pred: 
    :return: 
    """
    beta = 0.8

    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    pred_labels = K.one_hot(K.argmax(y_pred, 1), num_classes=K.shape(y_true)[1])
    return -K.sum((beta * y_true + (1. - beta) * pred_labels) *
                  K.log(y_pred), axis=-1)


def forward(P):
    """
    Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach
    CVPR17 https://arxiv.org/abs/1609.03683
    :param P: noise model, a noisy label transition probability matrix
    :return: 
    """
    P = K.constant(P)

    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        return -K.sum(y_true * K.log(K.dot(y_pred, P)), axis=-1)

    return loss


def backward(P):
    """
    Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach
    CVPR17 https://arxiv.org/abs/1609.03683
    :param P: noise model, a noisy label transition probability matrix
    :return: 
    """
    P_inv = K.constant(np.linalg.inv(P))

    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        return -K.sum(K.dot(y_true, P_inv) * K.log(y_pred), axis=-1)

    return loss


def lid(logits, k=20):
    """
    Calculate LID for each data point in the array.

    :param logits:
    :param k: 
    :return: 
    """
    batch_size = tf.shape(logits)[0]
    # n_samples = logits.get_shape().as_list()
    # calculate pairwise distance
    r = tf.reduce_sum(logits * logits, 1)
    # turn r into column vector
    r1 = tf.reshape(r, [-1, 1])
    D = r1 - 2 * tf.matmul(logits, tf.transpose(logits)) + tf.transpose(r1) + \
        tf.ones([batch_size, batch_size])

    # find the k nearest neighbor
    D1 = -tf.sqrt(D)
    D2, _ = tf.nn.top_k(D1, k=k, sorted=True)
    D3 = -D2[:, 1:]  # skip the x-to-x distance 0 by using [,1:]

    m = tf.transpose(tf.multiply(tf.transpose(D3), 1.0 / D3[:, -1]))
    v_log = tf.reduce_sum(tf.log(m + K.epsilon()), axis=1)  # to avoid nan
    lids = -k / v_log

    return lids


def lid_paced_loss(alpha=1.0):
    """TO_DO
    Class wise lid pace learning, targeting classwise asymetric label noise.

    Args:      
      alpha: lid based adjustment paramter: this needs real-time update.
    Returns:
      Loss tensor of type float.
    """

    # def loss(y_true, y_pred):
    #     return K.categorical_crossentropy(y_pred, y_true)
    #
    # return loss
    if alpha == 1.0:
        def loss(y_true, y_pred):
            y_true = K.clip(y_true, 0.01 / 9, 0.99)
            return -K.sum(y_true * K.log(y_pred), axis=-1)

        return loss
    else:
        def loss(y_true, y_pred):
            pred_labels = K.one_hot(K.argmax(y_pred, 1), num_classes=K.shape(y_true)[1])
            y_new = alpha * y_true + (1. - alpha) * pred_labels
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
            return -K.sum(y_new * K.log(y_pred), axis=-1)

        return loss