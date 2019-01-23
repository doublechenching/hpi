#encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras import backend as K
import tensorflow as tf


def weight_binary_ce(cls_weight):
    def w_ce(y_true, y_pred):
        weights = tf.constant(cls_weight, dtype='float32')
        pos_w = tf.zeros_like(y_true, dtype='float32') + weights
        loss = pos_w * y_true * K.log(y_pred) + \
            (1.0 - y_true) * K.log(1.0 - y_pred)

        return -loss

    return w_ce


def focal_loss(gamma=2, alpha=2):
    """focal loss, focal hard samples and seld-adaptive weight

    Args:
    -----
        gamma: n次方
        alpha: 权重项
    Returns:
    --------
        focal loss function
    Eamples:
    -------
        >>>model_prn.compile(optimizer=optimizer, loss=[focal_loss(alpha=2, gamma=2)])
    """
    def focal_loss_fixed(y_true, y_pred):
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        loss = -(K.mean(alpha * K.pow(1. - pt, gamma)
                        * K.log(pt + K.epsilon()), axis=-1))
        return loss

    return focal_loss_fixed


def roc_auc_loss(y_true, y_pred):
    """ ROC AUC Score.
    Approximates the Area Under Curve score, using approximation based on
    the Wilcoxon-Mann-Whitney U statistic.
    Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
    Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.
    Measures overall performance for a full range of threshold levels.
    Arguments:
        y_pred: `Tensor`. Predicted values.
        y_true: `Tensor` . Targets (labels), a probability distribution.
    """
    with tf.name_scope("RocAucScore"):
        pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
        neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))
        pos = tf.expand_dims(pos, 0)
        neg = tf.expand_dims(neg, 1)
        # original paper suggests performance is robust to exact parameter choice
        gamma = 0.2
        p = 3
        difference = tf.zeros_like(pos * neg) + pos - neg - gamma
        masked = tf.boolean_mask(difference, difference < 0.0)

        return tf.reduce_sum(tf.pow(-masked, p))


def f1_score(y_true, y_pred, threshold=0.5):
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)

    return K.mean(f1)


def binary_recall(y_true, y_pred, threshold=0.5):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1.0), threshold), K.floatx())
    tp = K.sum(K.cast(y_true * y_pred, 'float'))
    fn = K.sum(K.cast(y_true * (1-y_pred), 'float'))
    recall = tp / (tp + fn)

    return recall


def binary_precision(y_true, y_pred, threshold=0.5):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1.0), threshold), K.floatx())

    tp = K.sum(K.cast(y_true * y_pred, 'float'))
    fp = K.sum(K.cast((1-y_true) * y_pred, 'float'))

    percision = tp / (tp + fp)

    return percision



