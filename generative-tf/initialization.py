import tensorflow as tf
import numpy as np

def xavier_glorot_initialization(in_dim, out_dim, distribution='normal'):
    """
    Xavier Glorot and Yoshua Bengio (2010)
    Understanding the difficulty of training deep feedforward neural networks. 
    International Conference on Artificial Intelligence and Statistics.
    """

    if distribution == 'uniform':
        extreme = np.sqrt(12.0 / (in_dim + out_dim))
        return tf.random_uniform(
            (in_dim, out_dim), minval=-extreme, maxval=extreme, dtype=tf.float32)
    elif distribution == 'normal' or distribution == 'gaussian':
        stddev = np.sqrt(2.0 / (in_dim + out_dim))
        return tf.random_normal(
            (in_dim, out_dim), stddev=stddev, dtype=tf.float32)


def he_initialization(in_dim, out_dim, activation, alpha=None):
    """
    Kaiming He et al. (2015)
    Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. 
    arXiv preprint arXiv:1502.01852.
    """

    gain = None
    if activation == 'linear' or activation == 'sigmoid':
        gain = 1.0
    elif activation == 'relu':
        gain = np.sqrt(2.0)
    elif activation == 'leaky_relu' and alpha is not None:
        gain = np.sqrt(2.0 / (1 + alpha**2))

    if gain is None:
        raise ValueError("{} is an supported activation for He initialization".format(activation))

    stddev = gain * np.sqrt(1.0 / in_dim)
    return tf.random_normal(
            (in_dim, out_dim), stddev=stddev, dtype=tf.float32)
