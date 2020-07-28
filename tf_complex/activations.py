from math import pi

import tensorflow as tf
from tensorflow.keras.layers import Layer


def zrelu(x):
    r"""zReLU activation function.

    This is described in [C2020].
    zrelu(x) = x if phase(x) in (0; pi/2), 0 otherwise

    Arguments:
        - x (tf.Tensor): a complex-valued tensor.

    Returns:
        - tf.Tensor: a complex-valued tensor.
    """
    with tf.name_scope('zrelu'):
    # inspired by https://github.com/MRSRL/complex-networks-release/blob/master/complex_utils.py#L188
        phase = tf.math.angle(x)
        le = tf.less_equal(phase, pi / 2)
        ge = tf.greater_equal(phase, 0)
        set_condition = tf.logical_and(le, ge)
        y = tf.zeros_like(x)
        output = tf.where(set_condition, x, y)
    return output

def crelu(x):
    r"""CReLU activation function.

    This is described in [C2020].
    crelu(x) = ReLU(Re(x)) + i ReLU(Im(x))

    Arguments:
        - x (tf.Tensor): a complex-valued tensor.

    Returns:
        - tf.Tensor: a complex-valued tensor.
    """
    with tf.name_scope('crelu'):
        real = tf.math.real(x)
        imag = tf.math.imag(x)
        output = tf.complex(
            tf.nn.relu(real),
            tf.nn.relu(imag),
        )
    return output

def cardioid(x):
    r"""cardioid activation function.

    This is described in [C2020].
    cardioid(x) = (1/2) * x * (1 + cos(phase(x)))

    Arguments:
        - x (tf.Tensor): a complex-valued tensor.

    Returns:
        - tf.Tensor: a complex-valued tensor.
    """
    with tf.name_scope('cardioid'):
        phase = tf.math.angle(x)
        output = 0.5 * x * (1 + tf.cos(phase))
    return output


class ModReLU(Layer):
    """ModReLU activation function.

    This is described in [C2020].
    modrelu(x; b) = ReLU(|x| + b) * e^(i phase(x))

    We consider that the bias b is uniform accross a given feature map, which
    means that its dimension is the number of channels.

    Attributes:
        - bias (tf.Tensor): the bias `b` used in the function.

    Caveats:
        This activation function currently only works in the channel last mode.
    """
    def __init__(self, bias_initializer=None, bias_regularizer=None, bias_constraint=None):
        if bias_initializer is None:
            bias_initializer = tf.constant_initializer(0.0)
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.bias_constraint = bias_constraint

    def build(self, input_shape):
        self.bias = self.add_weight(
            'bias',
            shape=[1, 1, 1, input_shape[-1]],
            trainable=True,
            dtype=tf.float32,
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
        )

    def call(self, inputs):
        phase = tf.math.angle(inputs)
        modulus = tf.abs(inputs)
        output_modulus = tf.nn.relu(modulus + self.bias)
        i = tf.constant(1j)
        output = output_modulus * tf.exp(i * phase)
        return output
