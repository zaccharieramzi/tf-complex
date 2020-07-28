from math import pi

import tensorflow as tf
from tensorflow.keras.layers import Layer


def linear(x):
    """No activation, activation function.

    Arguments:
        - x (tf.Tensor): a complex-valued tensor.

    Returns:
        - tf.Tensor: a complex-valued tensor.
    """
    return x

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
        output = 0.5 * x * tf.cast((1 + tf.cos(phase)), x.dtype)
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
    def __init__(self, bias_initializer=None, bias_regularizer=None, bias_constraint=None, **kwargs):
        super(ModReLU, self).__init__(**kwargs)
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
        phase = tf.cast(tf.math.angle(inputs), inputs.dtype)
        modulus = tf.abs(inputs)
        output_modulus = tf.nn.relu(modulus + self.bias)
        i = tf.constant(1j, dtype=inputs.dtype)
        output = tf.cast(output_modulus, inputs.dtype) * tf.exp(i * phase)
        return output

class ComplexActivation(Layer):
    """Complex activation function.

    This mimics the tf.keras.layers.Actvation class, allowing to select an
    activation function with a string key or initializing it with a custom
    callable.

    Parameters:
        - activation (str or callable or None): if string, then it's the
            identifier to an implemented complex activation function. If callable
            then it's an activation function taking as input a complex valued
            tensor and returning a complex valued tensor of the same shape.
            If None, the linear activation function is chosen.

    Attributes:
        - activation (callable): the activation function.
    """
    activation_key_to_fun = {
        'zrelu': zrelu,
        'cardioid': cardioid,
        'crelu': crelu,
        'linear': linear
    }
    def __init__(self, activation, **kwargs):
        super(ComplexActivation, self).__init__(**kwargs)
        if isinstance(activation, str):
            try:
                self.activation = ComplexActivation.activation_key_to_fun[activation]
            except KeyError:
                raise ValueError(
                    f'{activation} is not a built-in complex activation function, \
                    chose from {ComplexActivation.activation_key_to_fun.keys()}'
                )
        elif activation is None:
            self.activation = linear
        elif callable(activation):
            self.activation = activation
        else:
            TypeError(
                'Could not interpret activation function identifier: {}'.format(
                    repr(activation)
                )
            )

    def call(self, inputs):
        output = self.activation(inputs)
        return output
