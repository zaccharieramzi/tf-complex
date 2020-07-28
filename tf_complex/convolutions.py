import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D

from .activations import ComplexActivation

class ComplexConv2D(Layer):
    conv_types = [
        'real_real',
        'real_imag',
        'imag_real',
        'imag_imag',
    ]
    def __init__(self, n_filters, kernel_size, activation=None, trainable=True, name=None, dtype=None, dynamic=False, **conv_kwargs):
        super(ComplexConv2D, self).__init__(
            trainable=trainable,
            name=name,
            dtype=dtype,
            dynamic=dynamic,
        )
        self.activation = ComplexActivation(activation)
        self.n_filters_total = n_filters
        self.kernel_size = kernel_size
        conv_kwargs.update(dict(
            filters=self.n_filters_total // 2,  # we follow the convention
            # established here:
            # https://github.com/MRSRL/complex-networks-release/blob/master/complex_utils.py#L13
            kernel_size=self.kernel_size,
            activation=None,
        ))
        self.convs = {
            conv_type: Conv2D(
                name=f'{conv_type}_conv2d',
                **conv_kwargs,
            ) for conv_type in ComplexConv2D.conv_types
        }

    def call(self, inputs):
        real = tf.math.real(inputs)
        imag = tf.math.imag(inputs)
        output_real = self.convs['real_real'](real) - self.convs['imag_imag'](imag)
        output_imag = self.convs['imag_real'](real) + self.convs['real_imag'](imag)
        output = tf.complex(output_real, output_imag)
        output = self.activation(output)
        return output
