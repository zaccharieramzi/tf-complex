import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D

from .activations import ComplexActivation

class ComplexConv2D(Layer):
    r"""Complex convolution.

    This is defined in [C2020].
    Parameters:
        n_filters (int): the equivalent number of filters used for a real
            convolution. As per the convention defined in the code for [C2020],
            this means that each convolution will actually have `n_filters // 2`
            filters.
        kernel_size (int): the size of the convolution kernels.
        activation (str or callable or None): the activation function to use for
            this complex convolution. Must be defined via tf_complex.activations
            is using a string. If None, the linear activation function is used.
            Defaults to None.
        trainable (bool): whether the layer's variables should be trainable.
            Defaults to True.
        name (str): name of the layer. Defaults to None.
        dtype (tf.dtype or str): the dtype of the layer's computations and
            weights. Defaults to None.
        dynamic (bool): if the layer is to be run eargerly. Defaults to False.
        **conv_kwargs: keyword arguments for the convolutions initializations.

    Attributes:
        n_filters_total (int): the number of filters in total for the
            convolutions. Corresponds to the parameter `n_filters`.
        activation (tf_complex.activations.ComplexActivation): the activation
            function.
        convs (dict str -> tf.keras.layers.Conv2D): the different convolution
            layers used to perform the underlying complex convolutions.
    """
    conv_types = [
        'real',
        'imag',
    ]
    def __init__(
            self,
            n_filters,
            kernel_size,
            activation=None,
            use_bias=True,
            trainable=True,
            name=None,
            dtype=None,
            dynamic=False,
            **conv_kwargs,
        ):
        super(ComplexConv2D, self).__init__(
            trainable=trainable,
            name=name,
            dtype=dtype,
            dynamic=dynamic,
        )
        self.activation = ComplexActivation(activation)
        self.use_bias = use_bias
        self.n_filters_total = n_filters
        self.kernel_size = kernel_size
        conv_kwargs.update(dict(
            filters=self.n_filters_total // 2,  # we follow the convention
            # established here:
            # https://github.com/MRSRL/complex-networks-release/blob/master/complex_utils.py#L13
            kernel_size=self.kernel_size,
            use_bias=False,
            activation=None,
        ))
        self.convs = {
            conv_type: Conv2D(
                name=f'{conv_type}_conv2d',
                **conv_kwargs,
            ) for conv_type in ComplexConv2D.conv_types
        }
        if self.use_bias:
            self.biases = {
                dense_type: self.add_weight(
                    name=f'{dense_type}_dense_bias',
                    shape=[self.n_filters_total // 2],
                    initializer=conv_kwargs.get('bias_initializer', 'zeros'),
                    regularizer=conv_kwargs.get('bias_regularizer', None),
                    constraint=conv_kwargs.get('bias_constraint', None),
                ) for dense_type in ComplexConv2D.conv_types
            }

    def call(self, inputs):
        real = tf.math.real(inputs)
        imag = tf.math.imag(inputs)
        output_real = self.convs['real'](real) - self.convs['imag'](imag)
        output_imag = self.convs['imag'](real) + self.convs['real'](imag)
        output = tf.complex(output_real, output_imag)
        output = self.activation(output)
        return output
