import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

from .activations import ComplexActivation


class ComplexDense(Layer):
    r"""Complex fully-connected layer.

    Parameters:
        units (int): dimensionality of the output space
        activation (str or callable or None): the activation function to use for
            this complex dense layer. Must be defined via tf_complex.activations
            is using a string. If None, the linear activation function is used.
            Defaults to None.
        trainable (bool): whether the layer's variables should be trainable.
            Defaults to True.
        name (str): name of the layer. Defaults to None.
        dtype (tf.dtype or str): the dtype of the layer's computations and
            weights. Defaults to None.
        dynamic (bool): if the layer is to be run eargerly. Defaults to False.
        **dense_kwargs: keyword arguments for the dense layers initializations.

    Attributes:
        activation (tf_complex.activations.ComplexActivation): the activation
            function.
        denses (dict str -> tf.keras.layers.Dense): the different dense
            layers used to perform the underlying complex multiplications.
        biases (dict str -> tf.Variable): the different biases used to perform
            the underlying complex additions. Only present if use_bias.
    """
    dense_types = [
        'real',
        'imag',
    ]
    def __init__(
            self,
            units,
            activation=None,
            use_bias=True,
            trainable=True,
            name=None,
            dtype=None,
            dynamic=False,
            **dense_kwargs,
        ):
        super(ComplexDense, self).__init__(
            trainable=trainable,
            name=name,
            dtype=dtype,
            dynamic=dynamic,
        )
        self.activation = ComplexActivation(activation)
        self.units = units
        self.use_bias = use_bias
        dense_kwargs.update(dict(
            units=self.units,
            use_bias=False,
            activation=None,
        ))
        self.denses = {
            dense_type: Dense(
                name=f'{dense_type}_dense',
                **dense_kwargs,
            ) for dense_type in ComplexDense.dense_types
        }
        if self.use_bias:
            self.biases = {
                dense_type: self.add_weight(
                    name=f'{dense_type}_dense_bias',
                    shape=[self.units],
                    initializer=dense_kwargs.get('bias_initializer', 'zeros'),
                    regularizer=dense_kwargs.get('bias_regularizer', None),
                    constraint=dense_kwargs.get('bias_constraint', None),
                ) for dense_type in ComplexDense.dense_types
            }


    def call(self, inputs):
        real = tf.math.real(inputs)
        imag = tf.math.imag(inputs)
        output_real = self.denses['real'](real) - self.denses['imag'](imag)
        output_imag = self.denses['imag'](real) + self.denses['real'](imag)
        if self.use_bias:
            output_real = output_real + self.biases['real']
            output_imag = output_imag + self.biases['imag']
        output = tf.complex(output_real, output_imag)
        output = self.activation(output)
        return output
