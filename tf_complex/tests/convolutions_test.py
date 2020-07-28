import tensorflow as tf
from tensorflow.test import TestCase

from tf_complex.activations import ComplexActivation
from tf_complex.convolutions import ComplexConv2D


class TestComplexConv2D(TestCase):
    def test_init_and_call(self):
        n_filters = 32
        inputs = tf.zeros([16, 32, 32, 8], dtype=tf.complex64)
        for activation_str in ComplexActivation.activation_key_to_fun.keys():
            conv = ComplexConv2D(
                n_filters,
                3,
                padding='same',
                activation=activation_str,
            )
            outputs = conv(inputs)
            self.assertEqual(inputs.shape[:-1], outputs.shape[:-1])
            self.assertEqual(n_filters // 2, outputs.shape[-1])
