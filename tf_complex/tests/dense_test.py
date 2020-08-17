import tensorflow as tf
from tensorflow.test import TestCase

from tf_complex.activations import ComplexActivation
from tf_complex.dense import ComplexDense


class TestComplexConv2D(TestCase):
    def test_init_and_call(self):
        units = 32
        inputs = tf.zeros([16, 32], dtype=tf.complex64)
        for activation_str in ComplexActivation.activation_key_to_fun.keys():
            dense = ComplexDense(
                units,
                activation=activation_str,
            )
            outputs = dense(inputs)
            self.assertEqual(inputs.shape[0], outputs.shape[0])
            self.assertEqual(units, outputs.shape[-1])
