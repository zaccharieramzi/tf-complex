import pytest
import tensorflow as tf
from tensorflow.test import TestCase

from tf_complex.activations import ModReLU, ComplexActivation


class TestModReLU(TestCase):
    def test_init_and_call(self):
        modrelu = ModReLU()
        inputs = tf.zeros([16, 32, 32, 8])
        outputs = modrelu(inputs)
        self.assertEqual(inputs.shape, outputs.shape)

class TestComplexActivation(TestCase):
    def test_init_and_call(self):
        inputs = tf.zeros([16, 32, 32, 8])
        for activation_str in ComplexActivation.activation_key_to_fun.keys():
            activation = ComplexActivation(activation_str)
            outputs = activation(inputs)
            self.assertEqual(inputs.shape, outputs.shape)
        activation = ComplexActivation(None)
        outputs = activation(inputs)
        self.assertEqual(inputs.shape, outputs.shape)

    def test_error_when_unknown(self):
        with pytest.raises(ValueError):
            ComplexActivation('fake_activation')
