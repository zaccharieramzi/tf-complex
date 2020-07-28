# tf-complex

[![Build Status](https://travis-ci.com/zaccharieramzi/tf-complex.svg?branch=master)](https://travis-ci.com/zaccharieramzi/tf-complex)

This package was inspired by the work of Elizabeth Cole et al.: [Image Reconstruction using an Unrolled DL Architecture including Complex-Valued Convolution and Activation Functions](https://arxiv.org/abs/2004.01738).
Please cite their work appropriately if you use this package.
The code for their work is available [here](https://github.com/MRSRL/complex-networks-release).

## Installation

You can install `tf-complex` using pypi:

```
pip install tf-complex
```

## Example use

You can define a complex convolution in the following way to use in one of your models:

```python
from tf_complex.convolutions import ComplexConv2D

conv = ComplexConv2D(
  16,
  3,
  padding='same',
  activation='crelu',
)
```
