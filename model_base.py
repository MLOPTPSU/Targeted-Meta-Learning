"""
This is the ResNet base model, based on Google implementation.
We use tf.keras layers to build the model so they can be adopted to the Bilevel class.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 'v2'
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


################################################################################
# Convenience functions for building the ResNet model.
################################################################################
def batch_norm_build(data_format):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  return tf.keras.layers.BatchNormalization(
      axis=1 if data_format == 'channels_first' else 3, trainable=True,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, fused=False)

def batch_norm(inputs, training, layer):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  return layer(inputs=inputs, training=training)

def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(tensor=inputs,
                           paddings=[[0, 0], [0, 0], [pad_beg, pad_end],
                                     [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(tensor=inputs,
                           paddings=[[0, 0], [pad_beg, pad_end],
                                     [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, layer, kernel_size, strides, data_format):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return layer(inputs)

def conv2d_fixed_padding_build(filters, kernel_size, strides, data_format):
  """Strided 2-D convolution with explicit padding."""
  return tf.keras.layers.Conv2D(
      filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.keras.initializers.VarianceScaling(),
      data_format=data_format)



################################################################################
# ResNet block definitions.
################################################################################
def _building_block_v1_build(filters, projection_shortcut, strides,
                       data_format):
  layers = []
  if projection_shortcut is not None:
    shortcut_layer = projection_shortcut()
    layers.append(shortcut_layer)

  conv_layer = conv2d_fixed_padding_build(
      filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)
  layers.append(conv_layer)
  batch_norm_layer = batch_norm_build(data_format=data_format)
  layers.append(batch_norm_layer)
  conv_layer1 = conv2d_fixed_padding_build(
      filters=filters, kernel_size=3, strides=1,
      data_format=data_format)
  layers.append(conv_layer1)
  batch_norm_layer1 = batch_norm_build(data_format=data_format)
  layers.append(batch_norm_layer1)

  return layers

def _building_block_v1(inputs, layers, training, projection_shortcut, strides,
                       data_format):
  """A single block for ResNet v1, without a bottleneck.
  Convolution then batch normalization then ReLU as described by:
    Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.

    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block; shape should match inputs.
  """
  layers_copy = list(layers).copy()
  shortcut = inputs

  if projection_shortcut is not None:
    shortcut = conv2d_fixed_padding(
          inputs=inputs, layer=layers_copy.pop(0),
          kernel_size=1, strides=strides, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs, layer=layers_copy.pop(0), kernel_size=3, strides=strides,
      data_format=data_format)
  inputs = batch_norm(inputs=inputs, training=training, layer=layers_copy.pop(0))
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, layer=layers_copy.pop(0), kernel_size=3, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs=inputs, training=training, layer=layers_copy.pop(0))
  inputs += shortcut
  inputs = tf.nn.relu(inputs)

  return inputs


def _building_block_v2_build(filters, projection_shortcut, strides,
                       data_format):

  layers = []
  batch_norm_layer = batch_norm_build(data_format=data_format)
  layers.append(batch_norm_layer)
  if projection_shortcut is not None:
    shortcut_layer = projection_shortcut()
    layers.append(shortcut_layer)
  
  conv_layer = conv2d_fixed_padding_build(
      filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)
  layers.append(conv_layer)
  batch_norm_layer1 = batch_norm_build(data_format=data_format)
  layers.append(batch_norm_layer1)
  conv_layer1 = conv2d_fixed_padding_build(
      filters=filters, kernel_size=3, strides=1,
      data_format=data_format)
  layers.append(conv_layer1)

  return layers

def _building_block_v2(inputs, layers, training, projection_shortcut, strides,
                       data_format):
  """A single block for ResNet v2, without a bottleneck.
  Batch normalization then ReLu then convolution as described by:
    Identity Mappings in Deep Residual Networks
    https://arxiv.org/pdf/1603.05027.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block; shape should match inputs.
  """
  layers_copy = list(layers).copy()

  shortcut = inputs
  inputs = batch_norm(inputs=inputs, training=training, layer=layers_copy.pop(0))
  inputs = tf.nn.relu(inputs)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = conv2d_fixed_padding(
          inputs=inputs, layer=layers_copy.pop(0),
          kernel_size=1, strides=strides, data_format=data_format)

  inputs = conv2d_fixed_padding(
        inputs=inputs, layer=layers_copy.pop(0),
        kernel_size=3, strides=strides, data_format=data_format)

  inputs = batch_norm(inputs=inputs, training=training, layer=layers_copy.pop(0))
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
        inputs=inputs, layer=layers_copy.pop(0), 
        kernel_size=1, strides=1, data_format=data_format)
  return inputs + shortcut 

def _bottleneck_block_v1_build(filters, projection_shortcut,
                         strides, data_format):
  layers = []

  if projection_shortcut is not None:
    shortcut_layer = projection_shortcut()
    layers.append(shortcut_layer)

  conv_layer = conv2d_fixed_padding_build(
      filters=filters, kernel_size=1, strides=1,
      data_format=data_format)
  layers.append(conv_layer)
  batch_norm_layer = batch_norm_build(data_format=data_format)
  layers.append(batch_norm_layer)

  conv_layer1 = conv2d_fixed_padding_build(
      filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)
  layers.append(conv_layer1)
  batch_norm_layer1 = batch_norm_build(data_format=data_format)
  layers.append(batch_norm_layer1)

  conv_layer2 = conv2d_fixed_padding_build(
      filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format)
  layers.append(conv_layer2)
  batch_norm_layer2 = batch_norm_build(data_format=data_format)
  layers.append(batch_norm_layer2)

  return layers

def _bottleneck_block_v1(inputs, layers, training, projection_shortcut,
                         strides, data_format):
  """A single block for ResNet v1, with a bottleneck.
  Similar to _building_block_v1(), except using the "bottleneck" blocks
  described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block; shape should match inputs.
  """
  layers_copy = list(layers).copy()
  shortcut = inputs

  if projection_shortcut is not None:
    shortcut = conv2d_fixed_padding(
          inputs=inputs, layer=layers_copy.pop(0),
          kernel_size=1, strides=strides, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs, layer=layers_copy.pop(0), kernel_size=1, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs=inputs, training=training, layer=layers_copy.pop(0))
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, layer=layers_copy.pop(0), kernel_size=3, strides=strides,
      data_format=data_format)
  inputs = batch_norm(inputs=inputs, training=training, layer=layers_copy.pop(0))
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, layer=layers_copy.pop(0), kernel_size=1, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs=inputs, training=training, layer=layers_copy.pop(0))
  inputs += shortcut
  inputs = tf.nn.relu(inputs)

  return inputs

def _bottleneck_block_v2_build(filters, projection_shortcut,
                         strides, data_format):
  """A single block for ResNet v2, with a bottleneck.
  Similar to _building_block_v2(), except using the "bottleneck" blocks
  described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
  Adapted to the ordering conventions of:
    Batch normalization then ReLu then convolution as described by:
      Identity Mappings in Deep Residual Networks
      https://arxiv.org/pdf/1603.05027.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block; shape should match inputs.
  """
  layers=[]
  batch_norm_layer = batch_norm_build(data_format=data_format)
  layers.append(batch_norm_layer)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut_layer = projection_shortcut()
    layers.append(shortcut_layer)

  conv_layer = conv2d_fixed_padding_build(
      filters=filters, kernel_size=1, strides=1,
      data_format=data_format)
  layers.append(conv_layer)

  batch_norm_layer1 = batch_norm_build(data_format=data_format)
  layers.append(batch_norm_layer1)
  conv_layer1 = conv2d_fixed_padding_build(
      filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)
  layers.append(conv_layer1)

  batch_norm_layer2 = batch_norm_build(data_format=data_format)
  layers.append(batch_norm_layer2)
  conv_layer2 = conv2d_fixed_padding_build(
      filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format)
  layers.append(conv_layer2)

  return layers

def _bottleneck_block_v2(inputs, layers, training, projection_shortcut,
                         strides, data_format):
  """A single block for ResNet v2, with a bottleneck.
  Similar to _building_block_v2(), except using the "bottleneck" blocks
  described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
  Adapted to the ordering conventions of:
    Batch normalization then ReLu then convolution as described by:
      Identity Mappings in Deep Residual Networks
      https://arxiv.org/pdf/1603.05027.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block; shape should match inputs.
  """
  layers_copy = list(layers).copy()
  shortcut = inputs
  inputs = batch_norm(inputs=inputs, training=training, layer=layers_copy.pop(0))
  inputs = tf.nn.relu(inputs)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = conv2d_fixed_padding(
          inputs=inputs, layer=layers_copy.pop(0),
          kernel_size=1, strides=strides, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs, layer=layers_copy.pop(0),
      kernel_size=1, strides=1, data_format=data_format)

  inputs = batch_norm(inputs=inputs, training=training, layer=layers_copy.pop(0))
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs, layer=layers_copy.pop(0), 
      kernel_size=3, strides=strides, data_format=data_format)

  inputs = batch_norm(inputs=inputs, training=training, layer=layers_copy.pop(0))
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs, layer=layers_copy.pop(0),
      kernel_size=1, strides=1, data_format=data_format)

  return inputs + shortcut



def block_layer_build(filters, bottleneck, block_fn_build, blocks, strides,
                name, data_format):

  block_layers = []
  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = filters * 4 if bottleneck else filters

  def projection_shortcut_build():
    return conv2d_fixed_padding_build(
        filters=filters_out, kernel_size=1, strides=strides,
        data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and strides
  layers = block_fn_build(filters=filters, projection_shortcut=projection_shortcut_build,
                          strides=strides, data_format=data_format)
  block_layers.append(layers)
  for _ in range(1, blocks):
    layers = block_fn_build(filters=filters, projection_shortcut=None, strides=1, data_format=data_format)
    block_layers.append(layers)

  return block_layers

def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, name, data_format, block_layers):
  """Creates one layer of blocks for the ResNet model.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    bottleneck: Is the block created a bottleneck block.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block layer.
  """
  block_layers_copy = list(block_layers).copy()
  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = filters * 4 if bottleneck else filters

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(inputs=inputs, layers=block_layers_copy.pop(0), 
                    training=training, projection_shortcut=True,
                    strides=strides, data_format=data_format)

  for _ in range(1, blocks):
    inputs = block_fn(inputs=inputs, layers=block_layers_copy.pop(0),
                      training=training, projection_shortcut=None, 
                      strides=1, data_format=data_format)

  return tf.identity(inputs, name)