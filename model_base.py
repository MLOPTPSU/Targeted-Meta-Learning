"""ResNet model.
Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class ResNet1(object):
  """ResNet model."""

  def __init__(self, is_training, data_format, batch_norm_decay, batch_norm_epsilon, activate_before_residual=False):
    """ResNet constructor.
    Args:
      is_training: if build training or inference model.
      data_format: the data_format used during computation.
                   one of 'channels_first' or 'channels_last'.
    """
    self._batch_norm_decay = batch_norm_decay
    self._batch_norm_epsilon = batch_norm_epsilon
    self._is_training = is_training
    assert data_format in ('channels_first', 'channels_last')
    self._data_format = data_format
    self._activate_before_residual = activate_before_residual


  def _residual_v1_build(self,
                          x,
                          kernel_size,
                          in_filter,
                          out_filter,
                          stride):
    """Residual unit with 2 sub layers, using Plan A for shortcut connection."""
    layers = []
    with tf.name_scope('residual_v1') as name_scope:
      orig_x = x

      if stride > 1:
        x, zero_pad0 = self._padding(x, padding=kernel_size-1)
        layers.append([zero_pad0])
      x, conv1 = self._conv(x, kernel_size, out_filter, stride)
      x, batch_norm1 = self._batch_norm(x)
      x, relu1 = self._relu(x)

      x, conv2 = self._conv(x, kernel_size, out_filter, 1)
      x, batch_norm2 = self._batch_norm(x)

      if layers:
        layers[0].extend([conv1, batch_norm1, relu1, conv2, batch_norm2])
      else:
        layers.append([conv1, batch_norm1, relu1, conv2, batch_norm2])

      if in_filter != out_filter:
        orig_x, avg_pool1 = self._avg_pool(orig_x, stride, stride)
        layers.append([avg_pool1])
        pad = (out_filter - in_filter) // 2
        if self._data_format == 'channels_first':
          orig_x = tf.pad(orig_x, [[0, 0], [pad, pad], [0, 0], [0, 0]])
        else:
          orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0], [pad, pad]])

      x, relu2 = self._relu(tf.add(x, orig_x))
      layers.append([relu2])
      tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
      return x, layers
  
  def _residual_v1(self,
                    x,
                    in_filter,
                    out_filter,
                    layers=[]):
    """Residual unit with 2 sub layers, using Plan A for shortcut connection."""

    assert len(layers) > 0,'The layers list is empty. make sure you build the model first'
    orig_x = x
    for l in layers[0]:
      x = l(x)

    if in_filter != out_filter:
      orig_x = layers[1][0](orig_x)
      pad = (out_filter - in_filter) // 2
      if self._data_format == 'channels_first':
        orig_x = tf.pad(orig_x, [[0, 0], [pad, pad], [0, 0], [0, 0]])
      else:
        orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0], [pad, pad]])

    x = layers[-1][0](tf.add(x, orig_x))
    return x

  def _residual_v2_build(self,
                          x,
                          kernel_size,
                          in_filter,
                          out_filter,
                          stride):
    """Residual unit with 2 sub layers with preactivation, plan A shortcut."""
    layers = []
    with tf.name_scope('residual_v2') as name_scope:
      if self._activate_before_residual:
        x, batch_norm1 = self._batch_norm(x)
        x, relu1 = self._relu(x)
        orig_x = x
      else:
        orig_x = x
        x, batch_norm1 = self._batch_norm(x)
        x, relu1= self._relu(x)
      layers.append([batch_norm1, relu1])

      if stride > 1:
        x, zero_pad0 = self._padding(x, padding=kernel_size-1)
        layers.append([zero_pad0])
      x, conv1 = self._conv(x, kernel_size, out_filter, stride)

      x, batch_norm2 = self._batch_norm(x)
      x, relu2 = self._relu(x)
      x, conv2 = self._conv(x, kernel_size, out_filter, 1)
      if len(layers) > 1:
        layers[1].extend([conv1, batch_norm2, relu2, conv2])
      else:
        layers.append([conv1, batch_norm2, relu2, conv2])
    
      if in_filter != out_filter:
        pad = (out_filter - in_filter) // 2
        orig_x, avg_pool1 = self._avg_pool(orig_x, stride, stride)
        layers.append([avg_pool1])
        if self._data_format == 'channels_first':
          orig_x = tf.pad(orig_x, [[0, 0], [pad, pad], [0, 0], [0, 0]])
        else:
          orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0], [pad, pad]])

      x = tf.add(x, orig_x)

      tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
      return x, layers

  def _residual_v2(self,
                  x,
                  in_filter,
                  out_filter,
                  layers=[]):
    assert len(layers) > 0,'The layers list is empty. make sure you build the model first'
    if self._activate_before_residual:
      for l in layers[0]:
        x = l(x)
      orig_x = x
    else:
      orig_x = x
      for l in layers[0]:
        x = l(x)
    
    for l in layers[1]:
      x = l(x)
    
    if in_filter != out_filter:
      pad = (out_filter - in_filter) // 2
      orig_x = layers[2][0](orig_x)
      if self._data_format == 'channels_first':
        orig_x = tf.pad(orig_x, [[0, 0], [pad, pad], [0, 0], [0, 0]])
      else:
        orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0], [pad, pad]])

    x = tf.add(x, orig_x)
    return x

  def _bottleneck_residual_v2_build(self,
                                    x,
                                    kernel_size,
                                    in_filter,
                                    out_filter,
                                    stride):
    """Bottleneck residual unit with 3 sub layers, plan B shortcut."""
    layers=[]
    with tf.name_scope('bottle_residual_v2') as name_scope:
      if self._activate_before_residual:
        x, batch_norm1 = self._batch_norm(x)
        x, relu1 = self._relu(x)
        orig_x = x
      else:
        orig_x = x
        x, batch_norm1 = self._batch_norm(x)
        x, relu1= self._relu(x)
      layers.append([batch_norm1, relu1])

      x, conv1 = self._conv(x, 1, out_filter // 4, stride, is_atrous=True)

      x, batch_norm2 = self._batch_norm(x)
      x, relu2 = self._relu(x)
      x, conv2 = self._conv(x, kernel_size, out_filter // 4, 1, is_atrous=True)

      x, batch_norm3 = self._batch_norm(x)
      x, relu3 = self._relu(x)
      x, conv3= self._conv(x, 1, out_filter, 1, is_atrous=True)

      layers.append([conv1, batch_norm2, relu2, conv2, batch_norm3, relu3, conv3])
      
      if in_filter != out_filter:
        orig_x, conv4 = self._conv(orig_x, 1, out_filter, stride, is_atrous=True)
        layers.append([conv4])

      x = tf.add(x, orig_x)

      tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
      return x, layers
  
  def _bottleneck_residual_v2(self,
                              x,
                              in_filter,
                              out_filter,
                              layers=[]):
    assert len(layers) > 0,'The layers list is empty. make sure you build the model first'
    if self._activate_before_residual:
      for l in layers[0]:
        x = l(x)
      orig_x = x
    else:
      orig_x = x
      for l in layers[0]:
        x = l(x)
    
    for l in layers[1]:
      x = l(x)
    
    if in_filter != out_filter:
      orig_x = layers[2][0](orig_x)
    x = tf.add(x, orig_x)

    return x

  def _conv(self, x, kernel_size, filters, strides, is_atrous=False):
    """Convolution."""

    padding = 'SAME'
    if not is_atrous and strides > 1:
      padding = 'VALID'
    layer = tf.layers.Conv2D(
        kernel_size=kernel_size,
        filters=filters,
        strides=strides,
        padding=padding,
        use_bias=False,
        data_format=self._data_format)
    return layer(x) , layer

  def _batch_norm(self, x):
    if self._data_format == 'channels_first':
      axis = 1 
    else:
      axis = -1
    # For taking the second order gradient, we need to set
    # fused in the Batch Normalization layer to be False.
    # Otherwise, it will end up in an infinite loop.
    layer = tf.layers.BatchNormalization(
                  axis=axis,
                  momentum=self._batch_norm_decay,
                  epsilon=self._batch_norm_epsilon,
                  fused=False)
    return layer(x), layer

  def _relu(self, x):
    layer = tf.keras.layers.ReLU()
    return layer(x), layer

  def _fully_connected(self, x, out_dim, activation=None):
    with tf.name_scope('fully_connected') as name_scope:
      layer =  tf.layers.Dense(units=out_dim, activation=activation)
      out = layer(x)
    tf.logging.info('image after unit %s: %s', name_scope, out.get_shape())
    return out, layer

  def _avg_pool(self, x, pool_size, stride):
    with tf.name_scope('avg_pool') as name_scope:
      layer = tf.layers.AveragePooling2D(
          pool_size, stride, 'SAME', data_format=self._data_format)
      out = layer(x)
    tf.logging.info('image after unit %s: %s', name_scope, out.get_shape())
    return out, layer

  def _global_avg_pool(self, x, build=False):
    with tf.name_scope('global_avg_pool') as name_scope:
      assert x.get_shape().ndims == 4
      if self._data_format == 'channels_first':
        x = tf.reduce_mean(x, [2, 3])
      else:
        x = tf.reduce_mean(x, [1, 2])
    if build:
      tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
    return x

  def _padding(self, x, padding):
    pad_beg = padding // 2
    pad_end = padding - pad_beg
    layer = tf.keras.layers.ZeroPadding2D(padding=(pad_beg, pad_end), data_format=self._data_format)
    return layer(x), layer









class ResNetOld(object):
  """ResNet model."""

  def __init__(self, is_training, data_format, batch_norm_decay, batch_norm_epsilon):
    """ResNet constructor.
    Args:
      is_training: if build training or inference model.
      data_format: the data_format used during computation.
                   one of 'channels_first' or 'channels_last'.
    """
    self._batch_norm_decay = batch_norm_decay
    self._batch_norm_epsilon = batch_norm_epsilon
    self._is_training = is_training
    assert data_format in ('channels_first', 'channels_last')
    self._data_format = data_format

  def forward_pass(self, x):
    raise NotImplementedError(
        'forward_pass() is implemented in ResNet sub classes')

  def _residual_v1(self,
                   x,
                   kernel_size,
                   in_filter,
                   out_filter,
                   stride,
                   activate_before_residual=False):
    """Residual unit with 2 sub layers, using Plan A for shortcut connection."""

    del activate_before_residual
    with tf.name_scope('residual_v1') as name_scope:
      orig_x = x

      x = self._conv(x, kernel_size, out_filter, stride)
      x = self._batch_norm(x)
      x = self._relu(x)

      x = self._conv(x, kernel_size, out_filter, 1)
      x = self._batch_norm(x)

      if in_filter != out_filter:
        orig_x = self._avg_pool(orig_x, stride, stride)
        pad = (out_filter - in_filter) // 2
        if self._data_format == 'channels_first':
          orig_x = tf.pad(orig_x, [[0, 0], [pad, pad], [0, 0], [0, 0]])
        else:
          orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0], [pad, pad]])

      x = self._relu(tf.add(x, orig_x))

      tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
      return x

  def _residual_v2(self,
                   x,
                   in_filter,
                   out_filter,
                   stride,
                   activate_before_residual=False):
    """Residual unit with 2 sub layers with preactivation, plan A shortcut."""

    with tf.name_scope('residual_v2') as name_scope:
      if activate_before_residual:
        x = self._batch_norm(x)
        x = self._relu(x)
        orig_x = x
      else:
        orig_x = x
        x = self._batch_norm(x)
        x = self._relu(x)

      x = self._conv(x, 3, out_filter, stride)

      x = self._batch_norm(x)
      x = self._relu(x)
      x = self._conv(x, 3, out_filter, 1)

      if in_filter != out_filter:
        pad = (out_filter - in_filter) // 2
        orig_x = self._avg_pool(orig_x, stride, stride)
        if self._data_format == 'channels_first':
          orig_x = tf.pad(orig_x, [[0, 0], [pad, pad], [0, 0], [0, 0]])
        else:
          orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0], [pad, pad]])

      x = tf.add(x, orig_x)

      tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
      return x

  def _bottleneck_residual_v2(self,
                              x,
                              in_filter,
                              out_filter,
                              stride,
                              activate_before_residual=False):
    """Bottleneck residual unit with 3 sub layers, plan B shortcut."""

    with tf.name_scope('bottle_residual_v2') as name_scope:
      if activate_before_residual:
        x = self._batch_norm(x)
        x = self._relu(x)
        orig_x = x
      else:
        orig_x = x
        x = self._batch_norm(x)
        x = self._relu(x)

      x = self._conv(x, 1, out_filter // 4, stride, is_atrous=True)

      x = self._batch_norm(x)
      x = self._relu(x)
      # pad when stride isn't unit
      x = self._conv(x, 3, out_filter // 4, 1, is_atrous=True)

      x = self._batch_norm(x)
      x = self._relu(x)
      x = self._conv(x, 1, out_filter, 1, is_atrous=True)

      if in_filter != out_filter:
        orig_x = self._conv(orig_x, 1, out_filter, stride, is_atrous=True)
      x = tf.add(x, orig_x)

      tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
      return x

  def _conv(self, x, kernel_size, filters, strides, is_atrous=False):
    """Convolution."""

    padding = 'SAME'
    if not is_atrous and strides > 1:
      pad = kernel_size - 1
      pad_beg = pad // 2
      pad_end = pad - pad_beg
      if self._data_format == 'channels_first':
        x = tf.pad(x, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
      else:
        x = tf.pad(x, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
      padding = 'VALID'
    return tf.layers.conv2d(
        inputs=x,
        kernel_size=kernel_size,
        filters=filters,
        strides=strides,
        padding=padding,
        use_bias=False,
        data_format=self._data_format)

  def _batch_norm(self, x):
    if self._data_format == 'channels_first':
      data_format = 'NCHW'
    else:
      data_format = 'NHWC'
    return tf.contrib.layers.batch_norm(
        x,
        decay=self._batch_norm_decay,
        center=True,
        scale=True,
        epsilon=self._batch_norm_epsilon,
        is_training=self._is_training,
        fused=True,
        data_format=data_format)

  def _relu(self, x):
    return tf.nn.relu(x)

  def _fully_connected(self, x, out_dim):
    with tf.name_scope('fully_connected') as name_scope:
      x = tf.layers.dense(x, out_dim)

    tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
    return x

  def _avg_pool(self, x, pool_size, stride):
    with tf.name_scope('avg_pool') as name_scope:
      x = tf.layers.average_pooling2d(
          x, pool_size, stride, 'SAME', data_format=self._data_format)

    tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
    return x

  def _global_avg_pool(self, x):
    with tf.name_scope('global_avg_pool') as name_scope:
      assert x.get_shape().ndims == 4
      if self._data_format == 'channels_first':
        x = tf.reduce_mean(x, [2, 3])
      else:
        x = tf.reduce_mean(x, [1, 2])
    tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
    return x



