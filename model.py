"""
The main model for bilevel structure using tf.keras.Model class

__author__ = "MM. Kamani"
"""

import tensorflow as tf
import model_base as mb

class BilevelModel(tf.keras.Model):
    
    def perturb_model_weights(self, loss, lr, old_vs_name=None, prefix=''):
        if self.layers:
            flatten_layers = self._flatten_nested_list(self.layers)
            for layer in flatten_layers:
                self.perturb_layer_weights(layer, loss, lr, old_vs_name, prefix)
        else:
            print('The model {} does not have any layers'.format(self.name))


    def perturb_layer_weights(self, layer, loss, lr, old_vs_name=None, prefix=''):
        if layer.trainable_weights:
            gradients = tf.gradients(loss, layer.trainable_weights, stop_gradients=layer.trainable_weights)
            new_weights = []
            for w,g in zip(layer.weights, gradients):
                old_name = self._get_name(w.name, old_vs_name)
                new_name = self._set_name(old_name, old_vs_name, prefix)
                tmp = tf.identity( w - lr * g, name=new_name)
                new_weights.append(tmp)
                tf.compat.v1.add_to_collection(
                    tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                    tmp
                )
            layer._trainable_weights = new_weights
            try:
                layer.kernel = new_weights[0]
                if layer.bias is not None:   
                    layer.bias = new_weights[1]
            except AttributeError:
                try:
                    if layer.scale:
                        layer.gamma = new_weights[0]
                        layer.beta = new_weights[1]
                    else:
                        layer.beta = new_weights[0]
                except:
                    weights_name_str = ', '.join([x.name for x in layer.trainable_weights])
                    raise KeyError('Weights {} has not been updated!'.format(weights_name_str))
                pass
        return
            

    def _get_name(self, name, old_vs_name):
        if old_vs_name:
            base_name = name.split(old_vs_name + '/',1)
            if len(base_name) > 1:
                return base_name[1].split(':')[0]
            else:
                raise ValueError('The variable named {} is not in the variable scope {}'.format(name, old_vs_name))
        else:
            return name.split(':')[0]


    def _set_name(self, old_name, old_vs_name, prefix):
        curr_vs_name = tf.compat.v1.get_variable_scope().name
        if ((curr_vs_name == old_vs_name) & (not prefix)):
            raise ValueError('Current name cannot be the same as previous variable name. Please set it in the new variable scope or set a prefix.')
        elif curr_vs_name == old_vs_name:
            return prefix + '/' + old_name
        else:
            return old_name

    def _flatten_nested_list(self, nested_list):
        if not nested_list:
            return nested_list
        if not isinstance(nested_list[0], list):
            return [nested_list[0]] + self._flatten_nested_list(nested_list[1:])
        return self._flatten_nested_list(nested_list[0]) + self._flatten_nested_list(nested_list[1:])


class BilevelLenet(BilevelModel):
    def __init__(self, num_class=2):
        super(BilevelLenet, self).__init__()
        self.num_class = num_class
        self.conv1 = tf.keras.layers.Conv2D(filters=6,
                                            kernel_size=5, 
                                            activation=tf.nn.relu)
        self.max_pool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)

        self.conv2 = tf.keras.layers.Conv2D(filters=16,
                                            kernel_size=5,
                                            activation=tf.nn.relu)
        self.max_pool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)

        self.flatten = tf.keras.layers.Flatten()

        self.fc1 = tf.keras.layers.Dense(units=120, activation=tf.nn.relu)
        self.fc2 = tf.keras.layers.Dense(units=84,  activation=tf.nn.relu)
        self.fc3 = tf.keras.layers.Dense(units=self.num_class)

    def call(self, input, input_data_format='channels_last'):
        del input_data_format
        input = input/128-1
        out = input
        for layer in self.layers:
            out = layer(out)
        return out

class BilevelRegression(BilevelModel):
    def __init__(self, num_class=2):
        super(BilevelRegression, self).__init__()
        self.num_class = num_class
        self.fc1 = tf.keras.layers.Dense(units=120, activation=tf.nn.relu)
        self.fc2 = tf.keras.layers.Dense(units=84,  activation=tf.nn.relu)
        # self.fc3 = tf.keras.layers.Dense(units=50,  activation=tf.nn.relu)
        self.fc4 = tf.keras.layers.Dense(units=self.num_class)

    def call(self, input, input_data_format='channels_last'):
        del input_data_format
        out = input
        for layer in self.layers:
            out = layer(out)
        return out


class BilevelResNet(BilevelModel):
  """ Resnet model with residual layers."""

  def __init__(self,
               resnet_size, 
               data_format=None, 
               num_classes=11, #For Cifar 10
               num_filters=16,
               kernel_size=3,
               conv_stride=1,
               first_pool_size=None,
               first_pool_stride=None,
               block_strides=[1, 2, 2],
               resnet_version=mb.DEFAULT_VERSION,
               dtype=mb.DEFAULT_DTYPE):
    super(BilevelResNet, self).__init__()
    self.resnet_size = resnet_size

    if self.resnet_size % 6 != 2:
      raise ValueError('resnet_size must be 6n + 2:', resnet_size)
    num_blocks = (resnet_size - 2) // 6

    if resnet_version in ['bv1', 'bv2']:
      self.bottleneck = True
      self.version = resnet_version[1:]
    else:
      self.bottleneck = False
      self.version = resnet_version

    if self.version not in ('v1', 'v2'):
      raise ValueError(
          'Resnet version should be 1 or 2.')
  
    if not data_format:
      self.data_format = (
          'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
    else:
      self.data_format = data_format
  
    
    if self.bottleneck:
      if self.version == 'v1':
        self.block_fn = mb._bottleneck_block_v1
        self.block_fn_build = mb._bottleneck_block_v1_build
      else:
        self.block_fn = mb._bottleneck_block_v2
        self.block_fn_build = mb._bottleneck_block_v2_build
    else:
      if self.version == 'v1':
        self.block_fn = mb._building_block_v1
        self.block_fn_build = mb._building_block_v1_build
      else:
        self.block_fn = mb._building_block_v2
        self.block_fn_build = mb._building_block_v2_build

    if dtype not in mb.ALLOWED_TYPES:
      raise ValueError('dtype must be one of: {}'.format(mb.ALLOWED_TYPES))

    self.num_classes = num_classes
    self.num_filters = num_filters
    self.kernel_size = kernel_size
    self.conv_stride = conv_stride
    self.first_pool_size = first_pool_size
    self.first_pool_stride = first_pool_stride
    self.block_sizes = [num_blocks] * 3
    self.block_strides = block_strides
    # self.dtype = dtype
    self.pre_activation = self.version == 'v2'

    self.resnet_layers = []
    with self._model_variable_scope():
      conv_layer = mb.conv2d_fixed_padding_build(
          filters=self.num_filters, kernel_size=self.kernel_size,
          strides=self.conv_stride, data_format=self.data_format)
      self.resnet_layers.append(conv_layer)

      if self.version == 'v1':
        batch_norm_layer = mb.batch_norm_build(self.data_format)
        self.resnet_layers.append(batch_norm_layer)

      for i, num_blocks in enumerate(self.block_sizes):
        num_filters = self.num_filters * (2**i)
        block_layers = mb.block_layer_build(
            filters=num_filters, bottleneck=self.bottleneck,
            block_fn_build=self.block_fn_build, blocks=num_blocks,
            strides=self.block_strides[i],
            name='block_layer{}'.format(i + 1), data_format=self.data_format)
        self.resnet_layers.append(block_layers)

      if self.pre_activation:
        batch_norm_layer = mb.batch_norm_build(self.data_format)
        self.resnet_layers.append(batch_norm_layer)
      
      dense_layer = tf.keras.layers.Dense(units=self.num_classes)
      self.resnet_layers.append(dense_layer)


  def _custom_dtype_getter(self, getter, name, shape=None, dtype=mb.DEFAULT_DTYPE,
                          *args, **kwargs):
    """Creates variables in fp32, then casts to fp16 if necessary.
    This function is a custom getter. A custom getter is a function with the
    same signature as tf.get_variable, except it has an additional getter
    parameter. Custom getters can be passed as the `custom_getter` parameter of
    tf.variable_scope. Then, tf.get_variable will call the custom getter,
    instead of directly getting a variable itself. This can be used to change
    the types of variables that are retrieved with tf.get_variable.
    The `getter` parameter is the underlying variable getter, that would have
    been called if no custom getter was used. Custom getters typically get a
    variable with `getter`, then modify it in some way.
    This custom getter will create an fp32 variable. If a low precision
    (e.g. float16) variable was requested it will then cast the variable to the
    requested dtype. The reason we do not directly create variables in low
    precision dtypes is that applying small gradients to such variables may
    cause the variable not to change.
    Args:
    getter: The underlying variable getter, that has the same signature as
    tf.get_variable and returns a variable.
    name: The name of the variable to get.
    shape: The shape of the variable to get.
    dtype: The dtype of the variable to get. Note that if this is a low
    precision dtype, the variable will be created as a tf.float32 variable,
    then cast to the appropriate dtype
    *args: Additional arguments to pass unmodified to getter.
    **kwargs: Additional keyword arguments to pass unmodified to getter.
    Returns:
    A variable which is cast to fp16 if necessary.
    """

    if dtype in mb.CASTABLE_TYPES:
      var = getter(name, shape, tf.float32, *args, **kwargs)
      return tf.cast(var, dtype=dtype, name=name + '_cast')
    else:
      return getter(name, shape, dtype, *args, **kwargs)

  def _model_variable_scope(self):
    """Returns a variable scope that the model should be created under.
    If self.dtype is a castable type, model variable will be created in fp32
    then cast to self.dtype before being used.
    Returns:
    A variable scope for the model.
    """

    return tf.compat.v1.variable_scope('resnet_model',
                      custom_getter=self._custom_dtype_getter)
  def call(self, inputs, training):
    """Add operations to classify a batch of input images.
    Args:
      inputs: A Tensor representing a batch of input images.
      training: A boolean. Set to True to add operations required only when
        training the classifier.
    Returns:
      A logits Tensor with shape [<batch_size>, self.num_classes].
    """
    temp_layers = list(self.resnet_layers).copy()
    with self._model_variable_scope():
      if self.data_format == 'channels_first':
        # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
        # This provides a large performance boost on GPU. See
        # https://www.tensorflow.org/performance/performance_guide#data_formats
        inputs = tf.transpose(a=inputs, perm=[0, 3, 1, 2])

      inputs = mb.conv2d_fixed_padding(inputs, temp_layers.pop(0), 
                                    self.kernel_size, self.conv_stride, self.data_format)
      inputs = tf.identity(inputs, 'initial_conv')

      # We do not include batch normalization or activation functions in V2
      # for the initial conv1 because the first ResNet unit will perform these
      # for both the shortcut and non-shortcut paths as part of the first
      # block's projection. Cf. Appendix of [2].
      if self.version == 'v1':
        inputs = mb.batch_norm(inputs, training, temp_layers.pop(0))
        inputs = tf.nn.relu(inputs)

      if self.first_pool_size:
        max_pool = tf.keras.layers.MaxPool2D(
            pool_size=self.first_pool_size,
            strides=self.first_pool_stride, padding='SAME',
            data_format=self.data_format)
        inputs = max_pool(inputs)
        inputs = tf.identity(inputs, 'initial_max_pool')

      for i, num_blocks in enumerate(self.block_sizes):
        num_filters = self.num_filters * (2**i)
        inputs = mb.block_layer(
            inputs=inputs, filters=num_filters, bottleneck=self.bottleneck,
            block_fn=self.block_fn, blocks=num_blocks,
            strides=self.block_strides[i], training=training,
            name='block_layer{}'.format(i + 1), data_format=self.data_format,block_layers=temp_layers.pop(0))

      # Only apply the BN and ReLU for model that does pre_activation in each
      # building/bottleneck block, eg resnet V2.
      if self.pre_activation:
        inputs = mb.batch_norm(inputs, training, temp_layers.pop(0))
        inputs = tf.nn.relu(inputs)

      # The current top layer has shape
      # `batch_size x pool_size x pool_size x final_size`.
      # ResNet does an Average Pooling layer over pool_size,
      # but that is the same as doing a reduce_mean. We do a reduce_mean
      # here because it performs better than AveragePooling2D.
      axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
      inputs = tf.reduce_mean(input_tensor=inputs, axis=axes, keepdims=True)
      inputs = tf.identity(inputs, 'final_reduce_mean')

      inputs = tf.squeeze(inputs, axes)
      inputs = temp_layers.pop(0)(inputs)
      inputs = tf.identity(inputs, 'final_dense')
      return inputs

class BilevelConvNet(BilevelModel):
  def __init__(self, num_class=2):
    super(BilevelConvNet, self).__init__()
    self.num_class = num_class
    self.conv1 = tf.keras.layers.Conv2D(filters=32,
                                  kernel_size=3,
                                  activation=tf.nn.relu)
    self.conv2 = tf.keras.layers.Conv2D(filters=64,
                                  kernel_size=3,
                                  activation=tf.nn.relu)
    self.max_pool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
    self.dropout1 = tf.keras.layers.Dropout(0.25)

    self.conv3 = tf.keras.layers.Conv2D(filters=128,
                                  kernel_size=3,
                                  activation=tf.nn.relu)
    self.max_pool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
    self.conv4 = tf.keras.layers.Conv2D(filters=128,
                                  kernel_size=3,
                                  activation=tf.nn.relu)
    self.max_pool3 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
    self.dropout2 = tf.keras.layers.Dropout(0.25)

    self.flatten = tf.keras.layers.Flatten()

    self.fc1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
    self.dropout3 = tf.keras.layers.Dropout(0.25)
    # self.fc2 = tf.keras.layers.Dense(units=84,  activation=tf.nn.relu)
    self.fc3 = tf.keras.layers.Dense(units=self.num_class)

  def call(self, input, training=True):
    del training
    input = input/128-1
    out = input
    for layer in self.layers:
      out = layer(out)
    return out

if __name__ == "__main__":
  m = BilevelConvNet()
  x = tf.keras.Input(shape=[32,32,3],dtype=tf.float32)
  y = m(x, True)
  print(m.summary())
  print("The input size is: {}".format(x.shape))
  print("The output size is: {}".format(y.shape))