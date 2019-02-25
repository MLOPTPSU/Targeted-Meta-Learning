import tensorflow as tf
from model_base import ResNet1


class BilevelModel(tf.keras.Model):
    
    def perturb_model_weights(self, loss, lr, old_vs_name=None, prefix=''):
        if self.layers:
            for layer in self.layers:
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
                tmp = tf.identity( w - lr * g,
                                    name=new_name
                )
                new_weights.append(tmp)
                tf.add_to_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    tmp
                )
            layer._trainable_weights = new_weights
            try:
                layer.kernel = new_weights[0]
                if layer.bias is not None:   
                    layer.bias = new_weights[1]
            except AttributeError:
                try:
                    layer.gamma = new_weights[0]
                    layer.beta = new_weights[1]
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
        curr_vs_name = tf.get_variable_scope().name
        if ((curr_vs_name == old_vs_name) & (not prefix)):
            raise ValueError('Current name cannot be the same as previous variable name. Please set it in the new variable scope or set a prefix.')
        elif curr_vs_name == old_vs_name:
            return prefix + '/' + old_name
        else:
            return old_name
    

class BilevelLenet(BilevelModel):
    def __init__(self):
        super(BilevelLenet, self).__init__()
        self.conv1 = tf.layers.Conv2D(filters=6,
                                     kernel_size=5,
                                     activation=tf.nn.relu)
        self.max_pool1 = tf.layers.MaxPooling2D(pool_size=2, strides=2)

        self.conv2 = tf.layers.Conv2D(filters=16,
                                     kernel_size=5,
                                     activation=tf.nn.relu)
        self.max_pool2 = tf.layers.MaxPooling2D(pool_size=2, strides=2)

        self.flatten = tf.layers.Flatten()

        self.fc1 = tf.layers.Dense(units=120, activation=tf.nn.relu)
        self.fc2 = tf.layers.Dense(units=84,  activation=tf.nn.relu)
        self.fc3 = tf.layers.Dense(units=2)

    def call(self, input, input_data_format='channels_last'):
        del input_data_format
        out = input
        for layer in self.layers:
            out = layer(out)
        return out

class BilevelResNet(BilevelModel, ResNet1):
  def __init__(self,
               num_layers,
               is_training,
               batch_norm_decay,
               batch_norm_epsilon,
               data_format='channels_last',
               version='v1'):
    super(BilevelResNet,self).__init__()
    ResNet1.__init__(self, is_training,data_format,batch_norm_decay,batch_norm_epsilon)
    
    self.n = (num_layers - 2) // 6
    # Add one in case label starts with 1. No impact if label starts with 0.
    self.num_classes = 2
    self.filters = [16, 16, 32, 64]
    self.strides = [1, 2, 2]
    
    self._model_layers = []
    #Build the model
    inputs = tf.placeholder(tf.float32, [None,32,32,3])
    x, conv1 = self._conv(inputs, 3, 16, 1)
    x, batch_norm1 = self._batch_norm(x)
    x, relu1 = self._relu(x)

    self._model_layers.append([conv1, batch_norm1, relu1])
    # Use basic (non-bottleneck) block and ResNet V1 (post-activation).
    if version == 'v1':
      self.res_func_build = self._residual_v1_build
      self.res_func = self._residual_v1
    elif version == 'v2':
      self.res_func_build = self._residual_v2_build
      self.res_func = self._residual_v2
    elif version == 'bv2':
      self.res_func_build = self._bottleneck_residual_v2_build
      self.res_func = self._bottleneck_residual_v2
  
    # 3 stages of block stacking.
    for i in range(3):
      with tf.name_scope('stage'):
        for j in range(self.n):
          if j == 0:
            # First block in a stage, filters and strides may change.
            x, res_layers = self.res_func_build(x, 3, self.filters[i], self.filters[i + 1],
                         self.strides[i])
          else:
            # Following blocks in a stage, constant filters and unit stride.
            x, res_layers = self.res_func_build(x, 3, self.filters[i + 1], self.filters[i + 1], 1)
          self._model_layers.append(res_layers)

    x = self._global_avg_pool(x, True)
    x, dense1 = self._fully_connected(x, self.num_classes)
    self._model_layers.append([dense1])

  def call(self, x, input_data_format):
    """Build the core model within the graph."""
    if self._data_format != input_data_format:
      if input_data_format == 'channels_last':
        # Computation requires channels_first.
        x = tf.transpose(x, [0, 3, 1, 2])
      else:
        # Computation requires channels_last.
        x = tf.transpose(x, [0, 2, 3, 1])

    # Image standardization.
    # x = x / 128 - 1
    
    for l in self._model_layers[0]:
      x = l(x)


    # 3 stages of block stacking.
    for i in range(3):
      for j in range(self.n):
        if j == 0:
          # First block in a stage, filters and strides may change.
          x = self.res_func(x, self.filters[i], self.filters[i + 1],
                        self._model_layers[1 + i*self.n])
        else:
          # Following blocks in a stage, constant filters and unit stride.
          x = self.res_func(x, self.filters[i + 1], self.filters[i + 1],
                        self._model_layers[1 + i*self.n + j])

    x = self._global_avg_pool(x)
    x = self._model_layers[-1][0](x)

    return x

