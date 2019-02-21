import tensorflow as tf


# def perturb_weights(Layer, loss, lr, old_vs_name=None, prefix=''):
#         if Layer.weights:
#             gradients = tf.gradients(loss, Layer.weights)
#             new_weights = []
#             for w,g in zip(Layer.weights, gradients):
#                 old_name = get_name(w.name, old_vs_name=old_vs_name)
#                 new_name = prefix + old_name
#                 tmp = tf.identity( w - lr * g,
#                                     name=new_name
#                 )
#                 new_weights.append(tmp)
#                 tf.add_to_collection(
#                     tf.GraphKeys.TRAINABLE_VARIABLES,
#                     tmp
#                 )
#             Layer._trainable_weights = new_weights
            


# def get_name(name, old_vs_name):
#     if old_vs_name:
#         base_name = name.split(old_vs_name + '/',1)
#         if len(base_name) > 1:
#             return base_name[1].split(':')[0]
#         else:
#             raise ValueError('The variable named {} is not in the variable scope {}'.format(name, old_vs_name))
#     else:
#         return name.split(':')[0]


# def set_name(old_name, old_vs_name, prefix):
#     curr_vs_name = tf.get_variable_scope().name
#     if ((curr_vs_name == old_vs_name) & (not prefix)):
#         raise ValueError('Current name cannot be the same as previous variable name. Please set it in the new variable scope or set a prefix.')
#     elif curr_vs_name == old_vs_name:
#         return prefix + '/' + old_name
#     else:
#         return old_name


class BilevelModel(tf.keras.Model):
    
    def perturb_model_weights(self, loss, lr, old_vs_name=None, prefix=''):
        if self.layers:
            for layer in self.layers:
                self.perturb_layer_weights(layer, loss, lr, old_vs_name, prefix)
        else:
            print('The model {} does not have any layers'.format(self.name))


    def perturb_layer_weights(self, layer, loss, lr, old_vs_name=None, prefix=''):
        if layer.weights:
            gradients = tf.gradients(loss, layer.weights)
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
            #TODO: change this behaviour. Dangerous!
            layer.kernel = new_weights[0]
            layer.bias = new_weights[1]
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

    def call(self, input):
        out = input
        for layer in self.layers:
            out = layer(out)
        return out

