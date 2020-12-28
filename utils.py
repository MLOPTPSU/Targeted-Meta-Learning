"""
Utitlity functions to be used in the main program

__author__ = "MM. Kamani"
"""
import collections
import six

import tensorflow as tf
import numpy as np
import csv

from tensorflow.python.platform import tf_logging as logging
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training import device_setter
# from tensorflow.contrib.learn.python.learn import run_config
from tensorflow.python.client import device_lib


# TODO(b/64848083) Remove once uid bug is fixed

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


class RunConfig(tf.estimator.RunConfig):
  def uid(self, whitelist=None):
    """Generates a 'Unique Identifier' based on all internal fields.
    Caller should use the uid string to check `RunConfig` instance integrity
    in one session use, but should not rely on the implementation details, which
    is subject to change.
    Args:
      whitelist: A list of the string names of the properties uid should not
        include. If `None`, defaults to `_DEFAULT_UID_WHITE_LIST`, which
        includes most properties user allowes to change.
    Returns:
      A uid string.
    """
    # if whitelist is None:
    #   whitelist = run_config._DEFAULT_UID_WHITE_LIST

    state = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
    # Pop out the keys in whitelist.
    # for k in whitelist:
    #   state.pop('_' + k, None)

    ordered_state = collections.OrderedDict(
        sorted(state.items(), key=lambda t: t[0]))
    # For class instance without __repr__, some special cares are required.
    # Otherwise, the object address will be used.
    if '_cluster_spec' in ordered_state:
      ordered_state['_cluster_spec'] = collections.OrderedDict(
         sorted(ordered_state['_cluster_spec'].as_dict().items(),
                key=lambda t: t[0])
      )
    return ', '.join(
        '%s=%r' % (k, v) for (k, v) in six.iteritems(ordered_state)) 


class ExamplesPerSecondHook(session_run_hook.SessionRunHook):
  """Hook to print out examples per second.

    Total time is tracked and then divided by the total number of steps
    to get the average step time and then batch_size is used to determine
    the running average of examples per second. The examples per second for the
    most recent interval is also logged.
  """

  def __init__(
      self,
      batch_size,
      every_n_steps=100,
      every_n_secs=None,):
    """Initializer for ExamplesPerSecondHook.

      Args:
      batch_size: Total batch size used to calculate examples/second from
      global time.
      every_n_steps: Log stats every n steps.
      every_n_secs: Log stats every n seconds.
    """
    if (every_n_steps is None) == (every_n_secs is None):
      raise ValueError('exactly one of every_n_steps'
                       ' and every_n_secs should be provided.')
    self._timer = basic_session_run_hooks.SecondOrStepTimer(
        every_steps=every_n_steps, every_secs=every_n_secs)

    self._step_train_time = 0
    self._total_steps = 0
    self._batch_size = batch_size

  def begin(self):
    self._global_step_tensor = training_util.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          'Global step should be created to use StepCounterHook.')

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return basic_session_run_hooks.SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):
    _ = run_context

    global_step = run_values.results
    if self._timer.should_trigger_for_step(global_step):
      elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
          global_step)
      if elapsed_time is not None:
        steps_per_sec = elapsed_steps / elapsed_time
        self._step_train_time += elapsed_time
        self._total_steps += elapsed_steps

        average_examples_per_sec = self._batch_size * (
            self._total_steps / self._step_train_time)
        current_examples_per_sec = steps_per_sec * self._batch_size
        # Average examples/sec followed by current examples/sec
        logging.info('%s: %g (%g), step = %g', 'Average examples/sec',
                     average_examples_per_sec, current_examples_per_sec,
                     self._total_steps)


class NanFinderHook(session_run_hook.SessionRunHook):
  def __init__(
      self,
      params_list,
      every_n_steps=1,
      every_n_secs=None,):

    if (every_n_steps is None) == (every_n_secs is None):
      raise ValueError('exactly one of every_n_steps'
                       ' and every_n_secs should be provided.')
    self._timer = basic_session_run_hooks.SecondOrStepTimer(
        every_steps=every_n_steps, every_secs=every_n_secs)

    self._step_train_time = 0
    self._total_steps = 0
    self._params_list = params_list
    self._nan_list = tf.reduce_any([tf.reduce_any(tf.is_nan(p)) for p in self._params_list])
    self._inf_list = tf.reduce_any([tf.reduce_any(tf.is_inf(p)) for p in self._params_list])

  def begin(self):
    self._global_step_tensor = training_util.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          'Global step should be created to use StepCounterHook.')

  def before_run(self, run_context):  # pylint: disable=unused-argument
    global_step = run_context.session.run(self._global_step_tensor)

    if self._timer.should_trigger_for_step(global_step):
      nan_list_ev = run_context.session.run(self._nan_list)
      inf_list_ev = run_context.session.run(self._inf_list)
      # nan_inds = [i for i, x in enumerate(nan_list_ev) if x.any()]
      # nan_params = [self._params_list[i] for i in nan_inds]
      
      logging.info('Are params are nan or inf before run:')
      logging.info(nan_list_ev)
      logging.info(inf_list_ev)

      self._timer.update_last_triggered_step(global_step)

  def after_run(self, run_context, run_values):
    nan_list_ev  = run_context.session.run(self._nan_list)
    inf_list_ev = run_context.session.run(self._inf_list)
    # nan_inds = [i for i, x in enumerate(nan_list_ev) if x.any()]
    # nan_params = [self._params_list[i] for i in nan_inds]
    logging.info('Are params are nan or inf after run:')
    logging.info(nan_list_ev)
    logging.info(inf_list_ev)

class WeightUpdateHook(tf.estimator.SessionRunHook):
  def __init__(
    self,
    weight,
    update_weight,
    index):
    self.weight = weight
    self.update_weight = update_weight
    self.index = index
    self.sample_weight_dict = {}
    self.sample_weight = tf.placeholder(tf.float32, self.weight.shape)
    self.update_op = [tf.assign(self.weight, self.sample_weight)]

  def before_run(self, run_context):
    # self.inds = run_context.session.run(self.index)
    # sample_weight_value = [self.sample_weight_dict[i] if i in self.sample_weight_dict else 0.0 for i in self.inds]
    sample_weight_value = np.zeros(self.weight.shape)
    run_context.session.run(self.update_op, feed_dict={self.sample_weight:sample_weight_value})

  # def after_run(self,run_context,run_values):
  #   # weight_values = run_context.session.run(self.update_weight)
    # self.sample_weight_dict.update(zip(self.inds, weight_values))


class WeightUpdateHook1(tf.estimator.SessionRunHook):
  def __init__(
    self,
    weight,
    update_weight,
    every_n_steps=10,
    every_n_secs=None,
    log_every_n_step=100):

    self.weight = weight
    self.update_weight = update_weight
    self.log_every_n_step = log_every_n_step
    # self.sample_weight_dict = {}
    # self.sample_weight = tf.placeholder(tf.float32, self.weight.shape)
    self.update_op = [tf.compat.v1.assign(self.weight, self.update_weight)]
    

    if (every_n_steps is None) == (every_n_secs is None):
      raise ValueError('exactly one of every_n_steps'
                       ' and every_n_secs should be provided.')
    self._timer = basic_session_run_hooks.SecondOrStepTimer(
        every_steps=every_n_steps, every_secs=every_n_secs)

  
  def begin(self):
    self._global_step_tensor = training_util.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          'Global step should be created to use StepCounterHook.')

  def before_run(self, run_context):
    global_step = run_context.session.run(self._global_step_tensor)
    if self._timer.should_trigger_for_step(global_step):
      # self.inds = run_context.session.run(self.index)
      # sample_weight_value = [self.sample_weight_dict[i] if i in self.sample_weight_dict else 0.0 for i in self.inds]
      # sample_weight_value = np.zeros(self.weight.shape)
      run_context.session.run(self.update_op)
      if global_step % self.log_every_n_step == 0:
        new_weight_value = np.squeeze(run_context.session.run(self.update_weight))
        logging.info('Weights updated in step {}'.format(global_step))
        logging.info('New weights are: {}'.format(new_weight_value))
      # with open('weights.csv', 'w') as f:
      #   writer = csv.writer(f)
      #   writer.writerow(new_weight_value)
      self._timer.update_last_triggered_step(global_step)
  # def after_run(self,run_context,run_values):
  #   weight_values = run_context.session.run(self.update_weight)
  #   self.sample_weight_dict.update(zip(self.inds, weight_values))


class dict2obj:
    def __init__(self, **entries):
        self.__dict__.update(entries)