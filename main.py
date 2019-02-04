from __future__ import division
from __future__ import print_function

import argparse
import functools
import itertools
import os
import json
from collections import namedtuple

import cifar10
import model_pool
import utils
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def get_model_fn(features, labels, mode, params):
	"""Returns a function that will build the TargetedLearning framework."""

	"""Model body.

	Args:
		features: a list of tensors
		labels: a list of tensors
		mode: ModeKeys.TRAIN or EVAL
		params: Hyperparameters suitable for tuning
	Returns:
		A EstimatorSpec object.
	"""
	is_training = (mode == tf.estimator.ModeKeys.TRAIN)
	weight_decay = params.weight_decay
	momentum = params.momentum

	train_features = features[0]
	train_labels = labels[0]
	val_features = features[1]
	val_labels = labels[1]
	losses = []
	gradvars = []
	preds = []

	# channels first (NCHW) is normally optimal on GPU and channels last (NHWC)
	# on CPU. The exception is Intel MKL on CPU which is optimal with
	# channels_last.
	num_gpus = len(utils.get_available_gpus())
	data_format = params.data_format
	if not data_format:
		if num_gpus == 0:
			data_format = 'channels_last'
		else:
			data_format = 'channels_first'


	# Building the main model

	with tf.variable_scope('model') as var_scope:
		main_loss, main_preds = _tower_fn(
			is_training, weight_decay, train_features, train_labels,
			data_format, params.num_layers, params.batch_norm_decay,
			params.batch_norm_epsilon)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, var_scope.name)
		# Get the params of the model
		main_params = tf.trainable_variables(scope=var_scope.name)
		
	with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as var_scope1:
		target_loss, target_preds = _tower_fn(
			is_training, weight_decay, val_features, val_labels,
			data_format, params.num_layers, params.batch_norm_decay,
			params.batch_norm_epsilon)
		target_loss = tf.reduce_mean(target_loss)

	with tf.variable_scope('sample_weight') as var_scope_weight:
		weight = tf.get_variable('weight',tf.shape(main_loss))
		train_op = [
			tf.assign(weight, tf.zeros(tf.shape(weight)))
			]
		target_param = tf.trainable_variables(scope=var_scope_weight.name)	
	
	main_loss = tf.reduce_mean(weight * main_loss)

	update_weight = tf.clip_by_value(-tf.gradients(target_loss, target_param),
																	clip_value_min=0.0)
	
	sum_weight = tf.reduce_sum(update_weight)
	if sum_weight != 0:
		update_weight /= sum_weight
	
	train_op.append(tf.assing(weight, update_weight))
	main_grad = tf.gradient(main_loss, main_params)
	gradvars = zip(main_grad, main_params)

	examples_sec_hook = utils.ExamplesPerSecondHook(
		params.train_batch_size, every_n_steps=10)

	tensors_to_log = {'Target loss': target_loss, 'Main loss':main_loss}

	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log, every_n_iter=100)

	train_hooks = [logging_hook, examples_sec_hook]

	optimizer = tf.train.MomentumOptimizer(
		learning_rate=params.learning_rate, momentum=momentum)

	# Create single grouped train op
	train_op.append(
		optimizer.apply_gradients(
			gradvars, global_step=tf.train.get_global_step())
	)
	train_op.extend(update_ops)
	train_op = tf.group(*train_op)

	predictions = {
		'classes':
			tf.concat([p['classes'] for p in target_preds], axis=0),
		'probabilities':
			tf.concat([p['probabilities'] for p in target_preds], axis=0)
	}
	metrics = {
		'accuracy':
			tf.metrics.accuracy(val_labels, predictions['classes'])
	}

	return tf.estimator.EstimatorSpec(
		mode=mode,
		predictions=predictions,
		loss=main_loss,
		train_op=train_op,
		training_hooks=train_hooks,
		eval_metric_ops=metrics)


def _tower_fn(is_training, weight_decay, feature, label, data_format,
			  num_layers, batch_norm_decay, batch_norm_epsilon):
  """Build computation tower (Resnet).

  Args:
	is_training: true if is training graph.
	weight_decay: weight regularization strength, a float.
	feature: a Tensor.
	label: a Tensor.
	data_format: channels_last (NHWC) or channels_first (NCHW).
	num_layers: number of layers, an int.
	batch_norm_decay: decay for batch normalization, a float.
	batch_norm_epsilon: epsilon for batch normalization, a float.

  Returns:
	A tuple with the loss for the tower, the gradients and parameters, and
	predictions.

  """
  model = model_pool.ResNetCifar10(
	  num_layers,
	  batch_norm_decay=batch_norm_decay,
	  batch_norm_epsilon=batch_norm_epsilon,
	  is_training=is_training,
	  data_format=data_format)
  logits = model.forward_pass(feature, input_data_format='channels_last')
  tower_pred = {
	  'classes': tf.argmax(input=logits, axis=1),
	  'probabilities': tf.nn.softmax(logits)
  }

  tower_loss = tf.losses.sparse_softmax_cross_entropy(
	  logits=logits, labels=label)

  return tower_loss, tower_pred


def input_fn(data_dir,
			 subset,
			 batch_size,
			 use_distortion_for_training=True):
  """Create input graph for model.

  Args:
	data_dir: Directory where TFRecords representing the dataset are located.
	subset: one of 'train', 'validate' and 'eval'.
	batch_size: total batch size for training
	use_distortion_for_training: True to use distortions.
  Returns:
	two lists of tensors for features and labels
  """
  with tf.device('/cpu:0'):
	use_distortion = subset == 'train' and use_distortion_for_training
	dataset = cifar10.Cifar10DataSet(data_dir, num_shards, subset, use_distortion, redundancy)
	feature_shards, label_shards = dataset.make_batch(batch_size)
   
	return feature_shards, label_shards



def main(job_dir, data_dir, num_gpus, use_distortion_for_training,
			 log_device_placement, num_intra_threads, **hparams):
  # The env variable is on deprecation path, default is set to off.
  os.environ['TF_SYNC_ON_FINISH'] = '0'
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
  if hparams['config_path']:
	  TF_CONFIG = json.load(open(hparams['config_path'], "r"))
	  TF_CONFIG['model_dir'] = job_dir
	  os.environ['TF_CONFIG'] = json.dumps(TF_CONFIG)

  # Session configuration.
  sess_config = tf.ConfigProto(
	  allow_soft_placement=True,
	  log_device_placement=log_device_placement,
	  intra_op_parallelism_threads=num_intra_threads,
	  gpu_options=tf.GPUOptions(force_gpu_compatible=True))

  config = utils.RunConfig(
	  session_config=sess_config, model_dir=job_dir)
  config.replace(save_checkpoints_steps=1000)

  train_input_fn = functools.partial(
	  input_fn,
	  data_dir,
	  subset='train',
	  batch_size=hparams['train_batch_size'],
	  use_distortion_for_training=use_distortion_for_training)

  eval_input_fn = functools.partial(
	  input_fn,
	  data_dir,
	  subset='eval',
	  batch_size=hparams['eval_batch_size'])

  num_eval_examples = cifar10.Cifar10DataSet.num_examples_per_epoch('eval')
  if num_eval_examples % hparams['eval_batch_size'] != 0:
	  raise ValueError(
		  'validation set size must be multiple of eval_batch_size')

  train_steps = hparams['train_steps']
  eval_steps = num_eval_examples // hparams['eval_batch_size']

  train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=train_steps)
  eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=eval_steps, throttle_secs=60)

  classifier = tf.estimator.Estimator(
	  model_fn=get_model_fn,
	  config=config,
	  params=tf.contrib.training.HParams(
            is_chief=config.is_chief,
                **hparams))

  # Create experiment.
  tf.estimator.train_and_evaluate(
	  estimator=classifier,
	  train_spec=train_spec,
	  eval_spec=eval_spec)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
	  '--data-dir',
	  type=str,
	  required=True,
	  help='The directory where the CIFAR-10 input data is stored.')
  parser.add_argument(
	  '--job-dir',
	  type=str,
	  required=True,
	  help='The directory where the model will be stored.')
  parser.add_argument(
	  '--num-gpus',
	  type=int,
	  default=1,
	  help='The number of gpus used. Uses only CPU if set to 0.')
  parser.add_argument(
	  '--num-layers',
	  type=int,
	  default=44,
	  help='The number of layers of the model.')
  parser.add_argument(
	  '--train-steps',
	  type=int,
	  default=80000,
	  help='The number of steps to use for training.')
  parser.add_argument(
	  '--train-batch-size',
	  type=int,
	  default=128,
	  help='Batch size for training.')
  parser.add_argument(
	  '--eval-batch-size',
	  type=int,
	  default=100,
	  help='Batch size for validation.')
  parser.add_argument(
	  '--momentum',
	  type=float,
	  default=0.9,
	  help='Momentum for MomentumOptimizer.')
  parser.add_argument(
	  '--weight-decay',
	  type=float,
	  default=2e-4,
	  help='Weight decay for convolutions.')
  parser.add_argument(
	  '--learning-rate',
	  type=float,
	  default=0.1,
	  help="""\
	  This is the inital learning rate value. The learning rate will decrease
	  during training. For more details check the model_fn implementation in
	  this file.\
	  """)
  parser.add_argument(
	  '--use-distortion-for-training',
	  type=bool,
	  default=True,
	  help='If doing image distortion for training.')
  parser.add_argument(
	  '--num-intra-threads',
	  type=int,
	  default=0,
	  help="""\
	  Number of threads to use for intra-op parallelism. When training on CPU
	  set to 0 to have the system pick the appropriate number or alternatively
	  set it to the number of physical CPU cores.\
	  """)
  parser.add_argument(
	  '--num-inter-threads',
	  type=int,
	  default=0,
	  help="""\
	  Number of threads to use for inter-op parallelism. If set to 0, the
	  system will pick an appropriate number.\
	  """)
  parser.add_argument(
	  '--data-format',
	  type=str,
	  default=None,
	  help="""\
	  If not set, the data format best for the training device is used. 
	  Allowed values: channels_first (NCHW) channels_last (NHWC).\
	  """)
  parser.add_argument(
	  '--log-device-placement',
	  action='store_true',
	  default=True,
	  help='Whether to log device placement.')
  parser.add_argument(
	  '--batch-norm-decay',
	  type=float,
	  default=0.997,
	  help='Decay for batch norm.')
  parser.add_argument(
	  '--batch-norm-epsilon',
	  type=float,
	  default=1e-5,
	  help='Epsilon for batch norm.')







  args = parser.parse_args()

  if args.num_gpus > 0:
	assert tf.test.is_gpu_available(), "Requested GPUs but none found."
  if args.num_gpus < 0:
	raise ValueError(
		'Invalid GPU count: \"--num-gpus\" must be 0 or a positive integer.')

  if (args.num_layers - 2) % 6 != 0:
	raise ValueError('Invalid --num-layers parameter.')


  main(**vars(args))
