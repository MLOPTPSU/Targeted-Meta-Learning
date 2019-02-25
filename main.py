from __future__ import division
from __future__ import print_function

import argparse
import functools
import itertools
import os
import json
from collections import namedtuple

import load_dataset as ld
import model_pool
import model
import utils
import numpy as np
import six
from six.moves import xrange 
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python import debug as tf_debug


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


	if is_training:
		train_features = features[0]
		train_labels = labels[0]
		val_features = features[1]
		val_labels = labels[1]
	else:
		train_features = features[0]
		train_labels = labels[0]
		val_features = features[0]
		val_labels = labels[0]


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
	with tf.variable_scope('base_model') as var_scope:
		if params.dataset == 'mnist':
			base_model = model.BilevelLenet()
		else:
			base_model = model.BilevelResNet(params.num_layers,
																			 is_training,
																			 params.batch_norm_decay,
																			 params.batch_norm_epsilon,
																			 data_format,
																			 version='v1')
		base_model_logits = base_model(train_features, data_format)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, var_scope.name)
		# Get the params of the model
		base_model_params = tf.trainable_variables(scope=var_scope.name)

		# Add weights for each sample in the batch
		# weight = tf.get_variable('weight',[params.train_batch_size])
		class_weights = tf.get_variable(
			'class_weight',
			# [train_labels.shape[1],1],
			initializer=np.array([[0.0],[0.0]]).astype(np.float32)
		)
		# train_op = [tf.assign(weight, tf.zeros(tf.shape(weight)))]
		# train_op = [tf.assign(weight, sample_weights)]
		weight = tf.matmul(
			train_labels,
			class_weights
		)
		
		base_model_loss, base_model_preds = _loss_fn(base_model_logits, train_labels)
		
		base_model_loss_reduced = tf.reduce_mean(tf.squeeze(weight) * base_model_loss) + weight_decay * tf.add_n(
																			[tf.nn.l2_loss(v) for v in base_model_params])
	
	
	with tf.variable_scope('bilevel_model', reuse=tf.AUTO_REUSE) as var_scope1:
		base_model.perturb_model_weights(base_model_loss_reduced, params.learning_rate, var_scope.name)
		target_logits = base_model(val_features, data_format)
		target_params = tf.trainable_variables(scope=var_scope1.name)
		target_loss, target_preds = _loss_fn(target_logits, val_labels)
		target_loss = tf.reduce_mean(target_loss) + weight_decay * tf.add_n(
																			[tf.nn.l2_loss(v) for v in target_params])

	weight_gradient = tf.gradients(target_loss, weight)
	update_weight = tf.clip_by_value(weight - weight_gradient[0],
																	clip_value_min=0.0, clip_value_max=100.0)
	sum_weight = tf.reduce_sum(update_weight) +  2e-12
	update_weight /= sum_weight

	train_op = [
		tf.assign(
			class_weights,
			tf.matmul(
				tf.transpose(train_labels),
				update_weight
			)
		)
	]
	# update_weight = tf.nn.l2_normalize(update_weight)
	train_op = []
	
	base_model_loss_reduced = tf.reduce_mean(tf.squeeze(update_weight) * base_model_loss) + weight_decay * tf.add_n(
																			[tf.nn.l2_loss(v) for v in base_model_params])
	
	base_model_grads = tf.gradients(base_model_loss_reduced, base_model_params)
	# clipped_base_model_grads = [tf.clip_by_value(bmg,-1.0,1.0) for bmg in base_model_grads]
	# base_model_grads = [tf.where(tf.is_nan(g), tf.zeros_like(g), g) for g in base_model_grads]
	# base_model_grads = [tf.zeros_like(g) for g in base_model_grads]
	base_model_gradvars = zip(base_model_grads, base_model_params)
	optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
	train_op.append(
			optimizer.apply_gradients(
				base_model_gradvars, global_step=tf.train.get_global_step())
		)

	# nan_list = [tf.reduce_any(tf.is_nan(p)) for p in [base_model_loss_reduced]]
	# max_grad = [tf.reduce_max(g) for g in base_model_grads]
	weight_sum = tf.reduce_sum(update_weight * tf.cast(tf.argmax(train_labels, axis=1), tf.float32))
	examples_sec_hook = utils.ExamplesPerSecondHook(
		params.train_batch_size, every_n_steps=100)

	tensors_to_log = {'Target loss': target_loss, 'Main loss':base_model_loss_reduced, 'class_weights':class_weights}

	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log, every_n_iter=10)
	train_hooks = [logging_hook, examples_sec_hook]


	train_op.extend(update_ops)
	train_op = tf.group(*train_op)

	if is_training:
		accuracy = tf.metrics.accuracy(tf.argmax(val_labels, axis=1), target_preds['classes'])
	else:
		accuracy = tf.metrics.accuracy(tf.argmax(train_labels, axis=1), base_model_preds['classes'])
	metrics = {'accuracy': accuracy}
	tf.summary.scalar('accuracy', accuracy[1])

	return tf.estimator.EstimatorSpec(
		mode=mode,
		predictions=target_preds,
		loss=base_model_loss_reduced,
		train_op=train_op,
		training_hooks=train_hooks,
		eval_metric_ops=metrics)

def _loss_fn(logits, labels):
	model_preds = {
	  'classes': tf.argmax(input=logits, axis=1),
	  'probabilities': tf.nn.softmax(logits)
  }


	model_loss = tf.losses.softmax_cross_entropy(
		  logits=logits, onehot_labels=labels, reduction=tf.losses.Reduction.NONE)  
	# model_loss_reduced = tf.reduce_mean(model_loss)
	return  model_loss, model_preds


def input_fn(data_dir,
			 subset,
			 batch_size,
			 dataset='mnist',
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
		use_distortion = (subset == 'train') and use_distortion_for_training
		d = ld.ImbalancedDataset(data_dir=data_dir, subset=subset, use_distortion=use_distortion, dataset=dataset)
		feature, label = d.make_batch(batch_size)
		
	return feature, label



def main(job_dir, data_dir, num_gpus, use_distortion_for_training,
			 log_device_placement, num_intra_threads, **hparams):

  # Session configuration.
  sess_config = tf.ConfigProto(
	  allow_soft_placement=True,
	  log_device_placement=log_device_placement,
	  intra_op_parallelism_threads=num_intra_threads,
	  gpu_options=tf.GPUOptions(force_gpu_compatible=True))

  config = utils.RunConfig(
	  session_config=sess_config, model_dir=job_dir)
  # config = config.replace(save_checkpoints_steps=100)

  train_input_fn = functools.partial(
	  input_fn,
	  data_dir,
	  subset='train',
	  batch_size=hparams['train_batch_size'],
		dataset=hparams['dataset'],
	  use_distortion_for_training=use_distortion_for_training)

  eval_input_fn = functools.partial(
	  input_fn,
	  data_dir,
	  subset='test',
		dataset=hparams['dataset'],
	  batch_size=hparams['eval_batch_size'])

  

  train_steps = hparams['train_steps']
  eval_steps = 2000 // hparams['eval_batch_size']

  train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=train_steps)
  eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=eval_steps,   throttle_secs=6000)

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
		default=20,
		help='The number of layers of the model.')
	parser.add_argument(
		'--train-steps',
		type=int,
		default=8000,
		help='The number of steps to use for training.')
	parser.add_argument(
		'--train-batch-size',
		type=int,
		default=50,
		help='Batch size for training.')
	parser.add_argument(
		'--eval-batch-size',
		type=int,
		default=50,
		help='Batch size for validation.')
	parser.add_argument(
		'--momentum',
		type=float,
		default=0.9,
		help='Momentum for MomentumOptimizer.')
	parser.add_argument(
		'--weight-decay',
		type=float,
		default=2e-3,
		help='Weight decay for convolutions.')
	parser.add_argument(
		'--learning-rate',
		type=float,
		default=0.00001,
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
		default=False,
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
	parser.add_argument(
		'--dataset',
		type=str,
		choices=['mnist','cifar10'],
		default='mnist',
		help='Datset name to run the experiment on.'
	)

	args = parser.parse_args()

	if args.num_gpus > 0:
		assert tf.test.is_gpu_available(), "Requested GPUs but none found."
	if args.num_gpus < 0:
		raise ValueError(
			'Invalid GPU count: \"--num-gpus\" must be 0 or a positive integer.')

	if (args.num_layers - 2) % 6 != 0:
		raise ValueError('Invalid --num-layers parameter.')


	main(**vars(args))
