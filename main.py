"""
The main program to run the training and inference

__author__ = "MM. Kamani"
"""

from __future__ import division
from __future__ import print_function

import argparse
import functools
import itertools
import os
import json
from time import strftime
from collections import namedtuple

import load_dataset as ld
import model
import utils
import numpy as np
import six
from six.moves import xrange 
import tensorflow as tf
import logging


logger = tf.get_logger()
logger.setLevel(logging.INFO)


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
	out_lr = 0.1 #params.learning_rate


	train_features = features[0]
	train_labels = labels[0]
	if is_training:
		val_features = features[1]
		val_labels = labels[1]
	else:
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

	train_op = []

	# Building the base model
	with tf.compat.v1.variable_scope('base_model') as var_scope:
		if params.dataset == 'mnist':
			base_model = model.BilevelLenet(num_class=params.num_class)
		else:
			base_model = model.BilevelResNet(resnet_size=params.num_layers,
									num_classes=params.num_class,
									resnet_version=params.version)
		base_model_logits = base_model(train_features, is_training)
		update_ops = tf.compat.v1.get_collection(
							tf.compat.v1.GraphKeys.UPDATE_OPS, var_scope.name)
		extra_update_ops = base_model.get_updates_for(train_features)
		update_ops.extend(extra_update_ops)
		# Get the params of the model
		base_model_params = tf.compat.v1.trainable_variables(scope=var_scope.name)

		# Set initial weights
		class_init=np.array([[1.0/params.num_class] for _ in range(params.num_class)]).astype(np.float32)
		class_weights = tf.compat.v1.get_variable(
			'class_weight',
			initializer=class_init
		)

		weight = tf.matmul(
			tf.cast(tf.one_hot(train_labels,len(class_init),
								 on_value=1, off_value=0),tf.float32),
			class_weights
		)
		
		# Get the loss of the main model
		base_model_loss, base_model_preds = _loss_fn(base_model_logits, tf.one_hot(train_labels,params.num_class, on_value=1, off_value=0))
		base_model_loss_reduced = tf.reduce_mean(tf.squeeze(weight) * base_model_loss) + weight_decay * tf.add_n(
																			[tf.nn.l2_loss(v) for v in base_model_params])
	
	# Define the outer model's logits, which is the bilevel model
	with tf.compat.v1.variable_scope('bilevel_model', reuse=tf.compat.v1.AUTO_REUSE) as var_scope1:
		base_model.perturb_model_weights(base_model_loss_reduced, params.learning_rate, var_scope.name)
		target_logits = base_model(val_features, False)
		target_params = tf.compat.v1.trainable_variables(scope=var_scope1.name)
		target_loss, target_preds = _loss_fn(target_logits, tf.one_hot(val_labels,params.num_class, on_value=1, off_value=0))
		target_loss = tf.reduce_mean(target_loss) + weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in target_params])
	
	# Calculate the gradients with respect to the class weights and normalize it
	class_weight_gradient = tf.gradients(target_loss, class_weights)
	update_class_weights = tf.clip_by_value(class_weights - out_lr * class_weight_gradient[0],
																	clip_value_min=0.0, clip_value_max=100.0)
	sum_class_weights = tf.reduce_sum(update_class_weights) +  2e-12
	update_class_weights /= sum_class_weights

	# Update the weight every n steps.
	weight_update_hook = utils.WeightUpdateHook1(class_weights, update_class_weights, every_n_steps=10, log_every_n_step=params.log_freq)
	
	# Calculate the base model grads
	base_model_grads = tf.gradients(base_model_loss_reduced, base_model_params)
	base_model_gradvars = zip(base_model_grads, base_model_params)
	
	boundaries = [
		params.num_batches_per_epoch * x
		for x in np.array([91, 136, 182], dtype=np.int64)
	]
	staged_lr = [params.learning_rate * x for x in [1, 0.1, 0.01, 0.001]]

	learning_rate =tf.compat.v1.train.piecewise_constant(tf.compat.v1.train.get_global_step(), boundaries, staged_lr)

	# Define optimizer
	optimizer = tf.compat.v1.train.MomentumOptimizer(
		learning_rate=learning_rate, momentum=params.momentum)
	# optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
	train_op.append(
			optimizer.apply_gradients(
				base_model_gradvars, global_step=tf.compat.v1.train.get_global_step())
		)
	
	# Calculate metrics
	target_accuracy = tf.compat.v1.metrics.accuracy(val_labels, target_preds['classes'])
	accuracy = tf.compat.v1.metrics.accuracy(train_labels, base_model_preds['classes'])
	# The following metrics are for the binary classification scenario.
	# They should be adopted for multiclass classification tasks.
	if params.num_class ==2:
		train_labels_mask = tf.cast(train_labels,tf.bool)
		inverse_train_labels_mask = tf.cast(tf.math.logical_not(train_labels_mask),tf.float32)
		inverse_prediction_mask = tf.cast(tf.math.logical_not(tf.cast(base_model_preds['classes'], tf.bool)), tf.float32)
		recall_minor = tf.compat.v1.metrics.recall(inverse_train_labels_mask, inverse_prediction_mask)
		recall_major = tf.compat.v1.metrics.recall(train_labels, base_model_preds['classes'])
		precision_minor = tf.compat.v1.metrics.precision(inverse_train_labels_mask, inverse_prediction_mask)
		metrics = {'obj/accuracy': accuracy, 'metrics/recall_minor': recall_minor,
				   'metrics/recall_major':recall_major, 'metrics/precision_minor': precision_minor}
	else:
		metrics = {'obj/accuracy': accuracy}


	examples_sec_hook = utils.ExamplesPerSecondHook(
		params.train_batch_size, every_n_steps=params.log_freq)

	tensors_to_log = {'Target loss': target_loss, 'Main loss':base_model_loss_reduced,
					  'Target accuracy':target_accuracy[1], 'Main accuracy':accuracy[1],
					  'learning_rates': learning_rate, 'step': tf.compat.v1.train.get_global_step()
					}
	
	logging_hook = tf.estimator.LoggingTensorHook(
		tensors=tensors_to_log, every_n_iter=params.log_freq)
	train_hooks = [weight_update_hook, logging_hook, examples_sec_hook]


	train_op.extend(update_ops)
	train_op = tf.group(*train_op)

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


	model_loss = tf.compat.v1.losses.softmax_cross_entropy(
		  logits=logits, onehot_labels=labels, reduction=tf.losses.Reduction.NONE) 
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
  sess_config =  tf.compat.v1.ConfigProto(
	  allow_soft_placement=True,
	  log_device_placement=log_device_placement,
	  intra_op_parallelism_threads=num_intra_threads,
	  gpu_options=tf.compat.v1.GPUOptions(force_gpu_compatible=True))

  config = utils.RunConfig(
	  session_config=sess_config, model_dir=job_dir)

  if hparams['eval']:
    config = config.replace(save_checkpoints_steps=hparams['eval_freq'])

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
  eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=eval_steps, start_delay_secs=0, throttle_secs=0)

  classifier = tf.estimator.Estimator(
    model_fn=get_model_fn,
    config=config,
    params=utils.dict2obj(**hparams))

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
		help='The directory where the input data is stored.')
	parser.add_argument(
		'--job-dir',
		type=str,
		required=True,
		help='The directory where the model and logs will be stored.')
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
		default=0.01,
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
		choices=['mnist','cifar10','cifar100'],
		default='mnist',
		help='Datset name to run the experiment on.'
	)
	parser.add_argument(
		'--num-class',
		type=int,
		default=2,
		help='Num of classes in the dataset.'
	)
	parser.add_argument(
		'--version',
		type=str,
		choices=['v1','v2','bv2'],
		default='v1',
		help='Version of the ResNet network. Only wworks with non MNIST datasets.'
	)
	parser.add_argument(
      '--eval',
      action='store_true',
      default=False,
      help="""If present when running in a distributed environment will run on eval mode.""")
	parser.add_argument(
		'--eval-freq',
		type=int,
		default=1000,
		help='Frequency of performing evaluation on test dataset based on step numbers'
	)
	parser.add_argument(
		'--log-freq',
		type=int,
		default=100,
		help='Frequency of reporting logs for the training based on step numbers.'
	)
	parser.add_argument(
      '--num-training-samples',
	  type=int,
	  required=True,
      help="""Indicates number of training samples in datasets.""")

	args = parser.parse_args()

	if args.num_gpus > 0:
		assert tf.test.is_gpu_available(), "Requested GPUs but none found."
	if args.num_gpus < 0:
		raise ValueError(
			'Invalid GPU count: \"--num-gpus\" must be 0 or a positive integer.')

	args.num_batches_per_epoch = int(args.num_training_samples / args.train_batch_size)
	args.job_dir += strftime("_%Y-%m-%d_%H-%M-%S")
	if (args.num_layers - 2) % 6 != 0:
		raise ValueError('Invalid --num-layers parameter.')


	main(**vars(args))