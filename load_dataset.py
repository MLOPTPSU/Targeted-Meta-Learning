"""
Load datasets from TFRecords

__author__ = "MM. Kamani"
"""

import os
import numpy as np
import tensorflow as tf

class ImbalancedDataset(object):

  def __init__(self,
              data_dir,
              subset='train',
              dataset='mnist',
              use_distortion=True):

    self.data_dir = data_dir
    self.subset = subset
    self.dataset = dataset
    self.use_distortion = use_distortion
    if self.dataset == 'mnist':
      self.WIDTH = 28
      self.HEIGHT = 28
      self.DEPTH = 1
    elif self.dataset == 'cifar10':
      self.WIDTH = 32
      self.HEIGHT = 32
      self.DEPTH = 3

  def get_filenames(self):
    if self.subset in ['train', 'validation', 'test']:
      return [os.path.join(self.data_dir, self.subset + '.tfrecords')]
    else:
      raise ValueError('Invalid data subset "%s"' % self.subset)

  def parser(self, serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    features = tf.parse_single_example(
        serialized_example,
        features={
            'data': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['data'], tf.uint8)
    image.set_shape([self.DEPTH * self.HEIGHT * self.WIDTH])

    
    if self.dataset == 'mnist':
      image = tf.cast(
          tf.reshape(image, [self.HEIGHT, self.WIDTH]),
          tf.float32)
    elif self.dataset == 'cifar10':
      # Reshape from [depth * height * width] to [depth, height, width].
      image = tf.cast(tf.reshape(image, [self.HEIGHT, self.WIDTH, self.DEPTH]),tf.float32)
      image = self.preprocess(image)
    
    label = tf.cast(features['label'], tf.int32)
    return image, label

  def make_batch(self, batch_size):
    """Read the images and labels from 'filenames'."""
    filenames = self.get_filenames()
    dataset = tf.data.TFRecordDataset(filenames)

    dataset = dataset.repeat()

    # Parse records.
    dataset = dataset.map(
      self.parser, num_parallel_calls=batch_size)

    # Ensure that the capacity is sufficiently large to provide good random
    # shuffling.
    dataset = dataset.shuffle(buffer_size = 3 * batch_size)

    # Batch it up.
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()

    return image_batch, label_batch

  def preprocess(self, image):
    """Preprocess a single image in [height, wi dth, depth] layout."""
    if self.subset == 'train' and self.use_distortion:
      # Pad 4 pixels on each dimension of feature map, done in mini-batch
      image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
      image = tf.random_crop(image, [self.HEIGHT, self.WIDTH, self.DEPTH])
      image = tf.image.random_flip_left_right(image)
    return image

