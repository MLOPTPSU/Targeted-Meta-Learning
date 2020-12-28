"""
Load datasets from TFRecords

__author__ = "MM. Kamani"
"""

import os
import glob
import numpy as np
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt


class ImbalancedDataset():

  def __init__(self,
              data_dir,
              subset='train',
              dataset='mnist',
              validation_batch_size=10,
              use_distortion=True):

    self.data_dir = data_dir
    self.subset = subset
    self.dataset = dataset
    self.validation_batch_size = validation_batch_size
    self.use_distortion = use_distortion
    if self.dataset == 'mnist':
      self.WIDTH = 28
      self.HEIGHT = 28
      self.DEPTH = 1
    elif self.dataset in ['cifar10','cifar100']:
      self.WIDTH = 32
      self.HEIGHT = 32
      self.DEPTH = 3

  def get_filenames(self, subset):
    if subset in ['train', 'validation', 'test']:
      return [os.path.join(self.data_dir, subset + '.tfrecords')]
    else:
      raise ValueError('Invalid data subset "%s"' % subset)

  def parser(self, serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'data': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        })
    image = tf.io.decode_raw(features['data'], tf.uint8)
    # image = tf.cast(image, tf.float32) / 128.0 - 1
    image.set_shape([self.HEIGHT * self.WIDTH * self.DEPTH])
  
    # Reshape from [depth * height * width] to [depth, height, width].
    # image = tf.cast(tf.reshape(image, [self.HEIGHT, self.WIDTH, self.DEPTH]),tf.float32)
    image = tf.cast(
          tf.transpose(tf.reshape(image, [self.DEPTH, self.HEIGHT, self.WIDTH]), [1, 2, 0]),
          tf.float32)
    image = self.preprocess(image)
    # label = tf.cast(tf.one_hot(features['label'],2), tf.float32)
    label = tf.cast(features['label'], tf.int32)
    return image, label

  def make_batch(self, batch_size):
    """Read the images and labels from 'filenames'."""
    if self.subset == 'train':
      image_batch_train, label_batch_train = self._create_tfiterator(batch_size, self.subset)
      image_batch_validation, label_batch_validation = self._create_tfiterator(
                                                                self.validation_batch_size,
                                                                subset='validation'
                                                                )
      image_batch = [image_batch_train, image_batch_validation]
      label_batch = [label_batch_train, label_batch_validation]
    elif self.subset =='test':
      image_batch_test, label_batch_test = self._create_tfiterator(batch_size, self.subset)
      image_batch = [image_batch_test]
      label_batch = [label_batch_test]

    return image_batch, label_batch

  def _create_tfiterator(self, batch_size, subset):
    filenames = self.get_filenames(subset=subset)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.repeat()
    # Parse records.
    dataset= dataset.map(
      self.parser, num_parallel_calls=batch_size)

    # Ensure that the capacity is sufficiently large to provide good random
    # shuffling.
    dataset = dataset.shuffle(buffer_size = 3 * batch_size)

    # Batch it up.
    dataset= dataset.batch(batch_size)
    # iterator = dataset.make_one_shot_iterator()
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    image_batch, label_batch = iterator.get_next()
    return image_batch, label_batch

  def preprocess(self, image):
    """Preprocess a single image in [height, width, depth] layout."""
    if self.subset == 'train' and self.use_distortion and self.dataset in ['cifar10','cifar100']:
      # Pad 4 pixels on each dimension of feature map, done in mini-batch
      image = tf.image.resize_with_crop_or_pad(image, 40, 40)
      image = tf.image.random_crop(image, [self.HEIGHT, self.WIDTH, self.DEPTH])
      image = tf.image.random_flip_left_right(image)

    # elif self.dataset =='mnist':
    #   image = tf.image.resize_image_with_crop_or_pad(image, 32, 32)
    return image

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data-dir',
      type=str,
      default='./data/mnist_23_09',
      help='Directory to download and extract dataset to.')
  parser.add_argument(
      '--dataset',
      type=str,
      default='mnist',
      choices=['cifar10','mnist','adult'],
      help='The dataset to load')
  args = parser.parse_args()
  d = ImbalancedDataset(data_dir=args.data_dir, subset='train', use_distortion='False', dataset=args.dataset)
  feature, label = d.make_batch(1)
  plt.imshow(feature[0][0,:,:,:])
  plt.text(1, 1, "The new label is: {}".format(label[0][0]), color='white', bbox=dict(fill=False, edgecolor='red', linewidth=2))
  plt.show()