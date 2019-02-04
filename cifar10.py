"""CIFAR-10 data set.

See http://www.cs.toronto.edu/~kriz/cifar.html.
"""
import os

import tensorflow as tf

HEIGHT = 32
WIDTH = 32
DEPTH = 3


class Cifar10DataSet(object):
  """Cifar10 data set.

  Described by http://www.cs.toronto.edu/~kriz/cifar.html.
  """

  def __init__(self, data_dir, num_shards, subset='train', use_distortion=True, redundancy=0.0):
    self.data_dir = data_dir
    self.num_shards = num_shards
    self.subset = subset
    self.use_distortion = use_distortion
    self.redundancy = redundancy

  def get_filenames(self):
    if self.subset in ['train', 'validation', 'eval']:
      return [os.path.join(self.data_dir, self.subset + '.tfrecords')]
    else:
      raise ValueError('Invalid data subset "%s"' % self.subset)

  def parser(self, serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([DEPTH * HEIGHT * WIDTH])

    # Reshape from [depth * height * width] to [depth, height, width].
    image = tf.cast(
        tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
        tf.float32)
    label = tf.cast(features['label'], tf.int32)

    # Custom preprocessing.
    image = self.preprocess(image)

    return image, label

  def make_batch(self, batch_size):
    feature_shards = [[] for i in range(self.num_shards)]
    label_shards = [[] for i in range(self.num_shards)]
    """Read the images and labels from 'filenames'."""
    filenames = self.get_filenames()
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.repeat()

    # Parse records.
    dataset = dataset.map(
      self.parser, num_parallel_calls=int(batch_size / self.num_shards))

    # Potentially shuffle records.
    min_queue_examples = int(
      Cifar10DataSet.num_examples_per_epoch(self.subset) * 0.4 / self.num_shards)
    # Ensure that the capacity is sufficiently large to provide good random
    # shuffling.
    dataset = dataset.shuffle(buffer_size=min_queue_examples + int(3 * batch_size / self.num_shards))

    # Batch it up.
    dataset = dataset.batch(batch_size )
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()

    return image_batch, label_batch

  def preprocess(self, image):
    """Preprocess a single image in [height, wi dth, depth] layout."""
    if self.subset == 'train' and self.use_distortion:
      # Pad 4 pixels on each dimension of feature map, done in mini-batch
      image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
      image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])
      image = tf.image.random_flip_left_right(image)
    return image

  @staticmethod
  def num_examples_per_epoch(subset='train'):
    if subset == 'train':
      return 45000
    elif subset == 'validation':
      return 5000
    elif subset == 'eval':
      return 10000
    else:
      raise ValueError('Invalid data subset "%s"' % subset)
