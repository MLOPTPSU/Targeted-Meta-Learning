"""
Create Imbalance Dataset for two classes or a long-tailed dataset for many classes

__author__ = "MM. Kamani"
"""


import numpy as np
import tensorflow as tf
import argparse
import os
import sys
from tqdm import tqdm


def _int_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))




class ImbalancedDataset():
  def __init__(self, 
               train_size=5000, 
               test_size=2000,
               validation_size=10, 
               underrepresented_class_labels=[2],
               normal_class_labels=[3],
               ratio=0.9, 
               dataset='mnist'):
    self.train_size = train_size
    self.test_size = test_size
    self.validation_size = validation_size
    self.underrepresented_class_labels = underrepresented_class_labels
    self.normal_class_labels = normal_class_labels
    if any([i in self.normal_class_labels for i in self.underrepresented_class_labels]):
      raise ValueError('The underrepresented class labels and normal class labels have overlap!')
    self.class_labels = self.underrepresented_class_labels + self.normal_class_labels
    self.dataset = dataset
    self.ratio = ratio
    self.data = {}
    self.subsets = ['train', 'validation', 'test']
    self._make_imbalance()
      

  def _get_data(self):
    if self.dataset == 'mnist':
      (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif self.dataset == 'cifar10':
      (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    else:
      raise ValueError('Dataset {} is not defined'.format(self.dataset))
    return (x_train, np.squeeze(y_train)), (x_test, np.squeeze(y_test))

  def _make_imbalance(self):
    (x_train, y_train), (x_test, y_test) = self._get_data()
    data_shape = x_train.shape[1:]
    self._initialize_matrices(data_shape)
    n_train_samples_majority = int(self.train_size / (len(self.normal_class_labels) 
                                      + len(self.underrepresented_class_labels) * (1-self.ratio)))
    n_train_samples_minority = int(n_train_samples_majority * (1-self.ratio))
    for i, c in enumerate(self.class_labels):
      if c in self.underrepresented_class_labels:
        n_train_samples = n_train_samples_minority
      else:
        n_train_samples = n_train_samples_majority
      
      n_validation_samples = int(self.validation_size / len(self.class_labels))
      n_test_samples = int(self.test_size / len(self.class_labels))
      train_ind = np.squeeze(np.argwhere(y_train == c))
      test_ind = np.squeeze(np.argwhere(y_test == c))
      assert (len(train_ind) >= (n_train_samples + n_validation_samples)), "Number of samples requested is greater than number of samples provided"
      assert (len(test_ind) >= n_test_samples), "Number of test samples requested is greater than number of samples provided"
      
      self.data['train']['data'] = np.vstack((self.data['train']['data'],
                                            x_train[train_ind[:n_train_samples],:]))
      self.data['train']['label'] = np.append(self.data['train']['label'],
                                                i * np.ones(n_train_samples))
      self.data['validation']['data'] = np.vstack((self.data['validation']['data'],
                                                  x_train[train_ind[n_train_samples:n_train_samples+n_validation_samples],:]))
      self.data['validation']['label'] = np.append(self.data['validation']['label'],
                                                  i * np.ones(n_validation_samples))
      self.data['test']['data'] = np.vstack((self.data['test']['data'],
                                            x_test[test_ind[:n_test_samples],:]))
      self.data['test']['label'] = np.append(self.data['test']['label'],
                                                  i * np.ones(n_test_samples))
    self.data['train'] = self._make_shuffle(self.data['train'])
    self.data['validation'] = self._make_shuffle(self.data['validation'])
    self.data['test'] = self._make_shuffle(self.data['test'])


  def _make_shuffle(self, data_dict):
    perm_inds = np.random.permutation(len(data_dict['label']))
    data_dict['data'] = data_dict['data'][perm_inds,:]
    data_dict['label'] = data_dict['label'][perm_inds]
    return data_dict

  def _initialize_matrices(self,data_shape):
    for subset in self.subsets:
      self.data[subset] = {}
      self.data[subset]['data'] = np.array([]).reshape([0] + list(data_shape))
      self.data[subset]['label'] = np.array([]).reshape(0)

  def _convert_example(self, output_file, data_dict):
    """Converts a data_dict to TFRecords."""
    print('Generating %s' % output_file)
    with tf.compat.v1.python_io.TFRecordWriter(output_file) as record_writer:
      data = data_dict['data'].astype(np.int8)
      labels = data_dict['label'].astype(np.int64)
      num_entries_in_batch = len(labels)
      for i in tqdm(range(num_entries_in_batch)):
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'data': _bytes_feature(data[i].tobytes()),
                'label': _int_feature(labels[i]),
            }))
        record_writer.write(example.SerializeToString())
  
  def convert_to_tfrecords(self, data_dir='./data/mnist_23_09'):
    os.makedirs(data_dir, exist_ok=True)

    for subset in self.subsets:
      output_file = os.path.join(data_dir, subset + '.tfrecords')
      try:
        os.remove(output_file)
      except OSError:
        pass
      # Convert to tf.train.Example and write the to TFRecords.
      self._convert_example(output_file, self.data[subset])
    print('Done!')



class LongTailDataset():
  def __init__(self,
               n0=5000,
               validation_size=10,
               imf=1,
               dataset='cifar10'):
    self.n0 = n0
    self.validation_size = validation_size
    self.dataset = dataset
    if self.dataset in ['mnist','cifar10']:
      self.num_class=10
    elif self.dataset == 'cifar100':
      self.num_class=100
    self.imf= imf
    self.mu = (1 / self.imf) ** (1/(self.num_class-1))
    self.data = {}
    self.subsets = ['train', 'validation', 'test']
    self._make_long_tail()
      

  def _get_data(self):
    if self.dataset == 'mnist':
      (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif self.dataset == 'cifar10':
      (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif self.dataset == 'cifar100':
      (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    else:
      raise ValueError('Dataset {} is not defined'.format(self.dataset))
    return (x_train, np.squeeze(y_train)), (x_test, np.squeeze(y_test))

  def _make_long_tail(self):
    (x_train, y_train), (x_test, y_test) = self._get_data()
    data_shape = x_train.shape[1:]
    self._initialize_matrices(data_shape)
    n_validation_samples = int(self.validation_size / self.num_class)
    for i in range(self.num_class):
      n_train_samples = int(self.n0 * self.mu ** i)
      train_ind = np.squeeze(np.argwhere(y_train == i))
      rand_ind = np.random.permutation(len(train_ind))
      train_ind = train_ind[rand_ind]
      test_ind = np.squeeze(np.argwhere(y_test == i))
      self.data['train']['data'] = np.vstack((self.data['train']['data'],
                                            x_train[train_ind[:n_train_samples],:]))
      self.data['train']['label'] = np.append(self.data['train']['label'],
                                                i * np.ones(n_train_samples))
      self.data['validation']['data'] = np.vstack((self.data['validation']['data'],
                                                  x_train[train_ind[-n_validation_samples:],:]))
      self.data['validation']['label'] = np.append(self.data['validation']['label'],
                                                  i * np.ones(n_validation_samples))
      self.data['test']['data'] = np.vstack((self.data['test']['data'],
                                            x_test[test_ind,:]))
      self.data['test']['label'] = np.append(self.data['test']['label'],
                                                  i * np.ones(len(test_ind)))
    self.data['train'] = self._make_shuffle(self.data['train'])
    self.data['validation'] = self._make_shuffle(self.data['validation'])
    self.data['test'] = self._make_shuffle(self.data['test'])


  def _make_shuffle(self, data_dict):
    perm_inds = np.random.permutation(len(data_dict['label']))
    data_dict['data'] = data_dict['data'][perm_inds,:]
    data_dict['label'] = data_dict['label'][perm_inds]
    return data_dict

  def _initialize_matrices(self,data_shape):
    for subset in self.subsets:
      self.data[subset] = {}
      self.data[subset]['data'] = np.array([]).reshape([0] + list(data_shape))
      self.data[subset]['label'] = np.array([]).reshape(0)

  def _convert_example(self, output_file, data_dict):
    """Converts a data_dict to TFRecords."""
    print('Generating %s' % output_file)
    with tf.compat.v1.python_io.TFRecordWriter(output_file) as record_writer:
      data = data_dict['data'].astype(np.int8)
      labels = data_dict['label'].astype(np.int64)
      num_entries_in_batch = len(labels)
      for i in tqdm(range(num_entries_in_batch)):
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'data': _bytes_feature(data[i].tobytes()),
                'label': _int_feature(labels[i]),
            }))
        record_writer.write(example.SerializeToString())
  
  def convert_to_tfrecords(self, data_dir='./data/mnist_23_09'):
    os.makedirs(data_dir, exist_ok=True)

    for subset in self.subsets:
      output_file = os.path.join(data_dir, subset + '.tfrecords')
      try:
        os.remove(output_file)
      except OSError:
        pass
      # Convert to tf.train.Example and write the to TFRecords.
      self._convert_example(output_file, self.data[subset])
    print('Done!')

if __name__=="__main__":
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
      help='The dataset to transfer to TFRecords')
  parser.add_argument(
      '--train-size',
      type=int,
      default=5000,
      help='Number of training samples.')
  parser.add_argument(
      '--validation-size',
      type=int,
      default=20,
      help='Number of validation samples.')
  parser.add_argument(
      '--test-size',
      type=int,
      default=2000,
      help='Number of test samples.')
  parser.add_argument(
      '--ratio',
      type=float,
      default=0.9,
      help='Ratio of classes.')
  parser.add_argument(
      '--minority-labels',
      type=int,
      nargs='+',
      default=[2], 
      help='Minority Class labels')
  parser.add_argument(
      '--majority-labels',
      type=int,
      nargs='+',
      default=[3],
      help='Majority Class labels')
  args = parser.parse_args()
  dataset = ImbalancedDataset(train_size=args.train_size,
                              test_size=args.test_size,
                              validation_size=args.validation_size,
                              underrepresented_class_labels=args.minority_labels,
                              normal_class_labels=args.majority_labels,
                              ratio=args.ratio,
                              dataset=args.dataset)
  dataset.convert_to_tfrecords(args.data_dir)