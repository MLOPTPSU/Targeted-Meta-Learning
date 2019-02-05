"""
Create Imbalance Dataset for two classes

__author__ = "MM. Kamani"
"""


import numpy as np
import tensorflow as tf
import argparse
import os
import sys
from tqdm import tqdm


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))




class ImbalancedDataset():
  def __init__(self, train_size=5000, validation_size=5, class_labels=[2,3], ratio=0.9, dataset='mnist'):
    self.train_size = train_size
    self.validation_size = validation_size
    self.class_labels = class_labels
    self.dataset = dataset
    self.ratio = ratio
    self.train={}
    self.validation={}
    self.test={}

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
    for i, c in enumerate(self.class_labels):
      n_train_samples = int(self.train_size * self.ratio) if i==0 else int(self.train_size * (1-self.ratio))
      n_validation_samples = int(self.validation_size / len(self.class_labels))
      train_ind = np.squeeze(np.argwhere(y_train == c))
      assert (len(train_ind) >= self.ratio*(self.train_size + self.validation_size)), "Number of samples requested is greater than number of samples provided"
      test_ind = np.argwhere(y_test == c)
      if i == 0:
        self.train['data'] = x_train[train_ind[:n_train_samples],:]
        self.train['label'] = np.zeros(n_train_samples)
        self.validation['data'] =  x_train[train_ind[n_train_samples:n_train_samples+n_validation_samples],:]
        self.validation['label'] = np.zeros(n_validation_samples)
        self.test['data'] = x_train[test_ind,:]
        self.test['label'] = np.zeros(len(test_ind))
      else:
        self.train['data'] = np.concatenate((self.train['data'],x_train[train_ind[:n_train_samples],:]),axis=0)
        self.train['label'] = np.concatenate((self.train['label'], np.ones(n_train_samples)))
        self.validation['data'] =  np.concatenate((self.validation['data'],
                                  x_train[train_ind[n_train_samples:n_train_samples+n_validation_samples],:]), axis=0)
        self.validation['label'] = np.concatenate((self.validation['label'], np.ones(n_validation_samples)))
        self.test['data'] = np.concatenate((self.test['data'], x_train[test_ind,:]))
        self.test['label'] = np.concatenate((self.test['label'], np.ones(len(test_ind))))
    
    self.train = self._make_shuffle(self.train)
    self.validation = self._make_shuffle(self.validation)
    self.test = self._make_shuffle(self.test)


  def _make_shuffle(self, data_dict):
    perm_inds = np.random.permutation(len(data_dict['label']))
    data_dict['data'] = data_dict['data'][perm_inds,:]
    data_dict['label'] = data_dict['label'][perm_inds]
    return data_dict

  def _convert_example(self, output_file, data_dict):
    """Converts a data_dict to TFRecords."""
    print('Generating %s' % output_file)
    with tf.python_io.TFRecordWriter(output_file) as record_writer:
      data = data_dict['data']
      labels = data_dict['label']

      num_entries_in_batch = len(labels)
      for i in tqdm(range(num_entries_in_batch)):
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'data': _bytes_feature(data[i].tobytes()),
                'label': _float_feature(labels[i])
            }))
        record_writer.write(example.SerializeToString())
  
  def convert_to_tfrecords(self, data_dir='./data/mnist_23_09'):
    os.makedirs(data_dir, exist_ok=True)
    modes = ['train', 'validation', 'test']
    dataset = [self.train, self.validation, self.test]
    for mode, data_dict in zip(modes, dataset):
      output_file = os.path.join(data_dir, mode + '.tfrecords')
      try:
        os.remove(output_file)
      except OSError:
        pass
      # Convert to tf.train.Example and write the to TFRecords.
      self._convert_example(output_file, data_dict)
    print('Done!')
