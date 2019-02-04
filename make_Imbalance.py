#Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===================================================================

"""
Imbalance CIFAR10 Dataset for two classes

__author__ = "Sadegh Farhang"
"""


import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import argparse
import os
import sys


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()



class make_Imbalance():
    def __init__(self, clas1, clas2, level, x_train, y_train, x_test, y_test):
        self.clas1 = clas1
        self.clas2 = clas2
        self.level = level

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def get_index(self):
        clas1_location_train = np.where(self.y_train == self.clas1)
        clas1_location_test = np.where(self.y_test == self.clas1)

        clas2_location_train = np.where(self.y_train == self.clas2)
        clas2_location_test = np.where(self.y_test == self.clas2)

        ratio = (self.level) / (1 - self.level)

        No_clas2_train = len(clas1_location_train[0]) / ratio
        No_clas2_test = len(clas1_location_test[0]) / ratio

        Train_clas2_location = np.sort(np.random.choice(clas2_location_train[0], size=int(No_clas2_train),
                                                        replace=False))
        # Test_clas2_locattion = np.sort( np.random.choice( clas2_location_test[0], size=int(No_clas2_test),
        #                                                replace=False))

        # a1_tr = clas1_location_train[0].reshape( len( clas1_location_train[0] ), 1 )
        a1_tr = clas1_location_train[0]
        # a1_te = clas1_location_test[0].reshape( len( clas1_location_test[0] ), 1 )
        a1_te = clas1_location_test[0]

        # a2_tr = Train_clas2_location.reshape( len( Train_clas2_location ), 1 )
        a2_tr = Train_clas2_location
        # a2_te = Test_clas2_locattion.reshape( len (Test_clas2_locattion), 1 )
        # a2_te = Test_clas2_locattion
        a2_te = clas2_location_test[0]

        index_tr = np.concatenate((a1_tr, a2_tr))
        index_te = np.concatenate((a1_te, a2_te))

        return a1_tr, a1_te, a2_tr, a2_te, index_tr, index_te

    def create_feat_label(self):
        A = self.get_index()

        features_tr = self.x_train[A[4]]
        labels_tr = self.y_train[A[4]]

        features_te = self.x_test[A[5]]
        labels_te = self.y_train[A[5]]

        assert features_tr.shape[0] == labels_tr.shape[0]
        assert features_te.shape[0] == labels_te.shape[0]

        dataset_train = tf.data.Dataset.from_tensor_slices((features_tr, labels_tr))
        dataset_test = tf.data.Dataset.from_tensor_slices((features_te, labels_te))

        return dataset_train, dataset_test

