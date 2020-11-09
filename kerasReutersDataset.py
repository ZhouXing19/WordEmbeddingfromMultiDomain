# import warnings
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from keras.datasets import reuters
# warnings.filterwarnings('ignore')
#
# dataset = tf.keras.datasets.reuters.load_data(
#     path="reuters.npz",
#     num_words=None)

import tensorflow as tf
from keras import models, regularizers, layers, optimizers, losses, metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils, to_categorical

from keras.datasets import reuters