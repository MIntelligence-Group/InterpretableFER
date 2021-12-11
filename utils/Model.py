from tensorflow.keras import layers as L
from tensorflow.keras import Sequential
import tensorflow as tf
from tensorflow.keras import initializers

from tensorflow.keras.regularizers import l2,l1
import tensorflow as tf
from tensorflow.keras.models import model_from_json

import tensorflow as tf
import keras.backend as K

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.lines as mlines
import tensorflow as tf
import numpy as np
import datetime
from glob import glob
import matplotlib.pyplot as plt
from tensorflow import keras 
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

from PIL import Image
from IPython.display import display
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import shap
import lime
import glob
import pandas as pd
from sklearn.metrics import confusion_matrix
#import seaborn as sns

import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.cm as cm



def model_architecture(metrics,size = [256,256] + [3],learning_rate=0.0001):

  ## Block 1
  input = Input(shape=size)
  x =L.Conv2D(input_shape=size,filters=64,kernel_size=(3,3),padding="same", activation="relu")(input)
  x = L.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu")(x)
  # x = L.BatchNormalization()(x)
  x = L.MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
  
  ## Resuidal 1
  x_skip1 = x
  

  ## Block 2
  x = L.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(x)
  x = L.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(x)
  # x = L.BatchNormalization()(x)
  x = L.MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
  
  ## Resuidal 2
  x_skip1 = L.MaxPool2D(pool_size=(2,2),strides=(2,2))(x_skip1)
  x_skip2 = x
  

  ## Block 3
  x = L.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(x)
  x = L.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(x)
  x = L.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(x)
  # x = L.BatchNormalization()(x)
  x = L.MaxPool2D(pool_size=(2,2),strides=(2,2))(x)

  ## Resuidal 3
  x_skip1 = L.MaxPool2D(pool_size=(2,2),strides=(2,2))(x_skip1)
  x_skip2 = L.MaxPool2D(pool_size=(2,2),strides=(2,2))(x_skip2)
  x_skip3 = x
  


 ##Block 4
  x = L.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
  x = L.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
  x = L.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
  # x = L.BatchNormalization()(x)
  x = L.MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
   
  ##Resuidal 4
  x_skip1 = L.MaxPool2D(pool_size=(2,2),strides=(2,2))(x_skip1)
  x_skip2 = L.MaxPool2D(pool_size=(2,2),strides=(2,2))(x_skip2)
  x_skip3 = L.MaxPool2D(pool_size=(2,2),strides=(2,2))(x_skip3)
  x_skip4 = x
  
  

  ##Block 5
  x = L.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
  x = L.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
  x = L.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
  # x = L.BatchNormalization()(x)
  x = L.MaxPool2D(pool_size=(2,2),strides=(2,2))(x)

  ## Resuidal 5
  x_skip1 = L.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x_skip1)
  x_skip2 = L.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x_skip2)
  x_skip3 = L.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x_skip3)
  x_skip4 = L.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x_skip4)

  x_skip1 = L.MaxPool2D(pool_size=(2,2),strides=(2,2))(x_skip1)
  x_skip2 = L.MaxPool2D(pool_size=(2,2),strides=(2,2))(x_skip2)
  x_skip3 = L.MaxPool2D(pool_size=(2,2),strides=(2,2))(x_skip3)
  x_skip4 = L.MaxPool2D(pool_size=(2,2),strides=(2,2))(x_skip4)


  ## Combining all resuidal and main 
  x = L.Add()([x, x_skip1,x_skip2,x_skip3,x_skip4])


  ## Flattening
  x = L.Flatten()(x)
  x = L.Dense(units=2048,activation="relu")(x)
  x = L.Dense(units=2048,activation="relu")(x)
  x = L.Dense(units=7,activation="softmax")(x)


  model = Model(inputs=input, outputs=x)


  loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True,  reduction=tf.keras.losses.Reduction.NONE)
  opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)

  model.compile(optimizer=opt,
              loss=loss,
              metrics=tf.keras.metrics.CategoricalAccuracy(name='accuracy'))
  
  return model
  