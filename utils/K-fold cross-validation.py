from sklearn.model_selection import KFold, StratifiedKFold,RepeatedKFold
from tensorflow.keras import backend as K 
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
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
from  Model import model_architecture

labeling = {"AN":0,"DI":1,"FE":2,"HA":3,"NE":4,"SA":5,"SU":6}
names = {"AN":'ANGRY',"DI":'DISGUST',"FE":'FEAR',"HA":'HAPPY',"NE":'NEUTRAL',"SA":'SAD',"SU":'SURPRISE'}
classes = ['ANGRY','DISGUST','FEAR','HAPPY','NEUTRAL','SAD','SURPRISE']
num_classes = len(classes)
num_classes
data_ =glob.glob("/content/drive/MyDrive/Research/3_JAFFE_database/*")

img_data_list=[]
label = []
i=0
for f in data_:
  if(f[-4:]=="tiff"):
    img= Image.open(f)
    img =img.resize((224,224))
    np_img = np.array(img)
    np_img = np.stack((np_img,np_img, np_img), axis=2)
    img_data_list.append(np_img)

    lab = f[52:54]
    label.append(labeling[lab])
    i+=1


labels = np.array(label)
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = img_data/255
print(img_data.shape,labels.shape)


Y = to_categorical(labels, num_classes)

x,y = shuffle(img_data,Y, random_state=6)


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0,stratify = y )
datagen_train = ImageDataGenerator(horizontal_flip=True)
train_generator= datagen_train.fit(X_train)
                                    
                                                   
datagen_validation = ImageDataGenerator(horizontal_flip=True)
validation_generator=datagen_validation.fit(X_test)

epochs = 50
batch_size = 16
steps_per_epoch = len(X_train)//batch_size
validation_steps = len(X_test)//batch_size


METRICS = [
     
    tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
]

reduce = tf.keras.callbacks.ReduceLROnPlateau( monitor="loss",
                                                factor=0.6,
                                                patience=10,
                                                verbose=1,
                                                mode="auto",
                                                min_lr=0.00008)


acc_per_fold = []
loss_per_fold = []
kfold = KFold(n_splits= 10)

# K-fold Cross Validation model evaluation
fold_no = 1

for train, test in kfold.split(x):
    
    
    steps_per_epoch = len(x[train])//batch_size
    validation_steps = len(x[test])//batch_size
    # Define the model
    with tf.device('/device:GPU:0'): 
      model1 = model_architecture(METRICS,size = [112,112,3],learning_rate=0.0002)
      # for layer in model1.layers:
      #   layer.kernel_regularizer = keras.regularizers.l2(0.001) 



      # Generate a print
      print('------------------------------------------------------------------------')
      print(f'Training for fold {fold_no} ...')

      

 
      history = model1.fit(
          x[train],y[train],        
          validation_data=(x[test], y[test]),
          epochs= epochs  ,
          steps_per_epoch=steps_per_epoch,
          validation_steps= validation_steps,
          batch_size= batch_size,
          workers=2,
          callbacks=[reduce],)
      
      model1.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
      del model1  # deletes the existing model 
      model1 = load_model('my_model.h5')

      # Generate generalization metrics
      scores = model1.evaluate(x[test], y[test], batch_size= batch_size,verbose=1,workers=2)
      scores1 = model1.evaluate(x[train], y[train], batch_size= batch_size,verbose=1,workers=2)
      print(f'Score for fold {fold_no}: {model1.metrics_names[0]} of {scores[0]}; {model1.metrics_names[1]} of {scores[1]*100}%')
      acc_per_fold.append(scores[1] * 100)
      loss_per_fold.append(scores[0])
 
      # Increase fold number
      fold_no = fold_no + 1
      