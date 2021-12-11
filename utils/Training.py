from tensorflow.keras.models import model_from_json

import tensorflow as tf
import keras.backend as K

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.lines as mlines
import tensorflow as tf
import numpy as np
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

model1 = model_architecture(METRICS,size=[224,224,3],learning_rate=0.0003)
print(model1.summary())

epochs = 80
batch_size = 16
checkpoint = ModelCheckpoint("/content/drive/MyDrive/Research/model/model_deep_weights-jaffe.h5",monitor='val_accuracy',
                            save_weights_only = True,mode='max',verbose=1)

callbacks = [checkpoint]


with tf.device('/device:GPU:0'):
  history = model1.fit(
      datagen_train.flow(X_train,y_train, batch_size=batch_size),        
      validation_data=datagen_validation.flow(X_test, y_test,batch_size=batch_size),
      epochs=epochs,
    steps_per_epoch=steps_per_epoch,
      validation_steps= validation_steps,
      callbacks=callbacks)

model_json = model1.to_json()
with open("/content/drive/MyDrive/Research/model/model_deep-jaffe.json","w") as json_file:
    json_file.write(model_json)