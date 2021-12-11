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


json_file = open("/content/drive/MyDrive/Research/model/model_deep-jaffe.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model_main = model_from_json(loaded_model_json)
model_main.load_weights("/content/drive/MyDrive/Research/model/model_deep_weights-jaffe.h5")

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


epochs = 50
batch_size = 16
steps_per_epoch = len(X_train)//batch_size
validation_steps = len(X_test)//batch_size


out_layer=model_main.layers[-1].output
output_fn= K.function([model_main.input],out_layer)
o=output_fn([X_train])#1st argument of the .fit() function
print(o.shape)


scaler=StandardScaler()
scaler.fit(o)
o=scaler.transform(o)
print(o.shape)

pca=PCA(n_components=3)

pca.fit(o)

x_pca=pca.transform(o)
print(x_pca.shape)

X=x_pca[:,0]
Y=x_pca[:,1]
Z=x_pca[:,2]


y_arg = np.argmax(y_train,axis=1)
y_arg

fig= plt.figure(figsize=(12,12))
ax=plt.axes(projection='3d')
#ax.grid(color='white',linestyle='-.',linewidth=0.7)
plt.rcParams['axes.facecolor'] = 'white'

ax.scatter(X[y_arg==0],Y[y_arg==0],Z[y_arg==0],marker='*',color='green',s=100)
ax.scatter(X[y_arg==1],Y[y_arg==1],Z[y_arg==1],marker='P',color='blue',s=100)
ax.scatter(X[y_arg==2],Y[y_arg==2],Z[y_arg==2],marker='^',color='darkorange',s=100)
ax.scatter(X[y_arg==3],Y[y_arg==3],Z[y_arg==3],marker='o',color='red',s=100)
ax.scatter(X[y_arg==4],Y[y_arg==4],Z[y_arg==4],marker='o',color='cyan',s=100)
ax.scatter(X[y_arg==5],Y[y_arg==5],Z[y_arg==5],marker='o',color='yellow',s=100)
ax.scatter(X[y_arg==6],Y[y_arg==6],Z[y_arg==6],marker='o',color='pink',s=100)

ax.set_xlabel('x axis',fontsize=20)
ax.set_ylabel('y axis',fontsize=20)
ax.set_zlabel('z axis',fontsize=20)

a = mlines.Line2D([], [], linestyle='None', markersize=11, label='ANGRY', marker='*',color='green')
b = mlines.Line2D([], [], linestyle='None', markersize=10, label='DISGUST',   marker='P',color='blue')
c = mlines.Line2D([], [], linestyle='None', markersize=10, label='FEAR',  marker='^',color='darkorange')
d = mlines.Line2D([], [], linestyle='None', markersize=10, label='HAPPY', marker='o',color='red')
e = mlines.Line2D([], [], linestyle='None', markersize=10, label='NEUTRAL', marker='o',color='cyan')
f = mlines.Line2D([], [], linestyle='None', markersize=10, label='SAD', marker='o',color='yellow')
g = mlines.Line2D([], [], linestyle='None', markersize=10, label='SUPRISE', marker='o',color='pink')

plt.legend(handles=[a,b,c,d,e,f],fontsize=20, loc ="upper center", edgecolor=(0, 0, 0, 1.), facecolor=(1, 1, 1, 0.1), handletextpad=0,ncol=4)#,bbox_to_anchor=(0.5, 0.5, 0.45, 0.42))

plt.rcParams["axes.labelsize"] = 22
fig.text(0.5, 0, "Output Layer", ha='center',fontsize=18)

plt.savefig('Cluster.pdf',bbox_inches ="tight", pad_inches = 1, orientation ='landscape',dpi=1200,transparent=True)

plt.show()