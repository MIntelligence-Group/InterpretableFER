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


from tensorflow.keras.models import model_from_json

from DncShap import DnCShap
from Plots import plot_shap,display_gradcam
from Grad_cam import make_gradcam_heatmap


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


width= 224
height= 224
times = 9

# DncShap
for i in range(num_classes):
  count = 0
  for j in range(len(img_data)):
    data = img_data[j]
    label = labels[j]
    pred = np.argmax(model_main.predict(data.reshape(1,224,224,3)))
    if(label==pred and label==i):
      pred = classes[pred]
      true = classes[label] 
      shap_value = DnCShap(model_main,data,width,height,times)
      plot_shap(model_main,img_data,labels,classes,shap_value, j, width=224,height=224,true_only =True, percentile=70,to_save=True,fname ="/content/drive/MyDrive/Research/Result/ICVGIP 2021/trial/"+classes[label]+str(count))
      count+=1
    if(count==3):
      break


#Lime

explainer = lime_image.LimeImageExplainer()

for i in range(num_classes):
  for j in range(len(img_data)):
    data = img_data[j]
    label1 = labels[j]
    pred = np.argmax(model_main.predict(data.reshape(1,224,224,3)))
    if(label1==pred and label1==i):
      explanation = explainer.explain_instance(data, model_main.predict, top_labels=1, num_samples=2000)
      temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False)
      img = mark_boundaries(temp , mask)
      fig, ax = plt.subplots(1)
      ax.imshow(img)
      ax.set_title(classes[label1])
      ax.axis('off')
      fig.savefig("/content/drive/MyDrive/Research/Result/JAFEE/Lime/"+classes[label1],format="pdf",dpi=600)
      break
#Gradient shap

explainer = shap.GradientExplainer(model_main, img_data,local_smoothing=1)
for i in range(num_classes):
  for j in range(len(img_data)):
    data = img_data[j]
    label1 = labels[j]
    pred = np.argmax(model_main.predict(data.reshape(1,224,224,3)))
    if(label1==pred and label1==i):
      shap_values = explainer.shap_values(data.reshape(1,224,224,3))
      s = shap_values[pred][0].sum(axis=2)
      s#hap_value =np.uint8(255 * s)
      plot_shap(s, j,  percentile=0,to_save=True,fname ="/content/drive/MyDrive/Research/Result/JAFEE/Shap_grad/"+classes[label1])
      print(i)
      break

#Grad-cam
model = model_main
model.layers[-1].activation = None

for i in range(num_classes):
  for j in range(len(img_data)):
    data = img_data[j]
    label1 = labels[j]
    pred = np.argmax(model_main.predict(data.reshape(1,224,224,3)))
    if(label1==pred and label1==i):
      heatmap = make_gradcam_heatmap(data.reshape(1,224,224,3), model, model.layers[-5].name)
      display_gradcam(data, heatmap,to_save=True,fname ="/content/drive/MyDrive/Research/Result/JAFEE/Gradcam/"+classes[label1])
      break