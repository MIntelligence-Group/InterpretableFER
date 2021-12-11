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




def RecShap(model,data,width,height,x_s,x_e,y_s,y_e,pred_b,pred_f,score,arg_max,times,i,value):

  if(times==0):
    value[x_s:x_e,y_s:y_e] = score
    #print(times)
    return

  else:

    if(i==1):
      
      pred_b = pred_b
      arg_max = arg_max
      times = times

      data_1 = np.zeros([width,height,3])
      data_2 = np.zeros([width,height,3])

      y_m = (y_s + y_e)//2

      data_1[:,:y_m,:] = data[:,:y_m,:]
      data_2[:,y_m:,:] = data[:,y_m:,:]

      data_1 = data_1.reshape(1,width,height,3)
      data_2 = data_2.reshape(1,width,height,3)

      pred_1 = model.predict(data_1)[0][arg_max]
      pred_2 = model.predict(data_2)[0][arg_max]


      score_1 = (((pred_1-pred_b) + (pred_f-pred_2))/2)/2
      score_2 =  (((pred_2-pred_b) + (pred_f-pred_2))/2)/2

      times-=1
      RecShap(model,data,width,height,x_s,x_e,y_s,y_m,pred_b+pred_2,pred_1+pred_f,score_1,arg_max,times,0,value)
      RecShap(model,data,width,height,x_s,x_e,y_m,y_e,pred_b+pred_1,pred_2+pred_f,score_2,arg_max,times,0,value)

    if(i==0):
    
      pred_b = pred_b
      arg_max = arg_max
      times = times

      data_1 = np.zeros([width,height,3])
      data_2 = np.zeros([width,height,3])

      x_m = (x_s + x_e)//2

      data_1[:x_m,:,:] = data[:x_m,:,:]
      data_2[x_m:,:,:] = data[x_m:,:,:]

      data_1 = data_1.reshape(1,width,height,3)
      data_2 = data_2.reshape(1,width,height,3)

      pred_1 = model.predict(data_1)[0][arg_max]
      pred_2 = model.predict(data_2)[0][arg_max]


      score_1 = (((pred_1-pred_b) + (pred_f-pred_2))/2)/2
      score_2 =  (((pred_2-pred_b) + (pred_f-pred_2))/2)/2

      times-=1

      RecShap(model,data,width,height,x_s,x_m,y_s,y_e,pred_b+pred_2,pred_1+pred_f,score_1,arg_max,times,1,value)
      RecShap(model,data,width,height,x_m,x_e,y_s,y_e,pred_b+pred_1,pred_2+pred_f,score_2,arg_max,times,1,value)




def DnCShap(model,data,width,height,times):
   
  data_b = np.zeros([width,height,3])
  data_f = data
  data_1 = np.zeros([width,height,3])
  data_2 = np.zeros([width,height,3])
  
  x_m = width//2
  y_m = height//2
   
  data_1[0:x_m,:,:] = data[0:x_m,:,:]
  data_2[x_m:,:,:] = data[x_m:,:,:]

  data_f = data_f.reshape(1,width,height,3)
  data_1 = data_1.reshape(1,width,height,3)
  data_2 = data_2.reshape(1,width,height,3)
  data_b = data_b.reshape(1,width,height,3)


 
  pred = model.predict(data_f)
  arg_max = np.argmax(pred)
  pred_f = pred[0][arg_max]
  pred_b = model.predict(data_b)[0][arg_max]
  pred_1 = model.predict(data_1)[0][arg_max]
  pred_2 = model.predict(data_2)[0][arg_max]

  score_1 = ((pred_1-pred_b) + (pred_f-pred_2))/2
  score_2 = ((pred_2-pred_b) + (pred_f-pred_1))/2

  shap_value = np.zeros([width,height])


  times-=1
  RecShap(model,data,width,height,0,x_m,0,height,(pred_b+pred_2),(pred_1+pred_f),score_1,arg_max,times,1,shap_value)
  RecShap(model,data,width,height,x_m,width,0,height,(pred_b+pred_1),(pred_2+pred_f),score_2,arg_max,times,1,shap_value)

  return(shap_value)



