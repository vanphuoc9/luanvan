import IPython.display as ipd
from scipy.io import wavfile
import os
import pandas as pd
import librosa
import glob
import librosa.display
import matplotlib.pyplot as plt
import random
import numpy as np
import time
train = pd.read_csv('traintest.csv')
data_dir = os.getcwd()
def parser(row):
   # function to load files and extract features
   file_name = os.path.join(os.path.abspath(data_dir), 'audio_train', str(row.fname))
   # handle exception to check if there isn't a file which is corrupted
   try:
       X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
         # we extract mfcc feature from data
       mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
       # print(mfcc)
   except Exception as e:
      print("Error encountered while parsing file: ", file)
      return None, None

   feature = mfccs
   label = row.label
   print (file_name)
   return pd.Series([feature, label])

temp = train.apply(parser, axis=1)
temp.columns = ['feature', 'label']
# print(temp.head(2))
# print(temp.feature.tolist())

#Convert the data to pass it in our deep learning model
from sklearn.preprocessing import LabelEncoder
# # # from keras.utils import np_utils
X = np.array(temp.feature.tolist())
y = np.array(temp.label.tolist())
# #
print (X)

lb = LabelEncoder()
print(lb.fit_transform(y)[:5])
# y = np_utils.to_categorical(lb.fit_transform(y))
