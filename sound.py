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

ipd.Audio('audio_train/00ad7068.wav')

data, sampling_rate = librosa.load('audio_train/00ad7068.wav')
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data, sr=sampling_rate)

import time
train = pd.read_csv('train.csv')
data_dir = os.getcwd()
# def load_wave():
#     i = random.choice(train.index)
#
#     audio_name = train.fname[i]
#     path = os.path.join(data_dir, 'audio_train', str(audio_name))
#
#     print('label: ', train.label[i])
#     x, sr = librosa.load(path)
#
#
#     plt.figure(figsize=(12, 4))
#     librosa.display.waveplot(x, sr=sr)
#     plt.xlabel(train.label[i])
#     plt.show()
# #
# for i in range(10):
#     load_wave()
# print (train.head())
# print (train.index)
# print (data_dir)
# print (train.label.value_counts())

def parser(row):
   # function to load files and extract features
   file_name = os.path.join(os.path.abspath(data_dir), 'audio_train', str(row.fname))

   # handle exception to check if there isn't a file which is corrupted
      # here kaiser_fast is a technique used for faster extraction
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
   return [feature, label]

temp = train.apply(parser, axis=1)
temp.columns = ['feature', 'label']
print (temp.head(2))
