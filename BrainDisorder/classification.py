# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 09:31:56 2022

@author: user
"""

import os

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt 
import tensorflow as tf
from skimage.io import imread
from skimage.transform import resize
#import pandas as pd
from pathlib import Path
import json
from collections import OrderedDict
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        # print(e)
        pass

#### 
#asdasdasdasd

    
img_lst =[]
def feature_extraction(path): # 고정 parameters 
    #print(path)
    min_level_db= -100
    y = librosa.load(path,sr=16000)[0]
    S = librosa.feature.melspectrogram(y=y, n_mels=64, n_fft=320, hop_length=160) # 20ms/10ms
    norm_log_S = np.clip((librosa.power_to_db(S, ref=np.max)-min_level_db)/-min_level_db,0,1)# normalize_mel(librosa.power_to_db(S, ref=np.max))
    return norm_log_S

def spectrogram_feature(name,data):
    AE_model = tf.keras.models.load_model(r'C:\Workspace\1_Univ\CAU\3-1\Capstone\Server\BrainDisorder\pretrained_model\ae_model.h5') #feature 추출 Auto-Encoder 모델 load    
    librosa.display.specshow(data)
    plt.tight_layout()
    plt.savefig('./spectrogram/'+name+'.jpg')  # spectrogram 저장
    img = imread('./spectrogram/'+name+'.jpg') # spectrogram read
    img_lst.append(resize(img, (28, 28, 3))) # resize 후 분류에 사용 될 array  
    testdata = np.array(img_lst)      
    encoded_data = AE_model.predict(testdata) # 512개의 feature 추출
   # df = pd.DataFrame(encoded_data)
    #df.to_csv('./feature_data/'+name+'.csv', index=False)
    
    #classification(name,encoded_data)
    
    return encoded_data
    
def classification(name,encoded_data):  
    mlp_model = tf.keras.models.load_model(r'C:\Workspace\1_Univ\CAU\3-1\Capstone\Server\BrainDisorder\pretrained_model\brain_classification.h5') #classifier MLP 모델 load
    results = mlp_model.predict(encoded_data)*100
   # #print("result : ",results[0])
    
    return results

def json_write(file_name,results):
    label = ['Normal','Speech', 'Larynx','Brain' ]
    index_ =len(results)-1
    file_data = OrderedDict()
    #file_data["file path"] = os.path.dirname(args.data)
    file_data["file name"] = file_name
    file_data["classification results"] = {str(label[0]):str(results[index_][0]),str(label[1]):str(results[index_][1])}
    #file_data["분류 결과 음성"] = str(label[np.where(results[0] ==max(results[0]))[0][0]]) 
    with open('./results/'+file_name+'.json', 'w', encoding="utf-8") as make_file:
        json.dump(file_data, make_file, ensure_ascii=False, indent="\t")
    return file_data    
    
def model_classification(file):
    directory = ['./feature_data/','./spectrogram/','./results/']
    try: 
        for x in range(len(directory)):
            if not os.path.exists(directory[x]):
                os.makedirs(directory[x])
    except OSError: 
        return
   
    path = file
    #print(path)
    ##print(path)
    file_name = Path(path).stem
    data = feature_extraction(path)
    encoded_data = spectrogram_feature(file_name,data)  
    results = classification(file_name, encoded_data) 
    json = json_write(file_name,results)
    return results    

   