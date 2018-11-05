#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 18:49:48 2018

@author: adriano
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import urllib
import scipy.io.wavfile
import pydub
import os, glob
from numpy import fft as fft
from sklearn.tree import DecisionTreeClassifier
import librosa
import librosa.display
import keras
from keras.layers import Activation, Dense, Dropout, Conv2D,Flatten, MaxPooling2D
from keras.models import Sequential
import random
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import warnings


#CRIAÇÃO DA DATASET
dataset = []
dataset_teste = []
dataset_val = []

temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/cel/"

os.chdir(temp_folder)

listaDeAudios = []
teste = 91
val = 117
cont = 0
for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(len(listaDeAudios)):
    if cont < teste:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset.append((ps,0))#cel
    elif cont < val:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset_teste.append((ps,0))#cel
    else:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset_val.append((ps,0))#cel
    cont +=1
    
temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/cla/"

os.chdir(temp_folder)

listaDeAudios = []
teste = 73
val = 93
cont = 0

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(len(listaDeAudios)):
    if cont < teste:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset.append((ps,1))#cla
    elif cont < val:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset_teste.append((ps,1))#cla
    else:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset_val.append((ps,1))#cla
    cont +=1
    
temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/flu/"

os.chdir(temp_folder)

listaDeAudios = []
teste = 107
val = 137
cont = 0

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(len(listaDeAudios)):
    if cont < teste:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset.append((ps,2))#flu
    elif cont < val:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset_teste.append((ps,2))#flu
    else:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset_val.append((ps,2))#flu
    cont +=1
    
temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/gac/"

os.chdir(temp_folder)

listaDeAudios = []
teste = 104
val = 134
cont = 0

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(len(listaDeAudios)):
    if cont < teste:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset.append((ps,3))#gac
    elif cont < val:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset_teste.append((ps,3))#gac
    else:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset_val.append((ps,3))#gac
    cont +=1

temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/gel/"

os.chdir(temp_folder)

listaDeAudios = []
teste = 77
val = 100
cont = 0

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(len(listaDeAudios)):
    if cont < teste:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset.append((ps,4))#gel
    elif cont < val:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset_teste.append((ps,4))#gel
    else:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset_val.append((ps,4))#gel
    cont +=1

temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/org/"

os.chdir(temp_folder)

listaDeAudios = []
teste = 105
val = 135
cont = 0

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(len(listaDeAudios)):
    if cont < teste:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset.append((ps,5))#org
    elif cont < val:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset_teste.append((ps,5))#org
    else:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset_val.append((ps,5))#org
    cont +=1

temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/pia/"

os.chdir(temp_folder)

listaDeAudios = []
teste = 91
val = 117
cont = 0

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(len(listaDeAudios)):
    if cont < teste:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset.append((ps,6))#pia
    elif cont < val:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset_teste.append((ps,6))#pia
    else:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset_val.append((ps,6))#pia
    cont +=1

    
temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/sax/"

os.chdir(temp_folder)

listaDeAudios = []
teste = 84
val = 108
cont = 0

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(len(listaDeAudios)):
    if cont < teste:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset.append((ps,7))#sax
    elif cont < val:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset_teste.append((ps,7))#sax
    else:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset_val.append((ps,7))#sax
    cont +=1

    
temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/tru/"

os.chdir(temp_folder)

listaDeAudios = []
teste = 111
val = 143
cont = 0


for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(len(listaDeAudios)):
    if cont < teste:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset.append((ps,8))#tru
    elif cont < val:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset_teste.append((ps,8))#tru
    else:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset_val.append((ps,8))#tru
    cont +=1

temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/vio/"

os.chdir(temp_folder)

listaDeAudios = []
teste = 85
val = 109
cont = 0

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(len(listaDeAudios)):
    if cont < teste:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset.append((ps,9))#vio
    elif cont < val:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset_teste.append((ps,9))#vio
    else:
        y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
        ps=librosa.feature.melspectrogram(y=y,sr=sr)
        dataset_val.append((ps,9))#vio
    cont +=1
        
    
###############################################    
###############################################
    
def creatCallbacks(nameModel):

    weight_path ="{}_weights.best.hdf5".format(nameModel)

    checkpoint = ModelCheckpoint(weight_path, monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max', save_weights_only = True)

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, verbose=1, mode='min', 
                                       epsilon=0.0001, cooldown=3, min_lr=0.0001)

    callbacks_list = [checkpoint, reduceLROnPlat]

    return weight_path, callbacks_list





'''
D = dataset
random.shuffle(D)

train = D[:929]
test = D[929:1195]
val = D[1195:]
'''

X_train,y_train = zip(*dataset)
X_test,y_test = zip(*dataset_teste)
X_val,y_val = zip(*dataset_val)


# Reshape for CNN input
X_train = np.array([x.reshape( (128, 130, 1) ) for x in X_train])
X_test = np.array([x.reshape( (128, 130, 1) ) for x in X_test])
X_val = np.array([x.reshape( (128, 130, 1) ) for x in X_val])


# One-Hot encoding for classes
y_train = np.array(keras.utils.to_categorical(y_train, 10))
y_test = np.array(keras.utils.to_categorical(y_test, 10))
y_val = np.array(keras.utils.to_categorical(y_val, 10))

model = Sequential()

input_shape=(128, 130, 1)


model.add(Conv2D(64,(5,5),strides=(1, 1), kernel_initializer = 'glorot_normal',input_shape=input_shape))
#model.add(Dropout(rate=0.5))#
model.add(MaxPooling2D((4, 2), strides=(4, 2)))
#model.add(MaxPooling2D())
model.add(Activation('relu'))

model.add(Conv2D(128, (5, 5),kernel_initializer = 'glorot_normal', padding="valid"))
#model.add(MaxPooling2D())
model.add(MaxPooling2D((4, 2), strides=(4, 2)))
model.add(Activation('relu'))
#model.add(Dropout(rate=0.5))#

model.add(Conv2D(128, (5, 5), kernel_initializer = 'glorot_normal',padding="valid"))
model.add(Activation('tanh'))
#model.add(Dropout(rate=0.5))#

model.add(Flatten())
model.add(Dropout(rate=0.5))

model.add(Dense(64,kernel_initializer = 'glorot_normal'))

model.add(Activation('tanh'))
model.add(Dropout(rate=0.5))

model.add(Dense(10,kernel_initializer = 'glorot_normal'))
model.add(Activation('softmax'))

model.compile(
	optimizer="rmsprop",
	loss="categorical_crossentropy",
	metrics=['accuracy'])


weight_path_model, callbacks_list_model = creatCallbacks('NOME_MODELO25')

model.load_weights(weight_path_model)

history = model.fit(x=X_train,y=y_train,epochs=25,
                              batch_size=32,
                              validation_data=(X_val, y_val),
                              callbacks = callbacks_list_model)

model.load_weights(weight_path_model)
model.save('NOME_MODELO25.h5')

Y_pred = model.predict(X_test)

y_pred = np.argmax(Y_pred, axis=1)
Y_test = np.argmax(y_test, axis=1)

accuracy_score(Y_test, y_pred)


from sklearn.metrics import classification_report, confusion_matrix
print('Confusion Matrix')
cm = confusion_matrix(Y_test, y_pred)
print(cm)  
'
    
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


    
    
    
    
    
    
    
    