#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 21:22:02 2018

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

from numpy import fft as fft

temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/cel/"

os.chdir("/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/cel/")

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)


frames = []

for i in range(len(listaDeAudios)):
    rate,audData=scipy.io.wavfile.read(temp_folder+listaDeAudios[i])
    fourier=fft.fft(audData)
    channel1=fourier[:,0]
    channel2=fourier[:,1]
    aux = pd.DataFrame(channel1).describe()
    aux2 = pd.DataFrame(channel2).describe()
    
    df = pd.DataFrame({'count': [aux[0][0]],
                       'mean': [aux[0][1]],
                       'std': [aux[0][2]],
                       'min': [aux[0][3]],
                       '1qt': [aux[0][4]],
                       '2qt': [aux[0][5]],
                       '3qt': [aux[0][6]],
                       'max': [aux[0][7]],
                       'mean2': [aux2[0][1]],
                       'std2': [aux2[0][2]],
                       'min2': [aux2[0][3]],
                       '1qt2': [aux2[0][4]],
                       '2qt2': [aux2[0][5]],
                       '3qt2': [aux2[0][6]],
                       'max2': [aux2[0][7]],                       
                       'class':'cel'},index=[i])
    
    frames.append(df)

'''
#speectogram
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

sample_rate, samples = wavfile.read(temp_folder+'[cel][cla]0017__1.wav')

frequencies, times, spectrogram = signal.spectrogram(samples[0], sample_rate)

plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
'''





temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/cla/"


os.chdir("/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/cla/")

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)

k = 0
for i in range(len(frames[0]),(len(frames[0])+len(listaDeAudios))):
    rate,audData=scipy.io.wavfile.read(temp_folder+listaDeAudios[k])
    k+=1
    fourier=fft.fft(audData)
    channel1=fourier[:,0]
    '''
    channel2=fourier[:,1]
    aux = pd.DataFrame(channel1).describe()
    aux2 = pd.DataFrame(channel2).describe()
    df = pd.DataFrame({'count': [aux[0][0]],
                       'mean': [aux[0][1]],
                       'std': [aux[0][2]],
                       'min': [aux[0][3]],
                       '1qt': [aux[0][4]],
                       '2qt': [aux[0][5]],
                       '3qt': [aux[0][6]],
                       'max': [aux[0][7]],
                       'mean2': [aux2[0][1]],
                       'std2': [aux2[0][2]],
                       'min2': [aux2[0][3]],
                       '1qt2': [aux2[0][4]],
                       '2qt2': [aux2[0][5]],
                       '3qt2': [aux2[0][6]],
                       'max2': [aux2[0][7]], 
                       'class':'cla'},index=[i])
    '''
    frames.append(channel1) 
    

temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/flu/"

os.chdir("/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/flu/")

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)

k = 0
for i in range(len(frames[0]),(len(frames[0])+len(listaDeAudios))):
    rate,audData=scipy.io.wavfile.read(temp_folder+listaDeAudios[k])
    k+=1
    fourier=fft.fft(audData)
    channel1=fourier[:,0]
    '''
    channel2=fourier[:,1]
    aux = pd.DataFrame(channel1).describe()
    aux2 = pd.DataFrame(channel2).describe()
    df = pd.DataFrame({'count': [aux[0][0]],
                       'mean': [aux[0][1]],
                       'std': [aux[0][2]],
                       'min': [aux[0][3]],
                       '1qt': [aux[0][4]],
                       '2qt': [aux[0][5]],
                       '3qt': [aux[0][6]],
                       'max': [aux[0][7]],
                       'mean2': [aux2[0][1]],
                       'std2': [aux2[0][2]],
                       'min2': [aux2[0][3]],
                       '1qt2': [aux2[0][4]],
                       '2qt2': [aux2[0][5]],
                       '3qt2': [aux2[0][6]],
                       'max2': [aux2[0][7]], 
                       'class':'flu'},index=[i])
    
    '''
    frames.append(channel1) 

temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/gac/"

os.chdir("/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/gac/")

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)

k = 0
for i in range(len(frames[0]),(len(frames[0])+len(listaDeAudios))):
    rate,audData=scipy.io.wavfile.read(temp_folder+listaDeAudios[k])
    k+=1
    fourier=fft.fft(audData)
    channel1=fourier[:,0]
    channel2=fourier[:,1]
    aux = pd.DataFrame(channel1).describe()
    aux2 = pd.DataFrame(channel2).describe()
    df = pd.DataFrame({'count': [aux[0][0]],
                       'mean': [aux[0][1]],
                       'std': [aux[0][2]],
                       'min': [aux[0][3]],
                       '1qt': [aux[0][4]],
                       '2qt': [aux[0][5]],
                       '3qt': [aux[0][6]],
                       'max': [aux[0][7]],
                       'mean2': [aux2[0][1]],
                       'std2': [aux2[0][2]],
                       'min2': [aux2[0][3]],
                       '1qt2': [aux2[0][4]],
                       '2qt2': [aux2[0][5]],
                       '3qt2': [aux2[0][6]],
                       'max2': [aux2[0][7]], 
                       'class':'gac'},index=[i])
    frames.append(df) 

temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/gel/"

os.chdir("/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/gel/")

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)

k = 0
for i in range(len(frames[0]),(len(frames[0])+len(listaDeAudios))):
    rate,audData=scipy.io.wavfile.read(temp_folder+listaDeAudios[k])
    k+=1
    fourier=fft.fft(audData)
    channel1=fourier[:,0]
    channel2=fourier[:,1]
    aux = pd.DataFrame(channel1).describe()
    aux2 = pd.DataFrame(channel2).describe()
    df = pd.DataFrame({'count': [aux[0][0]],
                       'mean': [aux[0][1]],
                       'std': [aux[0][2]],
                       'min': [aux[0][3]],
                       '1qt': [aux[0][4]],
                       '2qt': [aux[0][5]],
                       '3qt': [aux[0][6]],
                       'max': [aux[0][7]],
                       'mean2': [aux2[0][1]],
                       'std2': [aux2[0][2]],
                       'min2': [aux2[0][3]],
                       '1qt2': [aux2[0][4]],
                       '2qt2': [aux2[0][5]],
                       '3qt2': [aux2[0][6]],
                       'max2': [aux2[0][7]], 
                       'class':'gel'},index=[i])
    frames.append(df) 

temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/org/"

os.chdir("/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/org/")

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)

k = 0
for i in range(len(frames[0]),(len(frames[0])+len(listaDeAudios))):
    rate,audData=scipy.io.wavfile.read(temp_folder+listaDeAudios[k])
    k+=1
    fourier=fft.fft(audData)
    channel1=fourier[:,0]
    channel2=fourier[:,1]
    aux = pd.DataFrame(channel1).describe()
    aux2 = pd.DataFrame(channel2).describe()
    df = pd.DataFrame({'count': [aux[0][0]],
                       'mean': [aux[0][1]],
                       'std': [aux[0][2]],
                       'min': [aux[0][3]],
                       '1qt': [aux[0][4]],
                       '2qt': [aux[0][5]],
                       '3qt': [aux[0][6]],
                       'max': [aux[0][7]],
                       'mean2': [aux2[0][1]],
                       'std2': [aux2[0][2]],
                       'min2': [aux2[0][3]],
                       '1qt2': [aux2[0][4]],
                       '2qt2': [aux2[0][5]],
                       '3qt2': [aux2[0][6]],
                       'max2': [aux2[0][7]], 
                       'class':'org'},index=[i])
    frames.append(df) 
    
temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/pia/"

os.chdir("/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/pia/")

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)

k = 0
for i in range(len(frames[0]),(len(frames[0])+len(listaDeAudios))):
    rate,audData=scipy.io.wavfile.read(temp_folder+listaDeAudios[k])
    k+=1
    fourier=fft.fft(audData)
    channel1=fourier[:,0]
    channel2=fourier[:,1]
    aux = pd.DataFrame(channel1).describe()
    aux2 = pd.DataFrame(channel2).describe()
    df = pd.DataFrame({'count': [aux[0][0]],
                       'mean': [aux[0][1]],
                       'std': [aux[0][2]],
                       'min': [aux[0][3]],
                       '1qt': [aux[0][4]],
                       '2qt': [aux[0][5]],
                       '3qt': [aux[0][6]],
                       'max': [aux[0][7]],
                       'mean2': [aux2[0][1]],
                       'std2': [aux2[0][2]],
                       'min2': [aux2[0][3]],
                       '1qt2': [aux2[0][4]],
                       '2qt2': [aux2[0][5]],
                       '3qt2': [aux2[0][6]],
                       'max2': [aux2[0][7]], 
                       'class':'pia'},index=[i])
    frames.append(df) 

temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/sax/"

os.chdir("/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/sax/")

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)

k = 0
for i in range(len(frames[0]),(len(frames[0])+len(listaDeAudios))):
    rate,audData=scipy.io.wavfile.read(temp_folder+listaDeAudios[k])
    k+=1
    fourier=fft.fft(audData)
    channel1=fourier[:,0]
    channel2=fourier[:,1]
    aux = pd.DataFrame(channel1).describe()
    aux2 = pd.DataFrame(channel2).describe()
    df = pd.DataFrame({'count': [aux[0][0]],
                       'mean': [aux[0][1]],
                       'std': [aux[0][2]],
                       'min': [aux[0][3]],
                       '1qt': [aux[0][4]],
                       '2qt': [aux[0][5]],
                       '3qt': [aux[0][6]],
                       'max': [aux[0][7]],
                       'mean2': [aux2[0][1]],
                       'std2': [aux2[0][2]],
                       'min2': [aux2[0][3]],
                       '1qt2': [aux2[0][4]],
                       '2qt2': [aux2[0][5]],
                       '3qt2': [aux2[0][6]],
                       'max2': [aux2[0][7]], 
                       'class':'sax'},index=[i])
    frames.append(df) 
    
temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/tru/"

os.chdir("/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/tru/")

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)

k = 0
for i in range(len(frames[0]),(len(frames[0])+len(listaDeAudios))):
    rate,audData=scipy.io.wavfile.read(temp_folder+listaDeAudios[k])
    k+=1
    fourier=fft.fft(audData)
    channel1=fourier[:,0]
    channel2=fourier[:,1]
    aux = pd.DataFrame(channel1).describe()
    aux2 = pd.DataFrame(channel2).describe()
    df = pd.DataFrame({'count': [aux[0][0]],
                       'mean': [aux[0][1]],
                       'std': [aux[0][2]],
                       'min': [aux[0][3]],
                       '1qt': [aux[0][4]],
                       '2qt': [aux[0][5]],
                       '3qt': [aux[0][6]],
                       'max': [aux[0][7]],
                       'mean2': [aux2[0][1]],
                       'std2': [aux2[0][2]],
                       'min2': [aux2[0][3]],
                       '1qt2': [aux2[0][4]],
                       '2qt2': [aux2[0][5]],
                       '3qt2': [aux2[0][6]],
                       'max2': [aux2[0][7]], 
                       'class':'tru'},index=[i])
    frames.append(df) 
    
temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/vio/"

os.chdir("/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/vio/")

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)

k = 0
for i in range(len(frames[0]),(len(frames[0])+len(listaDeAudios))):
    rate,audData=scipy.io.wavfile.read(temp_folder+listaDeAudios[k])
    k+=1
    fourier=fft.fft(audData)
    channel1=fourier[:,0]
    channel2=fourier[:,1]
    aux = pd.DataFrame(channel1).describe()
    aux2 = pd.DataFrame(channel2).describe()
    df = pd.DataFrame({'count': [aux[0][0]],
                       'mean': [aux[0][1]],
                       'std': [aux[0][2]],
                       'min': [aux[0][3]],
                       '1qt': [aux[0][4]],
                       '2qt': [aux[0][5]],
                       '3qt': [aux[0][6]],
                       'max': [aux[0][7]],
                       'mean2': [aux2[0][1]],
                       'std2': [aux2[0][2]],
                       'min2': [aux2[0][3]],
                       '1qt2': [aux2[0][4]],
                       '2qt2': [aux2[0][5]],
                       '3qt2': [aux2[0][6]],
                       'max2': [aux2[0][7]], 
                       'class':'vio'},index=[i])
    frames.append(df) 


dataset = pd.concat(frames)

dataset=dataset.reset_index()

dataset.drop(['index'],axis=1, inplace=True)
dataset.drop(['count'],axis=1,inplace=True)


#correlação
import seaborn as sns

correlacao=dataset.corr().abs().unstack().sort_values(kind='quicksort')
sns.heatmap(dataset.corr())
print(correlacao)

#BALANCEAMENTO
dataset['class'].value_counts()


temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/flu/"

y, sr = librosa.load((temp_folder+'[flu][cla]0346__2.wav'),duration = 3)

ps=librosa.feature.melspectrogram(y=y,sr=sr)

ps.shape

librosa.display.specshow(ps,y_axis='mel',x_axis='time')


X = dataset.iloc[:,0:14]
Y = dataset.iloc[:,14]

#preparação dos dados de treino
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()  
scaler.fit(X_train)
#DISCRETIZAÇÃO DA DATABASE
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)                    
                          
                          
#rede neural      


accuracy=[]
                          
for x in range(0,50):                          
                          
    from sklearn.neural_network import MLPClassifier
    
    clf = MLPClassifier(activation='tanh',solver='lbfgs',
                        hidden_layer_sizes=(15,15,15),
                        random_state=x)
    
    clf.fit(X_train, y_train)
    
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
    
    y_pred=clf.predict(X_test)
    
    accuracy.append(accuracy_score(y_test, y_pred))



#ARVORE DE DECISAO

'''
accuracy=[]
                          
for x in range(0,20):              
                
    clf = DecisionTreeClassifier(random_state=x)
    
    clf.fit(X_train, y_train)
    
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
    
    y_pred=clf.predict(X_test)
    
    accuracy.append(accuracy_score(y_test, y_pred))
'''






#MATRIZ DE CONFUSÃO
matriz_de_confusao = confusion_matrix(y_test, y_pred)
print(matriz_de_confusao)  

# PRECISÃO//RECALL'SENSIBILIDADE E ESPECIFICIDADE'//F1-SCORE//SUPPORT
print(classification_report(y_test, y_pred))  

#ACURÁCIA
print('Acurácia:',accuracy_score(y_test, y_pred))













