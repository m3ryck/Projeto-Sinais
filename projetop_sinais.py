#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 11:22:14 2018

@author: Adriano Brito && Alexandre Santos && José Eugênio
"""
#IMPORTANDO BIBLIOTECAS
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
from numpy.fft import fftshift, ifft
from numpy import real
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt

#FUNÇÕES PARA APLICAR O FILTRO PASSA-BAIXA
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

#PARAMETROS PARA A EXECUÇÃO DO FILTRO
order = 6
cutoff = 3500  # desired cutoff frequency of the filter, Hz


############################
'''
CRIAÇÃO DO DATASET
PARA FACILITAR A LEITURA
DOS DADOS POSTERIORMENTE
'''
############################

dataset_instrumento = pd.DataFrame()
dataset = pd.DataFrame()
colunas = []

for i in range(0,20000):
    colunas.append('atr' + str(i))
    dataset['atr' + str(i)] = 0
dataset['class'] = 'cel'    




#LENDO AUDIOS DA PASTA CEL
temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/cel/"

os.chdir(temp_folder)

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(0, len(listaDeAudios)):
    rate,audData = scipy.io.wavfile.read(temp_folder + listaDeAudios[i])
    fourier = fft.fft(audData[:,0])
    sinal_deslocado = np.fft.fftshift(fourier)
    filtro_pb = butter_lowpass_filter(sinal_deslocado, cutoff, rate, order)
    filtro_limitado = filtro_pb[55000:75000,]
    ptreal = real(filtro_limitado)
    dataset_instrumento = pd.DataFrame([ptreal], columns = colunas)
    dataset_instrumento['class']  = 'cel' 
    dataset = dataset.append(dataset_instrumento, ignore_index=True)
    print("cel")

#LENDO AUDIOS DA PASTA CLA
temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/cla/"

os.chdir(temp_folder)

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(0, len(listaDeAudios)):
    rate,audData = scipy.io.wavfile.read(temp_folder + listaDeAudios[i])
    fourier = fft.fft(audData[:,0])
    sinal_deslocado = np.fft.fftshift(fourier)
    filtro_pb = butter_lowpass_filter(sinal_deslocado, cutoff, rate, order)
    filtro_limitado = filtro_pb[55000:75000,]
    ptreal = real(filtro_limitado)
    dataset_instrumento = pd.DataFrame([ptreal], columns = colunas)
    dataset_instrumento['class']  = 'cla' 
    dataset = dataset.append(dataset_instrumento, ignore_index=True)
    print("cla")

#dataset.drop(axis = 0, index = 200, inplace = True)
  
#LENDO AUDIOS DA PASTA FLU
temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/flu/"

os.chdir(temp_folder)

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(0, len(listaDeAudios)):
    rate,audData = scipy.io.wavfile.read(temp_folder + listaDeAudios[i])
    fourier = fft.fft(audData[:,0])
    sinal_deslocado = np.fft.fftshift(fourier)
    filtro_pb = butter_lowpass_filter(sinal_deslocado, cutoff, rate, order)
    filtro_limitado = filtro_pb[55000:75000,]
    ptreal = real(filtro_limitado)
    dataset_instrumento = pd.DataFrame([ptreal], columns = colunas)
    dataset_instrumento['class']  = 'flu' 
    dataset = dataset.append(dataset_instrumento, ignore_index=True)
    print("flu")



#LENDO AUDIOS DA PASTA GAC
temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/gac/"

os.chdir(temp_folder)

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(0, len(listaDeAudios)):
    rate,audData = scipy.io.wavfile.read(temp_folder + listaDeAudios[i])
    fourier = fft.fft(audData[:,0])
    sinal_deslocado = np.fft.fftshift(fourier)
    filtro_pb = butter_lowpass_filter(sinal_deslocado, cutoff, rate, order)
    filtro_limitado = filtro_pb[55000:75000,]
    ptreal = real(filtro_limitado)
    dataset_instrumento = pd.DataFrame([ptreal], columns = colunas)
    dataset_instrumento['class']  = 'gac' 
    dataset = dataset.append(dataset_instrumento, ignore_index=True)
    print("gac")


#LENDO AUDIOS DA PASTA GEL   
temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/gel/"

os.chdir(temp_folder)

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(0, len(listaDeAudios)):
    rate,audData = scipy.io.wavfile.read(temp_folder + listaDeAudios[i])
    fourier = fft.fft(audData[:,0])
    sinal_deslocado = np.fft.fftshift(fourier)
    filtro_pb = butter_lowpass_filter(sinal_deslocado, cutoff, rate, order)
    filtro_limitado = filtro_pb[55000:75000,]
    ptreal = real(filtro_limitado)
    dataset_instrumento = pd.DataFrame([ptreal], columns = colunas)
    dataset_instrumento['class']  = 'gel' 
    dataset = dataset.append(dataset_instrumento, ignore_index=True)
    print("gel")
    

#LENDO AUDIOS DA PASTA ORG    
temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/org/"

os.chdir(temp_folder)

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(0, len(listaDeAudios)):
    rate,audData = scipy.io.wavfile.read(temp_folder + listaDeAudios[i])
    fourier = fft.fft(audData[:,0])
    sinal_deslocado = np.fft.fftshift(fourier)
    filtro_pb = butter_lowpass_filter(sinal_deslocado, cutoff, rate, order)
    filtro_limitado = filtro_pb[55000:75000,]
    ptreal = real(filtro_limitado)
    dataset_instrumento = pd.DataFrame([ptreal], columns = colunas)
    dataset_instrumento['class']  = 'org' 
    dataset = dataset.append(dataset_instrumento, ignore_index=True)
    print("org")
        

#LENDO AUDIOS DA PASTA PIA    
temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/pia/"

os.chdir(temp_folder)

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(0, len(listaDeAudios)):
    rate,audData = scipy.io.wavfile.read(temp_folder + listaDeAudios[i])
    fourier = fft.fft(audData[:,0])
    sinal_deslocado = np.fft.fftshift(fourier)
    filtro_pb = butter_lowpass_filter(sinal_deslocado, cutoff, rate, order)
    filtro_limitado = filtro_pb[55000:75000,]
    ptreal = real(filtro_limitado)
    dataset_instrumento = pd.DataFrame([ptreal], columns = colunas)
    dataset_instrumento['class']  = 'pia' 
    dataset = dataset.append(dataset_instrumento, ignore_index=True)
    print("pia")
    

#LENDO AUDIOS DA PASTA SAX    
temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/sax/"

os.chdir(temp_folder)

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(0, len(listaDeAudios)):
    rate,audData = scipy.io.wavfile.read(temp_folder + listaDeAudios[i])
    fourier = fft.fft(audData[:,0])
    sinal_deslocado = np.fft.fftshift(fourier)
    filtro_pb = butter_lowpass_filter(sinal_deslocado, cutoff, rate, order)
    filtro_limitado = filtro_pb[55000:75000,]
    ptreal = real(filtro_limitado)
    dataset_instrumento = pd.DataFrame([ptreal], columns = colunas)
    dataset_instrumento['class']  = 'sax' 
    dataset = dataset.append(dataset_instrumento, ignore_index=True)
    print("sax")
    

#LENDO AUDIOS DA PASTA TRU  
temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/tru/"

os.chdir(temp_folder)

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(0, len(listaDeAudios)):
    rate,audData = scipy.io.wavfile.read(temp_folder + listaDeAudios[i])
    fourier = fft.fft(audData[:,0])
    sinal_deslocado = np.fft.fftshift(fourier)
    filtro_pb = butter_lowpass_filter(sinal_deslocado, cutoff, rate, order)
    filtro_limitado = filtro_pb[55000:75000,]
    ptreal = real(filtro_limitado)
    dataset_instrumento = pd.DataFrame([ptreal], columns = colunas)
    dataset_instrumento['class']  = 'tru' 
    dataset = dataset.append(dataset_instrumento, ignore_index=True)
    print("tru")


#LENDO AUDIOS DA PASTA VIO 
temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/vio/"

os.chdir(temp_folder)

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(0, len(listaDeAudios)):
    rate,audData = scipy.io.wavfile.read(temp_folder + listaDeAudios[i])
    fourier = fft.fft(audData[:,0])
    sinal_deslocado = np.fft.fftshift(fourier)
    filtro_pb = butter_lowpass_filter(sinal_deslocado, cutoff, rate, order)
    filtro_limitado = filtro_pb[55000:75000,]
    ptreal = real(filtro_limitado)
    dataset_instrumento = pd.DataFrame([ptreal], columns = colunas)
    dataset_instrumento['class']  = 'vio' 
    dataset = dataset.append(dataset_instrumento, ignore_index=True)
    print("vio")


#ADICIONANDO AO DATASET A COLUNA "CLASS" PARA CLASSIFICAÇÃO DOS AUDIOS    
dataset['class'].value_counts()   

#dataset.drop(axis = 0, index = 1000, inplace = True)

#SALVANDO DATASET EM UM CSV PARA LEITURA POSTERIOR
dataset.to_csv('irmas_dataframe_sinais.csv')

#LOCAL ONDE ESTA SALVO O DATASET
temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/vio/"

#RECEBENDO DATASET CRIADO PARA NAO TER QUE LER TODOS OS AUDIOS NOVAMENTE
dataset = pd.read_csv(temp_folder + 'irmas_dataframe_sinais.csv')
dataset.drop(['Unnamed: 0'], axis = 1, inplace = True)

#ELIMINANDO COLUNAS NA TENTATIVA DE MELHORAR A ACCURACIA
for i in range(0,5000):
    dataset.drop(['atr'+str(i)], axis = 1, inplace = True)
    dataset.drop(['atr'+str(i+44999)], axis = 1, inplace = True)
    

#DEFININDO ENTRADA E SAIDA
X = dataset
Y = X.iloc[:,49000]
X = X.drop(['class'], axis = 1)


#EXECUTANDO 20 VEZES O KNN PARA ENCONTRAR MELHOR SOLUÇÃO
tabela_acc=[]

for x in range(0,20):
    print('=================KNN=================')
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=x)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()  
    scaler.fit(X_train)
    #DISCRETIZAÇÃO DA DATABASE
    X_train = scaler.transform(X_train)  
    X_test = scaler.transform(X_test)  
    
    ###############################################################################
    #                                                                             #       
    #                               # KNN #                                       # 
    #                                                                             #           
    ###############################################################################
    
    from sklearn.neighbors import KNeighborsClassifier 
     
    classifier = KNeighborsClassifier(n_neighbors=1,
                                      metric='euclidean',
                                      )#, metric_params= {"V": 2}
    
    classifier.fit(X_train, y_train)  
    
    y_pred = classifier.predict(X_test)  
    
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  
    
    '''
    #MATRIZ DE CONFUSÃO
    cm=confusion_matrix(y_test,y_pred)
    print(cm)  
    
    # PRECISÃO//RECALL'SENSIBILIDADE E ESPECIFICIDADE'//F1-SCORE//SUPPORT
    print(classification_report(y_test, y_pred))  
    '''
    #ACURÁCIA
    accuracia = accuracy_score(y_test, y_pred)
    #print('Acurácia:',accuracia)
    
    tabela_acc.append(0)
    
    tabela_acc[x] = accuracia









