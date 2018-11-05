import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import os
from scipy.fftpack import fft
import numpy.fft as fftshift
from scipy.signal import butter, sosfilt, sosfreqz, lfilter, freqz
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy.fft import ifft
from numpy import real
from sklearn.metrics import confusion_matrix
import itertools

def findFiles(pasta, extensao):
    arquivosTxt = []
    caminhoAbsoluto = os.path.abspath(pasta)
    for pastaAtual, subPastas, arquivos  in os.walk(caminhoAbsoluto):
        arquivosTxt.extend([os.path.join(pastaAtual, arquivo) for arquivo in arquivos if arquivo.endswith('.wav')])
    return arquivosTxt

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog='False', btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y
    
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def createDataset():
    column = []
    dataset = pd.DataFrame()
    dataset['class'] = 0     
    for elem in range(0, 30000):
        dataset['column_' + str(elem)] = 0 
        column.append('column_' + str(elem))
    dataset.to_csv('dataset.csv')
    return dataset, column 


cel_cello = findFiles('IRMAS-TrainingData_red/cel', '.wav')
cla_clarinet = findFiles('IRMAS-TrainingData_red/cla', '.wav')
flu_flute = findFiles('IRMAS-TrainingData_red/flu', '.wav')
gac_acousticguitar = findFiles('IRMAS-TrainingData_red/gac', '.wav')
gel_eletronicguitar = findFiles('IRMAS-TrainingData_red/gel', '.wav')
org_organ = findFiles('IRMAS-TrainingData_red/org', '.wav')
pia_piano = findFiles('IRMAS-TrainingData_red/pia', '.wav')
sax_saxophone = findFiles('IRMAS-TrainingData_red/sax', '.wav')
tru_trumpet = findFiles('IRMAS-TrainingData_red/tru', '.wav')
vio_violin = findFiles('IRMAS-TrainingData_red/vio', '.wav')

dataset, column = createDataset()

dataset = pd.read_csv('dataset.csv')
classes = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio']
#test = '/home/suannyfabyne/Documentos/P6/Sinais_projeto/IRMAS-TrainingData_red/gel/[gel][jaz_blu]0890__2.wav'

# cutoffFourier = []
# fs ,data=wavfile.read(test)
# fourier=fft(data[:,1])
# print(fourier, 'fourier')
# print(data[:,1], 'data')
# print(fs, 'frequency')
# print(np.fft.fftshift(fourier), 'shift fourier')
# fourierdesloc = np.fft.fftshift(fourier)
# fourierfilter = butter_lowpass_filter(fourierdesloc, 3000.667, fs, order=5)
# print(fourierfilter, 'fourierfilter')
# for index, elem in enumerate(fourierfilter):
#     if(index > 50000 and index < 80000):
#         cutoffFourier.append(elem)
# dicionario = {}
# for index, elem in enumerate(cutoffFourier):
#     dicionario['column_'+str(index)] = elem
# dicionario['class'] = 'cel'
# datasetnew = pd.DataFrame(dicionario,index=[0])
# dataset = dataset.append(dataset_new)

# plt.plot(data[:,1])
# plt.show()
# plt.plot(fourier)
# plt.show()
# plt.plot(fourierdesloc)
# plt.show()
# plt.show()
# plt.plot(fourierfilter)
# plt.show()
# plt.plot(cutoffFourier)
# plt.show()

for ind, file in enumerate(cel_cello):  #CEL
    cutoffFourier = []
    fs,data=wavfile.read(file)
    fourier=fft(data[:,1])
    fourierdesloc = np.fft.fftshift(fourier)
    fourierfilter = butter_lowpass_filter(fourierdesloc, 3000.667, fs, order=5)
    cutoffFourier = fourierfilter[50000:80000,]
    cutoffFourier = real(cutoffFourier)
    dataset_new = pd.DataFrame([cutoffFourier], columns = column)
    dataset_new['class']  = 'cel' 
    dataset = dataset.append(dataset_new, ignore_index=True)
    print("cel:", ind)

print('-----------cabou 1----------')
     
for ind, file in enumerate(cla_clarinet):  #CLA
    cutoffFourier = []
    fs,data=wavfile.read(file)
    fourier=fft(data[:,1])
    fourierdesloc = np.fft.fftshift(fourier)
    fourierfilter = butter_lowpass_filter(fourierdesloc, 3000.667, fs, order=5)
    cutoffFourier = fourierfilter[50000:80000,]
    cutoffFourier = real(cutoffFourier)
    dataset_new = pd.DataFrame([cutoffFourier], columns = column)
    dataset_new['class']  = 'cla' 
    dataset = dataset.append(dataset_new, ignore_index=True)
    print("cla:", ind)

print('-----------cabou 2----------')
    
for ind, file in enumerate(flu_flute):  #FLU
    cutoffFourier = []
    fs,data=wavfile.read(file)
    fourier=fft(data[:,1])
    fourierdesloc = np.fft.fftshift(fourier)
    fourierfilter = butter_lowpass_filter(fourierdesloc, 3000.667, fs, order=5)
    cutoffFourier = fourierfilter[50000:80000,]
    cutoffFourier = real(cutoffFourier)
    dataset_new = pd.DataFrame([cutoffFourier], columns = column)
    dataset_new['class']  = 'flu' 
    dataset = dataset.append(dataset_new, ignore_index=True)
    print("flu:", ind)

print('-----------cabou 3----------')
    
for ind, file in enumerate(gac_acousticguitar):  #GAC
    cutoffFourier = []
    fs,data=wavfile.read(file)
    fourier=fft(data[:,1])
    fourierdesloc = np.fft.fftshift(fourier)
    fourierfilter = butter_lowpass_filter(fourierdesloc, 3000.667, fs, order=5)
    cutoffFourier = fourierfilter[50000:80000,]
    cutoffFourier = real(cutoffFourier)
    dataset_new = pd.DataFrame([cutoffFourier], columns = column)
    dataset_new['class']  = 'gac' 
    dataset = dataset.append(dataset_new, ignore_index=True)  
    print("gac:", ind)

print('-----------cabou 4----------')
        

for ind, file in enumerate(gel_eletronicguitar):  #CLA
    cutoffFourier = []
    fs,data=wavfile.read(file)
    fourier=fft(data[:,1])
    fourierdesloc = np.fft.fftshift(fourier)
    fourierfilter = butter_lowpass_filter(fourierdesloc, 3000.667, fs, order=5)
    cutoffFourier = fourierfilter[50000:80000,]
    cutoffFourier = real(cutoffFourier)
    dataset_new = pd.DataFrame([cutoffFourier], columns = column)
    dataset_new['class']  = 'gel' 
    dataset = dataset.append(dataset_new, ignore_index=True)    
    print("gel:", ind)

print('-----------cabou 5----------')

for ind, file in enumerate(org_organ):  #ORG
    cutoffFourier = []
    fs,data=wavfile.read(file)
    fourier=fft(data[:,1])
    fourierdesloc = np.fft.fftshift(fourier)
    fourierfilter = butter_lowpass_filter(fourierdesloc, 3000.667, fs, order=5)
    cutoffFourier = fourierfilter[50000:80000,]
    cutoffFourier = real(cutoffFourier)
    dataset_new = pd.DataFrame([cutoffFourier], columns = column)
    dataset_new['class']  = 'org' 
    dataset = dataset.append(dataset_new, ignore_index=True)     
    print("org:", ind)

print('-----------cabou 6----------')
    
for ind, file in enumerate(pia_piano):  #PIA
    cutoffFourier = []
    fs,data=wavfile.read(file)
    fourier=fft(data[:,1])
    fourierdesloc = np.fft.fftshift(fourier)
    fourierfilter = butter_lowpass_filter(fourierdesloc, 3000.667, fs, order=5)
    cutoffFourier = fourierfilter[50000:80000,]
    cutoffFourier = real(cutoffFourier)
    dataset_new = pd.DataFrame([cutoffFourier], columns = column)
    dataset_new['class']  = 'pia' 
    dataset = dataset.append(dataset_new, ignore_index=True)
    print("pia:", ind)
print('-----------cabou 7----------')
    
for ind, file in enumerate(sax_saxophone):  #SAX
    cutoffFourier = []
    fs,data=wavfile.read(file)
    fourier=fft(data[:,1])
    fourierdesloc = np.fft.fftshift(fourier)
    fourierfilter = butter_lowpass_filter(fourierdesloc, 3000.667, fs, order=5)
    cutoffFourier = fourierfilter[50000:80000,]
    cutoffFourier = real(cutoffFourier)
    dataset_new = pd.DataFrame([cutoffFourier], columns = column)
    dataset_new['class']  = 'sax' 
    dataset = dataset.append(dataset_new, ignore_index=True)
    print("sax:", ind)

print('-----------cabou 8----------')
    
for ind, file in enumerate(tru_trumpet):  #TRU
    cutoffFourier = []
    fs,data=wavfile.read(file)
    fourier=fft(data[:,1])
    fourierdesloc = np.fft.fftshift(fourier)
    fourierfilter = butter_lowpass_filter(fourierdesloc, 3000.667, fs, order=5)
    cutoffFourier = fourierfilter[50000:80000,]
    cutoffFourier = real(cutoffFourier)
    dataset_new = pd.DataFrame([cutoffFourier], columns = column)
    dataset_new['class']  = 'tru' 
    dataset = dataset.append(dataset_new, ignore_index=True)
    print("tru:", ind)

print('-----------cabou 9----------')
    
for ind, file in enumerate(vio_violin):  #VIO
    cutoffFourier = []
    fs,data=wavfile.read(file)
    fourier=fft(data[:,1])
    fourierdesloc = np.fft.fftshift(fourier)
    fourierfilter = butter_lowpass_filter(fourierdesloc, 3000.667, fs, order=5)
    cutoffFourier = fourierfilter[50000:80000,]
    cutoffFourier = real(cutoffFourier)
    dataset_new = pd.DataFrame([cutoffFourier], columns = column)
    dataset_new['class']  = 'vio' 
    dataset = dataset.append(dataset_new, ignore_index=True)
    print("vio:", ind)

print('-----------cabou 10----------')
    
    
 #print(dataset)
dataset.to_csv('dataset.csv')
 
 


