    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 23:46:00 2018

@author: adriano
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


###########base para treinamento#####################################
X = np.array([[1, 1, 0.75], [0.5, 0, 0.5], [0, 1, 0], [0.5, 0, 1]])

Y = np.array(['doente', 'saudavel', 'saudavel', 'doente'])
#####################################################################


dataset = pd.read_csv('Churn_Modelling.csv')
X_train,Y_train,x_test,y_test=pd.read_csv('Churn_Modelling.csv')

X=dataset.iloc[:,3:13].values
Y=dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,1:]


######### importando a biblio sklearn ###############################
from sklearn.naive_bayes import GaussianNB
#####################################################################

######### definindo o classificador #################################
clf=GaussianNB()
#####################################################################

######### Treinando ou ajustando os dados de entrada ################
clf.fit(X,Y)
#####################################################################

GaussianNB()

print(clf.predict([[0.5,0,0.25]]))

print(clf.score(X,Y))

#####################################################################
