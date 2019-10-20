#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 00:21:30 2019

@author: joao
"""
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix


#Load the Training and Test sets 
Train=np.loadtxt('TP1_train.tsv',delimiter='\t')
Test=np.loadtxt('TP1_test.tsv',delimiter='\t')

#Shuffles the test and train sets
np.random.shuffle(Train)
np.random.shuffle(Test)

#Separating the values of the sets from the labels
X=Train[:,:-1]
Y=Train[:,-1]

X_test=Test[:,:-1]
Y_test=Test[:,-1]

#standartize the train and test values
Xmeans=np.mean(X_test,axis=0)
Xstd=np.std(X_test,axis=0)
X_test=(X_test-Xmeans)/Xstd  

Xmeans=np.mean(X,axis=0)
Xstd=np.std(X,axis=0)
X=(X-Xmeans)/Xstd  

#Splits the training set in train and validation sets
#X_train,X_val,Y_train,Y_val=train_test_split(X,Y,test_size=0.33, shuffle=True,stratify=Y)

#Defines the range of the bandwidth cross validation testing
bandwidth=np.linspace(0.01,1,30)

#Cross Validation with 10 folds
kf = StratifiedKFold(n_splits=10)
folds=10
sc=[]
Vbw=[]
scores=[]
for bw in bandwidth:
    #Needs Completion
     tr_err = va_err = 0
     for tr_ix,va_ix in kf.split(X,Y):
         #Study how to use this function
         kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(X[tr_ix],Y[tr_ix])
         sc.append(kde.score(X[va_ix],Y[va_ix]))
     scores.append(np.sum(sc)/len(sc))
     sc=[]
     Vbw.append(bw)
     
bestBW=Vbw[np.argmax(scores)]
print("Best score->" + str(np.max(scores)) + " with bandwidth= " + str(bestBW))
pClass1=np.log(np.sum(Y_test)/len(Y_test))
pClass0=np.log(1-np.sum(Y_test)/len(Y_test))
kde = KernelDensity(kernel='gaussian', bandwidth=bestBW).fit(X,Y)
eval=kde.score_samples(X_test)
