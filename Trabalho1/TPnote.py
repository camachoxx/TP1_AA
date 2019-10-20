

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

#labels
Y_Train=Train[:,-1]
Y_Test=Test[:,-1]


#standartize the train and test values
Xmeans=np.mean(Test[:,:-1],axis=0)
Xstd=np.std(Test[:,:-1],axis=0)
X_Test=(Test[:,:-1]-Xmeans)/Xstd  

Xmeans=np.mean(Train[:,:-1],axis=0)
Xstd=np.std(Train[:,:-1],axis=0)
X_Train=(Train[:,:-1]-Xmeans)/Xstd  



#Defines the range of the bandwidth cross validation testing
bandwidth=np.arange(0.02,0.6,0.02)



#Cross Validation with 10 folds
kf = StratifiedKFold(n_splits=10)
folds=10
sc=[]
Vbw=[]
scores=[]
Bestbw_feat=[[],[]]  #vector of the best bw of each feat
Bestscores_feat=[[],[]]
kde_list=[[],[]]

#Cross validation testing
#######################################################################
def Kde_model(bw,data):
    #Returns the classifier list for the given bandwidth and data
    #The data must be [[feats],y]
    kde_list=[[],[]]
    
    data_0=data[data[:,-1]==0]
    
    data_1=data[data[:,-1]==1]
    
    #Class 0
    X_feats=data_0[:,:-1]
    
    Y=data_0[:,-1]
    for feat in range(X_feats.shape[1]):
        X_y=np.column_stack((X_feats[:,feat],Y))
        kde = KernelDensity(kernel='gaussian', bandwidth=bw)
        kde.fit(X_y)
        kde_list[0].append(kde)
    
    #Class 1    
    X_feats=data_1[:,:-1]
    Y=data_1[:,-1]
    for feat in range(X_feats.shape[1]):
        X_y=np.column_stack((X_feats[:,feat],Y))
        kde = KernelDensity(kernel='gaussian', bandwidth=bw)
        kde.fit(X_y)
        kde_list[1].append(kde)
    
    return kde_list

def Kde_class(kde,data):
    #Returns a score for the given kde and data
    #data must be [[feat],y]
    cla=[0,1]
    data_x=data[:,:-1]
    sc=[]
    score=[]
    data_y=[]
    for p in range(np.shape(data_x)[0]):
        for c in cla:
            for feat in range(np.shape(kde)[0]):
                k=kde[c][feat]
                sc.append(k.score(data_x[p,feat]))
            score.append(np.sum(sc))
        data_y.append(np.argmax(score))
        score=[]    
    
    
    return data_y


def KDE_cross_validation(kde,data):
    #cross validates the model for the given bandwidth vector
    #returns the best bandwidth
    #The data must be [x,y]
    X_y=data
    tr_err = va_err = 0
    sc=[]
    score=[]
    for bw in bandwidth:
        for tr_ix,va_ix in kf.split(X_y[:,0],X_y[:,1]):
            #Fit the data for a certain fold and save the value of the score
            kde=Kde_class(bw,X_y[tr_ix])
            sc.append(kde.score(X_y[va_ix]))
        score.append(np.mean(sc))
        sc=[]
    best_bw=bandwidth[np.argmax(score)]
    return best_bw

data=np.column_stack((X_Train,Y_Train))

kded=Kde_model(0.5,data)
y_scored=Kde_class(kded,data)



