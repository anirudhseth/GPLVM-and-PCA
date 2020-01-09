
import numpy as np
from numpy import genfromtxt
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import h5py

def scatter2D(X,y,title):
    labels = np.unique(y)
    Ncolors = len(labels)
    xx = np.arange(Ncolors)
    ys = [i+xx+(i*xx)**2 for i in range(Ncolors)]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    c = 1.0
    for label in labels:
        classIdx = np.where(y==label)[0]
        Xclass = X[classIdx,:]
        plt.scatter(Xclass[:,0],Xclass[:,1],linewidths=1,s=1,color=colors[label],marker='o',alpha=0.75)
        c += 1.
    plt.title(title+' Classes:'+str(len(labels)))
    plt.show()


def PCAalgo(latent_dim,y,labels,title):
        pca = PCA(n_components=latent_dim)
        X_pca = pca.fit_transform(y)
        X_pca = X_pca
        # plt.scatter(X_pca[:, 0], X_pca[:, 1], s=1, c=np.array(['r', 'g', 'b'])[np.where(labels)[1]])
        # plt.scatter(X_pca[:, 0], X_pca[:, 1])
        # plt.title(title)
        scatter2D(X_pca,labels,title)

lowerDim=2

#Oil Flow Datasets#
y1=pd.read_csv('Datasets/OilflowX.txt', header=None, sep='\s+').values
labels1=(pd.read_csv('Datasets/OilflowY.txt', header=None, sep='\s+').values)
labelstemp=[]
for i in labels1:
    labelstemp.append(np.where(i==1)[0][0])
labels1=np.asarray(labelstemp)
title1='PCA on Oil Flow Data'
PCAalgo(lowerDim,y1,labels1,title1)

#Vowel Datasets#    https://www.openml.org/d/58
y2 = genfromtxt('Datasets/vowelX.txt', delimiter=',')
labels2 = genfromtxt('Datasets/vowelY.txt', delimiter=',',dtype=np.int)
title2='PCA on Vowel Data'
PCAalgo(lowerDim,y2,labels2,title2)

#Olivetti faces Datasets#    https://scikit-learn.org/0.19/Datasetss/olivetti_faces.html
y3 = genfromtxt('Datasets/olivettifacesX.txt', delimiter=',')
y3= y3/255
title3='PCA on Olivetti faces Datasets'
labels3 = genfromtxt('Datasets/olivettifacesY.txt', delimiter=',',dtype=np.int)
PCAalgo(lowerDim,y3,labels3,title3)

#Wine Data Set#  https://archive.ics.uci.edu/ml/Datasetss/wine
y4= genfromtxt('Datasets/wineX.txt', delimiter=',')
labels4 = genfromtxt('Datasets/wineY.txt', delimiter=',',dtype=np.int)
labels4=labels4-1
title4='PCA on Wine Data Set'
PCAalgo(lowerDim,y4,labels4,title4)

# USPS Dataset# https://www.kaggle.com/bistaumanga/usps-dataset
with h5py.File(filename, 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]

y5=X_tr
labels5=y_tr
title5='PCA on USPS Dataset'
PCAalgo(lowerDim,y5,labels5,title5)