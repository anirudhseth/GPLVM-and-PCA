
import numpy as np
from numpy import genfromtxt
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import h5py

#Wine Data Set#  https://archive.ics.uci.edu/ml/Datasetss/wine
def getWineData():
    y= genfromtxt('Datasets/wineX.txt', delimiter=',')
    labels = genfromtxt('Datasets/wineY.txt', delimiter=',',dtype=np.int)
    labels=labels-1
    return y,labels

# USPS Dataset# https://www.kaggle.com/bistaumanga/usps-dataset
def getUSPSData():
    filename='Datasets/usps.h5'
    with h5py.File(filename, 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]
    y=X_tr
    labels=y_tr
    return y,labels

#Olivetti faces Datasets#    https://scikit-learn.org/0.19/Datasetss/olivetti_faces.html
def getOlivettiData():
    y = genfromtxt('Datasets/olivettifacesX.txt', delimiter=',')
    y= y/255
    labels = genfromtxt('Datasets/olivettifacesY.txt', delimiter=',',dtype=np.int)
    return y,labels

#Oil Flow Datasets#
def getOilFlowData():
    y=pd.read_csv('Datasets/OilflowX.txt', header=None, sep='\s+').values
    labels=(pd.read_csv('Datasets/OilflowY.txt', header=None, sep='\s+').values)
    labelstemp=[]
    for i in labels:
        labelstemp.append(np.where(i==1)[0][0])
    labels=np.asarray(labelstemp)
    return y,labels

#Vowel Datasets#    https://www.openml.org/d/58 
def getVowelDataset():
    y = genfromtxt('Datasets/vowelX.txt', delimiter=',')
    labels = genfromtxt('Datasets/vowelY.txt', delimiter=',',dtype=np.int)
    return y,labels

def plot2D(X,y,title):
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
    plt.title(title+' Classes:'+str(len(labels)), fontsize=10)
    plt.show()


def PCAandPlot(y,labels,title):
        latent_dim=2
        pca = PCA(n_components=latent_dim)
        X_pca = pca.fit_transform(y)
        X_pca = X_pca
        plot2D(X_pca,labels,title)



y,labels=getOilFlowData()
PCAandPlot(y,labels,'PCA on Oil Flow Data')

y,labels=getVowelDataset()
PCAandPlot(y,labels,'PCA on Vowel Data')

y,labels=getOlivettiData()
PCAandPlot(y,labels,'PCA on Olivetti faces Datasets')

y,labels=getWineData()
PCAandPlot(y,labels,'PCA on Wine Data Set')
  
y,labels=getUSPSData()
PCAandPlot(y,labels,'PCA on USPS Dataset')
