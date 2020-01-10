
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


# RBF kernel    
def rbf(X, sigma_f, length_scale, noise_coef=0.):

    
    num_points = X.shape[0]

    cov = np.dot(X, X.T)
    diag = np.diag(cov)

    # (x_n - x_m)' (x_n - x_m) = x_n'x_n + x_m'x_m - 2x_n'x_m
    cov_ = diag.reshape((num_points, 1)) + diag.reshape((1, num_points)) - 2 * cov

    return (sigma_f ** 2.) * np.exp(-1. / (2 * length_scale ** 2.) * cov_) + noise_coef * np.eye(num_points)

# Characteristic function for GP-LVM
def logLik(K, Y):
    D, N = Y.shape
    K_inv = np.linalg.inv(K)

    return -D*N/2*np.log(2*math.pi) - N/2*np.linalg.slogdet(K)[1] - 1/2*np.trace(np.dot(np.dot(K_inv,Y),Y.T))
 

def sievingGPLVMalgo(lowerDim, Y, alpha, beta, gamma):
    
    X, alpha, beta, gamma = GPLVMfit(Y, lowerDim, alpha, beta, gamma, num_iter=10, learn_rate=1e-5, verbose=False, log_every=1)

    return X


def GPLVMalgo(lowerDim, Y, labels, title, alpha, beta, gamma):
    
    X, alpha, beta, gamma = GPLVMfit(Y, lowerDim, alpha, beta, gamma, num_iter=1000, learn_rate=1e-5, verbose=False, log_every=1)

    scatter2D(X,labels,title)

    return X , Y , labels


def GPLVMfit (Y, latent_dim, alpha, beta, gamma, learn_rate=1e-6, num_iter=1000, verbose=True, log_every=50):

    # Initial guess for X (latent variable) using regular PCA

    pca = PCA(n_components = latent_dim)
    X = pca.fit_transform(Y)
    # Radial basis kernel for similarity matrix K
    K = rbf(X, alpha, gamma, beta)
    L = logLik(K, Y)

    logLik_lambda = lambda X_, alpha_, beta_, gamma_: logLik(rbf(X_, alpha_, gamma_, beta_),Y)

    # Parameters we wish to determine
    theta = [X, alpha, beta, gamma]

    # Compute gradients for each parameter
    dlogLik_dTheta = [grad(logLik_lambda, i) for i in range(len(theta))]

    # tqdm gives progress bar

    for i in notebook.tqdm(range(num_iter)):
                
        grads = [logLik_partial(*theta) for logLik_partial in dlogLik_dTheta]
        #[theta[j] + learn_rate * gradient for j, gradient in enumerate(grads)]

        theta[0] = theta[0] + learn_rate * grads[0]
        theta[1] = theta[1] + learn_rate * grads[1]
        theta[2] = theta[2] + 1e-15 * grads[2]
        theta[3] = theta[3] + 1e-7 * grads[3]

        if verbose and i % log_every == 0:
            print("Log-likelihood (iteration {}): {:.3f}".format(i + 1, logLik_lambda(*theta)))

    if verbose:
        print("Final log-likelihood: {:.3f}".format(logLik_lambda(*theta)))
    return tuple(theta)



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
