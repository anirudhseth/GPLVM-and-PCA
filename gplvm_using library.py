import numpy as np
from matplotlib import pyplot as plt
import GPy # import GPy package
import pandas as pd
from timeit import default_timer as timer
%config InlineBackend.figure_format = 'svg'
import numpy as np
from numpy import genfromtxt
from sklearn.decomposition import PCA , KernelPCA
from sklearn.manifold import MDS,TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import h5py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from autograd import grad
from tqdm import tqdm_notebook as tqdm
import math
import autograd.numpy as np
from sklearn.datasets import fetch_olivetti_faces

def KNNScore(x,y,title):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
        knn = KNeighborsClassifier(n_neighbors=1)
        # Fit a nearest neighbor classifier on the embedded training set
        knn.fit(X_train, y_train)
        # Compute the nearest neighbor accuracy on the embedded test set
        acc_knn = knn.score(X_test, y_test) #Return the mean accuracy on the given test data and labels.
        print(title+',Accuracy Score:'+str(acc_knn))

def getOilFlowData():
    y=pd.read_csv('Datasets/OilflowX.txt', header=None, sep='\s+').values
    labels=(pd.read_csv('Datasets/OilflowY.txt', header=None, sep='\s+').values)
    labelstemp=[]
    for i in labels:
        labelstemp.append(np.where(i==1)[0][0])
    labels=np.asarray(labelstemp)
    return y,labels

Y,labels=getOilFlowData()


input_dim = 2 # How many latent dimensions to use
kernel = GPy.kern.RBF(input_dim, 1, ARD=True) 

Q = input_dim

m_gplvm = GPy.models.GPLVM(Y, Q, kernel=GPy.kern.RBF(Q))
m_gplvm.kern.lengthscale = .2
m_gplvm.kern.variance = 1
m_gplvm.likelihood.variance = 1.

start = timer()
m_gplvm.optimize(messages=1, max_iters=1000)
end = timer()
print('Time for GPLVM:'+str(end - start))
m_gplvm.plot_latent(labels)
KNNScore(m_gplvm.latent_mean.values,labels,'GPLVM')


m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(Y, input_dim, num_inducing=30, missing_data=True)
start = timer()
m.optimize(messages=1, max_iters=1000)
end = timer()
print('Time for Bayesian GPLVM:'+str(end - start))
KNNScore(m.latent_space.mean.values,labels,'Bayesian GPLVM')

m_sp = GPy.models.SparseGPLVM(Y, input_dim, kernel=kernel, num_inducing=30)
# m_sp=GPy.models.SparseGPLVM(Y, input_dim, kernel=kernel,num_inducing=30,init='PCA')
from timeit import default_timer as timer
start = timer()
m_sp.optimize(messages=1, max_iters=1000)
end = timer()
print('Time for Sparse GPLVM:'+str(end - start))
KNNScore(m_sp.latent_space.values,labels,'Sparse GPLVM')

##https://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/MagnificationFactor.ipynb