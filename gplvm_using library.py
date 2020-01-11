import numpy as np
from matplotlib import pyplot as plt
import GPy # import GPy package
import pandas as pd

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

m_gplvm.optimize(messages=1, max_iters=10)
m_gplvm.plot_latent(labels)


m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(Y, input_dim, num_inducing=30, missing_data=True)
m.optimize(messages=1, max_iters=10)
m.plot_latent(labels)

##https://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/MagnificationFactor.ipynb