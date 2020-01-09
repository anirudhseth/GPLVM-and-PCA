
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class pcaAlgo:
    def __init__(self, latent_dim,y,labels,title):
        self.latent_dim=latent_dim
        self.y=pd.read_csv(y, header=None, sep='\s+').values
        self.labels=(pd.read_csv(labels, header=None, sep='\s+').values)
        self.title=title
    def fitandPLot(self):
        pca = PCA(n_components=self.latent_dim)
        X_pca = pca.fit_transform(self.y)
        X_pca = X_pca
        plt.scatter(X_pca[:, 0], X_pca[:, 1], s=1, c=np.array(['r', 'g', 'b'])[np.where(self.labels)[1]])
        plt.title(self.title)


