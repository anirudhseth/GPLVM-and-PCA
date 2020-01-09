
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

latent_dim = 2

# oil dataset

Y = pd.read_csv('Dataset1/train_data.txt', header=None, sep='\s+').values
labels = (pd.read_csv('Dataset1//labels.txt', header=None, sep='\s+').values)
pca = PCA(n_components=latent_dim)

X_pca = pca.fit_transform(Y)
X_pca = X_pca
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=1, c=np.array(['r', 'g', 'b'])[np.where(labels)[1]])
plt.title('PCA on Oil Flow Data')
