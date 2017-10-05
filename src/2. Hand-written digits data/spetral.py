# Created by Truong Phuc Anh (14520040@gm.uit.edu.vn)
# Kmeans cluster on hand-written digits data

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import spectral_clustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import load_digits

# load data
digits = load_digits()
n_digits = 10

# label using spectral cluster
graph = cosine_similarity(digits.data)
labels = spectral_clustering(graph, n_clusters=n_digits)

# result:
print("Result")
df = pd.DataFrame({'Labels':labels,'Truth labels':digits.target})
ct = pd.crosstab(df['Labels'],df['Truth labels'])
print(ct)
print("-----------------------------------------------------------------")

# visualize
pca_converter = PCA(n_components = 2)
# convert digits data to 2D points
data_2d = pca_converter.fit_transform(digits.data)
plt.scatter(data_2d[:,0], data_2d[:,1], c=labels)
plt.show()