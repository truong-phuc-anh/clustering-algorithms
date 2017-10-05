# Created by Truong Phuc Anh (14520040@gm.uit.edu.vn)
# Kmeans cluster on hand-written digits data

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
# load data
digits = load_digits()

# label using kmeans cluster
n_digits = 10
cluster = KMeans(n_clusters = n_digits)
labels = cluster.fit_predict(digits.data)

# result:
print("Result")
df = pd.DataFrame({'Labels':labels,'Truth labels':digits.target})
print(df)
ct = pd.crosstab(df['Labels'],df['Truth labels'])
print(ct)
print("-----------------------------------------------------------------")

# visualize
pca_converter = PCA(n_components = 2)
# convert digits data to 2D points
data_2d = pca_converter.fit_transform(digits.data)
plt.scatter(data_2d[:,0], data_2d[:,1], c=labels)
plt.show()