# Created by Truong Phuc Anh (14520040@gm.uit.edu.vn)
# Simple 

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# create input data (unlable)
n_samples = 2500
random_state = 1500
n_clusters = 2
x, y = make_blobs(n_samples=n_samples, n_features=n_clusters, centers=n_clusters, random_state=random_state)

# lable using KMeans
result = KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(x)

# visualize
plt.figure()
plt.scatter(x[:, 0], x[:, 1], c=result)

plt.show()