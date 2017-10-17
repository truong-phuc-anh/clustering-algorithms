# Created by Truong Phuc Anh (14520040@gm.uit.edu.vn)
# Simple implementation of Kmeans cluster

import matplotlib.pyplot as plt
import sklearn

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. Create data set
print "1. Creating data set..."
n_samples = 2000
random_state = 1500
n_clusters = 2
data, labels_true = make_blobs(n_samples=n_samples, n_features=n_clusters, centers=n_clusters, random_state=random_state)
print "Done"

# 2. Cluster with kmeans
print "2. Clustering..."
cluster = KMeans(n_clusters=n_clusters, random_state=random_state)
labels_pred = cluster.fit_predict(data)
print "Done"

# 3. Calculate evaluation using NMI (Normalized Mutual Infomation)
print "3. Calculating evaluation using NMI (Normalized Mutual Infomation)..."
NMI = sklearn.metrics.normalized_mutual_info_score(labels_pred, labels_true)
print "NMI = ", NMI
print "Done"

# 4. Visualize 
print "4. Visualizing..."
centers = cluster.cluster_centers_ # get centers of clusters
plt.figure('labels predit')
plt.title('Labels predited with kmeans')
plt.scatter(data[:, 0], data[:, 1], c=labels_pred)
plt.scatter(centers[:,0], centers[:, 1])

plt.figure('labels true')
plt.title('Labels true')
plt.scatter(data[:, 0], data[:, 1], c=labels_true)

plt.show()

