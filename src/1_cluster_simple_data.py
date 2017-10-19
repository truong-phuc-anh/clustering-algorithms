# Created by Truong Phuc Anh (14520040@gm.uit.edu.vn)
# Simple implementation of Kmeans cluster

import matplotlib.pyplot    as plt
from sklearn.cluster        import KMeans
from sklearn.datasets       import make_blobs
from evaluation             import evaluate_clustering

# 1. Creating data set
n_samples = 2000
random_state = 1500
n_clusters = 2
n_features = 2
data, labels_true = make_blobs(n_samples = n_samples, n_features = n_clusters, centers = n_clusters, random_state = random_state)

print ('n_samples       : %i' % n_samples)
print ('n_cluster       : %i' % n_clusters)
print ('feature space   : %id' % n_features)

# 2. Clustering and evaluating
print(75 * '-')
print('cluster\t\ttime\thomo\tcompl\tv-meas\tARI\tAMI')
kmeans_cluster = KMeans(n_clusters = n_clusters)
labels_pred = evaluate_clustering(kmeans_cluster, "kmeans", data, labels_true)
print(75 * '-')

# 3. Visualizing result
plt.figure()
plt.title('True labels')
plt.scatter(data[:,0], data[:,1], c = labels_true)

plt.figure()
plt.title("kmeans")
plt.scatter(data[:,0], data[:,1], c = labels_pred)

plt.show()