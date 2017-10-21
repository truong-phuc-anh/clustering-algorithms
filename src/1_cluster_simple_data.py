# Created by Truong Phuc Anh (14520040@gm.uit.edu.vn)
# Simple implementation of Kmeans cluster

import numpy
import matplotlib.pyplot    as plt
from time                   import time
from sklearn.cluster        import KMeans
from sklearn.datasets       import make_blobs, make_circles, make_s_curve
from evaluation             import evaluate_clustering

# 1. Kmeans is sensitive to initialization
n_samples = 5000
n_features = 2
n_clusters = 3
data, ground_truth = make_blobs(n_samples = n_samples, n_features = n_features, centers = n_samples % 3, center_box=(-20.0, 20.0), random_state=1500)
print ('n_samples       : %i' % n_samples)
print ('n_cluster       : %i' % n_clusters)
print ('feature space   : %id' % n_features)
fig = plt.figure()
ax = fig.add_subplot(2, 5, 1)
ax.scatter(data[:,0], data[:,1])
ax.set_title('data')
for i in xrange (9):
    kmeans_cluster = KMeans(n_clusters = n_clusters)
    start_time = time()
    kmeans_cluster.fit(data)
    exec_time = time() - start_time
    print('%-9s\t%.2fs' % (i, exec_time))
    ax = fig.add_subplot(2, 5, i + 2)
    ax.scatter(data[:,0], data[:,1], c = kmeans_cluster.labels_)
    ax.set_title('time: %.2fs' % (exec_time))

# 2. Only finds sphericalc clusters

data, something = make_circles(n_samples=1000)
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.scatter(data[:,0], data[:,1])
ax.set_title('data')

kmeans_cluster = KMeans(n_clusters = 2)
start_time = time()
kmeans_cluster.fit(data)
exec_time = time() - start_time
ax = fig.add_subplot(1, 2, 2)
ax.scatter(data[:,0], data[:,1], c = kmeans_cluster.labels_)
ax.set_title('time: %.2fs' % (exec_time))

plt.show()