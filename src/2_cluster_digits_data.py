# Created by Truong Phuc Anh (14520040@gm.uit.edu.vn)
# Kmeans cluster on hand-written digits data

import numpy
import matplotlib.pyplot        as plt
from sklearn                    import datasets
from comparison                 import cluster_and_compare

# 1. Loading data set
digits = datasets.load_digits()
data = digits.data
n_samples, n_features = data.shape
n_clusters = len(numpy.unique(digits.target)) 
labels_true = digits.target

print ('n_samples       : %i' % n_samples)
print ('n_cluster       : %i' % n_clusters)
print ('feature space   : %i' % n_features)

# 2. Clustering and evaluating
cluster_and_compare(n_clusters, data, labels_true)