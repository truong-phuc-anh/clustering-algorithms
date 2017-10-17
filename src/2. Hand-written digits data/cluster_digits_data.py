# Created by Truong Phuc Anh (14520040@gm.uit.edu.vn)
# Kmeans cluster on hand-written digits data

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import spectral_clustering
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load data set
print "1. Loading data set"
digits = load_digits()
print "Done"

# 2. Clustering with Kmeans
print "2. Clustering with Kmeans..."
n_digits = 10
kmeans_cluster = KMeans(n_clusters = n_digits)
kmeans_labels = kmeans_cluster.fit_predict(digits.data)
print "Done"

# 3. Cluster with Spectral
print "3. Clustering with Spectral..."
graph = cosine_similarity(digits.data)
spectral_labels = spectral_clustering(graph, n_clusters=n_digits)
print "Done"

# 4. Show result:
# just showing the comparasion between true labels and predited label 
# can't calculate evaluation because the true labels is value of this number, but the predited label is just like a group name
print "4. Showing result..."
print "     4.1.Kmeans true labels and predited labels comparison"
df = pd.DataFrame({'Labels':kmeans_labels,'Truth labels':digits.target})
ct = pd.crosstab(df['Labels'],df['Truth labels'])
print(ct)
print("-----------------------------------------------------------------")
print "     4.2.Spectral true labels and predited labels comparison"
df = pd.DataFrame({'Labels':spectral_labels,'Truth labels':digits.target})
ct = pd.crosstab(df['Labels'],df['Truth labels'])
print(ct)
print("-----------------------------------------------------------------")
print("Done")

# 5.Visualize with PCA compression
# pca to 2d so we don't really get what we need, but it sill ok (i mean we still can see some comparison in this result)
print "5. Visualizing with PCA compression..."
pca_converter = PCA(n_components = 2)
# convert digits data to 2D points
data_2d = pca_converter.fit_transform(digits.data)

plt.figure("Kmeans labels")
plt.title("Kmeans labels")
plt.scatter(data_2d[:,0], data_2d[:,1], c=kmeans_labels)

plt.figure("Spectral labels")
plt.title("Spectral labels")
plt.scatter(data_2d[:,0], data_2d[:,1], c=spectral_labels)

plt.figure("True labels")
plt.title("True labels")
plt.scatter(data_2d[:,0], data_2d[:,1], c=digits.target)

plt.show()
print "Done"