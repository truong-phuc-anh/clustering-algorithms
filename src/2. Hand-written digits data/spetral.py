# Created by Truong Phuc Anh (14520040@gm.uit.edu.vn)
# Kmeans cluster on hand-written digits data

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import spectral_clustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import load_digits

# load data
digits = load_digits()
n_digits = 10

# label using spectral cluster
graph = cosine_similarity(digits.data, Y=None, dense_output=True)
labels = spectral_clustering(graph, n_clusters=n_digits, eigen_solver='arpack')

# result:
print("Result")
df = pd.DataFrame({'Labels':labels,'Truth labels':digits.target})
ct = pd.crosstab(df['Labels'],df['Truth labels'])
print(ct)
print("-----------------------------------------------------------------")

# Ex. 
num = 3
print 'label predit: ', labels[num]
print 'expected label: ', digits.target[num]
plt.gray()
plt.matshow(digits.images[num])
plt.show()
