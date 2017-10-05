# Created by Truong Phuc Anh (14520040@gm.uit.edu.vn)
# Kmeans cluster on hand-written digits data

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits

# load data
digits = load_digits()

# label using kmeans cluster
n_digits = 10
kmeans = KMeans(n_clusters = n_digits)
labels = kmeans.fit_predict(digits.data)

# result:
print("Result")
df = pd.DataFrame({'Labels':labels,'Truth labels':digits.target})
ct = pd.crosstab(df['Labels'],df['Truth labels'])
print(ct)
print("-----------------------------------------------------------------")

# Ex. 
num = 2
print 'label predit: ', labels[num]
print 'expected label: ', digits.target[num]
plt.gray()
plt.matshow(digits.images[num])
plt.show()
