import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import fetch_lfw_people
from skimage.feature import local_binary_pattern

faces = fetch_lfw_people()
plt.matshow(faces.images[0])

lbp_image = local_binary_pattern(faces.images[0], 10, 10)
plt.matshow(lbp_image)
plt.show()