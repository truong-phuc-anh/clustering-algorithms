
# Created by Truong Phuc Anh (14520040@gm.uit.edu.vn)
# 16/10/2017
# Kmeans and Spetral cluster on faces data with HOG feature space

import matplotlib.pyplot as plt
import numpy
import sklearn
import sklearn.datasets
import sklearn.cluster
import skimage
import skimage.feature
import skimage.color
import os
from sklearn.metrics.pairwise import cosine_similarity

print('-----------------------------------------------------------------------------------------------------------')
print('\n\n')

# loads data set
print('Loading data set...')
faces = sklearn.datasets.fetch_lfw_people(min_faces_per_person = 40) # parameter is used for get a smaller dataset
print('Loaded data set\n\n')

print('Num images: ' + str(len(faces.images)) + '\n\n')

# settings for hog
hog_images = []
hog_images_file_path = './././data/faces/hog_images.npz'

# if hog feature has been computed before, use it
if os.path.isfile(hog_images_file_path):
    # loading hog feature from saved file
    print('Loading hog feature from ' + hog_images_file_path + '...')
    npz_file = numpy.load(hog_images_file_path)
    for file in npz_file.files:
        hog_images.append(npz_file[file])
    print( 'Loaded ' + str(len(hog_images)) + ' hog images\n\n' )

# if it has not been computed.
else:
    # computes hog feature
    print('Computting hog feature...')
    for image in faces.images:
        fd, hog_image = skimage.feature.hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True)
        hog_images.append(hog_image.flatten()) # save as 1D Array
    print('Computed hog feature\n\n')
    print('Num hog images: ' + str(len(hog_images)) + '\n\n')
    # saves it to file for the next time
    print('Saving hog to ' + hog_images_file_path + '...')
    numpy.savez(hog_images_file_path, *[hog_images[i] for i in range(len(hog_images))])
    print('Saved hog to ' + hog_images_file_path + '\n\n')

# config for cluster
n_persons = 60 # from http://vis-www.cs.umass.edu/lfw/#informatio
# clusters 
print('Kmeans Clustering...')
cluster = sklearn.cluster.KMeans(n_clusters = n_persons, random_state = 1500)
kmeans_labels = cluster.fit_predict(hog_images)
print('Clustered\n\n')

print('Spetral Clustering...')
graph = cosine_similarity(hog_images)
spetral_labels = sklearn.cluster.spectral_clustering(graph, n_clusters = n_persons)
print('Clustered\n\n')

print('-----------------------------------------------------------------------------------------------------------')

# show
plt.show()