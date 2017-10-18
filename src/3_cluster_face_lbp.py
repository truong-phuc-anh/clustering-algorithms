
# Created by Truong Phuc Anh (14520040@gm.uit.edu.vn)
# 06/10/2017
# Kmeans and Spetral cluster on faces data

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

# settings for LBP
radius = 3
n_points = 8 * radius
lbp_images = []
lbp_images_file_path = '././data/faces/lbp_images.npz'

# if lbp feature has been computed before, use it
if os.path.isfile(lbp_images_file_path):
    # loading lbp feature from saved file
    print('Loading lbp feature from ' + lbp_images_file_path + '...')
    npz_file = numpy.load(lbp_images_file_path)
    for file in npz_file.files:
        lbp_images.append(npz_file[file])
    print( 'Loaded ' + str(len(lbp_images)) + ' lbp images\n\n' )

# if it has not been computed.
else:
    # computes lbp feature
    print('Computting lbp feature...')
    for image in faces.images:
        lbp = skimage.feature.local_binary_pattern(image, n_points, radius) # compute lbs feature, 2D Array
        lbp_images.append(lbp.flatten()) # save as 1D Array
    print('Computed lbp feature\n\n')
    print('Num lbp images: ' + str(len(lbp_images)) + '\n\n')
    # saves it to file for the next time
    print('Saving LBP to ' + lbp_images_file_path + '...')
    numpy.savez(lbp_images_file_path, *[lbp_images[i] for i in range(len(lbp_images))])
    print('Saved LBP to ' + lbp_images_file_path + '\n\n')

# config for cluster
n_persons = 60 # from http://vis-www.cs.umass.edu/lfw/#informatio
# clusters 
print('Kmeans Clustering...')
cluster = sklearn.cluster.KMeans(n_clusters = n_persons, random_state = 1500)
kmeans_labels = cluster.fit_predict(lbp_images)
print('Clustered\n\n')

print('Spetral Clustering...')
graph = cosine_similarity(lbp_images)
spetral_labels = sklearn.cluster.spectral_clustering(graph, n_clusters = n_persons)
print('Clustered\n\n')

print('-----------------------------------------------------------------------------------------------------------')

# show
plt.show()