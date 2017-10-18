
# Created by Truong Phuc Anh (14520040@gm.uit.edu.vn)
# 06/10/2017
# Kmeans and Spetral cluster on faces data

import numpy
import matplotlib.pyplot as plt
import os
from sklearn                    import datasets
from skimage.feature            import local_binary_pattern
from comparison                 import cluster_and_compare

# 1. Loading data set
print('Loading data set...')
faces = datasets.fetch_lfw_people(min_faces_per_person = 40) # parameter is used for get a smaller dataset

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
# if it has not been computed.
else:
    # computes lbp feature
    print('Calculating lbp feature...')
    for image in faces.images:
        lbp = local_binary_pattern(image, n_points, radius) # compute lbs feature, 2D Array
        lbp_images.append(lbp.flatten()) # save as 1D Array
    # saves it to file for the next time
    print('Saving LBP to ' + lbp_images_file_path + '...')
    numpy.savez(lbp_images_file_path, *[lbp_images[i] for i in range(len(lbp_images))])

data = lbp_images
n_samples, n_features = numpy.shape(data)
n_clusters = len(numpy.unique(faces.target))
labels_true = faces.target

print ('n_samples       : %i' % n_samples)
print ('n_cluster       : %i' % n_clusters)
print ('feature space   : %i' % n_features)

# 2. Clustering and evaluating
cluster_and_compare(n_clusters, data, labels_true)