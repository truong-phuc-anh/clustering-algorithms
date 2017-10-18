
# Created by Truong Phuc Anh (14520040@gm.uit.edu.vn)
# 16/10/2017
# Kmeans and Spetral cluster on faces data with HOG feature space

import numpy
import matplotlib.pyplot as plt
import os
from sklearn                    import datasets
from skimage.feature            import hog
from comparison                 import cluster_and_compare

# loads data set
print('Loading data set...')
faces = datasets.fetch_lfw_people(min_faces_per_person = 40) # parameter is used for get a smaller dataset

# settings for hog
hog_images = []
hog_images_file_path = '././data/faces/hog_images.npz'

# if hog feature has been computed before, use it
if os.path.isfile(hog_images_file_path):
    # loading hog feature from saved file
    print('Loading hog feature from ' + hog_images_file_path + '...')
    npz_file = numpy.load(hog_images_file_path)
    for file in npz_file.files:
        hog_images.append(npz_file[file])
# if it has not been computed.
else:
    # computes hog feature
    print('Calculating hog feature...')
    for image in faces.images:
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True)
        hog_images.append(hog_image.flatten()) # save as 1D Array
    # saves it to file for the next time
    print('Saving hog to ' + hog_images_file_path + '...')
    numpy.savez(hog_images_file_path, *[hog_images[i] for i in range(len(hog_images))])

data = hog_images
n_samples, n_features = numpy.shape(data)
n_clusters = len(numpy.unique(faces.target))
labels_true = faces.target

print ('n_samples       : %i' % n_samples)
print ('n_cluster       : %i' % n_clusters)
print ('feature space   : %i' % n_features)

# 2. Clustering and evaluating
cluster_and_compare(n_clusters, data, labels_true)