import matplotlib.pyplot    as plt              # visualizing result
from time                   import time         # calculating excution time
from sklearn                import metrics      # calculating evaluation
from sklearn.decomposition  import PCA          # compression data to 2d

def evaluate_clustering(cluster, name, data, labels_true):
    print(75 * '-')
    print('cluster\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI')

    # Cluster
    start_time = time()
    cluster.fit(data)

    # Evaluate
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
        % (name, (time() - start_time), cluster.inertia_,
            metrics.homogeneity_score(labels_true, cluster.labels_),
            metrics.completeness_score(labels_true, cluster.labels_),
            metrics.v_measure_score(labels_true, cluster.labels_),
            metrics.adjusted_rand_score(labels_true, cluster.labels_),
            metrics.adjusted_mutual_info_score(labels_true,  cluster.labels_)))
    print(75 * '-')

    # Visualize
    pca_converter = PCA(n_components = 2)
    data_2d = pca_converter.fit_transform(data)

    plt.figure()
    plt.title(name)
    plt.scatter(data_2d[:,0], data_2d[:,1], c=cluster.labels_)

    plt.figure()
    plt.title('True labels')
    plt.scatter(data_2d[:,0], data_2d[:,1], c=labels_true)

    plt.show()
