import matplotlib.pyplot        as plt
from time                       import time
from sklearn                    import datasets
from sklearn.cluster            import KMeans, spectral_clustering, DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise   import cosine_similarity
from evaluation                 import evaluate_clustering, evaluate_with_predited_labels
from sklearn.decomposition      import PCA

def cluster_and_compare(n_clusters, data, labels_true):
    print(75 * '-')
    print('cluster\t\ttime\thomo\tcompl\tv-meas\tARI\tAMI')

    kmeans_cluster = KMeans(n_clusters = n_clusters)
    kmeans_labels = evaluate_clustering(kmeans_cluster, "kmeans", data, labels_true)

    start_time = time()
    graph = cosine_similarity(data)
    spectral_labels = spectral_clustering(graph, n_clusters = n_clusters)
    execution_time = time() - start_time
    evaluate_with_predited_labels(labels_true, spectral_labels, execution_time, "spectral")

    dbscan_cluster = DBSCAN(eps = 0.0595, min_samples = 10, metric='cosine')
    dbscan_labels = evaluate_clustering(dbscan_cluster, "DBSCAN", data, labels_true)

    agg_cluster = AgglomerativeClustering(n_clusters = n_clusters)
    agg_labels = evaluate_clustering(agg_cluster, "Agglomerative", data, labels_true)
    print(75 * '-')

    pca_converter = PCA(n_components = 2)
    data = pca_converter.fit_transform(data)

    fig = plt.figure()
    ax = fig.add_subplot(1, 5, 1)
    ax.scatter(data[:,0], data[:,1], c = labels_true)
    ax.set_title('data')

    ax = fig.add_subplot(1, 5, 2)
    ax.scatter(data[:,0], data[:,1], c = kmeans_labels)
    ax.set_title('kmeans')

    ax = fig.add_subplot(1, 5, 3)
    ax.scatter(data[:,0], data[:,1], c = spectral_labels)
    ax.set_title('spectral')

    ax = fig.add_subplot(1, 5, 4)
    ax.scatter(data[:,0], data[:,1], c = dbscan_labels)
    ax.set_title('dbscan')

    ax = fig.add_subplot(1, 5, 5)
    ax.scatter(data[:,0], data[:,1], c = agg_labels)
    ax.set_title('agglomerative')

    plt.show()


