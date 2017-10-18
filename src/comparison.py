from time                       import time
from sklearn                    import datasets
from sklearn.cluster            import KMeans, spectral_clustering, DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise   import cosine_similarity
from evaluation                 import evaluate_clustering, evaluate_with_predited_labels

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

    dbscan_cluster = DBSCAN()
    dbscan_labels = evaluate_clustering(dbscan_cluster, "DBSCAN", data, labels_true)

    agg_cluster = AgglomerativeClustering(n_clusters = n_clusters)
    agg_labels = evaluate_clustering(agg_cluster, "Agglomerative", data, labels_true)