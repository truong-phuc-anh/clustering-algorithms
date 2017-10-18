from time                   import time         # calculating excution time
from sklearn                import metrics      # calculating evaluation

def evaluate_clustering(cluster, name, data, labels_true):
    """Evalueate a clustering algorithm with some measure.
    
    Measures : Homogeneity, Completeness, V-measure, Adjusted Random, Adjusted Mutual Information.

    Parameters
    ----------
    cluster : one of sklearn.cluster object
        The cluster object

    name : string
        The name of cluster object used for displaying only.

    data : array
        The list of samples need to be clustered

    labels_true : 1D array (size = number of samples)
        The ground truth labels

    Returns
    -------
    labels_pred: labels of data after clustering
    """

    # Cluster
    start_time = time()
    labels_pred = cluster.fit_predict(data)
    execution_time = time() - start_time
    # Evaluate
    evaluate_with_predited_labels(labels_true, labels_pred, execution_time, name)

    return labels_pred

def evaluate_with_predited_labels(labels_true, labels_pred, execution_time, name):
    """Evalueate a predited labels with some measure.
    
    Measures : Homogeneity, Completeness, V-measure, Adjusted Random, Adjusted Mutual Information.

    Parameters
    ----------
    labels_true : 1D array (size = number of samples)
        The ground truth labels
        
    labels_pred : 1D array (size = number of samples)
        The labels after aplying clustering

    execution_time : float
        The execution time of clustering task

    name : string
        The name of clustering algorithm
    """
    print('%-9s\t%.2fs\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
        % (name, execution_time,
            metrics.homogeneity_score(labels_true, labels_pred),
            metrics.completeness_score(labels_true, labels_pred),
            metrics.v_measure_score(labels_true, labels_pred),
            metrics.adjusted_rand_score(labels_true, labels_pred),
            metrics.adjusted_mutual_info_score(labels_true,  labels_pred)))