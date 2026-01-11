from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

def perform_clustering(features, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(features)

def get_metrics(features, true_labels, pred_labels):
    return {
        "Silhouette": silhouette_score(features, pred_labels),
        "ARI": adjusted_rand_score(true_labels, pred_labels),
        "NMI": normalized_mutual_info_score(true_labels, pred_labels)
    }
