import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples
from matplotlib.ticker import FixedLocator, FixedFormatter


def cluster( model, encoded_data, original_data, k ):
    model.fit(encoded_data)
    print( model, ':\n\t', silhouette_score(encoded_data, model.labels_) )
    centroids = np.array([original_data.iloc[np.where(model.labels_ == c)[0],:].mean(axis=0) for c in range(k)])
    return pd.DataFrame(centroids, columns = original_data.columns)

def cluster_extrema(cluster_centroids):
    lower_bound = np.quantile(cluster_centroids, .25, axis = 0)
    upper_bound = np.quantile(cluster_centroids, .75, axis = 0)
    return cluster_centroids.mask(cluster_centroids < lower_bound, 'L').mask(cluster_centroids > upper_bound, 'U')

def match_difference( pd_a, pd_b, pd_matches ):
    return pd_a.loc[pd_matches.iloc[:,0],:].subtract(pd_b.loc[pd_matches.iloc[:,1],:].values, axis = 1)


   
def plot_values( cluster_list, values ):
    plt.figure(figsize=(8, 3))
    plt.plot(cluster_list, values, "bo-")
    plt.xlabel("$N Clusters$", fontsize=14)
    plt.ylabel("Metric", fontsize=14)
    plt.axis([min(cluster_list)-.5, max(cluster_list)+.5, min(values), max(values)])
    plt.show()
    
    

def explore_kmeans( cluster_list, data ):
    kmeans_per_k = [KMeans(n_clusters=k).fit(data)
                for k in cluster_list]
    inertias = [model.inertia_ for model in kmeans_per_k]
    silhouette_scores = [silhouette_score(data, model.labels_)
                     for model in kmeans_per_k]
    plot_values( cluster_list, inertias )
    plot_values( cluster_list, silhouette_scores )
    return kmeans_per_k


def explore_agglomerative( cluster_list, data ):
    models_per_n = [AgglomerativeClustering(n_clusters = n,linkage='ward').fit(data)
                for n in cluster_list]
    silhouette_scores = [silhouette_score(data, model.labels_)
                     for model in models_per_n]
    plot_values( cluster_list, silhouette_scores )
    return models_per_n

def plot_silhouettes( models, data ):
    ncols = 4
    nrows = int(np.ceil(len(models)/ ncols))

    plt.figure(figsize=(11, 9))
    for idx, model in enumerate(models):
        if isinstance(model, KMeans):
            k = model.cluster_centers_.shape[0]
        else:
            k = model.n_clusters_
        plt.subplot(nrows, ncols, idx + 1)

        y_pred = model.labels_
        silhouette_coefficients = silhouette_samples(data, y_pred)

        padding = data.shape[0] // 30
        pos = padding
        ticks = []
        for i in range(k):
            coeffs = silhouette_coefficients[y_pred == i]
            coeffs.sort()

            color = mpl.cm.Spectral(i / k)
            plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                              facecolor=color, edgecolor=color, alpha=0.7)
            ticks.append(pos + len(coeffs) // 2)
            pos += len(coeffs) + padding

        plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
        plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
        if k in range(2, 10, ncols):
            plt.ylabel("Cluster")

        if k in range((10-ncols), 10):
            plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.xlabel("Silhouette Coefficient")
        else:
            plt.tick_params(labelbottom=False)

        plt.axvline(x=silhouette_score(data, y_pred), color="red", linestyle="--")
        plt.title("$clusters={}$".format(k), fontsize=16)

    plt.show()