import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

def cluster( model, encoded_data, original_data ):
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
