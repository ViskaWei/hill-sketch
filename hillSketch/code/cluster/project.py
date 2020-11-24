import pandas as pd
import umap
import numpy as np
import copy
# from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.cluster import KMeans

def get_umap_pd(dfHH, ftr=None, isPlot=False):
    if ftr is None: ftr = dfHH.columns[:-4]
    try: df_umap=dfHH[ftr]
    except: df_umap=dfHH[list(map(str,ftr))]
    umapT = umap.UMAP(n_components=2,min_dist=0.0,n_neighbors=50, random_state=1178)
    umap_result = umapT.fit_transform(df_umap.values)
    dfHH['u1'] = umap_result[:,0]
    dfHH['u2'] = umap_result[:,1]
    if isPlot: sns.scatterplot('u1','u2',data=dfHH,alpha=0.7,s=10, color='k', marker="+")
    return umapT

def get_kmean_lbl(dfHH, N_cluster, u1 = 'u1', u2 = 'u2'):
    umap_result = dfHH.loc[:,[u1, u2 ]].values
    kmap = KMeans(n_clusters=N_cluster,n_init=30, algorithm='elkan',random_state=1178)
    kmap.fit(umap_result, sample_weight = None)
    dfHH[f'C{N_cluster}'] = kmap.labels_ + 1 
    return kmap