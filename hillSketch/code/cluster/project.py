from numba.cuda.api import mapped
import pandas as pd
import umap
import numpy as np
import copy
# from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def get_matUMAP(dfNorm,dfHH, nCluster, base, ratio=0.8,dimPCA=6, ftr=None, isPlot=True):
    umapT = get_umapT(dfHH,ratio=ratio,dimPCA=dimPCA, ftr=ftr, isPlot=isPlot)
    matUMAP = get_mapped(dfNorm,base,umapT)
    cluster_id, min_dist, kmap = get_cluster(matUMAP, nCluster)
    return matUMAP,  cluster_id, min_dist, kmap
    

def get_umapT(dfHH,ratio=0.8,dimPCA=6, ftr=None, isPlot=False):
    if ratio is not None: dfHH = dfHH[dfHH['ra']<ratio]  
    if ftr is None: ftr = dfHH.columns[:dimPCA]
    try: df_umap=dfHH[ftr]
    except: df_umap=dfHH[list(map(str,ftr))]
    umapT = umap.UMAP(n_components=2,min_dist=0.0,n_neighbors=50, random_state=227)
    umapT.fit(df_umap)
    if isPlot: 
        umap_result = umapT.transform(df_umap)
        plt.figure()
        plt.scatter( umap_result[:,0], umap_result[:,1],alpha=0.7,s=10, color='k', marker="+")
    return umapT

def get_cluster(matUMAP, nCluster):
    kmap = KMeans(n_clusters=nCluster,n_init=30, algorithm='elkan',random_state=227)
    kmap.fit(matUMAP, sample_weight = None)
    cluster_id = kmap.labels_ + 1
    min_dist = np.min(cdist(matUMAP, kmap.cluster_centers_, 'euclidean'), axis=1)    
    return cluster_id, min_dist, kmap

def get_mapped(dfNorm,base,umapT):
    dfNormB = dfNorm * base
    matUMAP=umapT.transform(dfNormB)   
    return matUMAP

