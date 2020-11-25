from numba.cuda.api import mapped
import pandas as pd
import umap
import numpy as np
import copy
# from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def get_umapT(dfHH,ratio=0.8,dimPCA=6, ftr=None, isPlot=False):
    if ratio is not None: dfHH = dfHH[dfHH['ra']<ratio]  
    if ftr is None: ftr = dfHH.columns[:dimPCA]
    try: df_umap=dfHH[ftr]
    except: df_umap=dfHH[list(map(str,ftr))]
    umapT = umap.UMAP(n_components=2,min_dist=0.0,n_neighbors=50, random_state=227)
    umapT.fit(df_umap)
    if isPlot: 
        umap_result = umapT.transform(df_umap)
        plt.scatter( umap_result[:,0], umap_result[:,1],alpha=0.7,s=10, color='k', marker="+")
    return umapT





def get_umap_pd(dfHH,dimPCA=6, ftr=None, isPlot=False):
    if ftr is None: ftr = dfHH.columns[:dimPCA]
    try: df_umap=dfHH[ftr]
    except: df_umap=dfHH[list(map(str,ftr))]
    umapT = umap.UMAP(n_components=2,min_dist=0.0,n_neighbors=50, random_state=227)
    umap_result = umapT.fit_transform(df_umap.values)
    dfHH['u1'] = umap_result[:,0]
    dfHH['u2'] = umap_result[:,1]
    if isPlot: sns.scatterplot('u1','u2',data=dfHH,alpha=0.7,s=10, color='k', marker="+")
    return umapT

def get_kmean_lbl(dfHH, N_cluster, u1 = 'u1', u2 = 'u2'):
    umap_result = dfHH.loc[:,[u1, u2 ]].values
    kmap = KMeans(n_clusters=N_cluster,n_init=30, algorithm='elkan',random_state=227)
    kmap.fit(umap_result, sample_weight = None)
    dfHH[f'C{N_cluster}'] = kmap.labels_ + 1 
    return kmap

def get_mapped(dfNorm,base,umapT):
    dfNormB = dfNorm * base
    mapped=umapT.transform(dfNormB)   
    return mapped

