import pandas as pd
import umap
import numpy as np
import copy
# from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt




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

def get_umapT(dfHH,dimPCA=6, ftr=None, isPlot=False):
    if ftr is None: ftr = dfHH.columns[:dimPCA]
    try: df_umap=dfHH[ftr]
    except: df_umap=dfHH[list(map(str,ftr))]
    umapT = umap.UMAP(n_components=2,min_dist=0.0,n_neighbors=50, random_state=227)
    umap_result = umapT.fit(df_umap.values)
    if isPlot: plt.scatter( umap_result[:,0], umap_result[:,1])
    return umapT

def get_kmean_lbl(dfHH, N_cluster, u1 = 'u1', u2 = 'u2'):
    umap_result = dfHH.loc[:,[u1, u2 ]].values
    kmap = KMeans(n_clusters=N_cluster,n_init=30, algorithm='elkan',random_state=227)
    kmap.fit(umap_result, sample_weight = None)
    dfHH[f'C{N_cluster}'] = kmap.labels_ + 1 
    return kmap

def get_df_mapped(dfHH,umapT,N_dim):
    # lb,ub=int(HH_pd['freq'][0]*lbr),int(HH_pd['freq'][0])
    # HH_pdc=HH_pd[HH_pd['freq']>lb]
    # print(f'lpdc: {len(HH_pdc)} lpd: {len(HH_pd)} ub:{ub} lb:{lb} HHratio:{lbr}')
    u_da=umapT.transform(dfHH[list(range(N_dim))])   
    dfHH['u1']=u_da[:,0]
    dfHH['u2']=u_da[:,1]
    sns.scatterplot('u1','u2',data=dfHH,alpha=0.7,s=10, color='k', marker="+")
    return dfHH
