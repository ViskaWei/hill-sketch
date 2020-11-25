# import os, sys
# import getpass
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns
# import copy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

########################### Loading #######################################

def load_data(DATASET, name='k', isVol=True):
    columns = [f'{name}{i}' for i in range(1,37)]
    if isVol: columns = columns + ['vol']
    if DATASET[-3:]=='csv':
        data = pd.read_csv(DATASET, names = columns)    
    elif DATASET[-4:]=='xlsx':
        data = pd.read_excel(DATASET, names = columns)    
    else:
        raise 'can only read csv or xlsx file'
    keep_columns = data.columns[(data.sum()!=0)]
    print(keep_columns)
    return data[keep_columns], keep_columns 

########################### Preprocessing #######################################

def get_pca(mat, dim=6):
    pca = PCA(n_components=dim)    
    matPCA=pca.fit_transform(mat)    
    print(matPCA.shape)
    return matPCA

def get_tsne(matPCA):
    matTSNE = TSNE(n_components=2).fit_transform(matPCA)
    print(matTSNE.shape)
    return matTSNE

def get_cluster(outTSNE, nCluster):
    kmap = KMeans(n_clusters=nCluster,n_init=30, algorithm='elkan',random_state=227)
    kmap.fit(outTSNE, sample_weight = None)
    cluster_id = kmap.labels_ + 1
    min_dist = np.min(cdist(outTSNE, kmap.cluster_centers_, 'euclidean'), axis=1)    
    return cluster_id, min_dist, kmap

def run_hill_simple(DATASET, nCluster=10, PCA_dim = 6,isCenter=True):
    data,keep_columns = load_data(DATASET)
    if isCenter: 
        dataPREPRO = data - data.mean().mean() 
    else:
        dataPREPRO = data
    matPCA = get_pca(dataPREPRO,dim=PCA_dim)
    matTSNE = get_tsne(matPCA)    
    cluster_id, min_dist, kmap = get_cluster(matTSNE, nCluster)
    data['t1'] = matTSNE[:,0]
    data['t2'] = matTSNE[:,1]    
    data[f'C{nCluster}'] = cluster_id
    data[f'M{nCluster}'] = min_dist
    grouped = data.groupby([f'C{nCluster}'])
    cid = grouped[f'M{nCluster}'].idxmin().values
    print('center Id:', cid)
    cMat = data.iloc[cid][keep_columns]
    return data,kmap,cMat

########################### Plotting #######################################

def plot_data(data,kmap):
    sns.scatterplot(
            x='t1', y='t2',
            hue= kmap.labels_+1 , marker='x',s=5,
            palette=sns.color_palette("muted", kmap.n_clusters),
            data=data,
            legend="full")
    plt.scatter(kmap.cluster_centers_[:,0],kmap.cluster_centers_[:,1], c='r')  


########################### Saving #######################################
def save_cluster_ids(data, nCluster, outDir=None,name='kMat'):
    if outDir is None: outDir = './'
    lbl = data[f'C{nCluster}']
    for i in range(1,nCluster+1):
        c = np.where(lbl==i)[0]+1   
        print(f'Cluster{i}: {c[:3]}..')
        np.savetxt(f'{outDir}{name}_C{nCluster}_c{i}.txt', c, fmt="%d")    
def save_centers(cMat,nCluster,outDir=None,name='cMat'):
    if outDir is None: outDir = './'
    cMat.to_csv(f'{outDir}{name}_C{nCluster}.csv') 