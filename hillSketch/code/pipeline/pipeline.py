import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from code.data.dataset import load_data
from code.data.bulk import prepro_data
from code.data.stream import get_encode_stream
from code.sketch.exact import get_HH_pd
from code.cluster.project import get_matUMAP



def run_hill_sketch(DATASET, nCluster, dimPCA, base, name='k', isVol=True, isCenter=True, \
                dtype='uint64', offset=1,isPlot=False):
    data,keep_columns,vol = load_data(DATASET,name=name, isVol=isVol)
    dfNorm = prepro_data(data, isCenter=isCenter, dimPCA=dimPCA,isPlot=isPlot,method='minmax')
    stream = get_encode_stream(dfNorm, base, dtype=dtype)
    dfHH = get_HH_pd(stream,base,dimPCA, dtype)
    matUMAP,cluster_id, min_dist, kmap = get_matUMAP(dfNorm,dfHH,nCluster, base,\
                                                 ratio=0.8,dimPCA=dimPCA, ftr=None, isPlot=isPlot)
    data[f'C{nCluster}'] = cluster_id
    data[f'M{nCluster}'] = min_dist
    if isVol: data['vol'] = vol
    grouped = data.groupby([f'C{nCluster}'])
    cid = grouped[f'M{nCluster}'].idxmin().values
    cMat = data.iloc[cid][keep_columns]
    print('center Id:', cid)
    data['t1'] = matUMAP[:,0]
    data['t2'] = matUMAP[:,1]    
    if isVol:
        cluster_vol = grouped['vol'].sum().values
        print('cluster volumn sum:', cluster_vol.sum().round(3)) 
        cMat['vol'] = cluster_vol
    cMat.index+=offset
    return data,kmap,cMat

def plot_data(data,kmap, cut = 2000, rng=50):
    f, axes = plt.subplots(2,2, figsize=(16,10))
    sns.scatterplot(ax=axes[0][0],
            x='t1', y='t2',
            hue= kmap.labels_+1 , marker='x',s=5,
            palette=sns.color_palette("muted", kmap.n_clusters),
            data=data,
            legend="full")
    axes[0][0].scatter(kmap.cluster_centers_[:,0],kmap.cluster_centers_[:,1], c='r') 
    axes[0][1].scatter(list(range(data.shape[0])), data[f'C{kmap.n_clusters}'])
    axes[1][1].scatter(list(range(data.shape[0])), data[f'C{kmap.n_clusters}'])
    axes[1][1].set_xlim(cut-rng,cut+rng)



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