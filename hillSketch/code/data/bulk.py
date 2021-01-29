# from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def prepro_data(data, isCenter=True, dimPCA=6,isPlot=True,method='minmax'):
    if isCenter: 
        dataPREPRO = data - data.mean().mean() 
    else:
        dataPREPRO = data
    matPCA = get_pca(data, dim= dimPCA)
    # matPCA = get_SVD(dataPREPRO, dimPCA = dimPCA, isPlot=isPlot)   
    matNorm = get_norm(matPCA, method=method, isPlot=isPlot)   
    # dfRebin = get_rebin(matNorm,base)
    return matNorm

def get_pca(mat, dim=6):
    pca = PCA(n_components=dim, random_state = 907)    
    matPCA=pca.fit_transform(mat)    
    print(matPCA.shape)
    return matPCA

######################## PCA ###########################
def get_SVD(data, dimPCA = 6, isPlot=False):
    cov = data.T.dot(data)    
    if isPlot: plt.matshow(cov)
    pc = get_pc(cov, dimPCA)
    matPCA = data.dot(pc)
    if isPlot: plt.matshow(matPCA.T , aspect='auto')
    return matPCA

def get_pc(cov, pca_comp):
    print(f"=============== PCA N_component: {pca_comp} ===============")
    u,s,v = np.linalg.svd(cov)
#     assert np.allclose(u, v.T)
    print('Explained Variance Ratio', np.round(s/sum(s),3))
    pc = u[:,:pca_comp]
    return pc

######################## NORM ###########################
def get_norm(matPCA, method='minmax', isPlot=False):
    if method=='minmax':
        try: 
            vmin,vmax = matPCA.min(), matPCA.max()
        except:
            vmin,vmax = np.min(matPCA), np.max(matPCA)

        matNorm =  (matPCA - vmin)/(vmax - vmin)
        if isPlot: plt.matshow(matNorm.T, aspect='auto')        
    else:
        raise 'select or implement norm method'
    return matNorm

# def get_rebin(dfNorm, base):
#     dfRebin=(dfNorm*(base-1)).round()
#     assert (dfRebin.min().min()>=0) & (dfRebin.max().max()<=base-1)
#     return dfRebin
