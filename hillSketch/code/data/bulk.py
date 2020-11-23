# from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def prepro_data(data, isCenter=True, dimPCA=6,isPlot=True,method='minmax'):
    if isCenter: 
        dataPREPRO = data - data.mean().mean() 
    else:
        dataPREPRO = data
    matPCA = get_PCA(dataPREPRO, dimPCA = dimPCA, isPlot=isPlot)   
    matNorm = get_norm(matPCA, method=method, isPlot=isPlot)    
    return matNorm

######################## PCA ###########################
def get_PCA(data, dimPCA = 6, isPlot=False):
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
        vmin,vmax = matPCA.min(), matPCA.max()
        matNorm =  (matPCA - vmin)/(vmax - vmin)
        if isPlot: plt.matshow(matNorm.T, aspect='auto')        
    else:
        raise 'select or implement norm method'
    return matNorm