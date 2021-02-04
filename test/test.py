import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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
    keep_data = data[keep_columns]
    vol = keep_data.pop('vol') if isVol else None
    return keep_data, keep_data.columns, vol 

def prepro_data(data, isCenter=True, isBig =False):
    data = data.values
    dim = data.shape[1]
    if isCenter: 
        lim = np.max(abs(data))
        print(f'centering with bound {lim}')
        dataPREPRO = (data + lim)/(2*lim)
    else:
        dataPREPRO = data
    if not isBig:
        matPCA = get_PCA(dataPREPRO, dimPCA= dim)
    else:
        matPCA = get_SVD(dataPREPRO, dimPCA =dim)
    return matPCA


def get_PCA(mat, dimPCA=6):
    pca = PCA(n_components=dimPCA, random_state = 227)    
    matPCA=pca.fit_transform(mat) 
    s = pca.explained_variance_ratio_
    # ss = np.round(s/sum(s),3)
    ss = np.round(s,3)

    print('Explained Variance Ratio', ss)
    f, axs = plt.subplots(1,3, figsize = (20,6))
    axs[0].plot(ss)
    axs[0].semilogy()
    axs[0].set_title('PCA')
    # axs[1].matshow(pca.get_covariance)
    axs[2].matshow(matPCA.T , aspect='auto')
    print(matPCA.shape)
    return matPCA

######################## PCA ###########################
def get_SVD(data, dimPCA = 6):
    f, axs = plt.subplots(1,3, figsize = (20,6))
    cov = data.T.dot(data)    
    axs[1].matshow(cov)
    pc = get_pc(cov, dimPCA, ax = axs[0])
    matPCA = data.dot(pc)
    axs[2].matshow(matPCA.T , aspect='auto')
    return matPCA

def get_pc(cov, dimPCA, ax = None):
    print(f"=============== PCA N_component: {dimPCA} ===============")
    u,s,v = np.linalg.svd(cov)
#     assert np.allclose(u, v.T)
    ss = np.round(s/np.sum(s),3)
    print('Explained Variance Ratio', ss)
    ax = ax or plt.gca()
    ax.plot(ss)
    ax.semilogy()
    ax.set_title('SVD')
    pc = u[:,:dimPCA]
    return pc
