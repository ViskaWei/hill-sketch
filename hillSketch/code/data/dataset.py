# import numpy as np
# import copy
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.cluster import KMeans
# from scipy.spatial.distance import cdist

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