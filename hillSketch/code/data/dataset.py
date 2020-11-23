# import numpy as np
# import copy
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.cluster import KMeans
# from scipy.spatial.distance import cdist

def load_data(DATASET):
    if DATASET[-3:]=='csv':
        data = pd.read_csv(DATASET)    
    elif DATASET[-4:]=='xlsx':
        data = pd.read_excel(DATASET)    
    else:
        raise 'can only read csv or xlsx file'
    keep_columns = data.columns[(data.sum()!=0)]
    print(keep_columns)
    return data[keep_columns], keep_columns
