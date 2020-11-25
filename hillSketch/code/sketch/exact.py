import time
import numpy as np
import pandas as pd
from collections import Counter
from code.data.stream import horner_decode


def get_exact_HH(stream):
    print(f'=============exact counting HHs==============')
    t0=time.time()
    exactHH=np.array(Counter(stream).most_common())
    t=time.time()-t0
    print('exact counting time:{:.2f}'.format(t))
    return exactHH[:,0], exactHH[:,1], t


def get_HH_pd(stream,base,ftr_len, dtype):
    HH,freq,t=get_exact_HH(stream)
    HHfreq=np.vstack((HH,freq))
    mat_decode_HH=horner_decode(HH,base,ftr_len, dtype)
    assert (mat_decode_HH.min().min()>=0) & (mat_decode_HH.max().max()<=base-1)
    dfHH=pd.DataFrame(np.hstack((mat_decode_HH,HHfreq.T)), columns=list(range(ftr_len))+['HH','freq']) 
    dfHH['rk']=dfHH['freq'].cumsum()
    dfHH['ra']=dfHH['rk']/dfHH['rk'].values[-1]
    return dfHH