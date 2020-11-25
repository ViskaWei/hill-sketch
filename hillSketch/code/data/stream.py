import numpy as np
import copy

def get_rebin(dfNorm, base):
    dfRebin=(dfNorm*(base-1)).round()
    assert (dfRebin.min().min()>=0) & (dfRebin.max().max()<=base-1)
    return dfRebin

def get_encode_stream(dfNorm, base, dtype='uint64'):
    mat = get_rebin(dfNorm,base)
    mat_encode=horner_encode(mat,base,dtype) 
    mat_decode=horner_decode(mat_encode,base,len(mat.keys()),dtype)  
    assert (mat_decode.min().min()>=0) & (mat_decode.max().max()<=base-1)
    try:
        assert np.sum(abs(mat_decode-mat.values))<=0.0001    
    except:
        print(np.nonzero(np.sum(abs(mat_decode-mat.values),axis=1)), np.sum(abs(mat_decode-mat.values)))
        raise 'overflow, try lower base or fewer features'     
    return mat_encode

def horner_encode(mat,base,dtype):
    r,c=mat.shape
    print('samples:',r,'ftrs:',c, 'base:',base)
    encode=np.zeros((r),dtype=dtype)
    for ii, key in enumerate(mat.keys()):
        val=(mat[key].values).astype(dtype)
        encode= encode + val*(base**ii)
#         print(ii,val, encode)
    return encode

def horner_decode(encode,base, dim,dtype):
    arr=copy.deepcopy(np.array(encode))
    decode=np.zeros((len(arr),dim), dtype=dtype)
    for ii in range(dim-1,-1,-1):
        digits=arr//(base**ii)
        decode[:,ii]=digits
        arr= arr% (base**ii)
#         print(digits,arr)
    return decode