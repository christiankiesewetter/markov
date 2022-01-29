import numpy as np

def scale(data, on_axis = 0):
    
    if len(data.shape) == 1:
        return data / data.sum()
    
    elif len(data.shape) == 2: 
        if on_axis == 0:
            return (data.T / data.sum(axis=1)).T
        if on_axis == 1:
            return (data.T / data.sum(axis=0)[...,np.newaxis]).T
    
    elif len(data.shape) == 3:
        if on_axis == 0:
            dsum = data.sum(axis=1).sum(axis=1)
            return data / dsum[...,np.newaxis, np.newaxis]
        if on_axis == 1:
            dsum = data.sum(axis=2)
            return data / dsum[...,np.newaxis]
        if on_axis == 2:
            dsum = data.sum(axis=1)
            return data / dsum[:,np.newaxis,:]



