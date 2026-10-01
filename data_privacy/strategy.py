import pandas as pd
import numpy as np
import pyqif as pyq

def intersection(C, D):
    '''Constructs the intersection attack channel for an adversary who
    has access to 2 channels C, D on the same inputs. Corresponds to a
    parallel composition of channels.'''
    #Pre: Channels C, D must have the same number of rows
    CD = pyq.compose.parallel(C.to_numpy(), D.to_numpy())
    # Recompute the columns
    cols = np.array(np.meshgrid(C.columns.names, D.columns.names)).T.reshape(-1, 2)
    return pd.DataFrame(data=CD, index=D.index, columns=cols)

def linkage(C, D):
    '''Computes the linkage attack matrix which utilises the
    correlation between observations on C and inputs on D.
    '''
    # Assume that C and D are channels created using make_channel
    return C.dot(D)

