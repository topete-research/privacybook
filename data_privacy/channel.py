import pandas as pd
import numpy as np
import math
from itertools import product
import pyqif as pyq
import privacybook.strategy as strategy

def compose(C, D):
    '''Constructs the perturbation channel corresponding to the composition of
    the channels C and D. Assumes that the columns of C match the rows of D.'''
    C_values = C.to_numpy()
    D_values = D.to_numpy()
    composition = np.matmul(C_values, D_values)
    return pd.DataFrame(data=composition, index=C.index, columns=D.columns)

def kroneckerN(C, n):
    '''Computes the n-fold Kronecker product of channel C with itself.'''
    # Use underlying pyqif library to kronecker the channel
    Ch = C.to_numpy()
    ChN = pyq.compose.kroneckerN(Ch, n)

    # Compute cartesian product of labels
    states = list(product(C.index, repeat=n))
    idx = pd.MultiIndex.from_tuples(
        states,
        names=[f"X{i+1}" for i in range(n)]
    )
    # Put channel back together with labels
    Kn = pd.DataFrame(ChN, index=idx, columns=idx)
    return Kn

def inputs_at_risk(C):
    '''
    Returns the row indices of the channel C for which the secrets can be identified completely.
    A secret is fully identified if there exists a column for which this secret is the only non-zero
    value in the column.
    '''
    # Assume C is a channel with secrets as row labels
    # We first count the number of non-zero values in each column
    counts = C.astype(bool).sum(axis=0)
    # Now we find the secrets that correspond to single non-zero values in a column
    inputs = counts.index[counts == 1]
    return inputs

def inputs_at_high_risk(C, p):
    '''Returns a list of row indices (secrets) for which the adversary can guess the secret
    with probability ≥ p under a uniform prior.
    '''
    # Compute the posterior matrix under uniform prior
    u = pyq.probab.uniform(C.shape[0])
    P = pyq.channel.posteriors(u, C.to_numpy())
    P_df = pd.DataFrame(data=P, index=C.index, columns=C.columns)
    maxes = P_df.max()
    return maxes.index[maxes >  p]

def inputs_at_linkage_risk(C, attrs):
    '''Returns the secrets which can be perfectly identified by the QIDs in attrs.
    '''
    # Assume that C is a channel returned by make_channel.
    # Assume also that attrs is a subset of the QIDs.

    # Group the secrets by the attrs
    groups = C.T.groupby(attrs).sum()

    # Count how many QIDs are unique to a secret
    summary = groups.astype(bool).T.sum()

    vulnerable_attrs = summary.index[summary == 1]
    vulnerable_secrets = C[vulnerable_attrs].astype(bool).sum(axis=1)
    return vulnerable_secrets.index[vulnerable_secrets == 1]

def inputs_at_inference_risk(C, D):
    '''Returns the list of of secrets that are at risk from being inferred
    through linking C and D.
    '''
    # Assume that C and D are channels created by make_channel.
    # Assume that the outputs of C correspond with the inputs of D.
    CD = strategy.linkage(C, D)
    return inputs_at_risk(CD)
