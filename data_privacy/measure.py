from scipy.spatial.distance import pdist, squareform
import pyqif as pyq
import numpy as np

def quasi_identification_risk(D):
    # Pre: D is a dataframe
    c = D.to_numpy() # Convert DataFrame to NumPy array
    u = pyq.probab.uniform(c.shape[0])

    # Return average risk across all individuals
    return pyq.measure.bayes.posterior(u, c)

#
# This function computes the bayes capacity of the channel, which is the
# sum of the column maxima.
#
def reidentification_risk(D):
    '''Computes the Bayes' capacity for the channel DBN, which corresponds to the
    leakage due to the channel for an adversary who wants to perform a reidentification
    attack (guessing the secret exactly in 1 try). Ranges from 1 (no leak) to #columns in
    channel (full leakage)'''
    return quasi_identification_risk(D)

#
# Computes the posterior vulnerability of the attribute with respect to 
# a uniform prior. This gives the probability of a successful property
# inference against a given sensitive attribute for an adversary with no
# prior knowledge.
#
def property_inference_risk(D, attr):
    '''Computes the risk of inferring a sensitive attribute in the dataset D.
    This assumes an adversary who has a uniform prior over all secrets.'''
    attr_i = D.index.names.index(attr)
    # Combine all rows with the same sensitive attribute
    # by computing their mean (equivalent to uniform prior)
    D_by_attr = D.groupby(level=[attr_i]).mean()
    u = pyq.probab.uniform(D_by_attr.shape[0])
    return pyq.measure.bayes.posterior(u, D_by_attr.to_numpy())

# 
# Computes the risk of guessing the secret given knowledge of the attributes
# in the attrs array, under a uniform prior. 
#
def linkability_risk(C, attrs):
    '''Outputs the risk of reconstructing the full secret in channel C,
    relative to linking via attributeList (quasi-identifiers).
    A row is considered at risk if its projection on the attributeList is unique.
    '''
    # For this we assume C is a channel produced by make_channel.
    # We assume attrs is a subset of the QIDs in C.

    # Group the secrets by the attrs
    groups = C.T.groupby(attrs).sum()

    # Count how many QIDs are unique to a secret
    summary = groups.astype(bool).T.sum()

    # Identify the vulnerable secrets as ones which contain a single non-zero value
    # Note that these are actually the QIDs since we have transposed the channel
    vulnerable_attrs = summary.index[summary == 1]

    # Add up all the probabilities for the vulnerable inputs.
    prob = groups.loc[vulnerable_attrs, ].max(axis=1).sum()

    # Now have to divide probability by size of secret space, equivalent to uniform prior
    num_secrets = C.index.size
    return prob / num_secrets


def __dist_arr(C, values):
    val_arr = np.array([ [i] for i in values ])
    dists = pdist(val_arr)
    # create the loss function which gives the abs distance between values of attr
    dists_arr = squareform(dists)
    return dists_arr

def average_minimum_error(C, attr):
    '''Computes the expected distance of the true value from the output value for
    the given attribute in the channel D.'''
    # C is a pandas dataframe representing a channel which takes secrets X to observations Y.
    # We assume that the given attribute is numeric so we can compute a mean error
    # between the guessed value of the attribute and the true value.
    values = C.index.get_level_values(attr).to_numpy()
    dists_arr = __dist_arr(C, values)
    avg_uncertainty = pyq.measure.ul.posterior(pyq.probab.uniform(len(values)), C.to_numpy(), dists_arr)
    return avg_uncertainty

def likely_interval(D, attr, c):
    '''Computes the probability that an adversary cannot infer the value of attr
    to within bound c of its true value.'''
    # The implementation of this uses the loss function from average_minimum_error
    # and converts the distances to bits depending on c.
    values = D.index.get_level_values(attr).to_numpy()
    dists_arr = __dist_arr(D, values)
    # convert distance matrix to bit matrix
    bits_arr = np.array( [ [ int(i<c) for i in row] for row in dists_arr])
    avg_uncertainty = pyq.measure.ul.posterior(pyq.probab.uniform(len(values)), D.to_numpy(), bits_arr)
    return avg_uncertainty

def bayes_total_mse(C, pi, q, n):
    '''Computes the MSE of the Bayesian estimate of the total statistic.
       Inputs:
         - C[x, z] is a dataframe representing the channel
         - pi[x] is the analyst's prior over true states x
         - q[x] is the contribution of state x to the statistic
         - pi is equal to the empirical distribution of the dataset
    '''
    p_z = pi @ C
    
    # a(z) = E[g(X) | Z=z]
    weighted = C.mul(pi, axis=0).mul(q, axis=0)
    a = weighted.sum(axis=0) / p_z
    
    # Error for each true x and possible observed z:
    # e(x,z) = a(z) - g(x)
    errors = pd.DataFrame(
        a.to_numpy()[None, :] - q.to_numpy()[:, None],
        index=C.index,
        columns=C.columns
    )
    b = (C * errors).sum(axis=1)
    m = (C * (errors**2)).sum(axis=1)
    n_x = n * pi
    mse = (n_x * (m - (b**2))).sum()
    return float(mse)

