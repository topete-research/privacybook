import pandas as pd
import numpy as np
import pyqif as pyq
import math

def bounded_noise(D, attr, noise):
    '''Creates a customised bounded noise channel using the given noise
    distribution. For example, noise which returns the same value with prob.
    1/2, and returns the value +- 1 with prob. 1/4 (each) would look like:
    noise=pd.Series([ 1/4, 1/2, 1/4], index=[-1, 0, 1]).'''
    try:
        attr_i = D.columns.names.index(attr)
        noise_labels = noise.index
        df = pd.DataFrame()
        for col_i in range(len(D.columns)):
            v = D.columns[col_i][attr_i] # the value of the attr to add noise to
            if int(v) != v:
                raise Exception("The given attribute is not numeric")
            # Create a new channel with labels as noisy attrs, values
            # are the probabilities.
            def get_noisy_qid(noise_i):
                qid = list(D.columns[col_i])
                qid[attr_i] += noise_labels[noise_i]
                return qid
            tuples = list([ get_noisy_qid(idx) for idx in range(len(noise_labels))])
            result = pd.MultiIndex.from_tuples(tuples, names=D.columns.names)
            multirow = pd.Series(noise.to_numpy(), index=result, name=D.columns[col_i])
            df = pd.concat([df, multirow.to_frame().T])
        df = df.fillna(0)
        # Fix up the names for the rows (secrets)
        df.index = df.index.set_names(D.columns.names)
        return df
    except ValueError:
        print("Error: no such attribute name: ", attr)
    except Exception as inst:
        print(inst)
    return None


def uniform_bounded_noise(D, attr, bound):
    '''Computes a bounded noise channel for perturbing the attribute in a given channel D
    using the given bound. For example, if bound=1 and attr="age", then each age in
    the dataset D should be perturbed by either adding 1, returning unchanged or subtracting 1,
    each with probability 1/3 (uniform).'''
    noise_prob = 1.0 / (2 * bound + 1) # uniform prob over values
    noise = pd.Series(noise_prob, index=list(range(-bound, bound+1)))
    return bounded_noise(D, attr, noise)


def random_response(D, attr, K, eps):
    '''Given the dataset channel D and an attribute attr, computes the random
    response channel applied to attr with range K and epsilon eps. Note that attr
    (and K) can be categorical (string valued) or numeric.'''
    # RR channel has entries q = 1 / (len(K) - 1 + e **eps) 
    # and single entry p = e**eps/(len(K) - 1 + e ** eps)
    q = 1 / (len(K) -1 + (math.e ** eps))
    p = q * (math.e ** eps)
    try:
        attr_i = D.columns.names.index(attr)
        df = pd.DataFrame()
        for col_i in range(len(D.columns)):
            v = D.columns[col_i][attr_i] # the value of the attr to add noise to
            #if int(v) != v:
            #    raise Exception("The given attribute is not numeric")
            qid = list(D.columns[col_i]) # the value of the QID for this column
            # We want to replace the value at index attr_i with the range of K
            tuples = [ qid[:attr_i] + [noise_i] + qid[attr_i+1:] for noise_i in K]
            result = pd.MultiIndex.from_tuples(tuples, names=D.columns.names)
            # Random response noise is almost all the same
            noise = [q] * len(K)
            # We need the index of v in K
            if not v in K:
                raise Exception("The attribute " + str(v) + " lies outside the range " + str(K))
            noise[K.index(v)] = p
            multirow = pd.Series(noise, index=result, name=D.columns[col_i])
            df = pd.concat([df, multirow.to_frame().T])
        df = df.fillna(0)
        # Fix up the names for the rows (secrets)
        df.index = df.index.set_names(D.columns.names)
        return df
    except ValueError:
        print("Error: no such attribute name: ", attr)
    except Exception as inst:
        print(inst)
    return None

def geometric(D, attr, K, eps):
    '''Given the dataset channel D and a numeric attribute attr, computes the geometric
    channel for adding noise to attr with range K and epsilon eps.'''
    # We start by just generating geometric noise as a numpy array
    G_channel = pyq.dp.ranged_geometric(K, eps)
    try:
        attr_i = D.columns.names.index(attr)
        df = pd.DataFrame()
        for col_i in range(len(D.columns)):
            v = D.columns[col_i][attr_i] # the value of the attr to add noise to
            if int(v) != v:
                raise Exception("The given attribute is not numeric")
            qid = list(D.columns[col_i]) # the value of the QID for this column
            # We want to replace the value at index attr_i with the range of K
            tuples = [ qid[:attr_i] + [noise_i] + qid[attr_i+1:] for noise_i in K]
            result = pd.MultiIndex.from_tuples(tuples, names=D.columns.names)
            # Get the noise from the correct row of the geometric mechanism
            # We need the index of v in K
            if not v in K:
                raise Exception("The attribute " + str(v) + " lies outside the range " + str(K))
            noise = G_channel[K.index(v)]
            multirow = pd.Series(noise, index=result, name=D.columns[col_i])
            df = pd.concat([df, multirow.to_frame().T])
        df = df.fillna(0)
        # Fix up the names for the rows (secrets)
        df.index = df.index.set_names(D.columns.names)
        return df
    except ValueError:
        print("Error: no such attribute name: ", attr)
    except Exception as inst:
        print(inst)
    return None

