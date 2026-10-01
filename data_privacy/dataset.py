import pandas as pd

#
# make_channel
#
# Inputs:
# - inputTable: dataframe representing a dataset
# - secretsList: array of column names representing secrets
# - qidsList: array of column names representing QIDs
#
# Returns:
#  - dataframe representing a channel: rows are labelled with
# secrets and columns are labelled with QIDs
#

def make_channel(inputTable, secretsList, qidsList):
    '''Create a channel from a tabular datasource given as a pandas frame.
    The row and column labels are also provided as secretsList and qidsList
    respectively.'''

    # Group by QIDs + secrets → counts
    counts = inputTable.groupby(qidsList + secretsList).size()

    # Reshape into QID (columns) × Secret (rows)
    channel = counts.unstack(level=qidsList, fill_value=0)

    # Convert counts into probabilities
    row_sums = channel.sum(axis=1)
    prob_channel = channel.div(row_sums, axis="index")

    return prob_channel


