import pandas as pd

def standardize(column):
    # Standardizes a column by z-score normalization
    
    return column - mean(column)/column.std

def on_hot(col):
    # Takes a column of categorical data and converts it into
    # a several columns by on_hot encoding.
    # Input: col - Categorical Data Column
    # Output: out - Pandas Dataframe of new on-hot columns
    out = pd.DataFrame()
    for cat in col.unique():
        out[cat] = (col == cat)*1
    return out
