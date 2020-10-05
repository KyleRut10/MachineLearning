import pandas as pd
import numpy as np

def run(data, option):
    if option == 1:
        return na_to_a(data)
    else:
        return na_remove(data)
    
def na_to_a(data):
    data = data.replace(["?"],'a')
    return data

def na_remove(data):
    data = data.replace(["?"],np.nan)
    return data.dropna(axis=0,how='any', inplace=True)

def z_stand(col):
    # Takes a Raw data Column and standardizes it by its z-score
    # Input: col - Raw Data Column
    # Output: Z-score of the Column
    return (col - col.mean())/col.std()
    
def on_hot(col):
    # Takes a column of categorical data and converts it into
    # a several columns by on_hot encoding.
    # Input: col - Categorical Data Column
    # Output: out - Pandas Dataframe of new on-hot columns
    out = pd.DataFrame()
    for cat in col.unique():
        out[cat] = (col == cat)*1
    return out
