import pandas as pd

def clean(rawdata):
    # A function to remove rows with question marks from the data
    
    raw = rawdata.replace(["?"],np.nan)
    data = raw.dropna(axis=0,how='any')
    
    return data
