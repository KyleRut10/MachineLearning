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
