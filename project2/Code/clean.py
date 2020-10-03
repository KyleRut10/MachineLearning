import pandas as pd

def run(data, option):
    if option == 1:
        return na_to_a(data)
    else:
        return na_remove(data)
    
def na_to_a(data)
    return data = rawdata.replace(["?"],'a')

def na_remove(data)
    data = data.replace(["?"],np.nan)
    return raw.dropna(axis=0,how='any')
