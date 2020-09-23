import pandas as pd

def run(data, option):
    if option == 1:
        return na.to.a(data)
    else:
        return na.remove(data)
    
def na.to.a(data)
    return data = rawdata.replace(["?"],'a')

def na.remove(data)
    data = data.replace(["?"],np.nan)
    return raw.dropna(axis=0,how='any')
