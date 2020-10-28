import pandas as pd
import os
import numpy as np
from Clean import *
from Standardize import *

# TODO: Brest cancer, soybean small

# glass dataset
def data_glass():
    # Read in the file
    df = pd.read_csv(os.path.join('..', '..', 'data', 'glass.csv'))
    # Clean data, by removing rows with missing atrributes
    clean(df)
    # Drop the class and index
    #df = df.drop(columns=['index'])
    df = df.drop(['index'], axis=1)
    # standardize the data
    for col in type_glass()[1]:
        df[col] = standardize(df[col])
    # Return the data
    return df

def type_glass():
    catigorical = []
    continuious = ['ri', 'na', 'mg', 'al', 'si', 'k', 'ca', 'ba', 'fe']
    return catigorical, continuious
def data_abalone():
    # Read in the file
    df = pd.read_csv(os.path.join('..', '..', 'data', 'abalone.csv'))
    # Clean data, by removing rows with missing atrributes
    clean(df)
    # standardize the data
    for col in type_abalone()[1]:
        df[col] = standardize(df[col])
    # Return the data
    return df

'''
def data_forestfire():
    # Read in the file
    df = pd.read_csv(os.path.join('..', '..', 'data', 'forestfires.csv'))
    # standardize the data
    for col in type_forestfire()[1]:
        df[col] = standardize(df[col])
    # Return the data
    return df


def data_hardware():
    # Read in the file
    df = pd.read_csv(os.path.join('..', '..', 'data', 'machine.csv'))
    df = df.drop(columns=['erp'])
    # standardize the data
    for col in type_hardware()[1]:
        df[col] = standardize(df[col])
    # Return the data
    return df
'''


