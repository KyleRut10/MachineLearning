import pandas as pd
import os
import numpy as np

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
    # get variables that are catigorical and continuious
    cat,cont = type_glass()
    # standardize the data
    for col in cont:
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
    df = df.drop(['sex'], axis=1)
    for col in type_abalone()[1]:
        df[col] = standardize(df[col])
    # standardize the regression column
    df['class'] = standardize(df['class'])
    # Return the data
    return df

def type_abalone():
    catigorical = ['sex']
    continuious = ['length', 'diameter', 'height', 'whole weight',
                   'shucked weight', 'viscera weight', 'shell weight']
    return catigorical, continuious


'''
def data_forestfire():
    # Read in the file
    df = pd.read_csv(os.path.join('..', '..', 'data', 'forestfires.csv'))
    # standardize the data
    for col in type_forestfire()[1]:
        df[col] = standardize(df[col])
    # Return the data
    return df

def type_forestfire():
    catigorical = ['x', 'y', 'month', 'day']
    continuious = ['ffmc', 'dmc', 'dc', 'isi', 'temp', 'rh', 'wind', 'rain']
    return catigorical, continuious


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

# Functions to clean and prep the data
def clean(rawdata):
    # A function to remove rows with question marks from the data
    
    raw = rawdata.replace(["?"],np.nan)
    data = raw.dropna(axis=0,how='any')
    
    return data


def standardize(col):
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

