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
    #df = df.drop(['sex'], axis=1)
    cat,cont = type_abalone()
    for col in cont:
        df[col] = standardize(df[col])
    # one hot encoding for catigorical
    for col_label in cat:
        df = one_hot(col_label, df) 
    # standardize the regression column
    df['response'] = standardize(df['response'])
    # Return the data
    return df

def type_abalone():
    catigorical = ['sex']
    continuious = ['length', 'diameter', 'height', 'whole weight',
                   'shucked weight', 'viscera weight', 'shell weight']
    return catigorical, continuious


def data_forestfire():
    # Read in the file
    df = pd.read_csv(os.path.join('..', '..', 'data', 'forestfires.csv'))
    cat,cont = type_forestfire()
    # standardize the data
    for col in cont:
        df[col] = standardize(df[col])
    # one hot coding on catagorical data
    for col_label in cat:
        df = one_hot(col_label, df)
    # Return the data
    return df

def type_forestfire():
    catigorical = ['x', 'y', 'month', 'day']
    continuious = ['ffmc', 'dmc', 'dc', 'isi', 'temp', 'rh', 'wind', 'rain']
    return catigorical, continuious


'''
def data_hardware():
    # Read in the file
    df = pd.read_csv(os.path.join('..', '..', 'data', 'machine.csv'))
    df = df.drop(columns=['erp'])
    # standardize the data
    for col in type_hardware()[1]:
        df[col] = standardize(df[col])
    # Return the data
    return df

def type_hardware():
    catigorical = ['vendor name', 'model name']
    continuious = ['myct', 'mmin', 'mmax', 'cach', 'chmin', 'chmax', 'respone']
    return catigorical, continuious
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

def one_hot(col_label, df):
    # Takes a column of categorical data and converts it into
    # a several columns by on_hot encoding.
    # Input: col - Categorical Data Column
    # Output: out - Pandas Dataframe of new on-hot columns
    # get values oin column
    col = df[col_label]
    # add new column for each unique value
    for cat in col.unique():
        new_label = '{}-{}'.format(col_label, cat)
        df[new_label] = (col == cat)*1
    # drop origional column
    df = df.drop(columns=[col_label])
    return df

