import pandas as pd
import os
from clean import *

# glass dataset
def data_glass():
    # Read in the file
    df = pd.read_csv(os.path.join('..', '..', 'data', 'glass.csv'))
    # Clean data, by removing rows with missing atrributes
    run(df, 0)
    # Drop the class and index
    df = df.drop(columns=['index', 'class'])
    # standardize the data
    for col in df.columns.values:
        df[col] = z_stand(df[col])
    # Return the data
    return df


def data_abalone():
    # Read in the file
    df = pd.read_csv(os.path.join('..', '..', 'data', 'abalone.csv'))
    # Clean data, by removing rows with missing atrributes
    run(df, 0)
    # Return the data
    return df


def data_vote():
    # Read in the file
    df = pd.read_csv(os.path.join('..', '..', 'data', 'house-votes-84.csv'))
    # Clean the data
    data = clean.run(df, 1)
    # Return the data
    return data


def data_segmentation():
    # Read in the file
    df = pd.read_csv(os.path.join('..', '..', 'data', 'segmentation.csv'))
    # Return the Data
    return df


def data_forestfire():
    # Read in the file
    df = pd.read_csv(os.path.join('..', '..', 'data', 'forestfire.csv'))
    # Return the data
    return df


def data_hardware():
    # Read in the file
    df = pd.read_csv(os.path.join('..', '..', 'data', 'machine.csv'))
    # Return the data
    return df


def data_test():
    data = {'x': [1.0,1.5,1.0,1.2,1.9,5.0,5.5,5.5], 
            'y': [1.0,1.2,1.5,1.4,1.7,5.0,6.0,5.4]}
    df = pd.DataFrame.from_dict(data)
    for col in df.columns.values:
        df[col] = z_stand(df[col])
    return df

