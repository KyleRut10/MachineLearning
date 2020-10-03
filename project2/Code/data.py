import pandas as pd
import os
from clean import *

# glass dataset
def data_glass():
    # Read in the file
    df = pd.read_csv(os.path.join('..', '..', 'data', 'glass.csv'))
    # Clean data, by removing rows with missing atrributes
    run(df, 0)
    # Return the data
    return df


def data_abalone():
    # Read in the file
    df = pd.read_csv(os.path.join('..', '..', 'data', 'abalone.csv'))
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
