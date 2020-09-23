import pandas as pd
import os

# glass dataset
def data_glass():
    df = pd.read_csv(os.path.join('..', 'data', 'glass.csv'))
    # TODO: Process the data
    return df


def data_abalone():
    df = pd.read_csv(os.path.join('..', 'data', 'abalone.csv'))
    # TODO: Process the data
    return df


def data_vote():
    df = pd.read_csv(os.path.join('..', 'data', 'house-votes-84.csv'))
    # TODO: Process the data
    return df


def data_segmentation():
    df = pd.read_csv(os.path.join('..', 'data', 'segmentation.csv'))
    # TODO: Process the data
    return df


def data_forestfire():
    df = pd.read_csv(os.path.join('..', 'data', 'forestfire.csv'))
    # TODO: Process the data
    return df


def data_hardware():
    df = pd.read_csv(os.path.join('..', 'data', 'machine.csv'))
    # TODO: Process the data
    return df
