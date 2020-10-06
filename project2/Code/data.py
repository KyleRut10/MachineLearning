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
    df = df.drop(columns=['index'])
    # standardize the data
    for col in df.columns.values:
        df[col] = z_stand(df[col])
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
    run(df, 0)
    # standardize the data
    for col in type_abalone()[1]:
        df[col] = z_stand(df[col])
    # Return the data
    return df


def type_abalone():
    catigorical = ['sex']
    continuious = ['length', 'diameter', 'height', 'whole weight',
                   'shucked weight', 'biscera weight', 'shell weight']
    return catigorical, continuious


def data_vote():
    # Read in the file
    df = pd.read_csv(os.path.join('..', '..', 'data', 'house-votes-84.csv'))
    # Clean the data
    df = run(df, 1)
    # move class to last column
    cols = list(df.columns.values)
    cols.remove('class')
    cols.append('class')
    df = df.reindex(columns=cols)
    # Return the data
    return df


def type_vote():
    catigorical = ['handicapped-infants', 'water-project-cost-sharing',
                   'adoption-of-the-budget-resolution-physician-fee-freeze',
                   'el-salvador-aid', 'religious-groups-in-schools',
                   'anti-satellite-test-ban', 'aid-to-nicaraguan-contras', 
                   'mx-missile', 'immigration', 'synfuels-corperation-cutback',
                   'education-spending', 'superfund-right-to-sue', 'crime',
                   'duty-free-exports', 
                   'export-administration-act-south-africa']
    continuious = []
    return catigorical, continuious


def data_segmentation():
    # Read in the file
    df = pd.read_csv(os.path.join('..', '..', 'data', 'segmentation.csv'))
    # move class to last column
    cols = list(df.columns.values)
    cols.remove('class')
    cols.append('class')
    df = df.reindex(columns=cols)
    # standardize the data
    for col in df.columns.values:
        df[col] = z_stand(df[col])
    # Return the Data
    return df


def type_segmentation():
    catigorical = []
    continuious = ['region-centroid-col', 'region-centroid-row',
                   'region-pixel-count', 'short-line-density-5', 
                   'short-line-density-2', 'vedge-mean,vedge-sd', 'hedge-mean',
                   'hedge-sd', 'intensity-mean', 'rawred-mean', 'rawblue-mean',
                   'rawgreen-mean', 'exred-mean', 'exblue-mean', 'exgreen-mean',
                   'value-mean', 'saturation-mean', 'hue-mean']
    return catigorical, continuious


def data_forestfire():
    # Read in the file
    df = pd.read_csv(os.path.join('..', '..', 'data', 'forestfires.csv'))
    # standardize the data
    for col in type_forestfire()[1]:
        df[col] = z_stand(df[col])
    # Return the data
    return df


def type_forestfire():
    catigorical = ['x', 'y', 'month', 'day']
    continuious = ['ffmc', 'dmc', 'dc', 'isi', 'temp', 'rh', 'wind', 'rain',
                   'area']
    return catigorical, continuious


def data_hardware():
    # Read in the file
    df = pd.read_csv(os.path.join('..', '..', 'data', 'machine.csv'))
    df = df.drop(columns=['erp'])
    # standardize the data
    for col in type_hardware()[1]:
        df[col] = z_stand(df[col])
    # Return the data
    return df


def type_hardware():
    catigorical = ['vendor name', 'model name']
    continuious = ['myct', 'mmin', 'mmax', 'cach', 'chmin', 'chmax', 'respone']
    return catigorical, continuious


def data_test():
    data = {'x': [1.0,1.5,1.0,1.2,1.9,5.0,5.5,5.5], 
            'y': [1.0,1.2,1.5,1.4,1.7,5.0,6.0,5.4]}
    df = pd.DataFrame.from_dict(data)
    for col in df.columns.values:
        df[col] = z_stand(df[col])
    return df

