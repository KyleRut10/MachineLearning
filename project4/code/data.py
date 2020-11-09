import pandas as pd
import os
import numpy as np
import random as rand

# classification
def data_glass():
    # Read in the file
    df = pd.read_csv(os.path.join('..', '..', 'data', 'glass.csv'))
    # Drop the class and index
    #df = df.drop(columns=['index'])
    df = df.drop(['index'], axis=1)
    # convert class strings to integers
    df['class'] = string_to_int(df['class'])
    # clean and standardize the data
    df = standardize(type_glass, df)
    # Return the data
    return df


def type_glass():
    catigorical = []
    continuious = ['ri', 'na', 'mg', 'al', 'si', 'k', 'ca', 'ba', 'fe']
    return catigorical, continuious

# regression
def data_abalone():
    # Read in the file
    df = pd.read_csv(os.path.join('..', '..', 'data', 'abalone.csv'))
    # clean and standardize the data
    df = standardize(type_abalone, df)
    return df


def type_abalone():
    catigorical = ['sex']
    continuious = ['length', 'diameter', 'height', 'whole weight',
                   'shucked weight', 'viscera weight', 'shell weight',
                   'response']
    return catigorical, continuious

# regression
# NOTE: Error values are not between [0,1], something is wrong
def data_forestfire():
    # Read in the file
    df = pd.read_csv(os.path.join('..', '..', 'data', 'forestfires.csv'))
    # clean and standardize the data
    df = standardize(type_forestfire, df)
    # Return the data
    return df


def type_forestfire():
    catigorical = ['x', 'y', 'month', 'day']
    continuious = ['ffmc', 'dmc', 'dc', 'isi', 'temp', 'rh', 'wind', 'rain',
                   'response']
    return catigorical, continuious

# regression
def data_hardware():
    # Read in the file
    df = pd.read_csv(os.path.join('..', '..', 'data', 'machine.csv'))
    df = df.drop(columns=['erp'])#, axis=1)
    # clean and standardize the data
    df = standardize(type_hardware, df)
    # Return the data
    return df


def type_hardware():
    catigorical = ['vendor name', 'model name']
    continuious = ['myct', 'mmin', 'mmax', 'cach', 'chmin', 'chmax', 'response']
    return catigorical, continuious

# classification
def data_breast():
    # Read in the file
    df = pd.read_csv(os.path.join('..', '..', 'data',
                                  'breast-cancer-wisconsin.csv'))
    # typecast whole dataframe to float, because for some reason it's reading
    # them in as strings
    df = clean(df)
    df = df.astype(float)
    # convert class strings to integers
    df['class'] = string_to_int(df['class'])
    # drop the index
    df = df.drop(['index'], axis=1)
    # clean and standardize the data
    df = standardize(type_breast, df)
    # Return the data
    return df


# NOTE: I'm not 100% sure these are correct
def type_breast():
    catigorical = ['clump thickness', 'uniformity of cell size',
                   'uniformity of cell shape', 'marginal adhesion',
                   'single epithelial cell size', 'bare nuclei',
                   'bland chromatin', 'normal nucleoli', 'mitosis']
    continuious = []
    return catigorical, continuious


# classification
def data_soybean_small():
    # Read in the file
    df = pd.read_csv(os.path.join('..', '..', 'data', 'soybean-small.csv'))
    # clean and standardize the data
    df = standardize(type_soybean_small, df)
    # convert class strings to integers
    df['class'] = string_to_int(df['class'])
    # Return the data
    return df


def type_soybean_small():
    catigorical = ['date', 'plant stand', 'percip', 'temp', 'hail',
                   'crop hist', 'area damaged', 'severity', 'seet tmt',
                   'germination', 'plant growth', 'leaves',
                   'leafspots halo', 'leafspots marg', 'leafspot size',
                   'leaf shread', 'leaf malf', 'leaf mild', 'stem',
                   'lodging', 'stem cankers', 'canker lesion',
                   'fruiting bodies', 'external decay', 'mycelium',
                   'int discolor', 'sclerotia', 'fruit pods', 'fruit spots',
                   'seed', 'mold growth', 'seed discolor', 'seed size',
                   'shriveling']
    continuious = []
    return catigorical, continuious


# Functions to clean and prep the data
def standardize(type_funct, df):
    # clean the data
    df = clean(df)
    # standardize the data
    cat,cont = type_funct()
    for col in cont:
        #print(col)
        #print(max(df[col]))
        df[col] = z_score_normalize(df[col])
        #print(max(df[col]))
    # one hot encoding for catigorical
    for col_label in cat:
        df = one_hot(col_label, df)
    return df


def clean(rawdata):
    # A function to remove rows with question marks from the data

    raw = rawdata.replace(["?"],np.nan)
    data = raw.dropna(axis=0,how='any')

    return data


def z_score_normalize(col):
    # Takes a Raw data Column and standardize it by its z-score
    # Input: col - Raw Data Column
    # Output: Z-score of the Column
    return (col - col.mean())/col.std()


def one_hot(col_label, df):
    # Takes a column of categorical data and converts it into
    # a several columns by on_hot encoding.
    # Input: col - Categorical Data Column
    # Output: out - Pandas Dataframe of new on-hot columns
    # make sure class stays in last
    identifier = df.columns.values[-1]
    # get values oin column
    col = df[col_label]
    # add new column for each unique value
    for cat in col.unique():
        new_label = '{}-{}'.format(col_label, cat)
        df[new_label] = (col == cat)*1
    # put the column in the last poistion again
    identifier_df = df.pop(identifier)
    df[identifier] = identifier_df
    # drop origional column
    df = df.drop([col_label], axis=1)
    return df


# convert the string classes to corresponding numerical values
def string_to_int(col):
    # get unique list of classes in column
    uni = col.unique().tolist()
    new_col = []  # hold new values
    # go through each value in column and assign it an interger value
    for val in col:
        new_col.append(uni.index(val))
    return new_col


def get_cross_validate_dfs(df):
    # Takes in a df and returns a list of 20 different ones
    # In each of the 10 lists, the first df is the testing one (1/10 of data)
    # and the second is the training one (9/10 of data)

    # randomize the data set
    index = [i for i in range(df.shape[0])]
    rand.shuffle(index)
    df = df.set_index([index]).sort_index()

    # assign each data point to one of ten groups
    test_indices = [[] for i in range(10)]
    for i in range(0,df.shape[0]):
        index = i%10
        test_indices[index].append(i)

    # make test and training data sets for each group
    dfs = []
    for indices in test_indices:
        group = []
        # add the testing points
        group.append(df.iloc[indices, :])
        # drop testing points and rest are for training
        train = df.drop(indices, axis=0)
        train.reset_index(inplace=True)
        group.append(train)
        dfs.append(group)

    return dfs


import subprocess
def write_to_clipboard(output):
    process = subprocess.Popen(
            'pbcopy', env={'LANG': 'en_US.UTF-8'}, stdin=subprocess.PIPE)
    process.communicate(output.encode('utf-8'))

