import numpy as np
import pandas as pd
import os
import math
import random


def FirstAlgorithm(rawdata, clean = 0):
  # A function to implement a basic machine learning algorithm
  #
  # Inputs 
  #   rawdata: A pandas dataframe where the first column
  #            contains the class and the following ones
  #            contain the attributes
  #   clean: A value specifying which of the cleaning
  #          functions should be used. Defaults to
  #          removing rows with na values
  
  # Data Cleaning Station
  if clean == 1:
    # Leave the ? as it's own answer (a for simplicity)
    # This method is also used if know unkown values
    # exist for the dataset in question
    data = rawdata.replace(["?"],'a')
  else:
    # Remove rows with ?s
    raw = rawdata.replace(["?"],np.nan)
    data = raw.dropna(axis=0,how='any')
  
  
  # Training and Testing Area
  
  # Values for splitting the data
  len_data = len(data)
  split = round(len_data/2)
  
  classes = data[0].unique().tolist()
  
  results = np.zeros(10)
  # Loops 5 times for performing 5 x 2 Cross Validation
  for i in range(0,4):
    shuffle = data.sample(frac = 1)
    left = shuffle[:split]
    right = shuffle[split:]
    
    # Training on the Left and Testing on the Right
    results[2*i] = test(right, train(left, classes), classes)
    
    # Training on the Right and Testing on the Left
    results[2*i + 1] = test(left, train(right, classes), classes)
    
  pass

def train(data, classes):
  # The training function for this algorithm
  #
  # Inputs - 
  #   data: The training data for this algorithm
  #   classes: The classes found in the original dataset
  # Output - storage: An Array holding the Q_C array
  #          and the F values for each attribute
  
  storage = {}
  
  Q_C = {}
  for i in range(0, len(classes)):
    Q_C[i] = sum(data[0] == classes[i])/len(data)
  
  storage[0] = Q_C
  
  for k in range(1, data.shape[1] - 1):
    levels = data[k].unique().tolist()
    matrix = np.zeros((len(classes),len(levels)))
    for i in range(0, len(classes)):
      nc = sum(data[0] == classes[i])
      for j in range(0, len(levels)):
        count = 0
        for r in range(0, len(data)):
          if (data.iloc[r, 0] == classes[i] and data.iloc[r, k] == levels[j]):
            count += 1
        matrix[i][j] = (count + 1)/(nc + data.shape[1] - 2)
    storage[2*k - 1] = levels
    storage[2*k] = matrix

  return(storage)

def test(data, params, classes):
  # The testing function for this algorithm
  #
  # Inputs - 
  #   data: The testing data for this algorithm
  #   params: The parameters calculated in training the algorithm
  #   which are used to classify the test set of 'data'
  # Output - confusion: The confusion matrix of the classification
  
  # Initialize our confusion Matrix
  confusion = np.zeros((len(classes), len(classes)))
  
  
  # Loop through the rows of our testing data
  for r in range(0, len(data)):
    
    # Initialize prediction value array
    Cx = np.ones(len(classes))
    
    # Populate with Classification Step
    for i in range(0, len(classes)):
      Cx[i] = Cx[i] * params[0][i]
      # Loop through Attributes
      for k in range(1, data.shape[1] - 1):
        lvl = params[2*k - 1].index(data.iloc[r,k])
        Cx[i] = Cx[i] * params[2*k][i][lvl]
      
    # Find Positions of the Prediction and Ground Truth
    pred = np.argmax(Cx)
    true = classes.index(data.iloc[r,0])
    
    # Increment the correct element of the confusion matrix
    confusion[pred][true] = confusion[pred][true] + 1

  return confusion


def bin_column(df, column, bin_points):
  # Puts values in a column into bins acording to bin_points
  #
  # Inputs - 
  #     df: The dataframe for the dataset
  #     column: the column name that is desired to bin
  #     bin_points: A list of the bin points, min is assumed to be less than the
  #                 first point and max is added on as infinity
  # Output - df: original dataframe but with column replaced with bin values
  
  # add infinity onto end of bin_points so it'll go until the max
  bin_points.append(math.inf)
  binned = []
  for data in df[column]:
    for i,bin_point in enumerate(bin_points):
      # check if less than point
      # don't need to check between, becuase if less than it'd have been
      # caught earlier
        if data < bin_point: 
          binned.append('bin{}'.format(i+1))
          # break out of loop, because will also be less than others
          #print(data, ' ', bin_point, ' ', binned[-1])
          break
  #print(binned)
  df[column] = binned
  return df



def scramble_features(df):
  # Shuffle values in ~10% of the attributes in the df
  #
  # Inputs - 
  #     df: The dataframe for the dataset
  # Output - df: original dataframe but with 10% of attributes shuffled
  
  # select how many attributes going to scramble
  # subtract 2 from len becuase class is also a column
  # add one at end becuase flooring and results could be 0
  num_atr_to_shuffle = math.floor(.1*(len(df.columns.values)-2))+1
  
  # randomly select those attributes
  atrs_to_shuffle = random.sample(set(df.columns.values[1:]), 
                                  num_atr_to_shuffle)
  print('reshuffling attribute(s): {}'.format(atrs_to_shuffle))

  # reshuffle values in the columns
  for atr in atrs_to_shuffle:
    df[atr] = np.random.permutation(df[atr].values)

  # return the dataframe with shuffled columns
  return df



# The datasets have been modified to have the class column first followed
# only by the attribute columns. (Index columns have been removed)

### Respository Datasets

## Breast Cancer
print('*** Brest Cancer Dataset ***')
breastdatastr = os.path.join('Data', 'breast-cancer-wisconsin.csv')
df = pd.read_csv(breastdatastr)
# drop index column
df = df.drop(columns=['index'])
# put class first
df = df.reindex(columns=['class', '1', '2', '3', '4', '5', '6', 
                            '7', '8', '9'])
# put columns back to integers
df.columns = range(df.shape[1])
FirstAlgorithm(df, 1)

# reshuffle
df = scramble_features(df)
FirstAlgorithm(df, 1)



## Glass
print('\n*** Glass Dataset ***')
glass_path = os.path.join('Data', 'glass.csv')
df = pd.read_csv(glass_path)
df = df.drop(columns=['index'])
# put class first
df = df.reindex(columns=['class', '1', '2', '3', '4', '5', '6', 
                         '7', '8', '9'])
# bin the columns
print('binning all attributes')
df = bin_column(df, '1', [1.515, 1.52, 1.525])
df = bin_column(df, '2', [12.75])
df = bin_column(df, '3', [1.5, 3])
df = bin_column(df, '4', [1, 1.7, 2.3])
df = bin_column(df, '5', [71, 72.5, 73.4])
df = bin_column(df, '6', [0.4, 1, 4])
df = bin_column(df, '7', [7.5, 9.5, 11.5, 13.25])
df = bin_column(df, '8', [.0001, 0.5, 1, 1.5, 2])
df = bin_column(df, '9', [.0001, .13, .2, .26])
# put columns back to integers
df.columns = range(df.shape[1])
FirstAlgorithm(df, 1)

# reshuffle
df = scramble_features(df)
FirstAlgorithm(df, 1)


## House Votes
print('\n*** House Votes Dataset ***')
house_path = os.path.join('Data', 'house-votes-84.csv')
df = pd.read_csv(house_path, header=None)
FirstAlgorithm(df, 1)

# reshuffle
df = scramble_features(df)
FirstAlgorithm(df, 1)



## Iris
print('\n*** Iris Dataset ***')
iris_path = os.path.join('Data', 'iris.csv')
df = pd.read_csv(iris_path)
# put class first
df = df.reindex(columns=['class', '1', '2', '3', '4'])
# bin the columns
print('binning all attributes')
df = bin_column(df, '1', [5.3, 6, 6.5])
df = bin_column(df, '2', [2.85, 3.2, 3.5])
df = bin_column(df, '3', [2])
df = bin_column(df, '4', [0.7, 1.6])
# put columns back to integers
df.columns = range(df.shape[1])
FirstAlgorithm(df, 1)
# reshuffle
df = scramble_features(df)
FirstAlgorithm(df, 1)


## Soybean Small
print('\n*** Soybean (Small) Dataset ***')
soybean_path = os.path.join('Data', 'soybean-small.csv')
df = pd.read_csv(soybean_path)
df = df.reindex(columns=['class', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                         'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                         'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                         'w', 'x', 'y', 'z'])
df.columns = range(df.shape[1])
FirstAlgorithm(df, 1)

# reshuffle
df = scramble_features(df)
FirstAlgorithm(df, 1)
