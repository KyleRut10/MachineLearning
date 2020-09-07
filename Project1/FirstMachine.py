import numpy as np
import pandas as pd
import os

def FirstAlgorithm(rawdata, clean = 0):
  # A function to implement a basic machine learning algorithm
  #
  # Inputs -
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
  
  results = []
  # Loops 5 times for performing 5 x 2 Cross Validation
  for i in range(0,4):
    shuffle = data.sample(frac = 1)
    left = shuffle[:split]
    right = shuffle[split:]
    
    # Training on the Left and Testing on the Right
    results[2i] = test(right, train(left))
    
    # Training on the Right and Testing on the Left
    results[2i + 1] = test(left, train(right))

def train(data):
  # The training function for this algorithm
  #
  # Input - data: The training data for this algorithm
  # Output - storage
  
  storage = {}
  
  classes = data[0].unique().tolist()
  
  Q_C = {}
  for i in classes:
    Q_C[i] = sum(data[0] == i)/len_data
  
  storage[0] = Q_C
  
  for k in range(1: data.shape[1] - 1):
    levels = data[k].unique.tolist()
    matrix = np.zeroes((len(classes),len(levels)))
    for i in range(0, len(classes)):
      nc = sum(data[0] = classes[i])
      for j in range(0, len(levels)):
        matrix[i,j] = (sum(data[0] = classes[i] && data[k] = levels[j])\
        + 1)/(nc + data.shape[1] - 2)
    storage[k] = matrix

return(storage)

def test(data, trained)
  # The testing function for this algorithm
  #
  # Inputs - 
  #   data: The testing data for this algorithm
  #   trained:
  # Output - result: The confusion matrix of the classification

  #For loop of classes


# The datasets have been modified to have the class column first followed
# only by the attribute columns. (Index columns have been removed)

### Respository Datasets

## Breast Cancer
breastdatastr = os.path.join('Data', 'breast-cancer-wisconsin.csv')
df = pd.read_csv(breastdatastr)
# drop index column
df = df.drop(columns=['index'])
# put class first
df = df.reindex(columns=['class', '1', '2', '3,', '4', '5', '6', 
                            '7', '8', '9'])
# put columns back to integers
df.columns = range(df.shape[1])
FirstAlgorithm(df, 1)

## Glass
glass_path = os.path.join('Data', 'glass.csv')
df = pd.read_csv(glass_path)
df = df.drop(columns=['index'])
# put class first
df = df.reindex(columns=['class', '1', '2', '3,', '4', '5', '6', 
                         '7', '8', '9'])
# put columns back to integers
df.columns = range(df.shape[1])
FirstAlgorithm(df, 1)


## House Votes
house_path = os.path.join('Data', 'house-votes-84.csv')
df = pd.read_csv(house_path, header=None)
FirstAlgorithm(df, 1)


## Iris
iris_path = os.path.join('Data', 'iris.csv')
df = pd.read_csv(iris_path)
# put class first
df = df.reindex(columns=['class', '1', '2', '3,', '4'])
# put columns back to integers
df.columns = range(df.shape[1])
FirstAlgorithm(df, 1)


## Soybean Small
soybean_path = os.path.join('Data', 'soybean-small.csv')
df = pd.read_csv(soybean_path)
df = df.reindex(columns=['class', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                         'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                         'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                         'w', 'x', 'y', 'z'])
df.columns = range(df.shape[1])
FirstAlgorithm(df, 1)

### Noisy Datasets
