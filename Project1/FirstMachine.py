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
  
  # Precalculated Values for Efficiency
  len_data = len(data)
  split = round(len_data/2)
  classes = data[0].unique().tolist()
  
  Q_C = {}
  for i in classes:
    Q_C[i] = sum(data[0] == i)/len_data
  
  results = []
  # Loops 5 times for performing 5 x 2 Cross Validation
  for i in range(0,4):
    shuffle = data.sample(frac = 1)
    left = shuffle[:split]
    right = shuffle[split:]
    
    # Training on the Left and Testing on the Right
    results[2i] = test(right, train(left, classes))
    
    # Training on the Right and Testing on the Left
    results[2i + 1] = test(left, train(right), Q_C)

def train(data, classes):
  # The training function for this algorithm
  #
  # Inputs - 
  #   data: The training data for this algorithm
  #   classes: The list of classes in the dataset
  # Output - matrix: Returns the functions F which specify
  #          the number of examples that match an attribute
  #          value (plus one) divided by the number of examples
  #          in the class (plus the total number of attributes)
  #          for each class in the training data
  
  matrix = np.zeros((len(classes),data.shape[1] - 1))
  
  return matrix

def test(data, matrix, Q_C)
  # The testing function for this algorithm
  #
  # Inputs - 
  #   data: The testing data for this algorithm
  #   matrix: The values used to predict a class given a
  #           set of attributes
  #   Q_C: The array of Q(C = ci) values to multiply
  # Output - result: The confusion matrix of the classification



# The datasets have been modified to have the class column first followed
# only by the attribute columns. (Index columns have been removed)

## Respository Datasets

# Breast Cancer
breastdatastr = os.path.join('Data', 'breast-cancer-wisconsin.csv')
<<<<<<< HEAD
brest_df = pd.read_csv(breastdatastr)
# drop index column
brest_df = brest_df.drop(columns=['index'])
# put class first
brest_df = brest_df.reindex(columns=['class', '1', '2', '3,', '4', '5', '6', 
                            '7', '8', '9'])
# put columns back to integers
brest_df.columns = range(brest_df.shape[1])
FirstAlgorithm(brest_df, 1)
=======
FirstAlgorithm(breastdatastr)
>>>>>>> c4250f6fa64b62427f6569b80458411703332c85

# Glass
glass_path = os.path.join('Data', 'glass.csv')
df = pd.read_csv(glass_path)
df = df.drop(columns=['index'])
# put class first
df = df.reindex(columns=['class', '1', '2', '3,', '4', '5', '6', 
                         '7', '8', '9'])
# put columns back to integers
df.columns = range(df.shape[1])
FirstAlgorithm(df, 1)


# House Votes
house_path = os.path.join('Data', 'house-votes-84.csv')
df = pd.read_csv(house_path, header=None)
FirstAlgorithm(df, 1)


# Iris
iris_path = os.path.join('Data', 'iris.csv')
df = pd.read_csv(iris_path)
# put class first
df = df.reindex(columns=['class', '1', '2', '3,', '4'])
# put columns back to integers
df.columns = range(df.shape[1])
FirstAlgorithm(df, 1)


# Soybean Small
soybean_path = os.path.join('Data', 'soybean-small.csv')
df = pd.read_csv(soybean_path)
df = df.reindex(columns=['class', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                         'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                         'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                         'w', 'x', 'y', 'z'])
df.columns = range(df.shape[1])
FirstAlgorithm(df, 1)
