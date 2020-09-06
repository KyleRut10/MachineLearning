import numpy as np
import pandas as pd
import os

def FirstAlgorithm(rawdata, clean = 0):
  # A function to implement a basic machine learning algorithm
  #
  # Inputs -
  #     rawdata: A pandas dataframe where the first column
  #              contains the class and the following ones
  #              contain the attributes
  #     clean: A value specifying which of the cleaning
  #            functions should be used. Defaults to
  #            removing rows with na values
  
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
    results[2i] = test(right, train(left))
    
    # Training on the Right and Testing on the Left
    results[2i + 1] = test(left, train(right))



# The datasets have been modified to have the class column first followed
# only by the attribute columns. (Index columns have been removed)

## Respository Datasets

# Breast Cancer
breastdatastr = os.path.join('Data', 'breast-cancer-wisconsin.csv')
FirstAlgorithm(breastdatastr)

# Glass

# House Votes

# Iris

# Soybean Small

## Noisy Datasets (Pulls from pandas datasets above and adds noise to them)

# Breast Cancer

# Glass

# House Votes

# Iris

# Soybean Small
