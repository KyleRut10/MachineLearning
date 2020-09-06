import numpy as np
import pandas as pd
import os

def FirstAlgorithm(file_loc, clean = 0):
  # A function to implement a basic machine learning algorithm
  #
  # Inputs -
  #     file_loc: The relative location of the csv file
  #     clean: A value specifying which of the cleaning
  #            functions should be used
  
  
  # Reading in the file
  rawdata = pd.read_csv(file_loc, header=None)
  
  # Data Cleaning Station
  if clean == 1:
    # Remove rows with ?s
    raw = rawdata.replace(["?"],np.nan)
    data = raw.dropna(axis=0,how='any')
  else:
    # Leave the ? as it's own answer (a for simplicity)
    # This method is also used if know unkown values
    # exist for the dataset in question
    data = rawdata.replace(["?"],'a')
  
  
  # Training and Testing Area
  # Precalculated Values for Efficiency
  len_data = len(data)
  split = round(len_data/2)
  if file_loc
  classes = data[data.shape[1] - 1].unique().tolist()
  
  Q_C = {}
  for i in classes:
    print(i)
    Q_C[i] = sum(data[data.shape[1]-1] == i)/len_data
  
  # Loops 5 times for performing 5 x 2 Cross Validation
  for i in range(0,4):
    shuffle = data.sample(frac = 1)
    left = shuffle[:split]
    right = shuffle[split:]
    
    # Training on the Left and Testing on the Right


# Function execution
# Breast Cancer
breastdatastr = os.path.join('Data', 'breast-cancer-wisconsin.csv')
FirstAlgorithm(breastdatastr, 1)
