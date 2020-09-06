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
  rawdata = pd.read_csv(file_loc)
  
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
  
  # Training and Testing for 5 x 2 Cross Validation
  len_data = len(data)
  for i in range(0,4):
    shuffle = data.sample(frac = 1)

  
  


#Function execution
breastdatastr = os.path.join('Data', 'breast-cancer-wisconsin.csv')
FirstAlgorithm(breastdatastr, 1)
