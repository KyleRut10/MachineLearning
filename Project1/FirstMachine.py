import numpy as np
import pandas as pd
import os
import random

def FirstAlgorithm(file_loc, clean):
  #
  
  
  # Reading in the file
  rawdata = pd.read_csv(file_loc)
  
  # Data Cleaning Station
  if clean == 1:
    # Remove rows with ?s
    data = RemoveQ(rawdata)
  else:
    # Leave the ? as it's own answer (a for simplicity)
    # This method is also used if know unkown values
    # exist for the dataset in question
    data = rawdata.replace(["?"],"NA")
  
  
  # Training and Testing for 5 x 2 Cross Validation
  len_data = len(data)
  for i in range(0,4):
    shuffle = rawdata
  
    

def RemoveQ(raw):
  # A function which removes ? values
  #
  # input - raw: Raw data to clean
  # output - data: Cleaned output data
  
  raw = raw.replace(["?"],"NA")
  data = raw.dropna()
  return data

breastdatastr = os.path.join('Data', 'breast-cancer-wisconsin.csv')
FirstAlgorithm(breastdatastr, 1)
