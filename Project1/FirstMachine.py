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
  elif clean == 2:
    # Replace ? with most probable answer for that class
    data = rawdata
  else:
    data = rawdata
  
  
  # Training and Testing for 5 x 2 Cross Validation
  len(data)
  for i in range(0,4):
    shuffle = rawdata
  
    

def RemoveQ(raw):
  # A function which removes ? values
  # input - raw: Raw data to clean
  # output - data: Cleaned output data
  raw = raw.replace(["?"],"NA")
  data = raw.dropna()
  return data
  
def ReplaceAvg(raw):
  # A function which replaces ? values
  # with the mean value of that
  # attribute for that class
  # input - raw: Raw data to clean
  # output - data: Cleaned output data
  return data
  
<<<<<<< HEAD
breastdatastr = 'C:\\Users\\Kyle\\MachineLearning\\Project1\\Data\\breast-cancer-wisconsin.csv'
=======
breastdatastr = os.path.join('Data', 'breast-cancer-wisconsin.csv')
>>>>>>> 5e4eacf32c1b894f5fefb712083438adee3c43c4
FirstAlgorithm(breastdatastr, 1)
