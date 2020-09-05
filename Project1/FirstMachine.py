import numpy as np
import pandas as pd
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
    data =
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
  
breastdatastr = 'C:\\Users\\Kyle\\MachineLearning\\Project1\\Data\\breast-cancer-wisconsin.csv'
FirstAlgorithm(breastdatastr, 1)
