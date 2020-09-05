import numpy as np
import pandas as pd
import os

def FirstAlgorithm(file_loc, classrow, clean):
  #
  
  
  # Reading in the file
  f = open(file_loc, "r")
  rawdata = f.read()
  
  # Data Cleaning Station
  if clean == 1:
    data = rawdata
    # Remove rows with ?s
  elif clean == 2:
    # Replace ? with most probable answer for that class
    data = rawdata
  else:
    data = rawdata
    
  # Training and Testing for 5 x 2 Cross Validation
  #for 
  
    

def RemoveQ(raw):
  # A function which removes ? values
  # input - raw: Raw data to clean
  # output - data: Cleaned output data
  return data
  
def ReplaceAvg(raw):
  # A function which replaces ? values
  # with the mean value of that
  # attribute for that class
  # input - raw: Raw data to clean
  # output - data: Cleaned output data
  return data
  
breastdatastr = os.path.join('Data', 'breast-cancer-wisconsin.csv')
FirstAlgorithm(breastdatastr, 11, 1)
