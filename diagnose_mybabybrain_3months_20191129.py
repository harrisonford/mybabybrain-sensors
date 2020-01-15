# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 15:20:39 2019

Script to load model, read sensor data, and provide diagnosis
@author: Francisco Mena

Input parameters:
    - filepath_sensor: Filepath to the folder with the sensor data.
    The folder should be located in the folder "Patient_data"
    - filepath_GMtimes: Filepath to the csv file with the start times (T0) and}
    end times of general movement periods

Other parameters:
    - scaler: scales the data. It is loaded together with the model
    - model: the classifier model.

Output:
    - Text file (csv) of an array with the diagnosis, each window of the signal 
    classified as normal movement (0) or abnormal movement (1)


@author: Francisco Mena
"""

import pickle
import numpy as np
import pandas as pd
import argparse
from functions_20191003 import classify_3months

#%%     Retrieve inputs of the model

descrp = "MyBabyBrain model to analyze accelerometer data - author: Francisco Mena"
parser = argparse.ArgumentParser( 
        description = descrp)

parser.add_argument("filepath_sensor", help = "Filepath to folder with sensor data")
parser.add_argument("filepath_GMtimes", help = "Filepath to csv file with general movement times T0 and Tf")

args = parser.parse_args()
print(args)



#%%     Load model

pkl_filename = 'pickle_model_3month_rf_20191002_123121.pkl'
with open(pkl_filename, 'rb') as file:
    scaler, model = pickle.load(file)



#%%

filepath_sensor = args.filepath_sensor

file_GMtimes = args.filepath_GMtimes

#file_GMtimes = "../zFinal/Patient_Data/3030CSV_GMtimes.csv"
df = pd.read_csv(file_GMtimes, sep = ',', index_col=False)

T0 = list(df.T0)
Tf = list(df.Tf)

#filepath_sensor = "../zFinal/Patient_Data/3030CSV"
y_pred = classify_3months(filepath_sensor, T0, Tf, scaler, model)

y_pred.astype(int)

np.savetxt("output.csv", y_pred, delimiter = ',', fmt = '%i')


#%%     Examples

### Here are examples to run as inputs in the model

## 1. Abnormal movement
#filepath = '../Patient_data/3030CSV'
#T0 = ['0:03:05', '0:04:12']
#Tf = ['0:03:24', '0:04:22']
#
## 2. Abnormal movement
#filepath = '../Patient_data/3087CSV'
#T0 = ['0:01:23', '0:01:45', '0:02:11', '0:02:27', '0:04:30']
#Tf = ['0:01:38', '0:02:04', '0:02:20', '0:03:08', '0:05:14']

## 3. Normal movement
#filepath = '../Patient_data/3090CSV'
#T0 = ['0:02:46', '0:03:15', '0:03:59']
#Tf = ['0:03:02', '0:03:28', '0:04:16']

#%%     Run model
    
#y_pred = classify_3months(filepath, T0, Tf, scaler, model)



