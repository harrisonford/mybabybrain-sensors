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
from functions_20191213 import classify_3months_20191213

#%%     Retrieve inputs of the model

descrp = "MyBabyBrain model to analyze accelerometer data - author: Francisco Mena"
parser = argparse.ArgumentParser( 
        description = descrp)

parser.add_argument("filepath_sensor", help = "Filepath to folder with sensor data")
parser.add_argument("filepath_GMtimes", help = "Filepath to csv file with general movement times T0 and Tf")
parser.add_argument("sampling_frequency", nargs='?', const = 2, default = 2, type = int, help = "Sampling frequency of the accelerometer. Default of 2 Hz")

args = parser.parse_args()
print(args)



#%%     Load model
pkl_filename = 'pickle_model_3month_rf_20191002_123121.pkl'
with open(pkl_filename, 'rb') as file:
    scaler, model = pickle.load(file)


#%%

filepath_sensor = args.filepath_sensor
file_GMtimes = args.filepath_GMtimes
freq = args.sampling_frequency

df = pd.read_csv(file_GMtimes, sep = ',', index_col = False)

T0 = list(df.T0)
Tf = list(df.Tf)

# freq = 100
# filepath_sensor = "../Patient_Data/sensor2_dbs"
y_pred = classify_3months_20191213(filepath_sensor, T0, Tf, scaler, model, freq=2)
y_pred.astype(int)
np.savetxt("output.csv", y_pred, delimiter = ',', fmt = '%i')


#%%
### Provide diagnosis
abnormal = 100*sum(y_pred)/len(y_pred)
#normal = 100 - abnormal


print("Analysis completed")
print("Un {:.1f}% de la data de los sensores presenta movimiento anormal".format(abnormal))
