# -*- coding: utf-8 -*-
"""
Created on Oct 3 2019

All the functions used for classification of 3-month old infants
"""

import pandas as pd
import os
import sys
from pathlib import Path

import numpy as np
from scipy import signal, stats


#%%

# Calculate the number of movements per second (2 peaks = 1 movement)
# Input: segnal = a time series
def numpeaks(segnal):
    
    stu = np.std(segnal)
    time = len(segnal)/100
    pkfq, _ = signal.find_peaks(segnal, height = stu)   #Find peaks

    vallfq, _ = signal.find_peaks(-segnal, height = stu)    #Find valleys

#    numpeaks = ( len(pkfq) + len(vallfq) )//(2 * time)

    numpeaks = ( len(pkfq) + len(vallfq) )/(2 * time)

    return numpeaks

#%%
    
#Working version of numpy gradient

def homemadegradient(vec, deltt):
    
    grad = [(vec[i+1]-vec[i-1])/(2*deltt) for i in range(1,len(vec)-1)]

    grad.insert(0, (vec[1] - vec[0])/deltt )
    
    grad.append( (vec[-1] - vec[-2])/deltt )
    
    return grad


#%%

def features_per_signal(dhrj, dhlj, dlrj, dllj, dchj, dhipj):

    freaks = list()
    cols = ["Acc_x [m/s^2]", "Acc_y [m/s^2]", "Acc_z [m/s^2]", 'Gyro_x [1/s]', 'Gyro_y [1/s]', 'Gyro_z [1/s]', 'jerk_x', 'jerk_y', 'jerk_z'];
        
    for colu in cols:
    
        freaks.append(stats.iqr( dhrj[colu]))
        freaks.append(stats.iqr( dhlj[colu]))
        freaks.append(stats.iqr( dlrj[colu]))
        freaks.append(stats.iqr( dllj[colu]))
        freaks.append(stats.iqr( dhipj[colu]))
        freaks.append(stats.iqr( dchj[colu]))
        
        freaks.append(stats.gmean( np.abs(dhrj[colu])))
        freaks.append(stats.gmean( np.abs(dhlj[colu])))
        freaks.append(stats.gmean( np.abs(dlrj[colu])))
        freaks.append(stats.gmean( np.abs(dllj[colu])))
        freaks.append(stats.gmean( np.abs(dhipj[colu])))
        freaks.append(stats.gmean( np.abs(dchj[colu])))                
        
        freaks.append(stats.hmean( np.abs(dhrj[colu])))
        freaks.append(stats.hmean( np.abs(dhlj[colu])))
        freaks.append(stats.hmean( np.abs(dlrj[colu])))
        freaks.append(stats.hmean( np.abs(dllj[colu])))
        freaks.append(stats.hmean( np.abs(dhipj[colu])))
        freaks.append(stats.hmean( np.abs(dchj[colu])))                
         
        freaks.append(np.quantile( np.abs(dhrj[colu]), 0.90))
        freaks.append(np.quantile( np.abs(dhlj[colu]), 0.90))
        freaks.append(np.quantile( np.abs(dlrj[colu]), 0.90))
        freaks.append(np.quantile( np.abs(dllj[colu]), 0.90))
        freaks.append(np.quantile( np.abs(dhipj[colu]), 0.90))
        freaks.append(np.quantile( np.abs(dchj[colu]), 0.90))    
    
        freaks.append(np.quantile( np.abs(dhrj[colu]), 0.20))
        freaks.append(np.quantile( np.abs(dhlj[colu]), 0.20))
        freaks.append(np.quantile( np.abs(dlrj[colu]), 0.20))
        freaks.append(np.quantile( np.abs(dllj[colu]), 0.20))
        freaks.append(np.quantile( np.abs(dhipj[colu]), 0.20))
        freaks.append(np.quantile( np.abs(dchj[colu]), 0.20))
    
    return freaks


#%%  Calculate the matrix of features

def featureMatrix(dhr, dhl, dlr, dll, dch, dhip, T0, Tf, deltt):

    x = "Acc_x [m/s^2]"
    y = "Acc_y [m/s^2]"
    z = "Acc_z [m/s^2]"
    gx = "Gyro_x [1/s]"
    gy = "Gyro_y [1/s]"
    gz = "Gyro_z [1/s]"   
    t = "Time [s]"
    cols = ["Acc_x [m/s^2]", "Acc_y [m/s^2]", "Acc_z [m/s^2]", 'Gyro_x [1/s]', 'Gyro_y [1/s]', 'Gyro_z [1/s]', 'jerk_x', 'jerk_y', 'jerk_z'];
#ii = 1

    # Loop over all the periods of General Movement
    for ii in range(len(T0)):

        
        T0i = T0[ii]
        Tfi = Tf[ii]
        
        timeHR = dhr[t]
        idx = (timeHR >= T0i) & (timeHR <= Tfi)
                
        dhru = dhr[idx].reset_index( drop = True)
        dhlu = dhl[idx].reset_index( drop = True)
        dlru = dlr[idx].reset_index( drop = True)
        dllu = dll[idx].reset_index( drop = True)
        dchu = dch[idx].reset_index( drop = True)
        dhipu = dhip[idx].reset_index( drop = True)

###     Add jerk
        dhru['jerk_x'] = np.gradient(dhru[x].values, 0.01)        
        dhru['jerk_y'] = np.gradient(dhru[y].values, 0.01)
        dhru['jerk_z'] = np.gradient(dhru[z].values, 0.01)
                
        dhlu['jerk_x'] = np.gradient(dhlu[x].values, 0.01)
        dhlu['jerk_y'] = np.gradient(dhlu[y].values, 0.01)
        dhlu['jerk_z'] = np.gradient(dhlu[z].values, 0.01)
    
        dlru['jerk_x'] = np.gradient(dlru[x].values, 0.01)
        dlru['jerk_y'] = np.gradient(dlru[y].values, 0.01)
        dlru['jerk_z'] = np.gradient(dlru[z].values, 0.01)
    
        dllu['jerk_x'] = np.gradient(dllu[x].values, 0.01)
        dllu['jerk_y'] = np.gradient(dllu[y].values, 0.01)
        dllu['jerk_z'] = np.gradient(dllu[z].values, 0.01)
    
        dchu['jerk_x'] = np.gradient(dchu[x].values, 0.01)
        dchu['jerk_y'] = np.gradient(dchu[y].values, 0.01)
        dchu['jerk_z'] = np.gradient(dchu[z].values, 0.01)
    
        dhipu['jerk_x'] = np.gradient(dhipu[x].values, 0.01)
        dhipu['jerk_y'] = np.gradient(dhipu[y].values, 0.01)
        dhipu['jerk_z'] = np.gradient(dhipu[z].values, 0.01)



###     Divide into segments of length deltt
        
        for shift in np.array([0,333,666]):
            for pp in range(0, int(len(dhru[t])/deltt)+1 ):           #If a time series is below 10 s, it is skipped    
                
                if(shift+deltt*(pp+1) < len(dhru[t])):
                    dhrj = dhru[shift+deltt*(pp) : shift+deltt*(pp+1)].copy()
                    dhlj = dhlu[shift+deltt*(pp) : shift+deltt*(pp+1)].copy()
                    dlrj = dlru[shift+deltt*(pp) : shift+deltt*(pp+1)].copy()
                    dllj = dllu[shift+deltt*(pp) : shift+deltt*(pp+1)].copy()
                    dchj = dchu[shift+deltt*(pp) : shift+deltt*(pp+1)].copy()
                    dhipj = dhipu[shift+deltt*(pp) : shift+deltt*(pp+1)].copy()
                
                elif( (shift==0)&(pp==0)&(deltt>len(dhru[t])) ):
                    dhrj = dhru.copy()
                    dhlj = dhlu.copy()
                    dlrj = dlru.copy()
                    dllj = dllu.copy()
                    dchj = dchu.copy()
                    dhipj = dhipu.copy()
                    
                else:
                    continue
    
    
###     Remove a posible drift in the signal
    
                for colu in cols:
                    dhrj.loc[:,colu] = signal.detrend(dhrj[colu], type = 'constant')
                    dhlj.loc[:,colu] = signal.detrend(dhlj[colu], type = 'constant')
                    dlrj.loc[:,colu] = signal.detrend(dlrj[colu], type = 'constant')
                    dllj.loc[:,colu] = signal.detrend(dllj[colu], type = 'constant')
                    dchj.loc[:,colu] = signal.detrend(dchj[colu], type = 'constant')
                    dhipj.loc[:,colu] = signal.detrend(dhipj[colu], type = 'constant')
                        
                        
    
###     Features 
                
                freaks = features_per_signal(dhrj, dhlj, dlrj, dllj, dchj, dhipj)
        
                if(shift == 0) and (pp == 0) and (ii == 0):
                    
                    matrix = np.array([freaks])
                
                else:
                    matrix = np.append(matrix, np.array([freaks]), axis = 0)


    return matrix



#%%
    
def classify_3months(filepath, T0, Tf, scaler, model):

    umbral = 80
    deltt = 500
    ### Reading sensor data
    
    folder_loc = Path(filepath)
    if os.path.isfile(folder_loc / "SensorData_RightUpperArm.csv"):
        dhr = pd.read_csv(folder_loc / "SensorData_RightUpperArm.csv", sep = ',', index_col=False)
        dhl = pd.read_csv(folder_loc / "SensorData_LeftUpperArm.csv", sep = ',', index_col=False)
        dlr = pd.read_csv(folder_loc / "SensorData_RightThigh.csv", sep = ',', index_col=False)
        dll = pd.read_csv(folder_loc / "SensorData_LeftThigh.csv", sep = ',', index_col=False)
        dch = pd.read_csv(folder_loc / "SensorData_ChestBottom.csv", sep = ',', index_col=False)
        dhip = pd.read_csv(folder_loc / "SensorData_Hip.csv", sep = ',', index_col=False)
        
    else: 
        sys.exit("Wrong filename in the folder. Make sure the files are correctly named")

###
    ### Convert time strings to seconds
    T0 = [60*int(x[-5:-3]) + int(x[-2:]) for x in T0]

    Tf = [60*int(x[-5:-3]) + int(x[-2:]) for x in Tf]

    ### Return the feature matrix
        
    X_test = featureMatrix(dhr, dhl, dlr, dll, dch, dhip, T0, Tf, deltt)
    
    X_test_scaled = scaler.transform(X_test)
    
    y_pred = model.predict(X_test_scaled)
    
    
###
    ### Provide diagnosis
    
    abnormal = 100*sum(y_pred)/len(y_pred)
    normal = 100 - abnormal
    

    if normal > umbral:
        print("A {:.1f}% of the data presents normal movement".format(normal))
    
    else:
        print("A {:.1f}% of the data presents abnormal movement".format(abnormal))
#        print("Please see a professional for a more detailed assessment")

        
    return y_pred

