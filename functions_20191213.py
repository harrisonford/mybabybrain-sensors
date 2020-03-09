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
def my_clean(an_element):
    if type(an_element) == str:
        return float("".join([c for c in an_element if c.isdigit()]))
    else:
        return an_element


def clean_the_shit(a_dataframe):

    new_data = a_dataframe  # make a copy
    for a_name in new_data.columns:
        the_array = new_data[a_name]
        new_data[a_name] = [my_clean(an_element) for an_element in the_array]
    return new_data


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


        if sum(dhrj[colu])==0:
            freaks.append( 0 )
        else:
            freaks.append(stats.hmean( np.abs(dhrj[colu])))
        ##
        if sum(dhlj[colu])==0:
            freaks.append( 0 )
        else:
            freaks.append(stats.hmean( np.abs(dhlj[colu])))
        ##
        if sum(dlrj[colu])==0:
            freaks.append( 0 )
        else:
            freaks.append(stats.hmean( np.abs(dlrj[colu])))
        ##
        if sum(dllj[colu])==0:
            freaks.append( 0 )
        else:
            freaks.append(stats.hmean( np.abs(dllj[colu])))
        ##
        if sum(dhipj[colu])==0:
            freaks.append( 0 )
        else:
            freaks.append(stats.hmean( np.abs(dhipj[colu])))
        ##
        if sum(dchj[colu])==0:
            freaks.append( 0 )
        else:
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

def featureMatrix(dhr, dhl, dlr, dll, dch, dhip, T0, Tf, deltt, freq):

    x = "Acc_x [m/s^2]"
    y = "Acc_y [m/s^2]"
    z = "Acc_z [m/s^2]"
    gx = "Gyro_x [1/s]"
    gy = "Gyro_y [1/s]"
    gz = "Gyro_z [1/s]"   
    t = "Time [s]"
    cols = ["Acc_x [m/s^2]", "Acc_y [m/s^2]", "Acc_z [m/s^2]", 'Gyro_x [1/s]', 'Gyro_y [1/s]', 'Gyro_z [1/s]', 'jerk_x', 'jerk_y', 'jerk_z'];

    flag = -1

#ii = 0
    # Loop over all the periods of General Movement
    for ii in range(len(T0)):
        print(ii)

        T0i = T0[ii]
        Tfi = Tf[ii]
        
        timeHR = dhr[t]
        idx = (timeHR >= T0i) & (timeHR <= Tfi)
        # if we dont have any data we just continue
        if not np.any(idx):
            continue

        # timeHR[idx]
                
        dhru = dhr[idx].reset_index( drop = True)
        dhlu = dhl[idx].reset_index( drop = True)
        dlru = dlr[idx].reset_index( drop = True)
        dllu = dll[idx].reset_index( drop = True)
        dchu = dch[idx].reset_index( drop = True)
        dhipu = dhip[idx].reset_index( drop = True)

###     Add jerk
        dT = 1/freq
        dhru['jerk_x'] = np.gradient(dhru[x].values, dT)  
        dhru['jerk_y'] = np.gradient(dhru[y].values, dT)
        dhru['jerk_z'] = np.gradient(dhru[z].values, dT)
                
        dhlu['jerk_x'] = np.gradient(dhlu[x].values, dT)
        dhlu['jerk_y'] = np.gradient(dhlu[y].values, dT)
        dhlu['jerk_z'] = np.gradient(dhlu[z].values, dT)
    
        dlru['jerk_x'] = np.gradient(dlru[x].values, dT)
        dlru['jerk_y'] = np.gradient(dlru[y].values, dT)
        dlru['jerk_z'] = np.gradient(dlru[z].values, dT)
    
        dllu['jerk_x'] = np.gradient(dllu[x].values, dT)
        dllu['jerk_y'] = np.gradient(dllu[y].values, dT)
        dllu['jerk_z'] = np.gradient(dllu[z].values, dT)
    
        dchu['jerk_x'] = np.gradient(dchu[x].values, dT)
        dchu['jerk_y'] = np.gradient(dchu[y].values, dT)
        dchu['jerk_z'] = np.gradient(dchu[z].values, dT)
    
        dhipu['jerk_x'] = np.gradient(dhipu[x].values, dT)
        dhipu['jerk_y'] = np.gradient(dhipu[y].values, dT)
        dhipu['jerk_z'] = np.gradient(dhipu[z].values, dT)



###     Divide into segments of length deltt
        
        for shift in np.array([0,333,666]):
            for pp in range(0, int(len(dhru[t])/deltt)+1 ):           #If a time series is below 10 s, it is skipped    
                
                
#                shift = 0; pp = 0; ii = 0
                if(shift+deltt*(pp+1) < len(dhru[t])):
                    dhrj = dhru[int(shift+deltt*(pp)) : int(shift+deltt*(pp+1))].copy()
                    dhlj = dhlu[int(shift+deltt*(pp)) : int(shift+deltt*(pp+1))].copy()
                    dlrj = dlru[int(shift+deltt*(pp)) : int(shift+deltt*(pp+1))].copy()
                    dllj = dllu[int(shift+deltt*(pp)) : int(shift+deltt*(pp+1))].copy()
                    dchj = dchu[int(shift+deltt*(pp)) : int(shift+deltt*(pp+1))].copy()
                    dhipj = dhipu[int(shift+deltt*(pp)) : int(shift+deltt*(pp+1))].copy()
                
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
        
                if(shift == 0) and (pp == 0) and (ii == 0) or (flag == -1):
#                    print("HOLY MOLI GUACAMOLI")
                    matrix = np.array([freaks])
                    flag = 1
                
                else:
                    matrix = np.append(matrix, np.array([freaks]), axis = 0)

    
    
    return matrix



#%%
    
def classify_3months_20191213(filepath, T0, Tf, scaler, model, freq):

#    umbral = 80
    deltt = freq*5  #5 seconds of measurements
#    deltt = 500    
    ### Reading sensor data

#filepath = filepath_sensor
    folder_loc = Path(filepath)
    if os.path.isfile(folder_loc / "SensorData_RightUpperArm.csv"):
        dhr = clean_the_shit(pd.read_csv(folder_loc / "SensorData_RightUpperArm.csv", sep = ',', index_col=False))
        dhl = clean_the_shit(pd.read_csv(folder_loc / "SensorData_LeftUpperArm.csv", sep = ',', index_col=False))
        dlr = clean_the_shit(pd.read_csv(folder_loc / "SensorData_RightThigh.csv", sep = ',', index_col=False))
        dll = clean_the_shit(pd.read_csv(folder_loc / "SensorData_LeftThigh.csv", sep = ',', index_col=False))
        dch = clean_the_shit(pd.read_csv(folder_loc / "SensorData_ChestBottom.csv", sep = ',', index_col=False))
        dhip = clean_the_shit(pd.read_csv(folder_loc / "SensorData_Hip.csv", sep = ',', index_col=False))
        
    else: 
        sys.exit("Wrong filename in the folder. Make sure the files are correctly named")

    ### We need to prune the arrays because lenghts may be irregular
    min_len = min(len(dhr), len(dhl), len(dlr), len(dll), len(dch), len(dhip))
    dhr = dhr[0:min_len]
    dhl = dhl[0:min_len]
    dlr = dlr[0:min_len]
    dll = dll[0:min_len]
    dch = dch[0:min_len]
    dhip = dhip[0:min_len]
    
    ### T0 and Tf already are in seconds

    ### Calculate the feature matrix
        
    X_test = featureMatrix(dhr, dhl, dlr, dll, dch, dhip, T0, Tf, deltt, freq)
    
    X_test_scaled = scaler.transform(X_test)
    
    y_pred = model.predict(X_test_scaled)
    
    
###
    ### Provide diagnosis
    
#    abnormal = 100*sum(y_pred)/len(y_pred)
#    normal = 100 - abnormal
    

#    if normal > umbral:
#        print("A {:.1f}% of the signal presents normal movement".format(normal))
#    
#    else:
#        print("A {:.1f}% of the signal presents abnormal movement".format(abnormal))
#        print("Please see a professional for a more detailed assessment")
    print("Analysis completed")
        
    return y_pred

