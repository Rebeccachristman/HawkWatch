#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: christman
"""

from __future__ import division, print_function, absolute_import
import sys, getopt
import os
from pathlib import Path
import argparse

import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from datetime import datetime, timedelta


#from sklearn.metrics import classification_report
#from sklearn.metrics import roc_curve, auc
#
#from sklearn.metrics import average_precision_score
#from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import confusion_matrix
#
#from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.metrics import mean_squared_error, r2_score
#from sklearn.metrics import accuracy_score


import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot') # Look Pretty

sys.path.append(os.getcwd())
import imtools as imtools


#column_name_list = ['File', 'Dir', 'Datetime', 'Camera', 'SeqNum', 'SeqLen', 'SeqNumDiff', 'Night', 'Mean', 'Std', 'TopCrop', 'BottomCrop', 'Carcass X', 'Carcass Y', 'Carcass Dist', 'Carcass Size', 'Obscuring Plants', 'NumObj', 'DistRank', 'Size', 'X', 'Y', 'Dist', 'Angle', 'Label' ]

# Funcation code from https://stackoverflow.com/questions/15771472/pandas-rolling-mean-by-time-interval
def time_offset_rolling_mean_df_ser(data_df_ser, window_i_s, min_periods_i=1, center_b=False):
    """ Function that computes a rolling mean

    Credit goes to user2689410 at http://stackoverflow.com/questions/15771472/pandas-rolling-mean-by-time-interval

    Parameters
    ----------
    data_df_ser : DataFrame or Series
         If a DataFrame is passed, the time_offset_rolling_mean_df_ser is computed for all columns.
    window_i_s : int or string
         If int is passed, window_i_s is the number of observations used for calculating
         the statistic, as defined by the function pd.time_offset_rolling_mean_df_ser()
         If a string is passed, it must be a frequency string, e.g. '90S'. This is
         internally converted into a DateOffset object, representing the window_i_s size.
    min_periods_i : int
         Minimum number of observations in window_i_s required to have a value.

    Returns
    -------
    Series or DataFrame, if more than one column

    >>> idx = [
    ...     datetime(2011, 2, 7, 0, 0),
    ...     datetime(2011, 2, 7, 0, 1),
    ...     datetime(2011, 2, 7, 0, 1, 30),
    ...     datetime(2011, 2, 7, 0, 2),
    ...     datetime(2011, 2, 7, 0, 4),
    ...     datetime(2011, 2, 7, 0, 5),
    ...     datetime(2011, 2, 7, 0, 5, 10),
    ...     datetime(2011, 2, 7, 0, 6),
    ...     datetime(2011, 2, 7, 0, 8),
    ...     datetime(2011, 2, 7, 0, 9)]
    >>> idx = pd.Index(idx)
    >>> vals = np.arange(len(idx)).astype(float)
    >>> ser = pd.Series(vals, index=idx)
    >>> df = pd.DataFrame({'s1':ser, 's2':ser+1})
    >>> time_offset_rolling_mean_df_ser(df, window_i_s='2min')
                          s1   s2
    2011-02-07 00:00:00  0.0  1.0
    2011-02-07 00:01:00  0.5  1.5
    2011-02-07 00:01:30  1.0  2.0
    2011-02-07 00:02:00  2.0  3.0
    2011-02-07 00:04:00  4.0  5.0
    2011-02-07 00:05:00  4.5  5.5
    2011-02-07 00:05:10  5.0  6.0
    2011-02-07 00:06:00  6.0  7.0
    2011-02-07 00:08:00  8.0  9.0
    2011-02-07 00:09:00  8.5  9.5
    """

    def calculate_mean_at_ts(ts):
        """Function (closure) to apply that actually computes the rolling mean"""
        if center_b == False:
            dslice_df_ser = data_df_ser[
                ts-pd.datetools.to_offset(window_i_s).delta+timedelta(0,0,1):
                ts
            ]
            # adding a microsecond because when slicing with labels start and endpoint
            # are inclusive
        else:
            dslice_df_ser = data_df_ser[
                ts-pd.datetools.to_offset(window_i_s).delta/2+timedelta(0,0,1):
                ts+pd.datetools.to_offset(window_i_s).delta/2
            ]
        if  (isinstance(dslice_df_ser, pd.DataFrame) and dslice_df_ser.shape[0] < min_periods_i) or \
            (isinstance(dslice_df_ser, pd.Series) and dslice_df_ser.size < min_periods_i):
            return dslice_df_ser.mean()*np.nan   # keeps number format and whether Series or DataFrame
        else:
            return dslice_df_ser.mean()

    if isinstance(window_i_s, int):
        mean_df_ser = pd.rolling_mean(data_df_ser, window=window_i_s, min_periods=min_periods_i, center=center_b)
    elif isinstance(window_i_s, basestring):
        idx_ser = pd.Series(data_df_ser.index.to_pydatetime(), index=data_df_ser.index)
        mean_df_ser = idx_ser.apply(calculate_mean_at_ts)

    return mean_df_ser






def main(argv):
    cwd = os.getcwd()
    print("\n*************** ")
    print("current working directory: ", cwd)

    window_size=str('12s')       # a sequence has 3 (some imagesets have 2) images
    threshold = 0.5      # threshold percentage for sliding window 
    imagedir=""
    csvdata=""
    csvout=""    
    csvauto=""
    csvmanual=""
    try:
   #   #opts, args = getopt.getopt(argv,"hi:o:",["imagedir=","ofile="])
       opts, args = getopt.getopt(argv,"hd:o:a:m:")
   
       #print 'Number of arguments:', len(sys.argv), 'arguments.'
       #print 'Argument List:', str(sys.argv)
       help_string = str(sys.argv[0]) + " -d <image directory> -a <auto classification file> -m <manual classification file>" 
    except getopt.GetoptError:
       print (help_string)
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print (help_string)
          sys.exit()
       elif opt in ("-d", "--imagedir"):
          imagedir = arg
          if not imagedir.endswith('\\') :
              imagedir = imagedir + '\\'
       elif opt in ("-a", "--csvautoclassify"):
          csvauto = arg
       elif opt in ("-m", "--csvmanualclassify"):
          csvmanual = arg
       elif opt in ("-o", "--csvout"):
          csvout = arg
       else:
          print (help_string)
          sys.exit()

    if not imagedir :
        imagedir = "D:\\Photo_set_1\\"
        print("Using default imagedir: ", imagedir)
    if not csvout : 
        csvout="..\\data\\hwi_rollup.csv"
        print("Using default model: ", csvout)
        
    csvdata = imagedir + "hwi_auto_classify_out.csv"

    print("Input label file: " + csvdata + "   Output rollup summary: ", csvout)
    if csvauto :
        print("Using auto classification file: ", csvauto)
    if csvmanual :
        print("Using manual classification file: ", csvauto)
        

    df_data = pd.read_csv(csvdata, sep=',', header=0)
    #df_data = df_data.drop(columns=['SeqNumDiff', 'Mean', 'Std', 'TopCrop', 'BottomCrop', 'Carcass X', 'Carcass Y', 'Carcass Dist', 'Carcass Size', 'Obscuring Plants', 'NumObj', 'DistRank', 'Size', 'X', 'Y', 'Dist', 'Angle' ])
    df_time_label = pd.DataFrame(columns=['Datetime', 'Label', 'RollMean'])
    df_time_label.Datetime = df_data.Datetime.str.replace(':', '-', 2)   # replace : in date with -
    df_time_label.Label = df_data.Label
    
    #df_time_label=
    print("Input labeled data...")
    print(df_time_label.head())

    df_time_label["Datetime"] = pd.to_datetime(df_time_label["Datetime"], format = '%Y-%m-%d %H:%M:%S') # Convert column type to be datetime
    #format = '%Y-%m-%d %H:%M:%S'
    #df['Datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format=format)
    #df_time_label.set_index(pd.DatetimeIndex(df_time_label['Datetime']))
    
    indexed_df = df_time_label.set_index(["Datetime"])           # Create a datetime index
    #indexed_df.rolling(window_size)                   # Create rolling windows
    #df_time_label.rolling(window_size)                   # Create rolling windows

    # ragged time indexed dataframe
    #df_time_label.RollMean = df_time_label.rolling(12).mean()   # is this 30 seconds?
    #df_time_label.RollMean = df_time_label['Label'].rolling(window_size, min_periods=1).mean()   #), center=True).mean()             # Then apply functions to rolling window
    #s = Series(df_time_label.Label, index=df_time_label.Datetime)
    #df_time_label.RollMean = rolling_mean(s, window=window_size)
    df_time_label.RollMean = time_offset_rolling_mean_df_ser(indexed_df, '15s', min_periods_i=1, center_b=False)

    df_time_label.RollMean.fillna(value=0, axis=0, inplace=True)

    print("Auto classified with rolling mean...")
    #print(df_time_label.head())
    #print(df_time_label.count)
    #print("Auto classified dataframe: ", df_time_label.count())
    print("Datetime data type: ", df_time_label.dtypes)

    fig=plt.figure()
    fig.show()
    ax=fig.add_subplot(111)
    
    ax.plot(df_time_label.Datetime, df_time_label.RollMean, color='red', alpha=0.5, linestyle='solid', marker='o', markerfacecolor='None', markersize=5)
    #ax.plot(df_time_label.Datetime, df_time_label.Label, color='purple', linestyle='None', marker='o', markerfacecolor='None', markersize=10)
    
    
    if csvmanual:
        df_manual = pd.read_csv(csvmanual, sep=',', header=0)
  
        df_manual_label = pd.DataFrame(columns=['Datetime', 'Label', 'RollMean'])
        df_manual_label.Datetime = df_manual.Datetime.str.replace(':', '-', 2)   # replace : in date with -
        #df_manual_label["Datetime"] = pd.to_datetime(df_manual_label["Datetime"]) # Convert column type to be datetime
        df_manual_label["Datetime"] = pd.to_datetime(df_manual_label["Datetime"], format = '%Y-%m-%d %H:%M:%S') # Convert column type to be datetime
        #format = '%Y-%m-%d %H:%M:%S'
        #df['Datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format=format)
        df_manual_label.set_index(pd.DatetimeIndex(df_manual_label['Datetime']))

        df_manual_label.Label = df_manual.Label

        df_manual_label.RollMean = df_manual_label['Label'].rolling_mean(window_size, min_periods=1)             # Then apply functions to rolling window
        df_manual_label.RollMean.fillna(value=0, axis=0, inplace=True)
        print("Manually classified with rolling mean...")
        #print(df_manual_label.head())
        #print("Manual dataframe: ", df_manual_label.count())
        print("Datetime data type: ", df_manual_label.dtypes)

        df_result = pd.merge(df_time_label, df_manual_label, on=['Datetime'], how='left', suffixes=['','_man'])
        #df_result.fillna(value=0, axis=0, inplace=True)
        print("Merged dataframe: ", df_result.count())

        #print("After merge...")
        #print(df_result.head())
        #print(df_manual_label.count)
        ax.plot(df_result.Datetime, df_result.RollMean_man, color='orange', linestyle='None', alpha=0.5,  marker='x', markersize=10)
        ax.plot(df_result.Datetime, df_result.Label_man, color='black', linestyle='None', marker='.', alpha=0.5, markersize=5)

    plt.show()

    # write out csv with labels
    #print("Writing rollup summary to output file: ", csvout)
    #df_final.to_csv(csvout, sep=',', index=False)      

    return

     
       
if __name__ == "__main__":
   main(sys.argv[1:])
