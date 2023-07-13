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
import pickle

from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from sklearn import preprocessing
from sklearn import decomposition
from sklearn import manifold

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report


from sklearn.metrics import roc_curve, auc

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score

#matplotlib.style.use('ggplot') # Look Pretty

sys.path.append(os.getcwd())
import imtools as imtools


column_name_list = ['File', 'Dir', 'Datetime', 'Camera', 'SeqNum', 'SeqLen', 'SeqNumDiff', 'Night', 'Mean', 'Std', 'TopCrop', 'BottomCrop', 'Carcass X', 'Carcass Y', 'Carcass Dist', 'Carcass Size', 'Obscuring Plants', 'NumObj', 'DistRank', 'Size', 'X', 'Y', 'Dist', 'Angle', 'Label' ]

def main(argv):
    cwd = os.getcwd()
    print("\n*************** ")
    print("current working directory: ", cwd)

    imagedir=""
    modelfile=""
    csvdata=""
    csvout=""    
    try:
   #   #opts, args = getopt.getopt(argv,"hi:o:",["imagedir=","ofile="])
       opts, args = getopt.getopt(argv,"hd:m:i:o:")
   
       #print 'Number of arguments:', len(sys.argv), 'arguments.'
       #print 'Argument List:', str(sys.argv)
       help_string = str(sys.argv[0]) + " -d <image directory> -m <model file> -i <input sequence data file> -o <classification output file>" 
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
       elif opt in ("-m", "--modelfile"):
          modelfile = arg
       elif opt in ("-i", "--csvdata"):
          csvdata = arg
       elif opt in ("-o", "--csvout"):
          csvout = arg
       else:
          print (help_string)
          sys.exit()

    if not imagedir :
        imagedir = "D:\\Photo_set_1\\"
        print("Using default imagedir: ", imagedir)
    if not csvdata :
        csvdata=imagedir+"hwi_sequence_out.csv"    
        print("Using default sequence data file: ", csvdata)
    if not modelfile : 
        modelfile="..\\data\\hwi_classifier_model.pkl"
        print("Using default model: ", modelfile)
    if not csvout : 
        csvout=imagedir+"hwi_auto_classify_out.csv"    
        print("Using default output file: ", csvout)

    print("Image directory: " + imagedir + "   trained model file: ", modelfile)
    print("Input Sequence data file: ", csvdata, "Classificiation output file: ", csvout)

    # load the pickled model
    print("Load pickled model from file: ", modelfile)  
    model_pkl = open(modelfile, 'rb')
    clf = pickle.load(model_pkl)
    # Close the pickle instances
    model_pkl.close()

    df_data = pd.read_csv(csvdata, sep=',', header=0)
    #df_data  = df_data[df_data['Night'] != 1]          # not sure if night images should be dropped
    #df_seq = df_data.drop(columns=['File', 'Dir', 'Datetime', 'Camera', 'SeqNum', 'SeqLen', 'SeqNumDiff', 'Night', 'TopCrop', 'BottomCrop', 'Carcass X', 'Carcass Y', 'Size', 'X', 'Y', 'DistRank'])
    #df_seq = df_data.drop(columns=['File', 'Dir', 'Datetime', 'Camera', 'SeqNum', 'SeqLen', 'SeqNumDiff', 'TopCrop', 'BottomCrop', 'Carcass X', 'Carcass Y', 'X', 'Y', 'DistRank'])
    #df_seq = df_data.drop(columns=['File', 'Dir', 'Datetime', 'Camera', 'SeqNum', 'SeqLen', 'SeqNumDiff', 'Night', 'Mean', 'Std', 'TopCrop', 'BottomCrop', 'Carcass X', 'Carcass Y', 'Carcass Dist', 'Carcass Size', 'Obscuring Plants', 'X', 'Y', 'DistRank'])
    #df_seq = df_data.drop(columns=['File', 'Dir', 'Datetime', 'Camera', 'SeqNum', 'SeqLen', 'SeqNumDiff', 'Night', 'Mean', 'Std', 'TopCrop', 'BottomCrop', 'Carcass X', 'Carcass Y', 'X', 'Y', 'DistRank'])

    df_seq = df_data.drop(columns=['File', 'Dir', 'Datetime', 'Camera', 'SeqNum', 'SeqLen', 'SeqNumDiff', 'Night', 'Mean', 'Std', 'TopCrop', 'BottomCrop', 'Carcass X', 'Carcass Y', 'X', 'Y', 'DistRank'])
    df_seq = df_seq.drop(columns=['Label'])  # use with the merged label file
    #df_seq = df_seq.drop(columns=['Carcass Size'])


    print("Input sequence data...")
    print(df_seq.head())

    # Fill or drop NaN. Are the number of objects zero for these? 
    df_seq.fillna(value=0, axis=0, inplace=True)
    #filtered_labels = df_data.loc[:, 'Label']
    
    #Normalize features
    #Normalizer(), MaxAbsScaler(), MinMaxScaler(), KernelCenterer(), and StandardScaler()
    do_Scaling = True
    if do_Scaling == True :
        print("Scaling the features. This needs to be the same as scaling done to train the model.")
        #pre_proc = preprocessing.MinMaxScaler()
        pre_proc = preprocessing.StandardScaler()
        #pre_proc = preprocessing.RobustScaler()   # Robust was slightly better than Standard
        #pre_proc = preprocessing.Normalizer()
        pre_proc.fit(df_seq)
        data_scaled = pre_proc.transform(df_seq)

    print(data_scaled.shape)
    print(data_scaled[0:5])
    
    label_predicted = clf.predict(data_scaled)
    df_label_predicted = pd.DataFrame(data=label_predicted, columns=['Label'])
#May need to concat the predicted labels with df_seq then do the left join with the original data    
    df_final = pd.merge(df_data, df_label_predicted, how='left',left_index=True, right_index=True) 



    # write out csv with labels
    print("Writing classificiation output file: ", csvout)
    df_final.to_csv(csvout, sep=',', index=False)      

    return

     
       
if __name__ == "__main__":
   main(sys.argv[1:])
