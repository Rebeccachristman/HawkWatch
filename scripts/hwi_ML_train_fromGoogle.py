#!/usr/bin/python
# -*- coding: utf-8 -*-
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


column_name_list = ['File', 'Dir', 'Datetime', 'Camera', 'SeqNum', 'SeqLen', 'SeqNumDiff', 'Night', 'Mean', 'Std', 'TopCrop', 'BottomCrop', 'Deer_X', 'Deer_Y', 'NumObj', 'DistRank', 'Size', 'X', 'Y', 'Dist', 'Angle', 'Label' ]

def main(argv):
    cwd = os.getcwd()
    print("\n*************** ")
    print("current working directory: ", cwd)

    csvtrain="..\\data\\tmp_labeled_data.csv"
    modelout="..\\data\\hwi_classifier_model.pkl"    
    try:
   #   #opts, args = getopt.getopt(argv,"hi:o:",["imagedir=","ofile="])
       opts, args = getopt.getopt(argv,"hi:m:")
   
       #print 'Number of arguments:', len(sys.argv), 'arguments.'
       #print 'Argument List:', str(sys.argv)
       help_string = str(sys.argv[0]) + "-i <labeled training data file> -m <model file>" 
    except getopt.GetoptError:
       print (help_string)
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print (help_string)
          sys.exit()
       elif opt in ("-i", "--labeleddatafile"):
          csvtrain = arg
       elif opt in ("-m", "--modeloutfile"):
          modelout = arg
       else:
          print (help_string)
          sys.exit()
    print("Labled training data file: " + csvtrain + "   trained model file: ", modelout)


    df_data=pd.read_csv(csvtrain, sep=',', header=0)
    #df_data  = df_data[df_data['Night'] != 1]          # not sure if night images should be dropped
    df_data = df_data.drop(columns=['File', 'Dir', 'Datetime', 'Camera', 'SeqNum', 'SeqLen', 'SeqNumDiff', 'Night', 'TopCrop', 'BottomCrop', 'Deer_X', 'Deer_Y', 'Size', 'X', 'Y', 'DistRank'])
    print("Labeled training data...")
    print(df_data.head())

    # Fill or drop NaN. Are the number of objects zero for these? 
    df_data.fillna(value=0, axis=0, inplace=True)
    filtered_labels = df_data.loc[:, 'Label']
    
    df_x = df_data.drop(columns=['Label'])
    # Keep df_result untouched for later use
    #df_result.count()   

    #print ("Exit for debug")
    #sys.exit()

    #split the data
    do_split=True
    if do_split:
        data_train, data_test, label_train, label_test = train_test_split(df_x, filtered_labels, test_size=0.05, random_state=7)
        print(data_train.index[0:5])
        print(label_train.index[0:5])
    else:
        data_train = df_result
        label_train = filtered_labels
        
    #Normalize features
    #Normalizer(), MaxAbsScaler(), MinMaxScaler(), KernelCenterer(), and StandardScaler()
    do_Scaling = True
    if do_Scaling == True & do_split == True:
        print("Scaling the features")
        #pre_proc = preprocessing.MinMaxScaler()
        pre_proc = preprocessing.StandardScaler()
        #pre_proc = preprocessing.RobustScaler()
        #pre_proc = preprocessing.Normalizer()
        pre_proc.fit(data_train)
        data_train = pre_proc.transform(data_train)
        data_test  = pre_proc.transform(data_test)

    print(data_train.shape)
    print(data_train[0:5])
    print(data_test.shape) 
    print(data_test[0:5])

    do_KNeighbors=False
    do_SVC=False   # Didn't work well with parameters from the capstone project. Accuracy was about 70%
    do_RFC=True
    
    do_GridSearch = True
    if do_GridSearch== True:
        max_iter_value=100000
        #X=data_train
        #y=label_train
    
        if do_KNeighbors:
            model_default = KNeighborsClassifier()
            print ("KNeighbors Classifier")
        elif do_SVC:
            kernel_value='linear'
            model_default = SVC(kernel=kernel_value)
            print ("Support Vector Classifier")
        elif do_RFC:
            model_default = RandomForestClassifier(random_state=0)
            print ("RandomforestClassifier")
        else:
            print ('!!!!! Pick a model !!!!!')
     
        if do_KNeighbors:
            tuned_parameters = {'n_neighbors': [2, 3, 4, 6, 8, 10]}
        elif do_SVC:
            #kernel_values=['linear', 'rbf']
            Cs = [0.001, 0.01, 0.1, 1, 10]
            #gamma isn't use for the linear kernel
            gammas = [0.0001, 0.001, 0.01, 0.1]
            #Cs = np.logspace(-5, 1, 2)
            #gammas = np.logspace(-2, 1, 2)
            #tuned_parameters = [{'kernel': 'linear', 'C': Cs, 'gamma': gammas}]
            tuned_parameters = [{'C': Cs, 'gamma': gammas}]   #gamma isn't used for kernel=linear
        elif do_RFC:
            #tuned_parameters = {'n_estimators': [500, 700, 1000], 'max_depth': [None, 1, 2, 3], 'min_samples_split': [2, 3, 4, 5]}
            #tuned_parameters = {'n_estimators': [100, 300, 500], 'max_features': ['sqrt'], 'max_depth': [None, 1, 2], 'min_samples_split': [2, 3]}
            tuned_parameters = {'n_estimators': [200, 300, 400]}
            #tuned_parameters = {'n_estimators': [10]}
    
        print ('tuned_parameters: ', tuned_parameters)   
        n_folds = 3
        print ("Beginning GridSearchCV")
        clf = GridSearchCV(model_default, tuned_parameters, cv=n_folds, verbose=1, refit=True)
        clf.fit(data_train, label_train)
        scores = clf.cv_results_['mean_test_score']
        scores_std = clf.cv_results_['std_test_score']
        print ('scores: ', scores)
        print ('scores_std: ', scores_std)
        label_predicted_train = clf.predict(data_train)
        label_predicted_test  = clf.predict(data_test)
        print ('BEST ESTIMATOR: ', clf.best_estimator_)
        best_estimator = clf.best_estimator_
        print (clf.cv_results_)      
    
    if do_RFC:
        print ('feature importances: ', clf.best_estimator_.feature_importances_)
    elif do_SVC:
        print ('coef: ', clf.best_estimator_.coef_)
        print ('coef: ', clf.best_estimator_.intercept_)
        
    #print 'decision path: ', clf.best_estimator_.decision_path(data_test)  

    # need to pickle to model to a file
    print("Pickle the trained model to file: ", modelout)  

    # Open the file to save as pkl file
    modelout_pkl = open(modelout, 'wb')
    pickle.dump(clf, modelout_pkl)
    # Close the pickle instances
    modelout_pkl.close()

    return

     
       
if __name__ == "__main__":
   main(sys.argv[1:])
