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
import math

import pandas as pd
import numpy as np
import pickle

import ast   # for literal eval of ObjectList
from collections import Iterable   # for flattening the dictionaries in ObjectList
from itertools import chain
from collections import ChainMap
import operator

from scipy.stats import skew

import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta

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
from sklearn.ensemble import AdaBoostClassifier

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


#column_name_list = ['File', 'Dir', 'Datetime', 'Camera', 'SeqNum', 'SeqLen', 'SeqNumDiff', 'Night', 'Mean', 'Std', 'TopCrop', 'BottomCrop', 'Carcass X', 'Carcass Y', 'Carcass Dist', 'Carcass Size', 'Obscuring Plants', 'NumObj', 'DistRank', 'Size', 'X', 'Y', 'Dist', 'Angle', 'Label' ]

def get_first_in_seq(imagefile, seqnum) :
    print(imagefile, seqnum)
    basename = os.path.splitext(imagefile)[0]   # strip off the file extension
    image_num = int(basename.split('_')[1])     # get the image number from IMG_xxxx
    return image_num - (seqnum - 1)             # calculate the first image number in the sequence

#attempts_to_flatten_objectlist 
                # somehow glue df_closest and df_second onto df_data where index=key
            ################## end loop            
        #tmp_indexes = df_data.index.get_values()
        #pd.merge(df_data, df_tmp, how='left',left_index=True, right_index=True)




            #df_closest = pd.DataFrame(list(ChainMap(*closest).items()))

            # Now append df_closest and df_second to df_data for row with index=key 
            
            #for d in data :
            #    flatten_dict = flatten{**{f'{k}': v for k, v in d.items()}}
                
            #flatten_dict = list(chain.from_iterable({key:item} if isinstance(item,Iterable) and
            #        not isinstance(item, str) else [item] for key, item in data))
            #flatten_dict = {k: v for k,v in val for val in data.items()}
            #df_tmp = pd.DataFrame(list(ChainMap(*data).items()))
            #L = []
            #count = 0
            #for d in data:
            #    temp = {}
            #    count = count + 1
            #    for key, val in d:
            #        temp.update(d[key])
            # 
            #    L.append(temp)

            
        #print("first object: ", objlist[0][0])
        #print("second object: ", objlist[0][1])

#tmp_indexes = df_pred.index.get_values()
        #df_tmp = pd.DataFrame(data=[objlist[:][0],objlist[:][1]],index=df_data.ObjectList.index, columns=['Region1', 'Region2'])
        # left join with the dataset that was split
        #df_final = pd.merge(df_data, df_tmp, how='left',left_index=True, right_index=True) 
        #print(df_final.head(3))


        #tmp_objlist = pd.Series(df_data.ObjectList).str.replace("'", "\"")

        #for key, value in tmp_objlist.items():
        #    data = json.loads(value)
             #print("{}: {}".format(key, value))
        
        #df_data.ObjectList.iteritems()
        #print("type of rec_objlist: ", type(rec_objlist))
        #rec_objlist = df_data.ObjectList
        #tmp_objlist = json.loads(pd.Series(df_data.ObjectList).str.replace("'", "\""))
        #tmp_objlist = pd.Series(df_data.ObjectList).tolist() # didn't work
        #list_dict_objects = json.loads(rec_objlist)
        #print("rec_objlist.dtype: ", rec_objlist.dtype)
        #region_props = tmp_objlist[0]
        #print("region_props type: ", type(region_props))

        #df_objlist = pd.DataFrame.from_records(df_data.ObjectList[:])
        #print(df_objlist.head())
        #objlist = df_data['ObjectList']
        #print(type(objlist))
        #print(type(objlist[0]))

        #df_objlist = pd.DataFrame.from_dict(df_data.ObjectList[:])
        #print(df_objlist.loc[:,:])
        #for row_dict in df_objlist.to_dict(orient='records'):
        #    print(row_dict)

        #object_list = []
        #object_list = df_data.ObjectList
        #print(object_list[0:5][0])
        #obj_dict_0 = {}
        #obj_dict_0=df_objlist.loc[0,:][0][0]
        #obj_dict_1=df_objlist.loc[0,:][1]
        #print("obj_dict_0: ", obj_dict_0)
        #print(df_objlist.loc[0,:][0])
        #print(df_objlist.head(5))
        #for x in obj_dict_0 :
        #    print(x['norm_dist_ROI_center'])

def timedelta_to_seconds(delta_string) :
    #Parse your string
    days, timestamp = delta_string.split(" days ")
    timestamp = timestamp[:len(timestamp)-7]
    
    #Generate datetime object
    t = datetime.datetime.strptime(timestamp,"%H:%M:%S") + datetime.timedelta(days=int(days))
    
    #Generate a timedelta
    delta = datetime.timedelta(days=t.day, hours=t.hour, minutes=t.minute, seconds=t.second)
    
    #Represent in Seconds
    return delta.total_seconds()


def main(argv):
    cwd = os.getcwd()
    print("\n*************** ")
    print("current working directory: ", cwd)

    csvtrain=""
    modelout=""    
    try:
   #   #opts, args = getopt.getopt(argv,"hi:o:",["imagedir=","ofile="])
       opts, args = getopt.getopt(argv,"hi:m:")
   
       #print 'Number of arguments:', len(sys.argv), 'arguments.'
       #print 'Argument List:', str(sys.argv)
       help_string = str(sys.argv[0]) + " -i <labeled training data file> -m <model file>" 
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

    if not csvtrain :
        csvtrain="..\\data\\tmp_labeled_data.csv"
        print("Using default training data file: ", csvtrain)
    if not modelout : 
        modelout="..\\data\\hwi_classifier_model.pkl"    
        print("Using default model output file: ", modelout)

    print("Labled training data file: " + csvtrain + "   trained model file: ", modelout)


    df=pd.read_csv(csvtrain, sep=',', header=0)
    #df_data  = df_data[df_data['Night'] != 1]          # not sure if night images should be dropped
    
    #df_data = df.assign(SeqFirst=lambda x:get_first_in_seq(x.File, x.SeqNum))    # allows pivot for sequences
    #df_data = df.assign(SeqFirst=lambda x: int(os.path.splitext(x.File)[0].split('_')[1]) - x.SeqNum-1)    # allows pivot for sequences

    df_data = df.fillna(value=0, axis=0)
    Feature_Engineering = False
    df_stats = pd.DataFrame()
    if not Feature_Engineering :
        print("START not feature engineering: ", df_data.columns.values)

        # It would be better to do this for the series, but the literal_eval may need to happen by row
        
        for i, row in enumerate(df_data.itertuples()) :
            #print("Index: ", row.Index)
            if row.NumObj == 0 :
                continue
           
            #bbox = ast.literal_eval(region.bbox)
            #centroid = ast.literal_eval(region.centroid)
            # the arrays from np.histogram have no commas in the CSV file
            hist1 = ast.literal_eval(','.join(row.hist1.split()))
            hist2 = ast.literal_eval(','.join(row.hist2.split()))
            moments_hu = ast.literal_eval(','.join(row.moments_hu.replace("[ ", "[").replace("  ", " ").split()))

            df_data.loc[row.Index, 'median1'] = np.median(hist1)
            df_data.loc[row.Index, 'std1'] = np.std(hist1)
            df_data.loc[row.Index, 'skew1'] = skew(hist1)
            df_data.loc[row.Index, 'median2'] = np.median(hist2)
            df_data.loc[row.Index, 'std2'] = np.std(hist2)
            df_data.loc[row.Index, 'skew2'] = skew(hist2)
            df_data.loc[row.Index, 'moments_hu_0'] = moments_hu[0]
            df_data.loc[row.Index, 'moments_hu_1'] = moments_hu[1]
            df_data.loc[row.Index, 'moments_hu_2'] = moments_hu[2]
            df_data.loc[row.Index, 'moments_hu_3'] = moments_hu[3]
            df_data.loc[row.Index, 'moments_hu_4'] = moments_hu[4]
            df_data.loc[row.Index, 'moments_hu_5'] = moments_hu[5]
            df_data.loc[row.Index, 'moments_hu_6'] = moments_hu[6]


        df_save_for_compare = df_data.copy(deep=True)
        df_data = df_data.drop(columns=['Dir',  'Camera', 'ROI', 'Datetime', 'Image1', 'Image2', 'SeqNum', 'SeqLen', 'Night', 'bbox', 'centroid', 'regionID', 'hist1', 'hist2', 'Side', ])
        df_data = df_data.drop(columns=['moments_hu', 'weighted_moments_hu1', 'weighted_moments_hu2'])

        # Hu moment arrays need to be converted

        df_data['TimeDiff2'] = pd.to_timedelta(df_data.TimeDiff2).dt.total_seconds()
        df_data['TimeDiff3'] = pd.to_timedelta(df_data.TimeDiff3).dt.total_seconds()
        df_data['TimeDiff1'] = pd.to_timedelta(df_data.TimeDiff1).dt.total_seconds()


        #df_data = df_data.drop(columns=['TimeDiff1', 'TimeDiff2', 'TimeDiff3'])
        print("end of not feature engineering: ", df_data.columns.values)
        print(df_data.head(2))
    else :
        # Do some feature engineering. This is the old feature set.
        
        # Drop the zero object rows to figure out the median
        df_tmp = df_data[df_data.NumObj != 0]     # Drop images with zero objects
        #df_tmp = df_data
        df_norm = pd.DataFrame(columns=['Dir', 'MedianArea'])
        
        for dirname in df_tmp.Dir.unique() :
            MedianArea = df_tmp[df_tmp.Dir==dirname].area.median()
            #print("Dir: ", dirname, "  MedianSize: ", MedianSize, "  MedianDist: ", MedianDist)

            df_norm = df_norm.append({'Dir':dirname, 'MedianArea':MedianArea}, ignore_index=True)

        #print("df_norm: ", df_norm)
        # now use the zero object rows
        df_data = pd.merge(df_data, df_norm, how='left', on='Dir')
        #print("In feature engineering, after merge...")
        #print(df_data.head(20))

        #df_data = df_data.assign(NormSize=lambda x: x.Size/x.MedianSize)
        #df_data = df_data.assign(NormDist=lambda x: x.Dist/x.MedianDist)

        df_data = df_data.assign(NormArea=lambda x: df_data.area/df_data.MedianArea)
        #df_junk = pd.DataFrame()
        #df_junk.MaxIntensity = np.where(df_data.mean_intensity1 < df_data.mean_intensity2, df_data.mean_intensity2, df_data.mean_intensity1)
        #df_data = pd.merge(df_data, df_junk,  left_index=True, right_index=True)

        #df_data.NormArea = np.log(df_data.NormArea)  # can't do this, the NumObj=0 rows are there
        
        #df_data = df_data.assign(NormDist=lambda x: df_data.DistMean/df_data.MedianDist)
        #df_data = df_data.assign(NormDistStd=lambda x: df_data.DistStd/df_data.MedianDist)
        
        for i, row in enumerate(df_data.itertuples()) :
            #print("Index: ", row.Index)
            if row.NumObj == 0 :
                continue
           
            #bbox = ast.literal_eval(region.bbox)
            #centroid = ast.literal_eval(region.centroid)
            # the arrays from np.histogram have no commas in the CSV file
            hist1 = ast.literal_eval(','.join(row.hist1.split()))
            hist2 = ast.literal_eval(','.join(row.hist2.split()))
            moments_hu = ast.literal_eval(','.join(row.moments_hu.replace("[ ", "[").replace("  ", " ").split()))

            #df_data.loc[row.Index, 'median1'] = np.median(hist1)
            #df_data.loc[row.Index, 'std1'] = np.std(hist1)
            df_data.loc[row.Index, 'skew1'] = skew(hist1)
            #df_data.loc[row.Index, 'median2'] = np.median(hist2)
            #df_data.loc[row.Index, 'std2'] = np.std(hist2)
            #df_data.loc[row.Index, 'skew2'] = skew(hist2)
            #df_data.loc[row.Index, 'moments_hu_0'] = moments_hu[0]
            #df_data.loc[row.Index, 'moments_hu_1'] = moments_hu[1]
            #df_data.loc[row.Index, 'moments_hu_2'] = moments_hu[2]
            #df_data.loc[row.Index, 'moments_hu_3'] = moments_hu[3]
            df_data.loc[row.Index, 'moments_hu_4'] = moments_hu[4]
            df_data.loc[row.Index, 'moments_hu_5'] = moments_hu[5]
            #df_data.loc[row.Index, 'moments_hu_6'] = moments_hu[6]



        df_save_for_compare = df_data.copy(deep=True)

        #print("In feature engineering, normalized columns added...")
        #print(df_data.head())

#Dir	Camera	ROI	Image1	Image2	Datetime	SeqNum	SeqLen	Night	TimeDiff1	TimeDiff2	TimeDiff3	NumObj	area	bbox	centroid	filled_area	regionID	mean_intensity1	mean_intensity2	moments_hu	orientation	weighted_moments_hu1	weighted_moments_hu2	hist1	hist2	EagleLabel	Side

        df_data = df_data.drop(columns=['Dir',  'Camera', 'ROI', 'Datetime', 'Image1', 'Image2'])
        df_data = df_data.drop(columns=['SeqNum', 'SeqLen', 'Night', 'NumObj'])
        df_data = df_data.drop(columns=['area', 'bbox', 'centroid', 'filled_area', 'regionID'])
        df_data = df_data.drop(columns=['mean_intensity1', 'mean_intensity2'])

        df_data = df_data.drop(columns=['moments_hu', 'weighted_moments_hu1',	'weighted_moments_hu2'])
        df_data = df_data.drop(columns=['hist1',	'hist2'])


        df_data = df_data.drop(columns=['MedianArea', 'Side'])  



        #df_data = df_data.drop(columns=['AngleStd','AngleSkew'])
        #df_data = df_data.drop(columns=['DiffMean', 'DiffStd', 'DiffMedian'])
        #df_data = df_data.drop(columns=['SizeMean', 'SizeStd', 'DistMean', 'DistStd', 'MedianSize', 'MedianDist'])
        #df_data = df_data.drop(columns=['NormSizeStd', 'NormDistStd'])  
        #df_data = df_data.drop(columns=['SizeSkew', 'DistSkew'])  


        df_data['TimeDiff2'] = pd.to_timedelta(df_data.TimeDiff2).dt.total_seconds()
        df_data['TimeDiff3'] = pd.to_timedelta(df_data.TimeDiff3).dt.total_seconds()
        df_data['TimeDiff1'] = pd.to_timedelta(df_data.TimeDiff1).dt.total_seconds()
        df_data = df_data.drop(columns=['TimeDiff1', 'TimeDiff2', 'TimeDiff3'])


        print("In feature engineering, extra columns dropped...")
        print(df_data.columns.values)
        #print(df_data.head())

    #print("remove exit")
    #sys.exit()
    df_data = df_data.fillna(value=0, axis=0)
    #print("Labeled training data...")
    #print(df_data.head())
    
    filtered_labels = df_data.loc[:, 'EagleLabel']
    print(filtered_labels.count())
    
    df_x = df_data.drop(columns=['EagleLabel'])
    # Keep df_result untouched for later use
    #df_result.count()   
    #df_data['Carcass Dist'] = df_data['Carcass Dist'].astype('category')
    #df_data['Carcass Size'] = df_data['Carcass Size'].astype('category')
    #df_data['Obscuring Plants'] = df_data['Obscuring Plants'].astype('category')


    #split the data
    do_split=True
    if do_split:
        data_train, data_test, label_train, label_test = train_test_split(df_x, filtered_labels, test_size=0.30) #, random_state=7)
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

    #print(data_train.shape)
    #print(data_train[0:5])
    #print(data_test.shape) 
    #print(data_test[0:5])

    do_KNeighbors=False
    do_SVC=False   # Didn't work well with parameters from the capstone project. Accuracy was about 70%
    do_RFC=True
    do_ADA=False

    
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
            model_default = RandomForestClassifier()    #random_state=0)
            print ("RandomforestClassifier")
        elif do_ADA:
            model_default = AdaBoostClassifier()
            print ("AdaboostClassifier")
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
            #tuned_parameters = {'n_estimators': [200, 600, 1000], 'max_depth': [None, 1, 2, 3], 'min_samples_split': [2, 3, 4, 5]}
            #tuned_parameters = {'n_estimators': [100, 300, 500], 'max_features': ['sqrt'], 'max_depth': [None, 1, 2], 'min_samples_split': [2, 3]}
            #tuned_parameters = {'n_estimators': [200, 300, 400]}   # this work best previously with 300 estimators as best
            #tuned_parameters = {'n_estimators': [250, 300, 350]}   # this work best previously with 300 estimators as best
            tuned_parameters = {'n_estimators': [350]}
        elif do_ADA:
            #tuned_parameters = {''learning_rate' [1.0, 2.0]: ,n_estimators': [50, 150, 350]}
            tuned_parameters = {'learning_rate': [0.5, 1.0, 2.0] ,'n_estimators': [50, 150, 300]}
    
        print ('tuned_parameters: ', tuned_parameters)   
        n_folds = 5
        #n_folds = 10
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

    label_predicted = clf.predict(data_test)
    #df_label_predicted = pd.DataFrame(data=label_predicted, columns=['Predicted Label'])

    average_precision = average_precision_score(label_test, label_predicted)
    
    print('Average precision-recall score: {0:0.2f}'.format(
          average_precision))
    
    precision, recall, _ = precision_recall_curve(label_test, label_predicted)
    
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(
              average_precision))
    plt.show()

    print("Accuracy Score")
    print (accuracy_score(label_test, label_predicted))
    print("\nConfusion Matrix")
    print(confusion_matrix(label_test, label_predicted))    # tn, fp, fn, tp 
    print("\nClassification Report")
    print(classification_report(label_test, label_predicted)) 
    
    print("Writing out evaluation labels and predictions")
    df_pred = pd.DataFrame(data=label_test)    # get indexes for split evaluation labels
    tmp_indexes = df_pred.index.get_values()
    print(tmp_indexes[0:5])
    df_tmp_pred = pd.DataFrame(data=label_predicted,index=tmp_indexes, columns=['Label_pred'])
    #df_pred = pd.concat(df_tmp_pred['Label_pred'], axis=1, ignore_index=True).copy()


    # left join with the dataset that was split
    df_final = pd.merge(df_tmp_pred, df_save_for_compare, how='left',left_index=True, right_index=True) 

    df_final.to_csv("tmp_ML_eval.csv", sep=',', index=False)  
    return

     
       
if __name__ == "__main__":
   main(sys.argv[1:])
