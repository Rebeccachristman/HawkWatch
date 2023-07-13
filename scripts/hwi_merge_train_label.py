#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import sys, getopt
import os
from pathlib import Path
import argparse

import pandas as pd
import numpy as np

sys.path.append(os.getcwd())
import imtools as imtools

def check_csv(filename):
   my_file = Path(filename)
   if my_file.is_file():
       # file exists
       usr_label = input("Output file exists. (a=append, o=overwrite, 9=exit):  ")
       if usr_label == 'a' :   # append to existing file
           print("Appending to file: ", filename)
           df = pd.read_csv(filename)
       elif usr_label == '9' :
           sys.exit()                # Exit program
       else: 
           df = pd.DataFrame()                #columns=column_name_list)
           print("Overwriting file: ", filename)        
   else:
       df = pd.DataFrame()                    #columns=column_name_list)

   return df


def main(argv):

   cwd = os.getcwd()
   print("\n*************** ")
   print("current working directory: ", cwd)

   datadir="..\\data\\"
   csvtable="..\\data\\hwi_merge_table.csv"
   csvout="..\\data\\tmp_labeled_data.csv"

   df_table = pd.DataFrame()
   df_table=pd.read_csv(csvtable, sep=',', header=0)
   print(df_table.head(5))

   total_label_ones=0
   total_row_count=0
   postmerge_label_ones=0
   postmerge_row_count=0

   df_train_tmp = pd.DataFrame() 
   df_label_tmp = pd.DataFrame()
   df_result= pd.DataFrame()
   df_merge = check_csv(csvout)
   for index, row in df_table.iterrows() :
       train_file = row[0]
       label_file = row[1]
       add_to_merge = row[2]
       print("reading train, label, add to merge: ", train_file, label_file, add_to_merge)
       if add_to_merge == True :
           df_train_tmp = pd.read_csv(train_file, sep=',', header=0, index_col=None)
           df_label_tmp = pd.read_csv(label_file, sep=',', header=0, index_col=None)
           total_row_count = total_row_count + len(df_label_tmp.index)
           total_label_ones = total_label_ones + len(df_label_tmp[df_label_tmp['EagleLabel']==1].index)

           #df_label_tmp.Dir = df_label_tmp.Dir + '\\'
           df_result = pd.merge(df_train_tmp, df_label_tmp, on=['Dir', 'Datetime', 'Camera', 'regionID'], how='inner', suffixes=['','_label'])


           postmerge_row_count = postmerge_row_count + len(df_result.index)
           postmerge_label_ones = postmerge_label_ones + len(df_result[df_result['EagleLabel']==1].index)
           print("row counts: ", total_row_count, postmerge_row_count, "  ones: ", total_label_ones, postmerge_label_ones)

           print("row counts: ", len(df_label_tmp.index), len(df_result.index), "  ones: ", len(df_label_tmp[df_label_tmp['EagleLabel']==1].index), len(df_result[df_result['EagleLabel']==1].index), "  file: ", label_file)
           df_merge = df_merge.append(df_result) #, ignore_index=True)



   df_merge = df_merge.drop(labels=['Image1_label', 'Image2_label', 'ROI_label', 'bbox_label'], axis=1)
   #print(df_merge.head(3))
   print("writing merged data to file: ", csvout)    
   df_merge.to_csv(csvout, sep=',', index=False)   
   
   final_row_count = postmerge_row_count + len(df_merge.index)
   final_label_ones = postmerge_label_ones + len(df_merge[df_merge['EagleLabel']==1].index)
   print("final row_count: ", final_row_count, "  label_ones: ", final_label_ones)
   
   return

     
       
if __name__ == "__main__":
   main(sys.argv[1:])
