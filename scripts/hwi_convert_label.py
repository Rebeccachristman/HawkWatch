#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: christman
"""

from __future__ import division, print_function, absolute_import
import sys, getopt
import os
from pathlib import Path

#import scipy
#from scipy import ndimage as ndi
from PIL import Image
from PIL.ExifTags import TAGS

import numpy as np
import pandas as pd
import argparse
import datetime
import matplotlib.pyplot as plt

import csv

sys.path.append(os.getcwd())
import imtools as imtools


# Thresholds for designating an image as after dark
dark_mean_threshold = 20
dark_std_threshold = 60

top_crop = int(32)
bottom_crop = int(top_crop + 25)

column_name_list = ['Dir', 'Datetime', 'Camera', 'Label']
#column_name_list = ['Dir', 'Camera', 'UpperX', 'UpperY', 'LowerX', 'LowerY', 'ImagePair', 'Datetime', 'SeqNum', 'SeqLen', 'Night', 'DiffMean', 'DiffStd', 'DiffMedian', 'NumObj', 'SizeMean','SizeStd','SizeSkew','DistMean','DistStd','DistSkew','AngleMean','AngleStd','AngleSkew' ]

sequence_list = ['ImgNum', 'Dir', 'Datetime', 'Camera', 'SeqNum', 'SeqLen']


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
            df = pd.DataFrame(columns=column_name_list)
        print("Overwriting file: ", filename)        
    else:
        df = pd.DataFrame(columns=column_name_list)
    return df


def check_range(imagenum, start_stop_pairs) :
    for start, stop in start_stop_pairs :
        if (imagenum >= start) and (imagenum <= stop):
            return 1
    return 0

def main(argv):
    basedir='E:\\\"Trail_Cameras\"\\Utah\\2016-2017\\'          # the space in Trail Cameras requires a quote
    basedir='E:\Trail_Cameras\\Utah\\2016-2017\\'          # the space in Trail Cameras requires a quote

    summary_sheet=basedir + "Utah_EVS_Camera_Database_2016-17.xlsx"

    labeldir="C:\\Users\\smith\\Documents\\Becky\\HawkWatch\\data\\"
    print("basedir: ", basedir)
    print("summary_sheet: ", summary_sheet)
    print("label files written to: ", labeldir)
    csvout=""
    try:
        opts, args = getopt.getopt(argv,"hd:o:")
        help_string = str(sys.argv[0]) + " -d <imagedir> -o <outputfile>" 
    except getopt.GetoptError:
        print (help_string)
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-h':
            print (help_string)
            sys.exit()
        elif opt in ("-d", "--imagedir"):
            imagedir = arg
            #if not imagedir.endswith('\\') :
            #    imagedir = imagedir + '\\'
        elif opt in ("-o", "--csvout"):
                csvout = arg
        else:
            print (help_string)
            sys.exit()

    print ("Image directory is ", imagedir)
    if not csvout:
        csvout = labeldir + "label_out_" + imagedir + ".csv"
    print ("CSV is ", csvout)
     
    df_tmp_summary = pd.read_excel(summary_sheet, sheet_name='Sheet1')

    df_summary = df_tmp_summary.loc[:, ['Camera','Begin Date', 'Carcass', 'Eagle Photo Numbers']]
    df_summary['Begin Date'] = pd.to_datetime(df_summary['Begin Date'])
    #print("head df_summary: ")
    #print(df_summary[0:3]) 

    # parse imagedir to get camera number and begin date to match with directory name
    toklst = imagedir.split("_")
    #print(toklst)
    camera_imagedir = int(toklst[0].split("Cam",1)[1])
    #print("camera_imagedir: ", camera_imagedir)
    begindate_imagedir = datetime.datetime.strptime(toklst[1], "%m%d%y")
    #print("begindate_imagedir: ", begindate_imagedir)
    carcass_imagedir = toklst[2]
    #print("carcass_imagedir: ", carcass_imagedir)
    
    # match imagedir info with entry in the summary sheet
    #print(df_summary['Camera'])
    df_match = df_summary[(df_summary.Camera == camera_imagedir) & (df_summary['Begin Date']==begindate_imagedir) ]
    #                  df[        df['first_name'].notnull() & (df['nationality'] == "USA")]
    # one row should be returned
    #print("match: ")
    #print(df_match[0:3]) 
    if df_match.empty :
        print("No matching entry in spreadsheet. Exiting...")
        sys.exit()
    
    # parse Eagle Photo Numbers to get the ranges with eagles
    eagle_photo_list = []
    eagle_start_stop = []
    no_eagle_flag = False
    df_match = df_match.fillna(0)
    if not df_match['Eagle Photo Numbers'].tolist()[0] == 0 :
    #if not df_match['Eagle Photo Numbers'] == np.NaN :
        eagle_photo_list = df_match['Eagle Photo Numbers'].tolist()[0].split(",")
        #print("eagle_photo_list: ", eagle_photo_list)

        split_pair =[]
        eagle_start_stop = []
        for pair in eagle_photo_list :
            split_pair = pair.strip().split("-")
            #print("split_pair: ", split_pair)

            if len(split_pair) == 1 :
                split_pair.append(split_pair[0])   # set stop=start if there's only a start
                
            eagle_start_stop.append([int(split_pair[0]), int(split_pair[1])])
    else :
        no_eagle_flag = True


    print("eagle_start_stop: ", eagle_start_stop)
    
    
    df_images = pd.DataFrame(columns=column_name_list)   # in place of check_csv()
    count = 0
    first_read = False
    save_images_in_seq = []
    df_save_sequence = pd.DataFrame(columns=sequence_list)
    
    imagedir = basedir+imagedir
    for file in os.listdir(imagedir):
       if not file.endswith('.JPG') :
           continue

       #df_test = df_images[(df_images['File']==file) & (df_images['Dir']==imagedir)]
       #if not df_test.empty :   # image is already labeled in the output file
       #    #print("File: ", imagedir+file, " labled in output file: ", csvout, "  Skipping..." )
       #    continue
              
       base_name = os.path.splitext(file)[0]
       # need to test the format of the file name to ensure IMG_xxxx
       image_num = int(base_name.split('_')[1])
       #print("base_name: ", base_name, "  image number: ", image_num)
       image_file = os.path.join(imagedir, file)
       #print("processing file: ", image_file)
       
       DateTime, CameraNumber, mode, seq_num, seq_len = imtools.get_all_photo_info(image_file)
       
       #print ("DateTime, CameraNumber:  ", DateTime, CameraNumber)          
       #print ("file, mode, sequence num, sequence length:  ", image_file, mode, seq_num, seq_len)          

       # if this is the first image in the run, start with first in sequence
       if first_read == False:
           # read the first image to get started
           #DateTime, CameraNumber, mode, seq_num, seq_len = imtools.get_all_photo_info(image_file)
           if seq_num == 1 :
               first_read = True
           else :
               print("skipping to first in sequence: ", image_num, seq_num, seq_len)
               continue
       #  Might need to move the dark check after reading all images in the sequence 
       #img=Image.open(image_file)
       #after_dark = 0
       #img_mean = round(np.mean(np.array(img)), 2)
       #img_std = round(np.std(np.array(img)), 2)
       #print("mean, std:  ", img_mean, img_std)
       #if img_mean < dark_mean_threshold and img_std < dark_std_threshold :
       #    print("Image is dark. Skip it")
       #    after_dark = 1
       #    #column_name_list = ['File', 'Directory', 'Image Number', 'Sequence Number', 'Sequence Length', 'After Dark', 'Image Array']
       #    df_images = df_images.append({'File':file, 'Dir':imagedir, 'Datetime':DateTime, 'Camera':CameraNumber, 'SeqNum': seq_num, 'SeqLen': seq_len, 'Night':1, 'Label':0}, ignore_index=True)
       #    continue
            
       df_save_sequence = df_save_sequence.append({'ImgNum':image_num,
                                                   'Dir':imagedir, 
                                                   'Datetime':DateTime, 
                                                   'Camera':CameraNumber, 
                                                   'SeqNum': seq_num, 
                                                   'SeqLen': seq_len, 
                                                   'Night':0, 
                                                   'Label':0}, ignore_index=True)

       # if this is the last image in a sequence of X out of X images, compute the delta images
       # to see what changed
       if seq_num < seq_len :  # get the next image in the sequence
           continue
       first_image_num = image_num - seq_len + 1
       #print("Beginning with image: ", first_image_num, "  ending with file: ", file)
       # Based on the first image number, see if any images in the sequence have an eagle
       if no_eagle_flag == True :
           label = 0
           #print("No eagle ranges for directory: ", imagedir)
       else:
           for seq_index in range(0, seq_len) :
               label = check_range(first_image_num+seq_index, eagle_start_stop)
               if label == 1 :  
                   #print("Start image ", first_image_num, " has label: ", label)    
                   break


       # Save the dataframe. All images in the sequence get the same label
       for seq_index in range(0, seq_len) :
           # get image number as integer and check against the eagle_start_stop list.
           # If one or more images in the sequence have an eagle, label the sequence as having an eagle
           #df_save_sequence.loc[seq_index, 'File']
           # leave 'Night' as NA

           df_images = df_images.append({'Dir':df_save_sequence.loc[seq_index,'Dir'], 
                                         'Datetime':df_save_sequence.loc[seq_index,'Datetime'], 
                                         'Camera':df_save_sequence.loc[seq_index,'Camera'], 
                                         'Label':label}, ignore_index=True)
           #df_images = df_images.append({'File':df_save_sequence.loc[seq_index, 'File'], 'Dir':df_save_sequence.loc[seq_index,'Dir'], 'Datetime':df_save_sequence.loc[seq_index,'Datetime'], 'Camera':df_save_sequence.loc[seq_index,'Camera'], 'SeqNum':df_save_sequence.loc[seq_index,'SeqNum'], 'SeqLen':df_save_sequence.loc[seq_index,'SeqLen'], 'Night':0, 'Label':label}, ignore_index=True)


       count = count + 1      
       save_images_in_seq.clear()
       df_save_sequence = pd.DataFrame(columns=sequence_list)
       df_save_sequence.drop(df_save_sequence.index[:], inplace=True)


    df_images.to_csv(csvout, sep=',', index=False)          


     
       
if __name__ == "__main__":
   main(sys.argv[1:])
