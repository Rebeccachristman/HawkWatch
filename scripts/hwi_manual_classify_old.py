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

import matplotlib.pyplot as plt

import csv

sys.path.append(os.getcwd())
import imtools as imtools


ext=[".jpg",".JPG",".png",".PNG",".tif",".TIF"]
img_downsize_scale = 1/8
#lines_to_crop = 32
lines_top_bottom = 32

# Thresholds for designating an image as after dark
dark_mean_threshold = 20
dark_std_threshold = 60

top_crop = int(32)
bottom_crop = int(top_crop + 25)


#image_list=[file, imagedir, DateTime, CameraNumber, AfterDark, label] 
column_name_list = ['File', 'Dir', 'Datetime', 'Camera', 'SeqNum', 'SeqLen', 'Night', 'Label']
sequence_list = ['File', 'Dir', 'Datetime', 'Camera', 'SeqNum', 'SeqLen']
#sequence_list = ['File', 'Dir', 'Datetime', 'Camera', 'SeqNum', 'SeqLen', 'SeqNumDiff', 'Night', 'Mean', 'Std']



columns = 2
rows = 1
#gl_cols = 3
#gl_rows = 3
gl_cols = 1  # need global qualifier
gl_rows = 3


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

def prep_image(image_file):
    img_gray_arr=np.array(Image.open(image_file).convert('L'))
    img_gray_arr_crop = img_gray_arr[top_crop:-bottom_crop, :]  # extra 25 to remove the company logo
    return img_gray_arr_crop

def main(argv):

   cwd = os.getcwd()
   print("\n*************** ")
   print("current working directory: ", cwd)

   imagedir="D:\\Photo_Set_4\\"
   csvout="..\\data\\label_output.csv"
   #csvtraining="label_training.csv"
   create_training_set = False
   start_imagenum = 0


   try:
      opts, args = getopt.getopt(argv,"hd:o:n:")
      #print 'Number of arguments:', len(sys.argv), 'arguments.'
      #print 'Argument List:', str(sys.argv)
      help_string = str(sys.argv[0]) + " -d <imagedir> -o <outputfile> -n <imagenum>" 
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
      elif opt in ("-o", "--csvout"):
         csvout = arg
      elif opt in ("-n", "--startimagenum"):
         start_imagenum = int(arg)
      else:
         print (help_string)
         sys.exit()
   print("Image directory: " + imagedir + "  csv output file: ", csvout, "   start image number: ", start_imagenum)


   print ("Image directory is ", imagedir)
   print ("CSV is ", csvout)


   
   # if output file exists, ask user to overwrite or append
   # if append, read the file in
   #df_images = pd.DataFrame()
   df_images = check_csv(csvout)
   print("head df_images: ", df_images[0:3]) 

   #if create_training_set :       # not sure what this was suppose to be for
   #    print("checking for training set file: ", csvtraining)
   #    df_training = check_csv(csvtraining)
   #    print("head df_training: ", df_training[0:3]) 

   count = 0
   first_read = False
   save_images_in_seq = []
   df_save_sequence = pd.DataFrame(columns=sequence_list)

   for file in os.listdir(imagedir):
       if not file.endswith(tuple(ext)):
           continue

       df_test = df_images[(df_images['File']==file) & (df_images['Dir']==imagedir)]
       if not df_test.empty :   # image is already labeled in the output file
           #print("File: ", imagedir+file, " labled in output file: ", csvout, "  Skipping..." )
           continue
              
       base_name = os.path.splitext(file)[0]
       # need to test the format of the file name to ensure IMG_xxxx
       image_num = int(base_name.split('_')[1])
       #print("base_name: ", base_name, "  image number: ", image_num)
       
       # command line arguement specified an image number to begin 
       if image_num < start_imagenum :
           continue
       
       image_file = os.path.join(imagedir, file)
       print("processing file: ", image_file)
       DateTime, CameraNumber = imtools.get_photo_info(image_file)
       print ("DateTime, CameraNumber:  ", DateTime, CameraNumber)          
       mode, seq_num, seq_len = imtools.get_sequence(image_file)
       print ("file, mode, sequence num, sequence length:  ", file, mode, seq_num, seq_len)          

       # if this is the first image in the run, start with first in sequence
       if first_read == False:
           if seq_num == 1 :
               first_read = True
           else :
               print("skipping to first in sequence: ", image_num, seq_num, seq_len)
               continue
                  
       #  Might need to move the dark check after reading all images in the sequence 
       img=Image.open(image_file)
       after_dark = 0
       img_mean = round(np.mean(np.array(img)), 2)
       img_std = round(np.std(np.array(img)), 2)
       print("mean, std:  ", img_mean, img_std)
       if img_mean < dark_mean_threshold and img_std < dark_std_threshold :
           print("Image is dark. Skip it")
           after_dark = 1
           #column_name_list = ['File', 'Directory', 'Image Number', 'Sequence Number', 'Sequence Length', 'After Dark', 'Image Array']
           df_images = df_images.append({'File':file, 'Dir':imagedir, 'Datetime':DateTime, 'Camera':CameraNumber, 'SeqNum': seq_num, 'SeqLen': seq_len, 'Night':1, 'Label':0}, ignore_index=True)
           continue
           

       img = np.array(prep_image(image_file))
       im2, cdf = imtools.histeq(img)
       # save images
       save_images_in_seq.append(im2) 
       df_save_sequence = df_save_sequence.append({'File':file, 'Dir':imagedir, 'Datetime':DateTime, 'Camera':CameraNumber, 'SeqNum': seq_num, 'SeqLen': seq_len, 'Night':0, 'Label':0}, ignore_index=True)

       # if this is the last image in a sequence of X out of X images, compute the delta images
       # to see what changed
       if seq_num < seq_len :  # get the next image in the sequence
           continue


       fig = plt.figure(figsize=(30, 10))
       for seq_index in range(0, seq_len) :
           #fig.add_subplot(gl_rows, gl_cols, seq_len*seq_index+1) 
           #ax = fig.add_subplot(gl_rows, gl_cols, gl_cols*seq_index+1)   # 3 columns, 3 rows
           ax = fig.add_subplot(1, 3, seq_index+1)

           if seq_index == 0 :
               #ax = fig.axes[0, 0]
               title = ax.set_title("Normalized Gray Scale", loc='center', y=1.1, fontsize=20)
           xlabel = df_save_sequence.loc[seq_index, 'File'] + "  Seq: "+str(seq_index+1)+"/"+str(seq_len)
           ax.set_xlabel(xlabel, fontsize=15)             #alpha=0.5)
           ax.get_xaxis().set_ticks([])
           ax.get_yaxis().set_ticks([])
           plt.imshow(save_images_in_seq[seq_index])



           #fig.add_subplot(rows, columns, 2) 
           #plt.imshow(im2)

       plt.show()
       if create_training_set :
           usr_label = input("Is it an eagle? (1=yes, 0=no, y=yes+add, n=no+add, 9=exit):  ")
       else :
           usr_label = input("Is it an eagle? (1=yes, 0=no, 9=exit):  ")
               
       if usr_label == '1' :
           label = 1
           print("It's an eagle!")
       elif usr_label == 'y' :
           label = 1
           print("It's an eagle!")
           #df_training = df_training.append({'File':file, 'Directory':imagedir, 'Datetime':DateTime, 'Camera':CameraNumber, 'AfterDark':after_dark, 'Label':label}, ignore_index=True)
       elif usr_label == 'n' :
           label = 0
           print("not an eagle!")
           #df_training = df_training.append({'File':file, 'Directory':imagedir, 'Datetime':DateTime, 'Camera':CameraNumber, 'AfterDark':after_dark, 'Label':label}, ignore_index=True)
       elif usr_label == '9' :
           # write out csv file and exit
           #write_to_csv(df_images)
           df_images.to_csv(csvout, sep=',', index=False)
           #df_training.to_csv(csvtraining, sep=',', index=False)
           sys.exit()                # Exit program
       else: 
            label = 0
            print("not an eagle")
            
       # Save the dataframe. All images in the sequence get the same label
       for seq_index in range(0, seq_len) :
           df_images = df_images.append({'File':df_save_sequence.loc[seq_index, 'File'], 'Dir':df_save_sequence.loc[seq_index,'Dir'], 'Datetime':df_save_sequence.loc[seq_index,'Datetime'], 'Camera':df_save_sequence.loc[seq_index,'Camera'], 'SeqNum':df_save_sequence.loc[seq_index,'SeqNum'], 'SeqLen':df_save_sequence.loc[seq_index,'SeqLen'], 'Night':0, 'Label':label}, ignore_index=True)


       count = count + 1      
       save_images_in_seq.clear()
       df_save_sequence = pd.DataFrame(columns=sequence_list)
       df_save_sequence.drop(df_save_sequence.index[:], inplace=True)

# two questions were annoying
           #if create_training_set :
           #    usr_label = input("Add to training set? (1=yes):  ")
           #    if usr_label == '1' :
           #        df_training = df_training.append({'File':file, 'Directory':imagedir, 'Datetime':DateTime, 'Camera':CameraNumber, 'AfterDark':after_dark, 'Label':label}, ignore_index=True)


            #fig.suptitle(image_file)
           #is_bird = True
           #if is_bird:
           #    print("That's a bird!")
           #    fig.suptitle(image_file+"\n"+ r"  A bird!", fontsize=20)
           #    #fig.suptitle(image_file+"\n"+ r"  A bird!", fontsize=20)
           #else:
           #    print("That's not a bird!")
           #    #fig.suptitle(image_file+"\n"+ r"  NOT a bird!", fontsize=20)
           #    fig.suptitle(image_file+"\n"+ r"  Not a bird!", fontsize=20)
               
           #row = list(image_file, is_bird)
           #outputfile_writer.writerow(row)
           

           #plt.show()
   df_images.to_csv(csvout, sep=',', index=False)          
   if create_training_set :
       df_images.to_csv(csvout, sep=',', index=False)
       
   #plt.show() 
   #csvfile.close()   
   return

     
       
if __name__ == "__main__":
   main(sys.argv[1:])
