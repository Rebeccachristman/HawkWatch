#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import sys, getopt
import os
from pathlib import Path

#import scipy
from scipy.ndimage import filters
from scipy.ndimage import measurements, morphology

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
dark_std_threshold = 30

diff_mean_threshold = 40
diff_std_threshold = 60


imagedir = ''
outfile = ''
#imagedir="C:\\Users\\smith\\Documents\\Becky\\HawkWatch\\Images\\"
imagedir="D:\\Photo_Set_1\\"

#csvdir="C:\\Users\\smith\\Documents\\Becky\\HawkWatch\\Data\\"
#csvout="label_output.csv"
csvout="subtract_images.csv"

csvtraining="subtract_training.csv"
create_training_set = False
#image_list=[file, imagedir, DateTime, CameraNumber, AfterDark, label] 
#column_name_list = ['File', 'Directory', 'Datetime', 'Camera', 'AfterDark', 'Label']
column_name_list = ['File', 'Directory', 'Base Image', 'Mean', 'Std', 'Norm Mean', 'Norm Std', 'Diff Mean', 'Diff Std']

# setup for displaying the images with matplotlib
   #fig=plt.figure(figsize=(8, 8))
gl_cols = 2
gl_rows = 2
gl_x = 0
gl_y = 0

#def write_to_csv(filename, df_images):
#    df_images.write_csv(filename, sep=',')
#    return
# Get a specific EXIF tag

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
    #img_gray=Image.open(image_file).draft('L', prep_shape)
    #img_gray_arr=np.array(img_gray) 

    img_gray_arr=np.array(Image.open(image_file).convert('L'))
    #print("draft size: ", img_gray.size, "  mode: ", img_gray.mode)

    # crop the top and bottom information lines so they don't add black to the cdf
    # should check to make sure the image has black top & bottom before cropping
    #img_gray_arr_crop = img_gray_arr[4:188, 0:256]  # dimensions for ()
    #lines_to_crop = int(lines_top_bottom*img_downsize_scale)
    lines_to_crop = int(lines_top_bottom)
    img_gray_arr_crop = img_gray_arr[lines_to_crop:-lines_to_crop-25, :]  # extra 25 to remove the company logo
    #print("img_gray_arr_crop: shape: ", img_gray_arr_crop.shape, "dtype: ", img_gray_arr_crop.dtype)
    im2, cdf = imtools.histeq(img_gray_arr_crop)
    return im2

def onclick(event):
    global gl_x
    global gl_y
    if event.xdata != None and event.ydata != None:
        gl_x = event.xdata
        gl_y = event.ydata
        #print(gl_x, gl_y) 
    return

# apparently morphology doesn't work with natural landscapes
def make_binary_image(image_file) :
    img_gray=np.array(Image.open(image_file).convert('L'))

    #img_binary=morphology.binary_opening(img_gray, np.ones((18,10)), iterations=2)
    img_binary = img_gray
    lines_to_crop = int(lines_top_bottom)
    img_crop = img_binary[lines_to_crop:-lines_to_crop-25, :]
    im2, cdf = imtools.histeq(img_crop)
    img_crop = 1*(im2<128)
    return img_crop

def main(argv):

   cwd = os.getcwd()
   print("\n*************** ")
   print("current working directory: ", cwd)
   
   try:
      #opts, args = getopt.getopt(argv,"hi:o:",["imagedir=","ofile="])
      opts, args = getopt.getopt(argv,"hb:",["imagebase="])
   
      #print 'Number of arguments:', len(sys.argv), 'arguments.'
      #print 'Argument List:', str(sys.argv)
   except getopt.GetoptError:
      print ("program.py -i <imagebase>")
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print ("program.py -b <imagebase>")
         sys.exit()
      elif opt in ("-b", "--imagebase"):
         imagebase = arg
      else:
         print ("program.py -b <imagebase>")
         sys.exit()

   #      outputfile = arg
   print ("Base image is ", imagebase)
   #print ("CSV is ", csvout)
   df_images = pd.DataFrame(columns=column_name_list)
   
   # if output file exists, ask user to overwrite or append
   # if append, read the file in
   #df_images = pd.DataFrame()
   #df_images = check_csv(csvout)
   #print("head df_images: ", df_images[0:3]) 
   count = 0  # for debug break out of loop early
   found_base_flag = False
   for file in os.listdir(imagedir):
       # skip over images before the base image file
       #if count > 200:
       #    break
       if not file.endswith(tuple(ext)):
           continue
       
       if file.lower() == imagebase.lower() :
           print("Found base image: ", imagebase)
           found_base_flag = True
           image_file = imagedir+file
           img=Image.open(image_file)
           ax = plt.gca()
           fig = plt.gcf()
           #figsiz=plt.figure(figsize=(14, 14))   # figure out how to make the image larger

           implot = ax.imshow(img)
           cid = fig.canvas.mpl_connect('button_press_event', onclick)
           plt.show()
           print("deer click coordinates: ", gl_x, gl_y)
           img_shape = np.array(img).shape
           # the Hawk Watch roadside images are (2048, 1536) 
           img_base_prep = prep_image(image_file)
           #img_base_binary = make_binary_image(image_file)
 
       if found_base_flag == False :
           print(" Before base image: ", imagebase, "Skipping file: ", file )
           continue

       image_file = os.path.join(imagedir, file)
       print("processing file: ", image_file)
       #DateTime, CameraNumber = imtools.get_photo_info(image_file)
       #print ("DateTime, CameraNumber:  ", DateTime, CameraNumber)          
       # Load the image file

       # Original image

       img=Image.open(image_file)
       after_dark = 0
       img_mean = round(np.mean(np.array(img)), 2)
       img_std = round(np.std(np.array(img)), 2)
       #print("mean, std:  ", img_mean, img_std)
       if img_mean < dark_mean_threshold and img_std < dark_std_threshold :
           #print("Image is dark. Skip it")
           continue
 
       # morphology doesn't work for a natural landscape
       #image_binary = make_binary_image(image_file)
       #labels, nbr_objects = measurements.label(image_binary)
       #print ("Number of objects: ", nbr_objects)

       
       im2 = prep_image(image_file)
       im2_mean = round(np.mean(np.array(im2)), 2)
       im2_std = round(np.std(np.array(im2)), 2)
       #print("mean, std:  ", img_mean, img_std)
       #if im2_mean < dark_mean_threshold and im2_std < dark_std_threshold :
       #    #print("Image is dark. Skip it")
       #    continue

       img_diff = im2 - img_base_prep
       img_diff_mean = round(np.mean(np.array(img_diff)), 2)
       img_diff_std = round(np.std(np.array(img_diff)), 2)
       img_diff_gaussian = filters.gaussian_filter(img_diff, 5)
       #diff_stats = [img_diff_mean, img_diff_std]
       #print("diff mean, std:  ", img_diff_mean, img_diff_std)
       #if img_diff_mean < diff_mean_threshold and img_diff_std < diff_std_threshold :
           #print("Difference image less than thresholds")
           #continue

       df_images = df_images.append({'File':file, 'Directory':imagedir, 'Base Image':imagebase, 'Mean': img_mean, 'Std': img_std, 'Norm Mean':im2_mean, 'Norm Std':im2_std, 'Diff Mean':img_diff_mean, 'Diff Std':img_diff_std}, ignore_index=True)

       show_each_image = True
       if show_each_image :
           fig=plt.figure(figsize=(14, 14))
           fig.add_subplot(gl_rows, gl_cols, 1) 
           plt.imshow(img)
           fig.add_subplot(gl_rows, gl_cols, 2) 
           #plt.imshow(image_binary)
           plt.imshow(im2)
           fig.add_subplot(gl_rows, gl_cols, 3) 
           #plt.imshow(img_diff)
           plt.imshow(img_diff_gaussian)

           fig.add_subplot(gl_rows, gl_cols, 4) 
           plt.scatter(np.array(df_images['Diff Mean']), np.array(df_images['Diff Std']), marker='.')
           plt.scatter(np.array(df_images.iloc[count-1,7]), np.array(df_images.iloc[count-1,8]), marker='X', c='red')
           plt.show()
           usr_label = input("Exit? (e=exit):  ")
           if usr_label == 'e' :
               sys.exit()
       count = count + 1
        
   df_images.to_csv('stats.csv', sep=',', index=False)          
   fig=plt.figure(figsize=(8, 8))
   fig.add_subplot(1, 1, 1) 
   #plt.scatter(np.array(df_images['Mean']), np.array(df_images['Std']), c='r')
   #plt.scatter(np.array(df_images['Norm Mean']), np.array(df_images['Norm Std']), c='g')
   plt.scatter(np.array(df_images['Diff Mean']), np.array(df_images['Diff Std']))
   plt.show()

   return 

       
if __name__ == "__main__":
   main(sys.argv[1:])
