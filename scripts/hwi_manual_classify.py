#!/usr/bin/python
# -*- coding: utf-8 -*-


from __future__ import division, print_function, absolute_import
import sys, getopt
import os
from pathlib import Path
import math

#import scipy
from scipy.stats import skew

from PIL import Image
from PIL.ExifTags import TAGS

import numpy as np
import pandas as pd
import argparse
from datetime import datetime, timedelta


import matplotlib.pyplot as plt
import matplotlib.patches as patches

import operator
import ast   # for literal eval of ObjectList
import re    # regular expressions
import csv

sys.path.append(os.getcwd())
import imtools as imtools
import ROF as ROF


ext=[".jpg",".JPG",".png",".PNG",".tif",".TIF"]
colors=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
sequence_list = ['File' 'Dir', 'Camera', 'ImgNum', 'Datetime', 'SeqNum', 'SeqLen', 'Night', 'DiffMean', 'DiffStd', 'DiffMedian']


imagedir = ''
outfile = ''
csvout="hwi_sequence_out.csv"
deer_coord_file="hwi_carcass_coordinates.csv"

# setup for displaying the images with matplotlib
gl_cols = 3
gl_rows = 3

# temporarily global
deer_x = 0
deer_y = 0
upperleft_x, upperleft_y = 0, 0
lowright_x, lowright_y = 0, 0

# move check_csv to imtools
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
           #df = pd.DataFrame(columns=column_name_list)
           df = pd.DataFrame()
           print("Overwriting file: ", filename)        
   else:
       df = pd.DataFrame()
       #df = pd.DataFrame(columns=column_name_list)

   return df



def get_imgstats(img) :
   imgstats = {'Mean': round(np.mean(np.array(img)), 2), 
               'Std': round(np.std(np.array(img)), 2),
               #'Skewness': round(skew(np.array(img)), 2),
               'Max': round(np.max(np.array(img)), 2),
               'Min': round(np.min(np.array(img)), 2),
               'Median': round(np.median(np.array(img)), 2),
               'Q25': round(np.percentile(np.array(img), 25), 2),
               'Q75': round(np.percentile(np.array(img), 75), 2),
               'Q85': round(np.percentile(np.array(img), 85), 2)
               }
   return imgstats

def get_timediff(t1, t2) :
    t1 = pd.to_datetime(t1, format = '%Y:%m:%d %H:%M:%S') # Convert column type to be datetime
    t2 = pd.to_datetime(t2, format = '%Y:%m:%d %H:%M:%S') # Convert column type to be datetime
    return t2 - t1

def get_image_pair(image_num, seq_index, seq_num, seq_len) :
    #image_pair = str(image_num) + '|'
    if seq_index < seq_len-1 :
        image_pair = (image_num, image_num + 1)
        #image_pair = image_pair + str(image_num + 1)
    else :    # wrap around 3|1 or 2|1
        image_pair = (image_num, image_num - seq_len + 1)
        #image_pair = image_pair + str(image_num-seq_len+1)
    return image_pair



def crop_image(image):
    global lowright_x, lowright_y
    global upperleft_x, upperleft_y

    img_arr=np.array(image)
    img_arr_crop = img_arr[int(upperleft_y):int(lowright_y), int(upperleft_x):int(lowright_x)]
        
    return img_arr_crop

def get_eagle_ranges(summary_sheet, imagedir) :  #camera, begin_date) :
    df_tmp_summary = pd.read_excel(summary_sheet, sheet_name='Sheet1')
    #df_tmp_summary = pd.read_excel(summary_sheet, sheet_name='Photo_Information')  #'Sheet1')

    df_summary = df_tmp_summary.loc[:, ['Camera','Begin Date', 'Carcass', 'Eagle Photo Numbers']]
    #df_summary = df_tmp_summary.loc[:, ['Set_ID', 'Begin Date', 'Eagle Photo Numbers']]
    df_summary['Begin Date'] = pd.to_datetime(df_summary['Begin Date'])
    #print("head df_summary: ")
    #print(df_summary[0:3]) 

    # parse imagedir to get camera number and begin date to match with directory name
    #basename = os.path.basename(imagedir)
    #print("basename: ", basename)
    toklst = os.path.basename(imagedir).split("_")
    print("toklst: ", toklst)
    camera_imagedir = int(toklst[0].split("Cam",1)[1])
    print("camera_imagedir: ", camera_imagedir)
    begindate_imagedir = datetime.strptime(toklst[1], "%m%d%y")
    #print("begindate_imagedir: ", begindate_imagedir)
    carcass_imagedir = toklst[2]
    print("carcass_imagedir: ", carcass_imagedir)
    df_match = df_summary[(df_summary.Camera == camera_imagedir) & (df_summary['Begin Date']==begindate_imagedir) ]
    
    # match imagedir info with entry in the summary sheet
    #print(df_summary['Camera'])
    # !!! Found problems with get_camera_info for 'CAM 277' and '279' in Utah 2016-2017
    
    #if camera[0].isalpha() :
    #    camera_number = int(camera.upper().split("CAM",1)[1])
    #elif camera[0].isnumeric() :
    #    camera.replace(" ", "")
    #    camera_number = str(camera)
    #else :
    #    print("PROBLEM WITH CAMERA NUMBER: [", camera, "]")
    #begin_date_tok = begin_date.split(" ")[0].split(":")
    #begin_date_str = begin_date_tok[1] + "/" + begin_date_tok[2] + "/" + begin_date_tok[0]
    #print("camera_number: ", camera_number, "  begin date: ", begin_date_str)
    #df_match = df_summary[(df_summary.Camera == camera_number) & (df_summary['Begin Date']==begin_date_str) ]

# !!!!!!!!!!!!!!!! hardwire for photo_set_1 to allow testing with thumb drive
    #df_match = df_summary[(df_summary.Set_ID == 'Photo_Set_1')]

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
    
    return eagle_start_stop

def check_range(imagenum, start_stop_pairs) :
    for start, stop in start_stop_pairs :
        if (imagenum >= start) and (imagenum <= stop):
            return 1
    return 0

def onclick(event):
    global gl_x
    global gl_y
    global gl_axes
    
    gl_x = 0
    gl_y = 0
    #gl_axes = np.nan()
    if event.xdata != None and event.ydata != None:
        gl_x = event.xdata
        gl_y = event.ydata
        gl_axes = event.inaxes
        print(gl_x, gl_y, gl_axes) 
    return


def main(argv):
   global deer_x
   global deer_y
   global lowright_x, lowright_y
   global upperleft_x, upperleft_y
   global gl_x, gl_y
   global gl_axes
   
   gl_x = gl_y = 0

   cwd = os.getcwd()
   #print("\n*************** ")
   #print("current working directory: ", cwd)

   #basedir='E:\\\"Trail_Cameras\"\\Utah\\2016-2017\\'          # the space in Trail Cameras requires a quote
   basedir='E:\Trail_Cameras\\Utah\\2016-2017\\'          # the space in Trail Cameras requires a quote
   summary_sheet=basedir + "Utah_EVS_Camera_Database_2016-17.xlsx"
   #basedir="D:\\"          # the space in Trail Cameras requires a quote
   #summary_sheet=basedir + "EVS_Machine_Learning_Spreadsheet_Modified.xlsx"

   labeldir="C:\\Users\\smith\\Documents\\Becky\\HawkWatch\\data\\"
   print("basedir: ", basedir)
   print("summary_sheet: ", summary_sheet)
   print("label files written to: ", labeldir)

   show_each_image = False
   start_imagenum = 1
   imagedir=""  
   csvinp=""
   inpfile=""
   csvout=""
   outfile=""
   overwrite_flag = False
   try:
      opts, args = getopt.getopt(argv,"hswd:i:o:n:")
      #print 'Number of arguments:', len(sys.argv), 'arguments.'
      #print 'Argument List:', str(sys.argv)
      help_string = str(sys.argv[0]) + " -s -w -d <imagedir> -i <inputfile> -o <outputfile> -n <imagenum>" 
   except getopt.GetoptError:
      print (help_string)
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print (help_string)
         sys.exit()
      elif opt in ("-s", "--showeachimage"):
         show_each_image = True
         print("Showing each image. To inhibit display of images, don't use -s flag")
      elif opt in ("-d", "--imagedir"):
         imagedir = arg
         if not imagedir.endswith('\\') :
             imagedir = imagedir + '\\'  
      elif opt in ("-i", "--csvinp"):
         inpfile = arg
      elif opt in ("-o", "--csvout"):
         outfile = arg
      elif opt in ("-n", "--startimagenum"):
         start_imagenum = int(arg)
      elif opt in ("-w", "--overwrite"):
         overwrite_flag = True
         print("Forcing overwrite of existing label file")
      else:
         print (help_string)
         sys.exit()

   if not imagedir :
       imagedir = "D:\\Photo_set_1\\"
       print("Using default imagedir: ", imagedir)
   if not inpfile :
       inpfile = "hwi_sequence_out.csv"    
       print("Using default input file: ", inpfile)
   if not outfile :
       outfile = "hwi_bbox_labels.csv"    
       print("Using default input file: ", outfile)

   csvinp = imagedir + inpfile    
   print("Using input file: ", csvinp)
       
   csvout = imagedir + outfile    
   print("Using output file: ", csvout)

   print("Show each image: ", show_each_image, "Image directory: ", imagedir )
   print("CSV output file: ", csvout, "   Start image number: ", start_imagenum)

# check the summary sheet to see if there are any eagles. exit if none.

   deer_coord = imagedir+deer_coord_file
   my_file = Path(deer_coord)
   if my_file.is_file():
       # file exists
       df_deer_coord = pd.read_csv(deer_coord)
   else :
       print ("Exiting. Missing carcass coordinate file: ", deer_coord)
       sys.exit()    

   # make sure x and y aren't switched
   upperleft_x =  df_deer_coord.loc[0, 'UpperX']
   upperleft_y = df_deer_coord.loc[0, 'UpperY']
   lowright_x = df_deer_coord.loc[0, 'LowerX']
   lowright_y = df_deer_coord.loc[0, 'LowerY']
   
          
   #df_images = pd.DataFrame()                 #columns=column_name_list)
   df_seq = pd.read_csv(csvinp)
  #print("head df_images: ", df_images[0:3])
   if not overwrite_flag :
       df_label = check_csv(csvout)
   else :
       df_label = pd.DataFrame()
       
# this really needs to be cleaned up! it's so bad
   first_read = False
   seq_time_stamp = []
   save_images_in_seq = []
   save_color_in_seq = []
   img_in_seq_diff = []
   img_in_seq_binary = []
   #labels = []
   #objects_centers = []
   save_morph_in_seq = []
   df_save_sequence = pd.DataFrame(columns=sequence_list)
   #df_object = pd.DataFrame(columns=object_list)       
   count = 0  # for debug break out of loop early
   seq_after_dark = 0
   #found_base_flag = False

# loop over files flagged with eagles in the summary sheet
# if the image isn't flagged with eagle, set all bboxes to 'no eagle'
   found_begin_date = False
   for file in os.listdir(imagedir):
       if not file.endswith(tuple(ext)):
           continue

       
       # Get the image number ranges containing eagles from the summary spreadsheet
       if not found_begin_date :
           image_file = os.path.join(imagedir, file)
           DateTime, CameraNumber, mode, seq_num, seq_len = imtools.get_all_photo_info(image_file)
           image_run = os.path.basename(os.path.normpath(imagedir))    # CamXXX_date_carcass
           print("image_run: ", image_run)
           eagle_start_stop = get_eagle_ranges(summary_sheet, image_run)   # CameraNumber, DateTime)
           found_begin_date = True
       
       base_name = os.path.splitext(file)[0]
       # need to test the format of the file name to ensure IMG_xxxx
       image_num = int(base_name.split('_')[1])
       #print("base_name: ", base_name, "  image number: ", image_num)
       if image_num < start_imagenum :
           continue
       # if this is the first image in the run, start with first in sequence
       if first_read == False:
           if seq_num == 1 :
               first_read = True
           else :
               print("skipping to first in sequence: ", image_num, seq_num, seq_len)
               continue

# get the corresponding sequence dataframe rows, one for each bbox
       df_current = df_seq[df_seq['Image1'] == image_num]
       #print(df_current)
       
# see if the image is in a range with eagles
       # if no eagle, set the label to zero in df_label and continue on
       if not check_range(image_num, eagle_start_stop) :
#Camera	Datetime	Dir	Image1	Image2	Night	NumObj	ROI	SeqLen	SeqNum	TimeDiff1	TimeDiff2	TimeDiff3	area	bbox	centroid	filled_area	hist1	hist2	histdiff	label	mean_intensity1	mean_intensity2	mean_intensitydiff	moments_hu	orientation	weighted_moments_hu1	weighted_moments_hu2	weighted_moments_hu_diff
           for row in df_current.itertuples() :
               df_label = df_label.append({'Dir':row.Dir, 
                                           'Camera':row.Camera,
                                           'Datetime':row.Datetime, 
                                           'Image1':row.Image1,
                                           'Image2':row.Image2,
                                           'ROI':row.ROI,
                                           'regionID':row.regionID,
                                           'bbox':row.bbox,
                                           'EagleLabel':0}, ignore_index=True)
           continue

       
       # If eagle, read in the image and it's diff partner for displaying
       
       image_file = os.path.join(imagedir, file)
       print("processing file: ", image_file)
       region_index = df_current.index.values
       diff_pair_file = os.path.join(imagedir, "IMG_" + str(int(df_current.loc[region_index[0], 'Image2'])).zfill(4) +".JPG")
       print("diff pair file: ", diff_pair_file)

       if int(df_current.loc[region_index[0], 'NumObj']) == 0 :
           #plt.show()
           # need to append record
           df_label = df_label.append({'Dir':df_current.loc[region_index[0], 'Dir'], 
                                       'Camera':df_current.loc[region_index[0], 'Camera'],
                                       'Datetime':df_current.loc[region_index[0], 'Datetime'], 
                                       'Image1':df_current.loc[region_index[0], 'Image1'],
                                       'Image2':df_current.loc[region_index[0], 'Image2'],
                                       'Side': 0,
                                       'ROI':df_current.loc[region_index[0], 'ROI'],
                                       'regionID':df_current.loc[region_index[0], 'regionID'],
                                       'bbox':df_current.loc[region_index[0], 'bbox'],
                                       'EagleLabel':0}, ignore_index=True)
           print("No bbox, label no eagle for image: ", df_current.loc[region_index[0], 'Image1'])
           continue

# in general this might not be a good idea, but it works for photo_set_1
# if there's only one bbox, label it an eagle
       #if int(df_current.loc[region_index[0], 'NumObj']) == 1 :
       #    #plt.show()
       #    # need to append record
       #    df_label = df_label.append({'Dir':df_current.loc[region_index[0], 'Dir'], 
       #                                'Camera':df_current.loc[region_index[0], 'Camera'],
       #                                'Datetime':df_current.loc[region_index[0], 'Datetime'], 
       #                                'Image1':df_current.loc[region_index[0], 'Image1'],
       #                                'Image2':df_current.loc[region_index[0], 'Image2'],
       #                                'Side': 0
       #                                'ROI':df_current.loc[region_index[0], 'ROI'],
       #                                'regionID':df_current.loc[region_index[0], 'regionID'],
       #                                'bbox':df_current.loc[region_index[0], 'bbox'],
       #                                'EagleLabel':1}, ignore_index=True)
       #    print("One bbox, label eagle for image: ", df_current.loc[region_index[0], 'Image1'])
       #    continue
       
       #  How does mean and std work on color images? The image is color here.
       img_gray=Image.open(image_file).convert('L')
       img1 = crop_image(img_gray)        # img comes out color and gray scale for ROI

       img_gray=Image.open(diff_pair_file).convert('L')
       img2 = crop_image(img_gray)        # img comes out color and gray scale for ROI

# Display pair of images with bboxes       
       fig = plt.figure(figsize=(16, 12))    
       ax1 = fig.add_subplot(2, 2, 1)
       ax2 = fig.add_subplot(2, 2, 2)
       title1 = ax1.set_title(image_file, loc='center', y=1.1, fontsize=20)
       title2 = ax2.set_title(diff_pair_file, loc='center', y=1.1, fontsize=20)

       ax1.imshow(img1, cmap=plt.cm.gray)   #, aspect='auto', cmap=plt.cm.spectral) 
       ax2.imshow(img2, cmap=plt.cm.gray)   #, aspect='auto', cmap=plt.cm.spectral) 

       ax_hist1 = fig.add_subplot(2, 2, 3)
       ax_hist2 = fig.add_subplot(2, 2, 4)



       
# iterate over regions
       for i, region in enumerate(df_current.itertuples()) :
           print("i: ", i, "  bbox: ", region.bbox, "  centroid: ", region.centroid)
           #if region.bbox == "" :
           #    continue
           
           bbox = ast.literal_eval(region.bbox)
           centroid = ast.literal_eval(region.centroid)
           # the arrays from np.histogram have no commas in the CSV file
           hist1 = ast.literal_eval(','.join(region.hist1.split()))
           hist2 = ast.literal_eval(','.join(region.hist2.split()))

           #hist1 = ast.literal_eval(re.sub('[\s+]', ',', region.hist1))
           #hist2 = ast.literal_eval(re.sub('[\s+]', ',', region.hist2))



           #if skew(hist1) > skew(hist2) :   # Use img1
           #    strstats1="Stats: " + str(region.regionID)+": "+str(round(np.median(hist1[1:]),4))+" "+str(round(np.std(hist1[1:]),4))+" "+str(round(skew(hist1[1:]),5))
           #else :
           #    strstats2="Stats 2: " + str(region.regionID)+": "+str(round(np.median(hist2[1:]),4))+" "+str(round(np.std(hist2[1:]),4))+" "+str(round(skew(hist2[1:]),5))
           strstats1="Med: "+str(round(np.median(hist1[1:]),4))+"  Std: "+str(round(np.std(hist1[1:]),4))+"  Skew: "+str(round(skew(hist1[1:]),5))
           strstats2="Med: "+str(round(np.median(hist2[1:]),4))+"  Std: "+str(round(np.std(hist2[1:]),4))+"  Skew: "+str(round(skew(hist2[1:]),5))

           ax_hist1.plot(hist1[1:], color=colors[i])
           ax_hist2.plot(hist2[1:], color=colors[i])
       

           #                         upper_x, upper_y,  width=x diff,   height=y diff
           rect1 = patches.Rectangle((bbox[1],bbox[0]),(bbox[3]-bbox[1]),(bbox[2]-bbox[0]),linewidth=4,edgecolor=colors[i],facecolor='none')
           rect2 = patches.Rectangle((bbox[1],bbox[0]),(bbox[3]-bbox[1]),(bbox[2]-bbox[0]),linewidth=4,edgecolor=colors[i],facecolor='none')

           ax1.add_patch(rect1)
           ax2.add_patch(rect2)

           #ax_hist1.suptitle(strstats1, color='black', fontsize=10)
           ax_hist1.annotate(strstats1, xy=(0.25, 0.75+0.05*i), xycoords='axes fraction', color=colors[i], fontsize=15)   
           ax_hist2.annotate(strstats2, xy=(0.25, 0.75+0.05*i), xycoords='axes fraction', color=colors[i], fontsize=15)

      #     morph_tmp = img_morph[bbox[0]:bbox[2], bbox[1]:bbox[3]]
       #plt.show()       

           ###################
       cid = fig.canvas.mpl_connect('button_press_event', onclick)
       print("mouse click cid: ", cid)

       plt.show()
       if (gl_x == 0 and gl_y == 0) :
           print ("You didn't click the carcass.")
           #sys.exit()
           
       user_x_coord = int(gl_x)
       user_y_coord = int(gl_y)
       side = 0
       if gl_axes == ax1 :
           side = 1
       elif gl_axes == ax2 :
           side = 2
       else :
           print("Neither axes selected")
           usr_label = input("Include image in training set? (i=include):  ")
           if usr_label != 'i' :
               print("Skipping to next image")
               continue
           
       print("User selected side: ", side)   
       fig.canvas.mpl_disconnect(cid)

#
# User clicks bbox with eagle, then append df_current with EagleLabel to df_label
#
       print("before highlight loop")
       save_bbox = ()
       label = 0    # if user doesn't click a bbox, the eagle label should be set to 0
       for i, region in enumerate(df_current.itertuples()) :
           bbox = ast.literal_eval(region.bbox)
           #                         upper_x, upper_y,  width=x diff,   height=y diff
           print("user coord: ", user_x_coord, user_y_coord, "  bbox: ", bbox)
           if user_x_coord > bbox[1] and user_x_coord < bbox[3] :
               if user_y_coord > bbox[0] and user_y_coord < bbox[2] :
                   # label this box with eagle=1
                   #rect1 = patches.Rectangle((bbox[1],bbox[0]),(bbox[3]-bbox[1]),(bbox[2]-bbox[0]),linewidth=4,edgecolor=colors[i],alpha=0.3)
                   #rect2 = patches.Rectangle((bbox[1],bbox[0]),(bbox[3]-bbox[1]),(bbox[2]-bbox[0]),linewidth=4,edgecolor=colors[i],alpha=0.3)
                   #ax1.add_patch(rect1)
                   #ax2.add_patch(rect2)
                   save_bbox = bbox
                   label = 1

           df_label = df_label.append({'Dir':region.Dir, 
                                       'Camera':region.Camera,
                                       'Datetime':region.Datetime, 
                                       'Image1':region.Image1,
                                       'Image2':region.Image2,
                                       'Side': side,
                                       'ROI':region.ROI,
                                       'regionID':region.regionID,
                                       'bbox':region.bbox,
                                       'EagleLabel':label}, ignore_index=True)
           label = 0


       print("clicked bbox: ", save_bbox)
       
    
       usr_label = input("Exit? (e=exit):  ")
       if usr_label == 'e' :
           print("writing to output file: ", csvout)
           df_label.to_csv(csvout, sep=',', index=False)  
           sys.exit()
    
    
       gl_x = gl_y = 0
       user_x_coord = user_y_coord = 0
       plt.close(fig)

 
   print("writing to output file: ", csvout)
   df_label.to_csv(csvout, sep=',', index=False)  
   return 

       
if __name__ == "__main__":
   main(sys.argv[1:])
