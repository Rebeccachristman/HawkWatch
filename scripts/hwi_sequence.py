#!/usr/bin/python
# -*- coding: utf-8 -*-


from __future__ import division, print_function, absolute_import
import sys, getopt
import os
from pathlib import Path
import math

#import scipy
from scipy.stats import skew
from scipy.ndimage import filters
from scipy.ndimage import measurements, morphology
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage.measure import regionprops 
from skimage.color import rgb2gray

from PIL import Image
from PIL.ExifTags import TAGS

import numpy as np
import pandas as pd
import argparse
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import operator

import csv

sys.path.append(os.getcwd())
import imtools as imtools
import ROF as ROF


ext=[".jpg",".JPG",".png",".PNG",".tif",".TIF"]


top_crop = int(32)
bottom_crop = int(top_crop + 25)

# Thresholds for designating an image as after dark
dark_mean_threshold = 20
dark_std_threshold = 60    # photo sets 2 & 3 processes with std=30

imagedir = ''
outfile = ''
csvout="hwi_sequence_out.csv"
deer_coord_file="hwi_carcass_coordinates.csv"

# used for 14-15 July tuning
#column_name_list = ['File', 'Dir', 'Datetime', 'Camera', 'SeqNum', 'SeqLen', 'SeqNumDiff', 'Night', 'Mean', 'Std', 'TopCrop', 'BottomCrop', 'Carcass X', 'Carcass Y', 'Carcass Dist', 'Carcass Size', 'Obscuring Plants', 'NumObj', 'DistRank', 'Size', 'X', 'Y', 'Dist', 'Angle' ]
#sequence_list = ['File', 'Dir', 'Datetime', 'Camera', 'SeqNum', 'SeqLen', 'SeqNumDiff', 'Night', 'Mean', 'Std']
#object_list = ['DistRank', 'Size', 'X', 'Y', 'Dist', 'Angle' ]

#column_name_list = ['Dir', 'Camera', 'UpperX', 'UpperY', 'LowerX', 'LowerY', 'ImagePair', 'Datetime', 'SeqNum', 'SeqLen', 'Night', 'DiffMean', 'DiffStd', 'DiffMedian', 'NumObj', 'SizeMean','SizeStd','SizeSkew','DistMean','DistStd','DistSkew','AngleMean','AngleStd','AngleSkew','TimeDiff1', 'TimeDiff2', 'TimeDiff3' ]
#column_name_list = ['Dir', 'Camera', 'ROI','Image1','Image2','Datetime','SeqNum','SeqLen','Night','TimeDiff1', 'TimeDiff2','TimeDiff3', 'NumObj',"area","bbox","centroid","filled_area","regionID","mean_intensity1","mean_intensity2","mean_intensitydiff","moments_hu","orientation","weighted_moments_hu1","weighted_moments_hu2", "weighted_moments_hu_diff","hist1","hist2","histdiff"]
column_name_list = ['Dir', 'Camera', 'ROI','Image1','Image2','Datetime','SeqNum','SeqLen','Night','TimeDiff1', 'TimeDiff2','TimeDiff3', 'NumObj',"area","bbox","centroid","filled_area","regionID","mean_intensity1","mean_intensity2","moments_hu","orientation","weighted_moments_hu1","weighted_moments_hu2","hist1","hist2"]

sequence_list = ['File' 'Dir', 'Camera', 'ImgNum', 'Datetime', 'SeqNum', 'SeqLen', 'Night', 'DiffMean', 'DiffStd', 'DiffMedian']
#object_list = ['DistRank', 'Size', 'X', 'Y', 'Dist', 'Angle' ]


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



def prep_image(image_color, image_gray):
    global deer_x
    global deer_y
    global lowright_x, lowright_y
    global upperleft_x, upperleft_y

# move reading the image to the main loop to avoid reading it once for the after-dark logic
# and a second time here
    #img_gray_arr=np.array(Image.open(image_file).convert('L'))
    img_color_arr=np.array(image_color)
    img_gray_arr=np.array(image_gray)
    # Conversion to grayscale moved to Image.open before dark check
    #img_gray_arr=Image.fromarray(image,mode='L')
    #img_gray_arr=np.array(image.convert('L'))
    #img_gray = rgb2gray(image)

    Crop_to_User_ROI = True
    if Crop_to_User_ROI :
        img_color_arr_crop = img_color_arr[int(upperleft_y):int(lowright_y), int(upperleft_x):int(lowright_x)]
        img_gray_arr_crop = img_gray_arr[int(upperleft_y):int(lowright_y), int(upperleft_x):int(lowright_x)]
    else :
        # crop the top and bottom information lines so they don't add black to the cdf
        # should check to make sure the image has black top & bottom before cropping
        img_color_arr_crop = img_color_arr[top_crop:-bottom_crop, :]  # extra 25 to remove the company logo
        img_gray_arr_crop = img_gray_arr[top_crop:-bottom_crop, :]  # extra 25 to remove the company logo
        
        
    img_gray_arr_crop, cdf = imtools.histeq(img_gray_arr_crop)
    UseDenoise = False
    if UseDenoise :
        # denoising works well but takes a long time to run
        denoised = ROF.denoise(img_gray_arr_crop, num_iter_max=20)
    else :
        denoised = img_gray_arr_crop
        
    return img_color_arr_crop, denoised




# apparently morphology doesn't work with natural landscapes
def make_binary_image(image, imgstats) :
    # Tuning on July 14-15 done with threshold of mean
    #img_binary = 1*(image > image.mean())
    
    # ROF denoising of the diff image does not much effect morphology
    #
    # Estimate the average noise standard deviation across color channels.
    # sigma_est = estimate_sigma(image, multichannel=False, average_sigmas=True)
    # Due to clipping in random_noise, the estimate will be a bit smaller than the
    # specified sigma.
    #print("Estimated Gaussian noise standard deviation = {}".format(sigma_est))
    #denoised = ROF.denoise(image, num_iter_max=50)

    # other experiments which take too long to run and don't work well
    #denoised = denoise_tv_chambolle(image, weight=0.1, multichannel=False)
    #denoised = denoise_bilateral(image, sigma_color=0.05, sigma_spatial=15, multichannel=False)
    #denoised = denoise_wavelet(image, multichannel=False)

    # Gaussian smears too much signal
    # use sigma proportional to the diagonal of the cropped image size
    #sig = 0.0005*np.sqrt(image.shape[0]**2 + image.shape[1]**2)
    #print("image shape: ", image.shape, "sig: ", sig)
    #denoised = filters.gaussian_filter(image, sigma=sig)
    
    denoised = image
    
    MorphologyBinaryOpening = False
    if MorphologyBinaryOpening :
        # This doesn't work for grass
        structure = np.ones((3,3))
        img_binary = morphology.binary_opening(denoised, structure, iterations=2).astype(np.int)
    else :
        #img_binary = 1*(denoised > imgstats['Mean'])   #imgmedian)  #+imgstd)
        img_binary = 1*(denoised > imgstats['Q75'])   #imgmedian)  #+imgstd)

    PlotBinary = False
    if PlotBinary :
        fig = plt.figure(figsize=(16, 8))
    
        ax1 = fig.add_subplot(121)
        plt.imshow(denoised)          #, cmap=plt.cm.spectral)    
        #ax2 = fig.add_subplot(132)
        #plt.imshow(img_binary_open)          #, cmap=plt.cm.spectral)
        ax3 = fig.add_subplot(122)
        plt.imshow(img_binary)
        plt.tight_layout()
        plt.show()

    return img_binary


def measure_objects(bin_image, imagediff, image1, image2) :
    
    # might be better to make the binary image in here
    label_im, nbr_labels = measurements.label(bin_image)
    #fig=plt.figure(figsize=(14, 14))
    #fig.add_subplot(1, 3, 1) 
    #plt.imshow(bin_image)   
    sizes = measurements.sum(bin_image, label_im, range(nbr_labels + 1))

    # there is a large number of pixels with the labels used for background, keep these
    mask_size = sizes < 10000
    #mask_size = sizes < 5000

    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0
    labels, label_unique_counts = np.unique(label_im, return_counts=True)
    label_clean = np.searchsorted(labels, label_im)

    # Remove label=0 for the background. This may not be a good idea.
    labels_tmp = []
    label_unique_counts_tmp = []
    for index in range(0, len(labels)) :
        if labels[index] > 0 :
            labels_tmp.append(labels[index])
            label_unique_counts_tmp.append(label_unique_counts[index])
    

    
    props  = regionprops(label_im, image1)    #, coordinates='xy')
    props2 = regionprops(label_im, image2)    #, coordinates='xy')
    propsdiff = regionprops(label_im, imagediff)    #, coordinates='xy')

    # centroid of first labeled object

    # sort the list of regions in place...
    #props.sort(key=operator.itemgetter('area'), reverse=True)

    ExtractProps = True
    if ExtractProps :
        #imgshape = image.shape
        #imgcenter = (imgshape[0]/2.0, imgshape[1]/2.0)
        #print("image shape: ", imgshape, "   center: ", imgcenter)
        #hypot_len = math.hypot(imgcenter[0],imgcenter[1])
        region_list = []
        for x in range(0, props.__len__()):
            #norm_dist = math.hypot(props[x].centroid[0]-imgcenter[0],props[x].centroid[1]-imgcenter[1])/hypot_len
            if not props[x].label == props2[x].label :
                print("!!!! Error: props labels don't match: ", props[x].label, props2[x].label, "   Exiting")
                sys.exit(2)
                
            imhist1, bins1 = np.histogram(props[x].intensity_image.flatten(),  bins=10, range=(5,255), density=True)
            imhist2, bins2 = np.histogram(props2[x].intensity_image.flatten(), bins=10, range=(5,255), density=True)
            #diffhist, bins2 = np.histogram(propsdiff[x].intensity_image.flatten(), bins=10, range=(5,255), density=False) #True)


            region_dict = {"area": props[x].area,
                  "bbox": props[x].bbox,
                  "centroid": props[x].centroid, 
                  #"convex_area": props[x].convex_area,
                  #"eccentricity": props[x].eccentricity,
                  #"equivalent_diameter": props[x].equivalent_diameter,
                  #"euler_number": props[x].euler_number,
                  #"extent": props[x].extent,
                  "filled_area": props[x].filled_area,
                  #"image": props[x].image,
                  "label": props[x].label,
                  #"major_axis_length": props[x].major_axis_length,
                  #"minor_axis_length": props[x].minor_axis_length,
                  #"max_intensity1": props[x].max_intensity, 
                  #"max_intensity2": props2[x].max_intensity, 
                  "mean_intensity1": props[x].mean_intensity, 
                  "mean_intensity2": props2[x].mean_intensity, 
                  #"mean_intensitydiff": propsdiff[x].mean_intensity, 
                  #"min_intensity1": props[x].min_intensity, 
                  #"min_intensity2": props2[x].min_intensity, 

                  "moments_hu": props[x].moments_hu,
                  "orientation": props[x].orientation,
                  #"perimeter": props[x].perimeter,
                  #"solidity": props[x].solidity,
                  #"slice1": props[x].slice,
                  #"slice2": props2[x].slice,
                  #"weighted_local_centroid1": props[x].weighted_local_centroid,
                  #"weighted_local_centroid2": props2[x].weighted_local_centroid,
                  "weighted_moments_hu1": props[x].weighted_moments_hu,
                  "weighted_moments_hu2": props2[x].weighted_moments_hu,
                  #"weighted_moments_hu_diff": propsdiff[x].weighted_moments_hu,
                  "hist1": imhist1,
                  "hist2": imhist2,
                  #"histdiff": diffhist
                  }

                  #"norm_dist_ROI_center": norm_dist
            region_list.append(region_dict)
        # No need to sort if each bbox is written out as a separate row
        # sort by area to get largest object first in list
        #region_list.sort(key=operator.itemgetter('area'))
    #print("long list distances: ")
    #for x in region_list :
    #    print(x['norm_dist_ROI_center'])

    #short_list = []
    #for i in range(0, props.__len__()):
    #    if i < 2 :        # save the two closest regions
    #        short_list.append(region_list[i])

    #print("short_list distances: ")
    #for x in short_list :
    #    print(x['norm_dist_ROI_center'])

#   label_centers_array = [x, y]
    ##fig.add_subplot(1, 3, 2) 
    ##plt.imshow(label_im, cmap=plt.cm.spectral) 
    ##fig.add_subplot(1, 3, 3) 
    #fig.add_subplot(1, 1, 1) 
    #plt.imshow(label_clean, cmap=plt.cm.spectral)
    #plt.plot(x, y, 'r.', markersize = 15)
    #plt.show()    

    #print("labels, count: ", len(label_unique_counts_tmp), label_unique_counts_tmp)
    #print (": ".join(map(str, labels)))
    #return region_list.__len__(), region_list, label_im
    return region_list, label_im
    #return props, label_im


def main(argv):
   global deer_x
   global deer_y
   global lowright_x, lowright_y
   global upperleft_x, upperleft_y

   cwd = os.getcwd()
   #print("\n*************** ")
   #print("current working directory: ", cwd)

   show_each_image = False
   start_imagenum = 1
   imagedir=""  
   csvout=""
   outfile=""
   overwrite_flag = False
   try:
      opts, args = getopt.getopt(argv,"hswd:o:n:")
      #print 'Number of arguments:', len(sys.argv), 'arguments.'
      #print 'Argument List:', str(sys.argv)
      help_string = str(sys.argv[0]) + " -s -w -d <imagedir> -o <outputfile> -n <imagenum>" 
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
      elif opt in ("-o", "--csvout"):
         outfile = arg
      elif opt in ("-n", "--startimagenum"):
         start_imagenum = int(arg)
      elif opt in ("-w", "--overwrite"):
         overwrite_flag = True
         print("Forcing overwrite of existing sequence files")
      else:
         print (help_string)
         sys.exit()

   #start_imagenum = 1540 #!!!!!!!!!!!for debug
   #print("!!!!!!!!!!!!!!!!! Remove start_imagenum = 1540")

   if not imagedir :
       imagedir = "D:\\Photo_set_1\\"
       print("Using default imagedir: ", imagedir)
   if not outfile :
       outfile = "hwi_sequence_out.csv"    
       print("Using default output file: ", outfile)
       
   csvout = imagedir + outfile    
   print("Using output file: ", csvout)

   print("Show each image: ", show_each_image, "Image directory: ", imagedir )
   print("CSV output file: ", csvout, "   Start image number: ", start_imagenum)

   deer_coord = imagedir+deer_coord_file
   my_file = Path(deer_coord)
   if my_file.is_file():
       # file exists
       df_deer_coord = pd.read_csv(deer_coord)
   else :
       print ("Exiting. Missing carcass coordinate file: ", deer_coord)
       sys.exit()    

   DeerCoordinates = False
   if DeerCoordinates :
       deer_x = df_deer_coord.loc[0, 'Carcass X']
       deer_y = df_deer_coord.loc[0, 'Carcass Y'] - top_crop    # adjust for cropping top to remove info banner
       print("File: ", deer_coord,"  Carcass coordinates: X: ", deer_x, "  Y: ", deer_y)           
       deer_dist = df_deer_coord.loc[0, 'Carcass Dist']
       deer_size = df_deer_coord.loc[0, 'Carcass Size']
       deer_plants = df_deer_coord.loc[0, 'Obscuring Plants']

   CarcassROI = True
   if CarcassROI :
        # make sure x and y aren't switched
        upperleft_x =  df_deer_coord.loc[0, 'UpperX']
        upperleft_y = df_deer_coord.loc[0, 'UpperY']
        lowright_x = df_deer_coord.loc[0, 'LowerX']
        lowright_y = df_deer_coord.loc[0, 'LowerY']
       
        # may need to have the user click the center of the carcass
        deer_x = int( (lowright_x - upperleft_x)/2.0)
        deer_y = int( (lowright_y - upperleft_y)/2.0)
           
   #      outputfile = arg
   #print ("Base image is ", imagebase)
   #print ("CSV is ", csvout)
   df_images = pd.DataFrame(columns=column_name_list)
   
   # if output file exists, ask user to overwrite or append
   # if append, read the file in
   if not overwrite_flag :
       df_images = check_csv(csvout)
       #print("head df_images: ", df_images[0:3])

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
   for file in os.listdir(imagedir):
       # skip over images before the base image file
       #if count > 200:
       #    break
       if not file.endswith(tuple(ext)):
           continue
       base_name = os.path.splitext(file)[0]
       # need to test the format of the file name to ensure IMG_xxxx
       image_num = int(base_name.split('_')[1])
       #print("base_name: ", base_name, "  image number: ", image_num)
       
       # command line arguement specified an image number to begin 
       if image_num < start_imagenum :
           continue
       
       image_file = os.path.join(imagedir, file)
       #print("processing file: ", image_file)

       DateTime, CameraNumber, mode, seq_num, seq_len = imtools.get_all_photo_info(image_file)
       #DateTime, CameraNumber = imtools.get_photo_info(image_file)
       #print ("DateTime, CameraNumber:  ", DateTime, CameraNumber)          
       #mode, seq_num, seq_len = imtools.get_sequence(image_file)
       #print ("file, mode, sequence num, sequence length:  ", file, mode, seq_num, seq_len)          

       # if this is the first image in the run, start with first in sequence
       if first_read == False:
           if seq_num == 1 :
               first_read = True
           else :
               print("skipping to first in sequence: ", image_num, seq_num, seq_len)
               continue
           

       # make sure the camera is in mode 'M', presumably motion trigger
       if not mode == 'M' :
           print("Big problems! Mode is: ", mode, "  should be M. Exiting.")
           sys.exit()

       #  How does mean and std work on color images? The image is color here.
       img_color=Image.open(image_file)
       img_gray=np.array(img_color.convert('L'))   # all of this to avoid hitting the disk twice
       img_color=np.array(img_color)
       #img_color=np.array(Image.open(image_file))

       #img=np.array(Image.open(image_file).convert('L'))
       
       #img = np.array(prep_image(image_file))
       img_color, img = prep_image(img_color, img_gray)        # img comes out color and gray scale for ROI
       #print("after cropping deer_x: ", deer_x, "  deer_y: ", deer_y)
       # save images
       # may want to use median here
       #print("raw single image mean, std:  ", img_mean, img_std)
       orig_stats = get_imgstats(img)
       #print("original grayscale stats: ", orig_stats)


       after_dark = 0
       if orig_stats['Mean'] < dark_mean_threshold and orig_stats['Std'] < dark_std_threshold :
           #print("Image is dark. Skip it")
           after_dark = 1
           seq_after_dark = 1

       save_images_in_seq.append(img) 
       save_color_in_seq.append(img) 

       df_save_sequence = df_save_sequence.append({'Dir':imagedir, 
                                                   'Camera':CameraNumber,
                                                   'ImgNum':image_num,
                                                   'Datetime':DateTime, 
                                                   'SeqNum': seq_num, 
                                                   'SeqLen': seq_len, 
                                                   'Night':after_dark}, ignore_index=True)


       # if this is the last image in a sequence of X out of X images, compute the delta images
       # to see what changed
       if seq_num < seq_len :  # get the next image in the sequence
           continue
       
       if seq_after_dark == 1 :  # one or more images in the sequence were flagged as after dark
          print("Sequence is flagged as after dark")
          seq_after_dark = 0 
          for seq_index in range(0, seq_len) :    # append the dark sequence to df_images
          #for seq_index in range(0, seq_len-1) :    # append the dark sequence to df_images
#'Dir', 'Camera', 'ImgNum1', 'ImgNum2', 'Datetime1', 'Datetime2', 'SeqNum1', 'SeqNum2',
               df_images = df_images.append({    #'File':df_save_sequence.loc[seq_index,'File'],
                                             'Dir':df_save_sequence.loc[seq_index,'Dir'], 
                                             'Camera':df_save_sequence.loc[seq_index,'Camera'], 
                                             'ROI': [(int(upperleft_x),int(upperleft_y)), (int(lowright_x),int(lowright_y))],
                                             #'UpperX':int(upperleft_x), 
                                             #'UpperY':int(upperleft_y), 
                                             #'LowerX':int(lowright_x), 
                                             #'LowerY':int(lowright_x), 
                                             #'ImagePair':get_image_pair(df_save_sequence.loc[seq_index,'ImgNum'], seq_index, seq_num, seq_len), 
                                             'Image1':get_image_pair(df_save_sequence.loc[seq_index,'ImgNum'], seq_index, seq_num, seq_len)[0], 
                                             'Image2':get_image_pair(df_save_sequence.loc[seq_index,'ImgNum'], seq_index, seq_num, seq_len)[1], 
                                             'Datetime':df_save_sequence.loc[seq_index,'Datetime'], 
                                             'SeqNum':df_save_sequence.loc[seq_index,'SeqNum'], 
                                             'SeqLen':df_save_sequence.loc[seq_index,'SeqLen'], 
                                             'Night':1}, ignore_index=True)

#sequence_list = ['File' 'Dir', 'Camera', 'ImgNum', 'Datetime', 'SeqNum', 'SeqLen', 'Night', 'DiffMean', 'DiffStd', 'DiffMedian']


          # reinitialize for next sequence
          count = count + 1  
          df_save_sequence.drop(df_save_sequence.index[:], inplace=True)  # reinitialize for each sequence
          continue
       seq_after_dark = 0
       
       # Save Datetime for last image in the sequence
       if len(seq_time_stamp) == 0 :  
           # if first image, put it's time stamp on the stack 3 time so the timediff calculations below don't blow up
           seq_time_stamp.append(df_save_sequence.loc[0,'Datetime'])
           seq_time_stamp.append(df_save_sequence.loc[0,'Datetime'])
           seq_time_stamp.append(df_save_sequence.loc[0,'Datetime'])
           
       seq_time_stamp.append(df_save_sequence.loc[seq_len-1,'Datetime'])

#### might not be used
       index = 0
#######################

       # Calculate difference squared images
       for seq_index in range(0, seq_len) :      # include wrap around
       #for seq_index in range(0, seq_len-1) :   # no wrap around
           #print("seq_index, seq_len: ", seq_index, seq_len)
           if (seq_index+1) < seq_len :
               # sqrt added 7-19 then removed 7-20
               # squaring diffs gives a spread between low (grass) and high (eagle) intensities
               img_diff_tmp = (save_images_in_seq[seq_index+1] - save_images_in_seq[seq_index])**2

               #print("image diff: seq_index: ", seq_index)
               df_save_sequence.loc[seq_index, 'SeqNumDiff'] =  str(seq_index+1)+"|"+str(seq_index)

           elif (seq_index+1) == seq_len :
               # sqrt added 7-19
               # 7-20 removed wrap around diff 3-1 for seqlen=3 and 2-1 for seqlen=2
               #print("wrap around sequence: shouldn't get here")
               img_diff_tmp = (save_images_in_seq[0] - save_images_in_seq[seq_index])**2
               #print("image diff wrap around: seq_index: ", seq_index)
               df_save_sequence.loc[seq_index, 'SeqNumDiff'] =  "0|"+str(seq_index)

           else :
               print("Error in storing image diffs: seq_index, seq_len: ", seq_index, seq_len)
               
           #### img_diff_norm and image_gaussian aren't used
           #img_diff_norm, cdf = imtools.histeq(img_diff_tmp)        
           #image_gaussian = filters.gaussian_filter(img_diff_norm, sigma=50)
           ######
           
           img_in_seq_diff.append(img_diff_tmp)

#!!!! Don't need a second loop here. The loops can be combined 
      # Determine morphology and store measurements in df_image for writting to csv    
# try combining loops
       #for seq_index in range(0, seq_len) :      # include wrap around
       #for seq_index in range(0, seq_len-1) :               

           diff_stats = get_imgstats(img_in_seq_diff[seq_index])
           #df_save_sequence.loc[seq_index, 'DiffMean'] =  diff_stats['Mean']
           #df_save_sequence.loc[seq_index, 'DiffStd']  =  diff_stats['Std']
           #df_save_sequence.loc[seq_index, 'DiffMedian']  =  diff_stats['Median']
           img_in_seq_binary.append(make_binary_image(img_in_seq_diff[seq_index], diff_stats))
           #nbr, sizes, centers, img_morph = measure_objects(img_in_seq_binary[seq_index])
           #nbr, sizes, centers, img_morph = measure_objects(save_images_in_seq[seq_index], img_in_seq_binary[seq_index])
           if (seq_index+1) < seq_len :
               regions, img_morph = measure_objects(img_in_seq_binary[seq_index], np.sqrt(img_in_seq_diff[seq_index]), save_images_in_seq[seq_index+1], save_images_in_seq[seq_index])
           else :
               regions, img_morph = measure_objects(img_in_seq_binary[seq_index], np.sqrt(img_in_seq_diff[seq_index]), save_images_in_seq[0], save_images_in_seq[seq_index])

           #regions, img_morph = measure_objects(img_in_seq_binary[seq_index], np.sqrt(img_in_seq_diff[seq_index]))
           #nbr = regions.__len__()
           nbr = regions.__len__()

           df_save_sequence.loc[seq_index, 'NumObj']  =  nbr
           save_morph_in_seq.append(img_morph)                # save the morph for displaying later
           
           PlotBBox = False
           if PlotBBox :
               color_stats_1 = []
               color_stats_2 = []
               fig_tmp = plt.figure(figsize=(16, 12))    
               ax_tmp = fig_tmp.add_subplot(1, 1, 1)
               #ax_tmp = fig_tmp.add_subplot(1, 2, 1)
               #ax_tmp.imshow(img_in_seq_binary[seq_index], cmap=plt.cm.gray)   #, aspect='auto', cmap=plt.cm.spectral) 
               ax_tmp.imshow(img_in_seq_binary[seq_index], aspect='auto', cmap=plt.cm.spectral) 

               #ax_hist = fig_tmp.add_subplot(1, 2, 2)

               for region in regions :
                   #ax_hist.plot(region['histdiff'], color="black")

                   #if region['mean_intensity1'] > region['mean_intensity2'] :
                   #    ax_hist.plot(region['hist1'], color="blue")
                   #else :
                   #    ax_hist.plot(region['hist2'], color="red")
                   #print("bbox: ", region['bbox'], "  centroid: ", region['centroid'])
                   if skew(region['hist1']) > skew(region['hist2']) :   # Use img1
                       strstats="Use Image 1: " + str(region['label'])+": "+str(round(np.median(region['hist1']),4))+" "+str(round(np.std(region['hist1']),4))+" "+str(round(skew(region['hist1']),5))
                   else :
                       strstats="Use Image 2: " + str(region['label'])+": "+str(round(np.median(region['hist2']),4))+" "+str(round(np.std(region['hist2']),4))+" "+str(round(skew(region['hist2']),5))


                   #print(strstats)
                   #print("img2: ", str2)
                   
                   # removed for getting image for ppt presentation
                   #ax_hist.plot(region['hist1'], color="blue")
                   #ax_hist.plot(region['hist2'], color="red")


                   bbox = region['bbox']
                   #                         upper_x, upper_y,  width=x diff,   height=y diff
                   rect = patches.Rectangle((bbox[1],bbox[0]),(bbox[3]-bbox[1]),(bbox[2]-bbox[0]),linewidth=5,edgecolor='r',facecolor='none')
                   ax_tmp.add_patch(rect)
                   #ax_tmp.annotate(str(region['label']), xy=(bbox[1],bbox[0]), xycoords='data', color='green', fontsize=25)

                   #                          (upper_y, upper_x, low_y, low_x) with rows=up&down=x, cols=right&left=y
                   # bounding box coordinates (min_row, min_col, max_row, max_col)
    #        img_color_arr_crop = img_color_arr[int(upperleft_y):int(lowright_y), int(upperleft_x):int(lowright_x)]
    # stats on grayscale aren't going to work. Try colors
    #    colours = img.getcolors(w*h)  #Returns a list [(pixel_count, (R, G, B))]
    
                   morph_tmp = img_morph[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                   #print("seq_index: ", seq_index, "  morph max: ",  np.max(img_morph[bbox[0]:bbox[2], bbox[1]:bbox[3]]))
                   #print("bbox: ", bbox)
                   #print("slice: ", bbox[0], ":", bbox[2], ",", bbox[1], ":", bbox[3])
                   if (seq_index+1) < seq_len :
                       # bounding box: (min_row, min_col, max_row, max_col)
                       #img_gray_arr_crop = img_gray_arr[bbox[1]:bbox[2], bbox[3]:bbox[4]]
                       color_stats_1.append(get_imgstats(save_images_in_seq[seq_index+1][bbox[0]:bbox[2], bbox[1]:bbox[3]]))
                       color_stats_2.append(get_imgstats(save_images_in_seq[seq_index][bbox[0]:bbox[2], bbox[1]:bbox[3]]))
                   elif (seq_index+1) == seq_len :   # wrap around
                       color_stats_1.append(get_imgstats(save_images_in_seq[0][bbox[0]:bbox[2], bbox[1]:bbox[3]]))
                       color_stats_2.append(get_imgstats(save_images_in_seq[seq_index][bbox[0]:bbox[2], bbox[1]:bbox[3]]))
                   else :
                       print("Error in looping over bbox color stats: seq_index, seq_len: ", seq_index, seq_len)
           #plt.show()       
    
           #df_region = pd.DataFrame.from_dict(region, orient='columns')
# figure out a better way to put the dictionary into the dataframe row
           #if regions.__len__ == 0 :
           if nbr == 0 :
               df_images = df_images.append({'Dir':df_save_sequence.loc[seq_index,'Dir'], 
                                         'Camera':df_save_sequence.loc[seq_index,'Camera'], 
                                         'ROI': [(int(upperleft_x),int(upperleft_y)), (int(lowright_x),int(lowright_y))],
                                         'Image1':get_image_pair(df_save_sequence.loc[seq_index,'ImgNum'], seq_index, seq_num, seq_len)[0], 
                                         'Image2':get_image_pair(df_save_sequence.loc[seq_index,'ImgNum'], seq_index, seq_num, seq_len)[1], 
                                         'Datetime':df_save_sequence.loc[seq_index,'Datetime'], 
                                         'SeqNum':df_save_sequence.loc[seq_index,'SeqNum'], 
                                         'SeqLen':df_save_sequence.loc[seq_index,'SeqLen'], 
                                         'Night':0, 
                                         'TimeDiff1': get_timediff(seq_time_stamp[-2], df_save_sequence.loc[seq_index,'Datetime']),
                                         'TimeDiff2': get_timediff(seq_time_stamp[-3], df_save_sequence.loc[seq_index,'Datetime']), 
                                         'TimeDiff3': get_timediff(seq_time_stamp[-4], df_save_sequence.loc[seq_index,'Datetime']), 
                                         'NumObj':0, 
                                         }, ignore_index=True)
           else :
               for region in regions :
                   df_images = df_images.append({'Dir':df_save_sequence.loc[seq_index,'Dir'], 
                                             'Camera':df_save_sequence.loc[seq_index,'Camera'], 
                                             'ROI': [(int(upperleft_x),int(upperleft_y)), (int(lowright_x),int(lowright_y))],
                                             'Image1':get_image_pair(df_save_sequence.loc[seq_index,'ImgNum'], seq_index, seq_num, seq_len)[0], 
                                             'Image2':get_image_pair(df_save_sequence.loc[seq_index,'ImgNum'], seq_index, seq_num, seq_len)[1], 
                                             'Datetime':df_save_sequence.loc[seq_index,'Datetime'], 
                                             'SeqNum':df_save_sequence.loc[seq_index,'SeqNum'], 
                                             'SeqLen':df_save_sequence.loc[seq_index,'SeqLen'], 
                                             'Night':0, 
                                             'TimeDiff1': get_timediff(seq_time_stamp[-2], df_save_sequence.loc[seq_index,'Datetime']),
                                             'TimeDiff2': get_timediff(seq_time_stamp[-3], df_save_sequence.loc[seq_index,'Datetime']), 
                                             'TimeDiff3': get_timediff(seq_time_stamp[-4], df_save_sequence.loc[seq_index,'Datetime']), 
                                             'NumObj':df_save_sequence.loc[seq_index,'NumObj'], 
                                             "area": region['area'],
                                             "bbox": region['bbox'],
                                             "centroid": region['centroid'], 
                                             "filled_area": region['filled_area'],
                                             "regionID": region['label'],
                                             "mean_intensity1": region['mean_intensity1'], 
                                             "mean_intensity2": region['mean_intensity2'], 
                                             #"mean_intensitydiff": region['mean_intensitydiff'], 
                                             "moments_hu": str(region['moments_hu']).replace('\n', ' ').replace('\r', ''),
                                             "orientation": region['orientation'],
                                             "weighted_moments_hu1": str(region['weighted_moments_hu1']).replace('\n', ' ').replace('\r', ''),
                                             "weighted_moments_hu2": str(region['weighted_moments_hu2']).replace('\n', ' ').replace('\r', ''),
                                             #"weighted_moments_hu_diff": region['weighted_moments_hu_diff'],
                                             "hist1": str(region['hist1']).replace('\n', ' ').replace('\r', ''),
                                             "hist2": str(region['hist2']).replace('\n', ' ').replace('\r', ''),
                                             #"histdiff": region['histdiff']
                                             }, ignore_index=True)



       #show_each_image = False   # pass in as a command line flag
       if show_each_image :
           fig = plt.figure(figsize=(14, 14))
           #ax = plt.axes
           #fig, axes = plt.subplots(gl_rows, gl_cols)

           for seq_index in range(0, seq_len) :
           #for seq_index in range(0, seq_len-1) :
               #fig.add_subplot(gl_rows, gl_cols, seq_len*seq_index+1) 
               ax = fig.add_subplot(gl_rows, gl_cols, gl_cols*seq_index+1)
               if seq_index == 0 :
                   #ax = fig.axes[0, 0]
                   title = ax.set_title("Grayscale Cropped", loc='center', y=1.1, fontsize=20)
               xlabel = str(get_image_pair(df_save_sequence.loc[seq_index,'ImgNum'], seq_index, seq_num, seq_len)) + "  Seq: "+str(seq_index+1)+"/"+str(seq_len)
               ax.set_xlabel(xlabel, fontsize=15)             #alpha=0.5)
               ax.get_xaxis().set_ticks([])
               ax.get_yaxis().set_ticks([])
               plt.imshow(save_images_in_seq[seq_index])
               #plt.imshow(img_in_seq_diff[seq_index])

               #fig.add_subplot(gl_rows, gl_cols, seq_len*seq_index+2) 
               #plt.imshow(img_in_seq_diff[seq_index])
               ax = fig.add_subplot(gl_rows, gl_cols, gl_cols*seq_index+2) 
               if seq_index == 0 :
                   #ax = fig.axes[0, 1]
                   title = ax.set_title("Differences Images", loc='center', y=1.1, fontsize=20)
               #fig.add_subplot(gl_rows, gl_cols, gl_cols*seq_index+2) 
               if (seq_index+1) < seq_len :
                   xlabel = "Subtract "+str(seq_index+2)+" and "+str(seq_index+1)
               elif (seq_index+1) == seq_len :
                   xlabel = "Subtract 1 and "+str(seq_index+1)
               ax.set_xlabel(xlabel, fontsize=15)             #alpha=0.5)
               ax.get_xaxis().set_ticks([])
               ax.get_yaxis().set_ticks([])
               plt.imshow(img_in_seq_diff[seq_index])


               ax = fig.add_subplot(gl_rows, gl_cols, gl_cols*seq_index+3) 
               if seq_index == 0 :
                   #ax = fig.axes[0, 2]
                   title = ax.set_title("Moving Objects", loc='center', y=1.1, fontsize=20)
               xlabel = str(int(df_save_sequence.loc[seq_index, 'NumObj'])) + " objects"
               ax.set_xlabel(xlabel, fontsize=15)             #alpha=0.5)
               ax.get_xaxis().set_ticks([])
               ax.get_yaxis().set_ticks([])

               plt.imshow(save_morph_in_seq[seq_index], cmap=plt.cm.spectral)
               ShowObjCenters = False
               if ShowObjCenters :
                   obj_coord = objects_centers[seq_index]
                   obj_x = [coord[0] for coord in obj_coord]
                   obj_y = [coord[1] for coord in obj_coord]
                   #obj_y = [y[slice(1)] for y in objects_centers[seq_index]]
                   #plt.plot(obj_x, obj_y, 'r.', markersize = 15, color='red')
                   #plt.plot(deer_x, deer_y, 'r.', marker='X', markersize = 10, color='red')
                   line_x = []
                   line_y = []
                   for line_index in range(0, len(obj_coord)) :
                       line_x = [deer_x, obj_x[line_index]]
                       line_y = [deer_y, obj_y[line_index]]
                       plt.plot(line_x, line_y, 'r.', marker='.', markersize = 10, color='red', linestyle='dashed', linewidth=1)
                   #plt.annotate('Deer', xy=(deer_x, deer_y), xytext='data', xycoords='data', color='green', arrowprops=dict(facecolor='green', shrink=0.01),horizontalalignment='center', verticalalignment='center')
                   plt.annotate('Deer', xy=(deer_x, deer_y), xycoords='data', color='red', fontsize=15, horizontalalignment='center', verticalalignment='top')

           plt.show()
           usr_label = input("Exit? (e=exit):  ")
           if usr_label == 'e' :
               print("writing to output file: ", csvout)
               df_images.to_csv(csvout, sep=',', index=False)  
               sys.exit()
           plt.close(fig)


       count = count + 1      
       save_images_in_seq.clear()
       save_color_in_seq.clear()
       save_morph_in_seq.clear()
       df_save_sequence.drop(df_save_sequence.index[:], inplace=True)  # reinitialize for each sequence

       img_in_seq_diff.clear()
       img_in_seq_binary.clear()
#       objects_centers.clear()
       
       # Checkpoint to file, use 200 for runs
       if ( count % 200 == 0 ) :
           checkpoint_filename = ".\\checkpoint\\"+"bird_sequence.chkpt-"+str(count)
           print('Writing checkpoint file: ', checkpoint_filename)
           df_images.to_csv(checkpoint_filename, sep=',', index=False)
 
   print("writing to output file: ", csvout)
   df_images.to_csv(csvout, sep=',', index=False)  
   return 

       
if __name__ == "__main__":
   main(sys.argv[1:])
