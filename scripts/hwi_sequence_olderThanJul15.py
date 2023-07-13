#!/usr/bin/python
# -*- coding: utf-8 -*-


from __future__ import division, print_function, absolute_import
import sys, getopt
import os
from pathlib import Path
import math

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

#lines_to_crop = 32
#lines_top_bottom = 32
top_crop = int(32)
bottom_crop = int(top_crop + 25)


# Thresholds for designating an image as after dark
dark_mean_threshold = 20
dark_std_threshold = 60    # photo sets 2 & 3 processes with std=30

diff_mean_threshold = 40
diff_std_threshold = 60


imagedir = ''
outfile = ''
#imagedir="C:\\Users\\smith\\Documents\\Becky\\HawkWatch\\Images\\"


#csvdir="C:\\Users\\smith\\Documents\\Becky\\HawkWatch\\Data\\"
#csvout="label_output.csv"
csvout="hwi_sequence_out.csv"
deer_coord_file="hwi_carcass_coordinates.csv"

column_name_list = ['File', 'Dir', 'Datetime', 'Camera', 'SeqNum', 'SeqLen', 'SeqNumDiff', 'Night', 'Mean', 'Std', 'TopCrop', 'BottomCrop', 'Carcass X', 'Carcass Y', 'Carcass Dist', 'Carcass Size', 'Obscuring Plants', 'NumObj', 'DistRank', 'Size', 'X', 'Y', 'Dist', 'Angle' ]
sequence_list = ['File', 'Dir', 'Datetime', 'Camera', 'SeqNum', 'SeqLen', 'SeqNumDiff', 'Night', 'Mean', 'Std']
object_list = ['DistRank', 'Size', 'X', 'Y', 'Dist', 'Angle' ]

# setup for displaying the images with matplotlib
# need global qualifier
gl_cols = 3
gl_rows = 3
gl_x = 0
gl_y = 0
gl_binary_threshold = 150
gl_sigma = 30
gl_size_lower_threshold = 500
gl_size_upper_threshold = 40000

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

def onclick(event):
    global gl_x
    global gl_y
    if event.xdata != None and event.ydata != None:
        gl_x = event.xdata
        gl_y = event.ydata
        #print(gl_x, gl_y) 
    return


def prep_image(image):
# move reading the image to the main loop to avoid reading it once for the after-dark logic
# and a second time here
#def prep_image(image_file):

    #img_gray_arr=np.array(Image.open(image_file).convert('L'))
    img_gray_arr=np.array(image.convert('L'))

    # crop the top and bottom information lines so they don't add black to the cdf
    # should check to make sure the image has black top & bottom before cropping
    img_gray_arr_crop = img_gray_arr[top_crop:-bottom_crop, :]  # extra 25 to remove the company logo
    return img_gray_arr_crop



# apparently morphology doesn't work with natural landscapes
def make_binary_image(image) :
    #img_binary=morphology.binary_opening(image, np.ones((4, 4)), iterations=2)
    #img_binary=morphology.binary_opening(image, np.ones((9, 5)), iterations=1)
    image_std = image.std()
    #print("image std: ", image_std)
    #image_gaussian = filters.gaussian_filter(image, sigma=10)
    #img_binary = 1*(image_gaussian < gl_binary_threshold)
    #img_binary = 1*(image_gaussian < image.mean())


    #img_binary = 1*(image < (image.mean()-image_std))
    img_binary = 1*(image > image.mean())
    #img_binary = 1*(image < 128)

    return img_binary


def measure_objects(bin_image) :
    # might be better to make the binary image in here
    label_im, nbr_labels = measurements.label(bin_image)
    #fig=plt.figure(figsize=(14, 14))
    #fig.add_subplot(1, 3, 1) 
    #plt.imshow(bin_image)   
    sizes = measurements.sum(bin_image, label_im, range(nbr_labels + 1))

    # there is a large number of pixels with the labels used for background, keep these
    #mask_size = ((sizes > gl_size_lower_threshold) & (sizes < gl_size_upper_threshold))
    #mask_size = sizes < gl_size_lower_threshold
    mask_size = sizes < 10000
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
    
    # Label centers is an array of tuples
    label_centers = measurements.center_of_mass(bin_image, labels=label_im, index=labels_tmp)
    #centers = measurements.center_of_mass(label_im, labels=label_clean, index=None)
    # x-axis going right to left in the image is the second element of the centers tuple
    # y-axis going up and down is the first element in the centers tuple
    x = []
    y = []
    label_centers_array = []
    for index in range(0, len(label_centers)) :
            x.append(int(label_centers[index][1]))
            y.append(int(label_centers[index][0]))
            # switch the coordinates to (x,y)=(right-left, up-down) to pass back
            label_centers_array.append([int(label_centers[index][1]), int(label_centers[index][0])])
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
    return len(label_unique_counts_tmp), label_unique_counts_tmp, label_centers_array, label_clean

def plot_points() :         # not working yet
    x = []
    y = []
    label_centers_array = []
    for index in range(0, len(label_centers)) :
            x.append(int(label_centers[index][1]))
            y.append(int(label_centers[index][0]))
            # switch the coordinates to (x,y)=(right-left, up-down) to pass back
            label_centers_array.append([int(label_centers[index][1]), int(label_centers[index][0])])
#    label_centers_array = [x, y]
    #fig.add_subplot(1, 3, 2) 
    #plt.imshow(label_im, cmap=plt.cm.spectral) 
    #fig.add_subplot(1, 3, 3) 
    fig.add_subplot(1, 1, 1) 
    plt.imshow(label_clean, cmap=plt.cm.spectral)
    plt.plot(x, y, 'r.', markersize = 15)

    plt.show()    
    return

def main(argv):

   cwd = os.getcwd()
   #print("\n*************** ")
   #print("current working directory: ", cwd)

   show_each_image = False
   start_imagenum = 1
   imagedir=""  
   csvout=""
   outfile=""
   try:
      opts, args = getopt.getopt(argv,"hsd:o:n:")
      #print 'Number of arguments:', len(sys.argv), 'arguments.'
      #print 'Argument List:', str(sys.argv)
      help_string = str(sys.argv[0]) + " -s -d <imagedir> -o <outputfile> -n <imagenum>" 
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
      else:
         print (help_string)
         sys.exit()

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

   deer_x = df_deer_coord.loc[0, 'Carcass X']
   deer_y = df_deer_coord.loc[0, 'Carcass Y'] - top_crop    # adjust for cropping top to remove info banner
   print("File: ", deer_coord,"  Carcass coordinates: X: ", deer_x, "  Y: ", deer_y)           
   deer_dist = df_deer_coord.loc[0, 'Carcass Dist']
   deer_size = df_deer_coord.loc[0, 'Carcass Size']
   deer_plants = df_deer_coord.loc[0, 'Obscuring Plants']
           
   #      outputfile = arg
   #print ("Base image is ", imagebase)
   #print ("CSV is ", csvout)
   df_images = pd.DataFrame(columns=column_name_list)
   
   # if output file exists, ask user to overwrite or append
   # if append, read the file in
   df_images = check_csv(csvout)
   #print("head df_images: ", df_images[0:3])
   
   first_read = False
   save_images_in_seq = []
   img_in_seq_diff = []
   img_in_seq_binary = []
   labels = []
   objects_centers = []
   save_morph_in_seq = []
   df_save_sequence = pd.DataFrame(columns=sequence_list)
   df_object = pd.DataFrame(columns=object_list)       
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
       img=Image.open(image_file)
       after_dark = 0
       img_mean = round(np.mean(np.array(img)), 2)
       img_std = round(np.std(np.array(img)), 2)
       #print("raw single image mean, std:  ", img_mean, img_std)
       if img_mean < dark_mean_threshold and img_std < dark_std_threshold :
           #print("Image is dark. Skip it")
           after_dark = 1
           seq_after_dark = 1
           #column_name_list = ['File', 'Directory', 'Image Number', 'Sequence Number', 'Sequence Length', 'After Dark', 'Image Array']
           #df_images = df_images.append({'File':file, 'Dir':imagedir, 'Datetime':DateTime, 'Camera':CameraNumber, 'SeqNum': seq_num, 'SeqLen': seq_len, 'Night':after_dark}, ignore_index=True)
           #continue
           

       #img = np.array(prep_image(image_file))
       img = np.array(prep_image(img))        # img comes out gray scale with metadata cropped off

       # save images
       save_images_in_seq.append(img) 
       df_save_sequence = df_save_sequence.append({'File':file, 'Dir':imagedir, 'Datetime':DateTime, 'Camera':CameraNumber, 'SeqNum': seq_num, 'SeqLen': seq_len, 'Night':after_dark}, ignore_index=True)

       # if this is the last image in a sequence of X out of X images, compute the delta images
       # to see what changed
       if seq_num < seq_len :  # get the next image in the sequence
           continue
       
       if seq_after_dark == 1 :  # one or more images in the sequence were flagged as after dark
          #print("Sequence is flagged as after dark")
          seq_after_dark = 0 
          for seq_index in range(0, seq_len) :    # append the dark sequence to df_images
               df_images = df_images.append({'File':df_save_sequence.loc[seq_index, 'File'], 'Dir':df_save_sequence.loc[seq_index,'Dir'], 'Datetime':df_save_sequence.loc[seq_index,'Datetime'], 'Camera':df_save_sequence.loc[seq_index,'Camera'], 'SeqNum':df_save_sequence.loc[seq_index,'SeqNum'], 'SeqLen':df_save_sequence.loc[seq_index,'SeqLen'], 'Night':1}, ignore_index=True)
          # reinitialize for next sequence
          count = count + 1  
          df_save_sequence.drop(df_save_sequence.index[:], inplace=True)  # reinitialize for each sequence
          continue
       seq_after_dark = 0
       
#### these might not be used
       index = 0
       last_row_index = df_images.shape[0]
       image_arr_len = img.shape   # image array length after prep_image
       #print("image_arr_len: ", image_arr_len)
#######################

       for seq_index in range(0, seq_len) :
           #print("seq_index, seq_len: ", seq_index, seq_len)
           if (seq_index+1) < seq_len :
               img_diff_tmp = (save_images_in_seq[seq_index+1] - save_images_in_seq[seq_index])**2
               #print("image diff: seq_index: ", seq_index)
               df_save_sequence.loc[seq_index, 'SeqNumDiff'] =  str(seq_index+1)+"|"+str(seq_index)

           elif (seq_index+1) == seq_len :
               img_diff_tmp = (save_images_in_seq[0] - save_images_in_seq[seq_index])**2
               #print("image diff wrap around: seq_index: ", seq_index)
               df_save_sequence.loc[seq_index, 'SeqNumDiff'] =  "0|"+str(seq_index)

           else :
               print("Error in storing image diffs: seq_index, seq_len: ", seq_index, seq_len)

           img_diff_norm, cdf = imtools.histeq(img_diff_tmp)        
           image_gaussian = filters.gaussian_filter(img_diff_norm, sigma=50)

           img_in_seq_diff.append(img_diff_tmp)
           
       for seq_index in range(0, seq_len) :
# Not really using these lists
           df_save_sequence.loc[seq_index, 'Mean'] =  round(np.mean(np.array(img_in_seq_diff[seq_index])), 2)
           df_save_sequence.loc[seq_index, 'Std']  =  round(np.std(np.array(img_in_seq_diff[seq_index])), 2)

           img_in_seq_binary.append(make_binary_image(img_in_seq_diff[seq_index]))
           nbr, sizes, centers, img_morph = measure_objects(img_in_seq_binary[seq_index])
           df_save_sequence.loc[seq_index, 'NumObj']  =  nbr
           save_morph_in_seq.append(img_morph)
           objects_centers.append(centers)

           # Save the dataframe unless this is the second in a sequence of 2. Differences 0-1 and 1-0 are identical.
           #if not ( (seq_index+1) == seq_len and seq_len == 2 ) :
           if True :   # write out duplicate data for seq_len = 2 because matching with label file needs both

               # save header record if header and detail records are being used
               #df_images = df_images.append({'File':df_save_sequence.loc[seq_index, 'File'], 'Dir':df_save_sequence.loc[seq_index,'Dir'], 'Datetime':df_save_sequence.loc[seq_index,'Datetime'], 'Camera':df_save_sequence.loc[seq_index,'Camera'], 'SeqNum':df_save_sequence.loc[seq_index,'SeqNum'], 'SeqLen':df_save_sequence.loc[seq_index,'SeqLen'], 'SeqNumDiff':df_save_sequence.loc[seq_index,'SeqNumDiff'] , 'Night':0, 'Mean':df_save_sequence.loc[seq_index,'Mean'], 'Std':df_save_sequence.loc[seq_index,'Std'], 'TopCrop': top_crop, 'BottomCrop':bottom_crop, 'Carcass X':deer_x, 'Carcass Y':deer_y, 'Carcass Dist':deer_dist, 'Carcass Size':deer_size, 'Obscuring Plants':deer_plants, 'NumObj':df_save_sequence.loc[seq_index,'NumObj'], 'DistRank':0}, ignore_index=True)
               #print("sorting for seq_index, nbr", seq_index, nbr)


               # distance is needed to determine the closest object
               # angle can be calculated from the coordinates in the machine learning program
               objects_dist = []
               objects_angle = []
               # objects found
               for obj_index in range(0, nbr) :
                   obj_coord = centers[obj_index]             # this worked with centers from above
                   obj_x = obj_coord[0]
                   obj_y = obj_coord[1]
                   obj_size = sizes[obj_index]
                   #print("distance coordinates: ", obj_coord[0], obj_coord[1])
                   obj_dist = int(math.sqrt((obj_x-deer_x)**2+(obj_y-deer_y)**2))

                   if obj_coord[0]== deer_x :
                       obj_coord[0] = obj_coord[0] + 1   # fudge to avoid divide by 0
                      
                   obj_angle = int(math.degrees(math.atan(-(obj_coord[1]-deer_y)/(obj_coord[0]-deer_x))))
      
                   objects_dist.append(obj_dist)
                   objects_angle.append(obj_angle)
                   df_object = df_object.append({'Size':sizes[obj_index], 'X':obj_coord[0], 'Y':obj_coord[1], 'Dist':obj_dist, 'Angle':obj_angle}, ignore_index=True)

                   #print("distance, angle: ", objects_dist[obj_index], objects_angle[obj_index])
                   # add a detail record to df_images for each object in the image
               # Use this for closest object data in single record. This case is no objects found.
               if nbr == 0 :
                   df_images = df_images.append({'File':df_save_sequence.loc[seq_index, 'File'], 'Dir':df_save_sequence.loc[seq_index,'Dir'], 'Datetime':df_save_sequence.loc[seq_index,'Datetime'], 'Camera':df_save_sequence.loc[seq_index,'Camera'], 'SeqNum':df_save_sequence.loc[seq_index,'SeqNum'], 'SeqLen':df_save_sequence.loc[seq_index,'SeqLen'], 'SeqNumDiff':df_save_sequence.loc[seq_index,'SeqNumDiff'] , 'Night':0, 'Mean':df_save_sequence.loc[seq_index,'Mean'], 'Std':df_save_sequence.loc[seq_index,'Std'], 'TopCrop': top_crop, 'BottomCrop':bottom_crop, 'Carcass X':deer_x, 'Carcass Y':deer_y, 'Carcass Dist':deer_dist, 'Carcass Size':deer_size, 'Obscuring Plants':deer_plants, 'NumObj':df_save_sequence.loc[seq_index,'NumObj'], 'DistRank':0}, ignore_index=True)
               else :
                   df_object['DistRank'] = df_object['Dist'].rank(ascending=True)
                   index_closest = 0
                   if (df_object.DistRank== 1).any() :
                       index_closest = df_object[df_object.DistRank == 1].index.tolist()[0]
                   elif (df_object.DistRank == 1.5).any() :
                        # This is arbitrary and not a good way to handle it.
                       index_closest = df_object[df_object.DistRank == 1.5].index.tolist()[0]
                   else :   # should never get here due to the nbr=0 check above
                       print("!!!!!!! ERROR: Found image with no closest object", df_save_sequence.loc[seq_index, 'File'])
                       df_images = df_images.append({'File':df_save_sequence.loc[seq_index, 'File'], 'Dir':df_save_sequence.loc[seq_index,'Dir'], 'Datetime':df_save_sequence.loc[seq_index,'Datetime'], 'Camera':df_save_sequence.loc[seq_index,'Camera'], 'SeqNum':df_save_sequence.loc[seq_index,'SeqNum'], 'SeqLen':df_save_sequence.loc[seq_index,'SeqLen'], 'SeqNumDiff':df_save_sequence.loc[seq_index,'SeqNumDiff'] , 'Night':0, 'Mean':df_save_sequence.loc[seq_index,'Mean'], 'Std':df_save_sequence.loc[seq_index,'Std'], 'TopCrop': top_crop, 'BottomCrop':bottom_crop, 'Carcass X':deer_x, 'Carcass Y':deer_y, 'Carcass Dist':deer_dist, 'Carcass Size':deer_size, 'Obscuring Plants':deer_plants, 'NumObj':df_save_sequence.loc[seq_index,'NumObj'], 'DistRank':0}, ignore_index=True)
                   # Add single record with closest object data
                   df_images = df_images.append({'File':df_save_sequence.loc[seq_index, 'File'], 'Dir':df_save_sequence.loc[seq_index,'Dir'], 'Datetime':df_save_sequence.loc[seq_index,'Datetime'], 'Camera':df_save_sequence.loc[seq_index,'Camera'], 'SeqNum':df_save_sequence.loc[seq_index,'SeqNum'], 'SeqLen':df_save_sequence.loc[seq_index,'SeqLen'], 'SeqNumDiff':df_save_sequence.loc[seq_index,'SeqNumDiff'] , 'Night':0, 'Mean':df_save_sequence.loc[seq_index,'Mean'], 'Std':df_save_sequence.loc[seq_index,'Std'], 'TopCrop': top_crop, 'BottomCrop':bottom_crop, 'Carcass X':deer_x, 'Carcass Y':deer_y, 'Carcass Dist':deer_dist, 'Carcass Size':deer_size, 'Obscuring Plants':deer_plants, 'NumObj':df_save_sequence.loc[seq_index,'NumObj'], 'DistRank':df_object.loc[index_closest, 'DistRank'], 'Size':df_object.loc[index_closest, 'Size'], 'X':df_object.loc[index_closest, 'X'], 'Y':df_object.loc[index_closest, 'Y'], 'Dist':df_object.loc[index_closest, 'Dist'], 'Angle':df_object.loc[index_closest, 'Angle']}, ignore_index=True)

               # Save all objects as detail records
               #for obj_index in range(0, nbr) :
               #    df_images = df_images.append({'File':df_save_sequence.loc[seq_index, 'File'], 'Dir':df_save_sequence.loc[seq_index,'Dir'], 'Datetime':df_save_sequence.loc[seq_index,'Datetime'], 'Camera':df_save_sequence.loc[seq_index,'Camera'], 'NumObj':df_save_sequence.loc[seq_index, 'NumObj'], 'DistRank':df_object.loc[obj_index, 'DistRank'], 'Size':df_object.loc[obj_index, 'Size'], 'X':df_object.loc[obj_index, 'X'], 'Y':df_object.loc[obj_index, 'X'], 'Dist':df_object.loc[obj_index, 'Dist'], 'Angle':df_object.loc[obj_index, 'Angle']}, ignore_index=True)
               df_object.drop(df_object.index[:], inplace=True)  # reinitialize for each pair in save_sequence


       #show_each_image = False   # pass in as a command line flag
       if show_each_image :
           fig = plt.figure(figsize=(14, 14))
           #ax = plt.axes
           #fig, axes = plt.subplots(gl_rows, gl_cols)

           for seq_index in range(0, seq_len) :
               #fig.add_subplot(gl_rows, gl_cols, seq_len*seq_index+1) 
               ax = fig.add_subplot(gl_rows, gl_cols, gl_cols*seq_index+1)
               if seq_index == 0 :
                   #ax = fig.axes[0, 0]
                   title = ax.set_title("Grayscale Cropped", loc='center', y=1.1, fontsize=20)
               xlabel = df_save_sequence.loc[seq_index, 'File'] + "  Seq: "+str(seq_index+1)+"/"+str(seq_len)
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
               df_images.to_csv(csvout, sep=',', index=False)  
               sys.exit()


       count = count + 1      
       save_images_in_seq.clear()
       save_morph_in_seq.clear()
       df_save_sequence.drop(df_save_sequence.index[:], inplace=True)  # reinitialize for each sequence

       img_in_seq_diff.clear()
       img_in_seq_binary.clear()
       objects_centers.clear()
       
       # Checkpoint to file
       if ( count % 200 == 0 ) :
           checkpoint_filename = ".\\checkpoint\\"+"bird_sequence.chkpt-"+str(count)
           print('Writing checkpoint file: ', checkpoint_filename)
           df_images.to_csv(checkpoint_filename, sep=',', index=False)
 
   print("writing to output file: ", csvout)
   df_images.to_csv(csvout, sep=',', index=False)  
   return 

       
if __name__ == "__main__":
   main(sys.argv[1:])
