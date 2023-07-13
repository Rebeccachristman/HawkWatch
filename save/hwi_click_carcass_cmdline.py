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
from scipy.ndimage import filters
from scipy.ndimage import measurements, morphology

from PIL import Image, ImageTK
from PIL.ExifTags import TAGS

import numpy as np
import pandas as pd
import argparse

from matplotlib import pyplot as plt
from matplotlib import style
matplotlib.use("TkAgg")
from matplotlib import matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

from matplotlib import style
style.use("ggplot")

import tkinter as tk
from tkinter import tkk

import csv

sys.path.append(os.getcwd())
import imtools as imtools


ext=[".jpg",".JPG",".png",".PNG",".tif",".TIF"]

deer_coord_file="hwi_carcass_coordinates.csv"

column_name_list = ['File', 'Directory', 'Datetime', 'Camera', 'Carcass X', 'Carcass Y', 'Carcass Dist', 'Carcass Size', 'Obscuring Plants']

gl_x = 0
gl_y = 0

def check_csv(filename):
   my_file = Path(filename)
   if my_file.is_file():
       # file exists
       usr_label = input("Output file exists. (o=overwrite, 9=exit):  ")
       if usr_label == '9' :
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
    gl_x = 0
    gl_y = 0
    if event.xdata != None and event.ydata != None:
        gl_x = event.xdata
        gl_y = event.ydata
        #print(gl_x, gl_y) 
    return


def main(argv):

   cwd = os.getcwd()
   print("\n*************** ")
   print("current working directory: ", cwd)

   imagedir = "D:\\junk\\" 
   imagebase = "IMG_0001.JPG"
   try:
      opts, args = getopt.getopt(argv,"hd:f:")
      #opts, args = getopt.getopt(argv,"hn:",["imagenum="])
   
      #print 'Number of arguments:', len(sys.argv), 'arguments.'
      #print 'Argument List:', str(sys.argv)
      help_string = str(sys.argv[0]) + " -d <input directory> -f <imagefile>" 
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
      elif opt in ("-f", "--imagefile"):
         imagebase = arg
      else:
         print (help_string)
         sys.exit()

   print ("Directory: ", imagedir)
   print ("Base image: ", imagebase)
   csvout = imagedir+deer_coord_file             # write CSV to image directory
   print ("Carcass coordinates output to file: ", csvout)
   df_images = pd.DataFrame(columns=column_name_list)
   
   # if output file exists, ask user to overwrite or exit
   df_images = check_csv(csvout)
     
   image_file = os.path.join(imagedir, imagebase)
   DateTime, CameraNumber = imtools.get_photo_info(image_file)
   print ("DateTime, CameraNumber:  ", DateTime, CameraNumber)          
       
   # Don't crop the baseline deer image. The original coordinates are needed.
   # The offset due to cropping should be handled by programs using deer_coordinates.csv    
   img=np.array(Image.open(image_file).convert('L'))    # Read in gray scale image
   img_shape = img.shape
   print("image shape: ", img_shape)
   #ax = plt.gca()
   #fig = plt.gcf()
   fig, ax = plt.subplots(1, 1, figsize=(14, 14))   # preferred alternative to gcf()
   ax.set_title('Click on the carcass, then close image:', fontsize=25)

   implot = ax.imshow(img)
   cid = fig.canvas.mpl_connect('button_press_event', onclick)
   plt.show()
   if (gl_x == 0 and gl_y == 0) :
       print ("You didn't click the carcass. Try again.")
       sys.exit()
       
   user_x_coord = int(gl_x)
   user_y_coord = int(gl_y)
   fig.canvas.mpl_disconnect(cid)
#column_name_list = ['File', 'Directory', 'Datetime', 'Camera', 'Carcass X', 'Carcass Y', 'Carcass Dist', 'Carcass Size', 'Obscuring Plants']
   print("Answer three questions")
   usr_label = ""
   while True :
       usr_label = input("Distance from camera to carcass close(=1), medium(=2), or far(=3). Enter 1, 2 or 3:  ")
       if not (usr_label == '1' or usr_label == '2' or usr_label == '3') : 
           print("entry needs to be 1, 2, or 3")
       else :
           break
   carcass_dist = int(usr_label)
   usr_label = ""

   while True :
       usr_label = input("Size of carcass small (like rabbit=1), medium (like coyote=2), or large (like deer=3). Enter 1, 2 or 3:  ")
       if not (usr_label == '1' or usr_label == '2' or usr_label == '3') : 
           print("entry needs to be 1, 2, or 3")
       else :
           break
   carcass_size = int(usr_label)
   usr_label = 0

   while True :
       usr_label = input("Plants obscuring the camera none(=1), some grass(=2), or significant plant obstruction(=3). Enter 1, 2 or 3:  ")
       if not (usr_label == '1' or usr_label == '2' or usr_label == '3') : 
           print("entry needs to be 1, 2, or 3")
       else :
           break
   obscuring_plants = int(usr_label)
      
   df_images = df_images.append({'File':imagebase, 'Directory':imagedir, 'Datetime':DateTime, 'Camera':CameraNumber, 'Carcass X':user_x_coord, 'Carcass Y':user_y_coord, 'Carcass Dist':carcass_dist, 'Carcass Size':carcass_size, 'Obscuring Plants':obscuring_plants}, ignore_index=True)

   print("user clicked carcass at coordinates: ", user_x_coord, user_y_coord)
   # add code to display the categorical choices for distance, size, and plants

   #fig.add_subplot(1, 2, 2) 
   fig, ax = plt.subplots(1, 1, figsize=(14, 14))   # preferred alternative to gcf()
   ax.set_title('You clicked on the red dot:', fontsize=25)
   plt.imshow(img)
   plt.plot(user_x_coord, user_y_coord, 'r.', markersize = 20)
   plt.show()      


   print("Writing coordinates to: ", csvout)
   df_images.to_csv(csvout, sep=',', index=False)          

   return 

       
if __name__ == "__main__":
   main(sys.argv[1:])
