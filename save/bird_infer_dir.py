#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import scipy
import numpy as np
import argparse

import matplotlib.pyplot as plt
import sys, getopt
import os
import csv

ext=[".jpg",".JPG",".png",".PNG",".tif",".TIF"]

model_bird_classifier="C:\\Users\\smith\\Documents\\Becky\\HawkWatch\\model\\epoch70-bird-classifier.tfl"
imagedir = ''
outfile = ''
imagedir="C:\\Users\\smith\\Documents\\Becky\\HawkWatch\\Images\\"
csvdir="C:\\Users\\smith\\Documents\\Becky\\HawkWatch\\Data\\"


def main(argv):

   cwd = os.getcwd()
   print("\n*************** ")
   print("current working directory: ", cwd)
   print("using bird-classifier model: ", model_bird_classifier)
   #try:
   #   #opts, args = getopt.getopt(argv,"hi:o:",["imagedir=","ofile="])
   #   opts, args = getopt.getopt(argv,"hi:o:",["imagedir=","ofile="])
   #
   #   #print 'Number of arguments:', len(sys.argv), 'arguments.'
   #   #print 'Argument List:', str(sys.argv)
   #except getopt.GetoptError:
   #   print ("bird_dir_infer.py -i <imagedir> -o <outputfile>")
   #   sys.exit(2)
   #for opt, arg in opts:
   #   if opt == '-h':
   #      print ("bird_dir_infer.py -i <imagedir> -o <outputfile>")
   #      sys.exit()
   #   elif opt in ("-i", "--imagedir"):
   #      imagedir = arg
   #   elif opt in ("-o", "--ofile"):
   #      outputfile = arg
   print ("Image directory is ", imagedir)



   #print ("CSV tracking directory is ", csvdir)

# setup for displaying the images with matplotlib
   #fig=plt.figure(figsize=(8, 8))
   columns = 2
   rows = 1
   


   #outputfile_writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
                            #quotechar='|', quoting=csv.QUOTE_MINIMAL)
   #row = list("Image", "Indicator")
   #outputfile_writer.writerow(row)

# Same network definition as before
   img_prep = ImagePreprocessing()
   img_prep.add_featurewise_zero_center()
   img_prep.add_featurewise_stdnorm()
   img_aug = ImageAugmentation()
   img_aug.add_random_flip_leftright()
   img_aug.add_random_rotation(max_angle=25.)
   img_aug.add_random_blur(sigma_max=3.)

   network = input_data(shape=[None, 32, 32, 3],
                        data_preprocessing=img_prep,
                        data_augmentation=img_aug)
   network = conv_2d(network, 32, 3, activation='relu')
   network = max_pool_2d(network, 2)
   network = conv_2d(network, 64, 3, activation='relu')
   network = conv_2d(network, 64, 3, activation='relu')
   network = max_pool_2d(network, 2)
   network = fully_connected(network, 512, activation='relu')
   network = dropout(network, 0.5)
   network = fully_connected(network, 2, activation='softmax')


   network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

   model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='bird-classifier.tfl.ckpt')

   model.load(model_bird_classifier)

   for file in os.listdir(imagedir):
       if file.endswith(tuple(ext)):
           image_file = os.path.join(imagedir, file)
           print("processing file: ", image_file)
# setup to write output csv file
           
           # Load the image file
           img = scipy.ndimage.imread(image_file)       
           fig=plt.figure(figsize=(14, 14))
           fig.add_subplot(rows, columns, 1) 
           plt.imshow(img)
           #plt.show()
           
           # Scale it to 32x32
           img = scipy.misc.imresize(img, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')
           fig.add_subplot(rows, columns, 2) 
           plt.imshow(img)
           #plt.show()
 
           # Predict
           prediction = model.predict([img])

           # Check the result.
           is_bird = np.argmax(prediction[0]) == 1

           #fig.suptitle(image_file)
           if is_bird:
               print("That's a bird!")
               fig.suptitle(image_file+"\n"+ r"  A bird!", fontsize=20)
               #fig.suptitle(image_file+"\n"+ r"  A bird!", fontsize=20)
           else:
               print("That's not a bird!")
               #fig.suptitle(image_file+"\n"+ r"  NOT a bird!", fontsize=20)
               fig.suptitle(image_file+"\n"+ r"  Not a bird!", fontsize=20)
               
           #row = list(image_file, is_bird)
           #outputfile_writer.writerow(row)
           

           plt.show()
          
   #plt.show() 
   #csvfile.close()   
       
       
if __name__ == "__main__":
   main(sys.argv[1:])
