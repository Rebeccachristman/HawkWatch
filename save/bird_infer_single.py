#!/usr/bin/python
# -*- coding: utf-8 -*-


#from __future__ import division, print_function, absolute_import

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

model_bird_classifier="C:\\Users\\smith\\Documents\\Becky\\HawkWatch\\model\\bird-classifier.tfl"
#model_bird_classifier="C:\\Users\\smith\\Documents\\Becky\\HawkWatch\\bird-sandbox\\bird-classifier.tfl.ckpt-27824"


def main(argv):
   imagedir = ''
   outputfile = ''
   cwd = os.getcwd()
   print("\n*************** ")
   print("current working directory: ", cwd)
   print("using bird-classifier model: ", model_bird_classifier)
   try:
      opts, args = getopt.getopt(argv,"hi:",["infile="])
      #print 'Number of arguments:', len(sys.argv), 'arguments.'
      #print 'Argument List:', str(sys.argv)
   except getopt.GetoptError:
      print ("bird_infer_single.py -i <infile>")
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print ("bird_infer_single.py -i <infile>")
         sys.exit()
      elif opt in ("-i", "--infile"):
         infile = arg

   print ("Infile is ", infile)

# setup for displaying the images with matplotlib
   #fig=plt.figure(figsize=(8, 8))
   columns = 2
   rows = 1
   
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

   model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='epoch70-bird-classifier.tfl.ckpt')

   model.load(model_bird_classifier)

         
   # Load the image file
   img = scipy.ndimage.imread(infile)       
   fig=plt.figure(figsize=(8, 8))
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
       fig.suptitle(infile+"\n"+ r"  A bird!", fontsize=20)
   else:
       print("That's not a bird!")
       fig.suptitle(infile+"\n"+ r"  NOT a bird!", fontsize=20)


   plt.show()
          

       
if __name__ == "__main__":
   main(sys.argv[1:])
