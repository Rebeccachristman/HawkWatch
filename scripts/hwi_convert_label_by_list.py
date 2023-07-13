# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 09:04:01 2018

@author: christman
"""

import sys, getopt
import os
import subprocess
from pathlib import Path

pgm_sequence = "C:\\Users\\smith\\Documents\\Becky\\HawkWatch\\scripts\\hwi_sequence.py"
pgm_crawl = "C:\\Users\\smith\\Documents\\Becky\\HawkWatch\\scripts\\hwi_crawl_dir.py"
filename_carcass_coord = "hwi_carcass_coordinates.csv" 
filename_sequence = "hwi_sequence_out.csv"

def list_image_dirs(startpath):

    lst_dirs = []
    exclude_prefixes = ('__', '.')  # exclusion prefixes

    for root, dirs, files in os.walk(startpath):   
        for f in files:
            if f.startswith('IMG_') and f.endswith('.JPG') :
            #if f.endswith('.JPG') :
                if not root.replace(startpath, '').startswith(exclude_prefixes):
                    #print(root)
                    lst_dirs.append(root+"\\") 
                break
        
    return lst_dirs

          
def main(argv):
    basedir=""
    try:
       opts, args = getopt.getopt(argv,"hd:")
       help_string = str(sys.argv[0]) + " -d <base directory>" 
    except getopt.GetoptError:
       print (help_string)
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print (help_string)
          sys.exit()
       elif opt in ("-d", "--basedir"):
          basedir = arg
          if not basedir.endswith('\\') :
              basedir = basedir + '\\'
       else:
          print (help_string)
          sys.exit()

    if not basedir :
        basedir = "E:\\Trail_Cameras\\Utah\\2016-2017\\"
        print("Using default base directory: ", basedir)
    
    # if the carcass file exists and there's no sequence file, call hwi_sequence
    for x in list_image_dirs(basedir) :
        #print("x: ", x)
        #get the last token in the directory name
        dirname = x.split("\\")[-2]
        #print("dirname: ", dirname)            
        #print("calling hwi_sequence for: ", x)
        cmd = "python hwi_convert_label.py -d " + dirname
        print("running cmd: ", cmd)
        output = subprocess.call(cmd, shell=True)

    #output = subprocess.check_output("dir "+basedir+ "; exit 0", stderr=subprocess.STDOUT, shell=True)
    #print("output: ", output)
    
    return
  
       
if __name__ == "__main__":
   main(sys.argv[1:])
