# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 09:04:01 2018

@author: christman
"""

import sys, getopt
import os
import subprocess
from pathlib import Path
import time
from datetime import datetime
from dateutil import relativedelta


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
    overwrite = ""
    overwrite_flag = False

    try:
       opts, args = getopt.getopt(argv,"hwd:")
       help_string = str(sys.argv[0]) + " -w -d <base directory>" 
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
       elif opt in ("-w", "--overwrite"):
           overwrite_flag = True
           overwrite = " -w "
           print("Forcing overwrite of existing sequence files")
       else:
          print (help_string)
          sys.exit()

    if not basedir :
        basedir = "E:\\Trail_Cameras\\Utah\\2016-2017\\"
        print("Using default base directory: ", basedir)
 
    #timestart = time.localtime()
    timestart = time.asctime( time.localtime(time.time()) )
    print("Start time: ", timestart)
    # if the carcass file exists and there's no sequence file, call hwi_sequence
    for x in list_image_dirs(basedir) :
        #print("x: ", x)
        carcass_file = Path(x + filename_carcass_coord)
        if carcass_file.exists():
            sequence_file = Path(x + filename_sequence)
            if not sequence_file.exists() or overwrite_flag :
                #print("calling hwi_sequence for: ", x)
                cmd = "python hwi_sequence.py -d " + x + overwrite
                print("running cmd: ", cmd)
                output = subprocess.call(cmd, shell=True)
                timeend = time.asctime( time.localtime(time.time()) )
                print("***Current time: ", timeend)

            else:
                print("Skipping... Sequence file exits in directory: ", x)
        else:
            print("Skipping... No carcass coordinate file in directory: ", x)

    #output = subprocess.check_output("dir "+basedir+ "; exit 0", stderr=subprocess.STDOUT, shell=True)
    #print("output: ", output)
    #timeend = datetime.now()
    #print("Elapsed time for entire loop: ", dateutil.relativedelta(timeend - timestart)
   
    return
  
       
if __name__ == "__main__":
   main(sys.argv[1:])
