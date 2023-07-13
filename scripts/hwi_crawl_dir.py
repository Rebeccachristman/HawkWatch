# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 09:04:01 2018

@author: christman
"""

import sys, getopt
import os

def list_files(startpath):

    exclude_prefixes = ('__', '.')  # exclusion prefixes

    for root, dirs, files in os.walk(startpath):   
        for f in files:
            if f.startswith('IMG_') and f.endswith('.JPG') :
            #if f.endswith('.JPG') :
                if not root.replace(startpath, '').startswith(exclude_prefixes):
                    print(root)
                break
        
    return

          
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
        basedir = "D:\\"
        #print("Using default base directory: ", basedir)
        
    list_files(basedir)
    
    return
  
       
if __name__ == "__main__":
   main(sys.argv[1:])
