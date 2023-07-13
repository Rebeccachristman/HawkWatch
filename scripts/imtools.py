# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 11:01:01 2018

@author: christman
"""
import numpy as np
import re
from PIL import Image
from PIL.ExifTags import TAGS


# Get EXIF tags
#for (k,v) in Image.open(sys.argv[1])._getexif().iteritems():   # Python 2
#for (k,v) in Image.open(image_file)._getexif().items():       # Python 3
#        print ('%s = %s' % (TAGS.get(k), v))

def get_photo_info(image_file) :
    try :
        exif = Image.open(image_file)._getexif()   # Make RECONYX Model PC800 Professional
        #print("type, exif: ", type(exif), exif)
        #MakerNote=get_field(exif,'MakerNote')
        #for (k,v) in exif.items():    #Python 3
        #    if TAGS.get(k) == "MakerNote" :
        #        MakerNote = v
        #    if TAGS.get(k) == "DateTime" :
        #        DateTimeStamp = v
                
    except:
        return None, None

    DateTime=get_field(exif,'DateTime')
    #print ("get_photo_info: DateTime:  ", DateTime)
    MakerNote=get_field(exif,'MakerNote')
    #print("get_photo_info: Makernote: ", MakerNote)
    CameraNumber=get_camera_number(MakerNote)
    #print ("get_photo_info: camera:  length: ", CameraNumber, len(CameraNumber))

    return DateTime, CameraNumber

# Returns all photo information needed by hwi_sequence. Combined function so image is read only once
def get_all_photo_info(image_file) :
    try :
        exif = Image.open(image_file)._getexif()   # Make RECONYX Model PC800 Professional
        #print("type, exif: ", type(exif), exif)
        #MakerNote=get_field(exif,'MakerNote')
        #for (k,v) in exif.items():    #Python 3
        #    if TAGS.get(k) == "MakerNote" :
        #        MakerNote = v
        #    if TAGS.get(k) == "DateTime" :
        #        DateTimeStamp = v
                
    except:
        return None, None, None, None, None

    DateTime=get_field(exif,'DateTime')
    #print ("get_photo_info: DateTime:  ", DateTime)
    makernote=get_field(exif,'MakerNote')
    #print("get_photo_info: Makernote: ", MakerNote)
    CameraNumber=get_camera_number(makernote)
    CameraNumber = CameraNumber.replace(" ", "")
    #print ("get_photo_info: camera:  length: ", CameraNumber, len(CameraNumber))
    mode = chr(makernote[12])
    seq_num = makernote[14]
    seq_out_of_images = makernote[16]
    #print("mode, sequence num, out_of_images: ", mode, seq_num, seq_out_of_images)
    return DateTime, CameraNumber, mode, seq_num, seq_out_of_images


# Get a specific EXIF tag
def get_field (exif,field) :
    #print("in get_field")
    try :
        #for (k,v) in exif.iteritems():    # Python 2
        for (k,v) in exif.items():    #Python 3
            if TAGS.get(k) == field:
                return v
    except:
        return None
    
    return None

# Search bytearray for the character string 'CAM' followed by a camera number
def get_camera_number (makernote) :
    camera_number = ""             # "Not Found"
    #print("length: ", len(makernote))
    index=0
    found_space = False
    for x in makernote:
        #print ("index: ", index)
        #if chr(x) == 'C'and chr(makernote[index+1]) == 'A' and chr(makernote[index+2]) == 'M':
        if index == 86 :
            # need to make the length dynamically determined
            #tmp_camera_number=makernote[index:index+9].decode()
            # strip non-alphanumeric characters
            #camera_number = re.sub("[^\\w]", "", tmp_camera_number)  #doesn't work
            #camera_number="CAM"
# CAM 277 12/07/16 Deer has space in the camera name (CAM 277)
            #if chr(makernote[index] == " ") :
            #    found_space = True
            #    j = 4
            #    camera_number = "CAM"
            #    print("get_camera_number: found space in camera number")
            #else :
            #    j = 0
            j = 0    
            while chr(makernote[index+j]).isalnum() :
                camera_number = camera_number + chr(makernote[index+j])
                j = j + 1

                    
            #print("get_camera_number: index, camera_number: ", index, camera_number)
            return camera_number
        index=index+1
    return camera_number

# In Reconyx makernotes, sequence is located at byte offset 0x0e=14 decimal
def get_sequence (image_file) :
    try :
        exif = Image.open(image_file)._getexif()   # Make RECONYX Model PC800 Professional
        #print("type, exif: ", type(exif), exif)
        #MakerNote=get_field(exif,'MakerNote')
        #for (k,v) in exif.items():    #Python 3
        #    if TAGS.get(k) == "MakerNote" :
        #        MakerNote = v
        #    if TAGS.get(k) == "DateTime" :
        #        DateTimeStamp = v
                
    except:
        return None, None

    makernote=get_field(exif,'MakerNote')
    #print("get_sequence: makernote: ", makernote)
    mode = chr(makernote[12])
    seq_num = makernote[14]
    seq_out_of_images = makernote[16]
    #print("mode, sequence num, out_of_images: ", mode, seq_num, seq_out_of_images)
    
    return mode, seq_num, seq_out_of_images


# Histogram equalization of a grayscale image
def histeq(im, nbr_bins=256) :
    # get image histogram
    imhist, bins = np.histogram(im.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum()      # cumulative distribution function
    cdf = 255 * cdf/cdf[-1]    # normalize
    
    # use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape), cdf


