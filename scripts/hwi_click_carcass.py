# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 17:20:47 2018

@author: christman
"""

import sys          # , getopt
import os
from pathlib import Path

from PIL import Image                    #, ImageTK
from PIL.ExifTags import TAGS

import numpy as np
import pandas as pd
import argparse


import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

#import matplotlib.animation as animation
from matplotlib import style
from matplotlib.widgets import RectangleSelector


import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

#from matplotlib import pyplot as plt

import csv

sys.path.append(os.getcwd())
import imtools as imtools

ext=[".jpg",".JPG",".png",".PNG",".tif",".TIF"]
deer_coord_file="hwi_carcass_coordinates.csv"

LARGE_FONT= ("Verdana", 12)
NORM_FONT= ("Verdana", 10)
SMALL_FONT= ("Verdana", 8)

DISTANCE = [
        ("Near", "1"),
        ("Intermediate", "2"),
        ("Far", "3"),
    ]
SIZE = [
        ("Small", "1"),
        ("Medium", "2"),
        ("Large", "3"),
    ]
PLANTS = [
        ("None", "1"),
        ("Some", "2"),
        ("Lots", "3"),
    ]

style.use("ggplot")

carcass_coord_file="hwi_carcass_coordinates.csv"
column_name_list = ['File', 'Directory', 'Datetime', 'Camera', 'UpperX', 'UpperY', 'LowerX', 'LowerY']

gl_upperleft_x, gl_upperleft_y = 0, 0
gl_lowright_x, gl_lowright_y = 0, 0

gl_x = [0]
gl_y = [0]
gl_imagedir=""
gl_imagefile=""
gl_savefile="D:\\"




class ClickCarcassApp:
    def __init__(self, master):
        #tk.Tk()
        frame = tk.Frame(master)
        #frame.pack()   #side="top", fill="both", expand = True)
        
        tk.Tk.iconbitmap(master, default="Iconshock-Stroke-Animals-Hawk.ico")
        tk.Tk.wm_title(master, "HWI Image Classification - Carcass Coordinate Entry")

        #self.fig = Figure()
        #self.ax = self.fig.add_subplot(111)
        #self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        #self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=2, rowspan=4, padx=5, sticky="nsew")


        #self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        #self.toolbar = NavigationToolbar2TkAgg(self.canvas, master)
        #self.toolbar.update()
        #self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        #self.canvas.show()

         
        self.button1 = tk.Button(master, 
                         text="QUIT", fg="red",
                         command=frame.quit)
        self.button1.grid(row=1, column=1) #, columnspan=10, sticky="new")
        #self.button1.pack(side="left")
        self.button2 = tk.Button(master,
                         text="Get Image to Select ROI",
                         command=lambda: self.get_imagedir(master))
        self.button2.grid(row=1, column=2) #, columnspan=10, sticky="new")
        #self.button2.pack(side="left")
        self.button3 = tk.Button(master,
                         text="Clear Current Image",
                         command=lambda: self.clear_image(master))
        #self.button3.pack(side="left")
        self.button3.grid(row=1, column=3) #, columnspan=10, sticky="new")
        self.button4 = tk.Button(master,
                         text="Write ROI to File",
                         command=self.write_to_csv)
        #self.button4.pack(side="left")
        self.button4.grid(row=1, column=4) #,columnspan=10, sticky="new")
        
        AskQuestions = False
        if AskQuestions :
            self.carcass_distance = tk.StringVar()
            self.carcass_distance.set("0") # initialize
    
            lb1 = tk.Label(master, text="Carcass distance: ", anchor=tk.W, justify=tk.RIGHT)
            lb1.grid(row=5, column=1)  #, anchor="e", columnspan=25, sticky="new")
            i = 0
            for text, dist in DISTANCE:
                b1 = tk.Radiobutton(master, text=text, variable=self.carcass_distance, value=dist, command=self.radiodist, anchor=tk.E, justify=tk.LEFT)
                b1.grid(row=5, column=2+i)
                i=i+1
    
    
            self.carcass_size = tk.StringVar()
            self.carcass_size.set("0") # initialize
            lb2 = tk.Label(master, text="Carcass size: ", anchor=tk.W, justify=tk.RIGHT)
            lb2.grid(row=7, column=1)  #, anchor="e", columnspan=25, sticky="new")
            i = 0
            for text, size in SIZE:
                b2 = tk.Radiobutton(master, text=text, variable=self.carcass_size, value=size, command=self.radiosize, anchor=tk.E, justify=tk.LEFT)
                b2.grid(row=7, column=2+i)
                i=i+1
    
    
            self.obscuring_plants = tk.StringVar()
            self.obscuring_plants.set("0") # initialize
    
            lb2 = tk.Label(master, text="Obscuring plants: ", anchor=tk.W, justify=tk.RIGHT)
            lb2.grid(row=9, column=1)  #, anchor="e", columnspan=25, sticky="new")
            i = 0
            for text, plant in PLANTS:
                b3 = tk.Radiobutton(master, text=text, variable=self.obscuring_plants, value=plant, command=self.radioplant, anchor=tk.E, justify=tk.LEFT)
                b3.grid(row=9, column=2+i)
                i=i+1


        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        #self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        # get_tk_widget grid used for radio buttons
        self.canvas.get_tk_widget().grid(row=10, column=0, columnspan=20, rowspan=40, padx=5, sticky="nsew")
        #self.canvas.show()

        self.canvas.show()                    
        #return
        
    def write_to_csv(self):
        csvout=gl_imagedir + '\\' + carcass_coord_file


        if gl_upperleft_x==0 and gl_upperleft_y==0 and gl_lowright_x==0 and gl_lowright_y==0 :
            self.popupmsg("Select a Region of Interest before saving")
            return


        UseQuestions = False
        if UseQuestions :
            if gl_x[-1] == 0 and gl_y[-1] == 0 :
                self.popupmsg("Click the carcass")
                return
            if self.carcass_distance.get() == "0" :
                self.popupmsg("Select a distance")
                return
            if self.carcass_size.get() == "0" :
                self.popupmsg("Select a size")
                return
            if self.obscuring_plants.get() == "0" :
                self.popupmsg("Select a value for obscuring plants")
                return

        print ("Carcass coordinates output to file: ", csvout)
        df = pd.DataFrame(columns=column_name_list)
        ## if output file exists, ask user to overwrite or exit
        #df_images = check_csv(csvout)
        #image_file = os.path.join(gl_imagedir, gl_imagefile)
        DateTime, CameraNumber = imtools.get_photo_info(gl_savefile)
        print ("DateTime, CameraNumber:  ", DateTime, CameraNumber)
        df = df.append({'File':gl_imagefile, 
                        'Directory':gl_imagedir, 
                        'Datetime':DateTime, 
                        'Camera':CameraNumber, 
                        'UpperX':int(gl_upperleft_x), 
                        'UpperY':int(gl_upperleft_y), 
                        'LowerX':int(gl_lowright_x), 
                        'LowerY':int(gl_lowright_y)}, ignore_index=True)

        # old carcass coordinates
        #df = df.append({'File':gl_imagefile, 'Directory':gl_imagedir, 'Datetime':DateTime, 'Camera':CameraNumber, 'Carcass X':gl_x[-1], 'Carcass Y':gl_y[-1], 'Carcass Dist':self.carcass_distance.get(), 'Carcass Size':self.carcass_size.get(), 'Obscuring Plants':self.obscuring_plants.get()}, ignore_index=True)

        df.to_csv(csvout, sep=',', index=False)   
        
        
    def popupmsg(self, msg):
        popup = tk.Tk()
        popup.wm_title("!")
        label = ttk.Label(popup, text=msg, font=NORM_FONT)
        label.pack(side="top", fill="x", pady=10)
        B1 = ttk.Button(popup, text="Okay", command = popup.destroy)
        B1.pack()
        popup.mainloop()

    def clear_image(self, master):
        print("in clear_image")
        self.fig.clf()

        self.canvas.get_tk_widget().destroy()
        self.ax.clear()
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().grid(row=10, column=0, columnspan=20, rowspan=40, padx=5, sticky="nsew")
        self.canvas.show()

        gl_upperleft_x, gl_upperleft_y = 0, 0
        gl_lowright_x, gl_lowright_y = 0, 0


        #self.canvas.grid(row=5, column=10)
        #self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        #self.toolbar = NavigationToolbar2TkAgg(self.canvas, master)
        #self.toolbar.update()
        #self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        print("Need to reinitialize coordinates")
        #self.carcass_distance.set("0") # initialize
        #self.carcass_size.set("0") # initialize
        #self.obscuring_plants.set("0") # initialize



    def get_imagedir(self, master):
        global gl_imagedir
        global gl_imagefile
        global gl_savefile
        global RS
        global gl_x
        global gl_y


        filename = ""
        print("file dialog for directory: ", gl_savefile)

        filename =  filedialog.askopenfilename(initialdir=os.path.basename(gl_savefile), 
                                               initialfile=os.path.basename(gl_savefile),
                                               #parent = get_parent_window(self),
                                               title = "Select file",
                                               filetypes = (("all files", "*.*"), ("jpeg files","*.jpg") ))
        if not filename :
            print("No file selected")
            return

        gl_savefile=filename
        print ("file name: ", filename) 
        gl_imagedir, gl_imagefile = os.path.split(filename)

        print('in get_imagedir, global imagedir: ', gl_imagedir, "  global imagefile: ", gl_imagefile)
        img=np.array(Image.open(filename))    # Read in gray scale image
        img_shape = img.shape
        print("image shape: ", img_shape)
        #gl_x.clear()
        #gl_y.clear()

        gl_upperleft_x, gl_upperleft_y = 0, 0
        gl_lowright_x, gl_lowright_y = 0, 0

        #cid = self.canvas.mpl_connect('button_press_event', self.onclick)
        
        # don't need to toggle the rectangle selector
        #cid = self.canvas.mpl_connect('key_press_event', toggle_selector)

        RS = RectangleSelector(self.ax, line_select_callback,
                                               drawtype='box', useblit=True,
                                               button=[1, 3],  # don't use middle button
                                               minspanx=5, minspany=5,
                                               spancoords='pixels',
                                               interactive=True)

        
        self.ax.set_title(filename, fontsize=10)
        self.fig.suptitle("Click and drag a rectangle around the carcass", fontsize=12)
        implot = self.ax.imshow(img, picker=True)

        #self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        #print(matplotlib.artist.getp(a))
        self.canvas.show()


    def radiodist(self) :
        print("in radiodist, value: ", self.carcass_distance.get())
        
    def radiosize(self) :
        print("in radiosize, value: ", self.carcass_size.get())
        
    def radioplant(self) :
        print("in radioplants, value: ", self.obscuring_plants.get())
        
    # old -- used for clicking the center coordinates of the carcass
    #def onclick(self, event):
    #    global gl_x
    #    global gl_y
    #    #gl_x = 0
    #    #gl_y = 0
    #    self.ax
    #    #print("in onclick")
    #    if event.xdata != None and event.ydata != None:
    #        gl_x.append(event.xdata)
    #        gl_y.append(event.ydata)
    #        print(gl_x[-1], gl_y[-1]) 
    #        self.ax.plot(gl_x[0:-1], gl_y[0:-1], 'bx', markersize = 10)
    #        self.ax.plot(gl_x[-1], gl_y[-1], 'r.', markersize = 10)
    #        self.canvas.show()
    #        
    #    return

    
            
            
############### end class
# used for selecting a rectangle ROI
# This callback didn't work as part of the ClickCarcassApp class. 
def line_select_callback(eclick, erelease):
    global gl_upperleft_x, gl_upperleft_y, gl_lowright_x, gl_lowright_y
    # eclick and erelease are the press and release events
    gl_upperleft_x, gl_upperleft_y = eclick.xdata, eclick.ydata
    gl_lowright_x, gl_lowright_y = erelease.xdata, erelease.ydata
    print("line_select_callback: (%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (gl_upperleft_x, gl_upperleft_y, gl_lowright_x, gl_lowright_y))
    #print("line_select_callback: The button you used were: %s %s" % (eclick.button, erelease.button))


# used for selecting a rectangle ROI    
def toggle_selector(self, event):
    print('toggle_selector: Key pressed.')
    if event.key in ['Q', 'q'] and RS.active:
        print('toggle_selector: RectangleSelector deactivated.')
        RS.set_active(False)
    if event.key in ['A', 'a'] and not RS.active:
        print('toggle_selector: RectangleSelector activated.')
        RS.set_active(True)


root = tk.Tk()
app = ClickCarcassApp(root)
root.geometry("800x600")
root.mainloop()