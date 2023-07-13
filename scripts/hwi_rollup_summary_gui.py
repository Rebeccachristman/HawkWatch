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

YESNO = [
        ("No", "0"),
        ("Yes", "1"),
        ("N/A", ""),
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


column_name_list = ['File', 'Directory', 'Datetime', 'Camera', 'Carcass X', 'Carcass Y', 'Carcass Dist', 'Carcass Size', 'Obscuring Plants']
gl_x = [0]
gl_y = [0]
gl_imagedir=""
gl_savedir="D:\\"
datadir="C:\\Users\\smith\\Documents\\Becky\\HawkWatch\\data\\"
rollupfile=datadir + "hwi_summary_rollup.csv"
mergefile=datadir + "hwi_merge_table.csv"
csvout=datadir + "hwi_rollup.csv"
csvauto = "hwi_auto_classify_out.csv"   # need to concatenate with gl_imagedir

window_size=12       # a sequence has 3 (some imagesets have 2) images
threshold = 0.5      # threshold percentage for sliding window 


class SummaryRollupApp:
    def __init__(self, master):
        # Set up the Tkinter frame
        #tk.Tk()   # Why is this commented out?
        frame = tk.Frame(master)
        #frame.pack()   #side="top", fill="both", expand = True)
        #frame = tk.Frame(master, width=300, height=300, background="bisque")   # didn't work

        tk.Tk.iconbitmap(master, default="Iconshock-Stroke-Animals-Hawk.ico")
        tk.Tk.wm_title(master, "HWI Image Classification - Summary Rollup")


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
                         text="Select Image Directory",
                         command=lambda: self.get_imagedir(master))
        self.button2.grid(row=1, column=2) #, columnspan=10, sticky="new")
        #self.button2.pack(side="left")
        self.button3 = tk.Button(master,
                         text="Clear Current Display",
                         command=lambda: self.clear_image(master))
        #self.button3.pack(side="left")
        self.button3.grid(row=1, column=3) #, columnspan=10, sticky="new")
        self.button4 = tk.Button(master,
                         text="Append to Summary File",
                         command=self.write_to_csv)
        #self.button4.pack(side="left")
        self.button4.grid(row=1, column=4) #,columnspan=10, sticky="new")
        
        #self.ask_questions(master)

        self.carcass_removed = tk.StringVar()
        self.carcass_removed.set("0") # initialize
    
        lb1 = tk.Label(master, text="Was the carcass removed during the run? ", anchor=tk.W, justify=tk.RIGHT)
        lb1.grid(row=5, column=1)  #, anchor="e", columnspan=25, sticky="new")
        i = 0
        for text, yesno in YESNO:
            b1 = tk.Radiobutton(master, text=text, variable=self.carcass_removed, value=yesno, command=self.radioremoved, anchor=tk.E, justify=tk.LEFT)
            b1.grid(row=5, column=2+i)
            i=i+1
    
        use_radiobuttons = False
        if use_radiobuttons :
    
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
# end use_radiobuttons
        # Set up the matplotlib figure with the Artist canvas
        self.fig = Figure()
        #self.fig.tight_layout()
        self.fig.set_figheight(5)
        self.fig.set_figwidth(8)
        #ax1 = plt.subplot2grid((4,3), (0,0), colspan=3, rowspan=2)
        #ax2 = plt.subplot2grid((4,3), (2,0), colspan=3, sharex=ax1)
        #ax3 = plt.subplot2grid((4,3), (3,0), colspan=3, sharex=ax1)
        self.ax1 = self.fig.add_subplot(311)
        self.ax2 = self.fig.add_subplot(312)
        self.ax3 = self.fig.add_subplot(313)

        #self.ax.autoscale_view()
        #self.rect = self.ax.patch  # a Rectangle instance
        #self.rect.set_facecolor('green')

        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().grid(row=10, column=0, columnspan=20, rowspan=40, padx=5, sticky="nsew")
        self.canvas.show()
        
    def write_to_csv(self):
        #csvout=gl_imagedir + '\\' + carcass_coord_file
        use_radiobuttons = False
        if use_radiobuttons :

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

        print ("Summary rollup append to file: ", csvout)
        df = pd.DataFrame(columns=column_name_list)
        df = df.append({'Directory':gl_imagedir, 'Begin Date':begin_date, 'End Date':end_date, 'Camera':CameraNumber}, ignore_index=True)
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
        self.ax1.clear()
        self.canvas.get_tk_widget().destroy()
        self.ax1 = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().grid(row=10, column=0, columnspan=20, rowspan=40, padx=5, sticky="nsew")
        self.canvas.show()

        gl_x.clear()
        gl_y.clear()


        #self.canvas.grid(row=5, column=10)
        #self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        #self.toolbar = NavigationToolbar2TkAgg(self.canvas, master)
        #self.toolbar.update()
        #self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        #self.carcass_distance.set("0") # initialize
        #self.carcass_size.set("0") # initialize
        #self.obscuring_plants.set("0") # initialize



    def get_imagedir(self, master):
        global gl_imagedir
        global gl_savedir
        global gl_x
        global gl_y


        dirname = ""
        print("file dialog for directory: ", gl_savedir)

        dirname =  filedialog.askdirectory(initialdir=gl_savedir, 
                                               #parent = get_parent_window(self),
                                               title = "Select image directory")
                                              
        if not dirname :
            print("No directory selected")
            return

        gl_savedir=dirname
        print ("directory name: ", dirname) 
        gl_imagedir = dirname

        # display rolling window plot
        self.display_auto_classification()

        cid = self.canvas.mpl_connect('button_press_event', self.onclick)
        self.ax1.clear()
        self.ax1 = self.fig.add_subplot(111)
        self.ax1.set_title(dirname, fontsize=10)
        #implot = self.ax.imshow(img, picker=True)
        #self.ask_questions(master)
        #self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        #print(matplotlib.artist.getp(a))
        #self.canvas.show()


    def radioremoved(self) :
        print("in radioremoved, value: ", self.carcass_removed.get())
        
    #def radiosize(self) :
    #    print("in radiosize, value: ", self.carcass_size.get())
        
    #def radioplant(self) :
    #    print("in radioplants, value: ", self.obscuring_plants.get())
        
    def display_auto_classification(self) :
        global gl_imagedir
        
        # read in auto_classify CSV
        if not gl_imagedir.endswith('\\') :
            imagedir = gl_imagedir + '\\'

        csvfile = imagedir + csvauto
        print("read auto classification file: ", csvfile)
        df_data = pd.read_csv(csvfile, sep=',', header=0)
        #df_data = df_data.drop(columns=['SeqNumDiff', 'Mean', 'Std', 'TopCrop', 'BottomCrop', 'Carcass X', 'Carcass Y', 'Carcass Dist', 'Carcass Size', 'Obscuring Plants', 'NumObj', 'DistRank', 'Size', 'X', 'Y', 'Dist', 'Angle' ])
        df_time_label = pd.DataFrame(columns=['Datetime', 'Label', 'RollMean'])
        df_time_label.Datetime = df_data.Datetime.str.replace(':', '-', 2)   # replace : in date with -
        df_time_label.Label = df_data.Label
        
        #df_time_label=
        #print("Input labeled data...")
        #print(df_time_label.head())
    
        df_time_label["Datetime"] = pd.to_datetime(df_time_label["Datetime"]) # Convert column type to be datetime
    
        
        #indexed_df = df_time_label.set_index(["Datetime"])           # Create a datetime index
        #indexed_df.rolling(window_size)                   # Create rolling windows
        #df_time_label.rolling(window_size)                   # Create rolling windows
    
        df_time_label.RollMean = df_time_label['Label'].rolling(window_size, center=True).mean()             # Then apply functions to rolling window
        df_time_label.RollMean.fillna(value=0, axis=0, inplace=True)
    
        print("Auto classified with rolling mean...")
        #print(df_time_label.head())
        #print(df_time_label.count)
        #print("Auto classified dataframe: ", df_time_label.count())
        print("Datetime data type: ", df_time_label.dtypes)
    
        #fig=plt.figure()
        #fig.show()
        fig = self.fig
        
        ax = self.ax1
        ax=fig.add_subplot(111)
        #ax.set_xlabel("Date")
        ax.set_ylabel("Rolling Mean with Window Size "+str(window_size), fontsize='small')
        for tick in ax.xaxis.get_major_ticks():
            #tick.label.set_fontsize(14) 
            # specify integer or one of preset strings, e.g.
            tick.label.set_fontsize('x-small')            #('x-small') 
            #tick.label.set_rotation('vertical')


        ticklabs = ax.get_xticklabels()    #.get_yticklabels()
        #ax.set_xticks()
        #ax.set_xticklabels(ticklabs, fontsize=4) # changes tick labels to 0 to 1.0
        ax.plot(df_time_label.Datetime, df_time_label.RollMean, color='red') #, alpha=0.5)
        #ax.plot(df_time_label.Datetime, df_time_label.Label, color='purple', linestyle='None', marker='o', markerfacecolor='None', markersize=10)
        self.canvas.show()

        
    def display_manual_classification(self) :
        # check data\hwi_merge_table.csv to see if there's a manually label file for this image directory
        # read mergefile
        # read in manual classify CSV
    
        df_manual = pd.read_csv(csvmanual, sep=',', header=0)
  
        df_manual_label = pd.DataFrame(columns=['Datetime', 'Label', 'RollMean'])
        df_manual_label.Datetime = df_manual.Datetime.str.replace(':', '-', 2)   # replace : in date with -
        df_manual_label["Datetime"] = pd.to_datetime(df_manual_label["Datetime"]) # Convert column type to be datetime
        df_manual_label.Label = df_manual.Label
        # rolling().sum()
        df_manual_label.RollMean = df_manual_label['Label'].rolling(window_size).mean()             # Then apply functions to rolling window
        df_manual_label.RollMean.fillna(value=0, axis=0, inplace=True)
        print("Manually classified with rolling mean...")
        #print(df_manual_label.head())
        #print("Manual dataframe: ", df_manual_label.count())
        print("Datetime data type: ", df_manual_label.dtypes)

        df_result = pd.merge(df_time_label, df_manual_label, on=['Datetime'], how='left', suffixes=['','_man'])
        #df_result.fillna(value=0, axis=0, inplace=True)
        print("Merged dataframe: ", df_result.count())

        #print("After merge...")
        #print(df_result.head())
        #print(df_manual_label.count)
        fig = self.fig
        ax = self.ax

        ax.plot(df_result.Datetime, df_result.RollMean_man, color='blue', linestyle='dashed', alpha=0.5)
        ax.plot(df_result.Datetime, df_result.Label_man, color='black', linestyle='None', marker='x', alpha=0.5, markersize=15)

        self.canvas.show()


    def onclick(self, event):
        global gl_x
        global gl_y
        #gl_x = 0
        #gl_y = 0
        self.ax
        #print("in onclick")
        if event.xdata != None and event.ydata != None:
            gl_x.append(event.xdata)
            gl_y.append(event.ydata)
            print(gl_x[-1], gl_y[-1]) 
            self.ax.plot(gl_x[0:-1], gl_y[0:-1], 'bx', markersize = 10)
            self.ax.plot(gl_x[-1], gl_y[-1], 'r.', markersize = 10)
            self.canvas.show()
            
        return




root = tk.Tk()
app = SummaryRollupApp(root)
root.geometry("800x600")
root.mainloop()