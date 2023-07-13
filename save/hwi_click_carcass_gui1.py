# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 14:41:19 2018

@author: christman
"""

#from __future__ import division, print_function, absolute_import
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


style.use("ggplot")

f = Figure()
#a = f.add_subplot(111)
save_axis = f.add_subplot(111)
a = save_axis.findobj

#a = f.add_subplot(111)

deer_coord_file="hwi_carcass_coordinates.csv"
column_name_list = ['File', 'Directory', 'Datetime', 'Camera', 'Carcass X', 'Carcass Y', 'Carcass Dist', 'Carcass Size', 'Obscuring Plants']

gl_x = [0]
gl_y = [0]
imagedir=""
imagefile=""
savefile="D:\\"



def popupmsg(msg):
    popup = tk.Tk()
    popup.wm_title("!")
    label = ttk.Label(popup, text=msg, font=NORM_FONT)
    label.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text="Okay", command = popup.destroy)
    B1.pack()
    popup.mainloop()
    


class takeInput(object):

    def __init__(self,requestMessage):
        self.root = tk.Tk()
        self.string = ''
        self.frame = tk.Frame(self.root)
        self.frame.pack()        
        self.acceptInput(requestMessage)

    def acceptInput(self,requestMessage):
        r = self.frame

        k = tk.Label(r,text=requestMessage)
        k.pack(side='left')
        self.e = tk.Entry(r,text='Name')
        self.e.pack(side='left')
        self.e.focus_set()
        b = tk.Button(r,text='okay',command=self.gettext)
        b.pack(side='right')

    def gettext(self):
        self.string = self.e.get()
        self.root.destroy()    
 
    def getString(self):
        return self.string

    def waitForInput(self):
        self.root.mainloop()

def getText(requestMessage):
    msgBox = takeInput(requestMessage)
    #loop until the user makes a decision and the window is destroyed
    msgBox.waitForInput()
    return msgBox.getString()



         

class ClickCarcass(tk.Tk):

    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.iconbitmap(self, default="Iconshock-Stroke-Animals-Hawk.ico")
        tk.Tk.wm_title(self, "HWI Image Classification - Carcass Coordinate Entry")
        
        
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        #self.frames = {}
        #for F in (StartPage, Show_Image_Page):
        #    frame = F(container, self)
        #    self.frames[F] = frame
        #    frame.grid(row=0, column=0, sticky="nsew")
        
        #self.frames = {}
        #self.show_frame(StartPage)

           # Try having only one page
       
 
    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

        
#class StartPage(tk.Frame):
class StartPage(tk.Tk):

    def __init__(self, *args, **kwargs):

# from click_carcass
    #def __init__(self, parent, controller):

        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.iconbitmap(self, default="Iconshock-Stroke-Animals-Hawk.ico")
        tk.Tk.wm_title(self, "HWI Image Classification - Carcass Coordinate Entry")
        
        
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        Frame.grid(row=0, column=0, sticky="nsew")
### end from click_carcass

        #tk.Frame.__init__(self,parent)

        label = tk.Label(self, text=("HWI Click Carcass "), font=LARGE_FONT)
        label.pack(pady=2,padx=2)

        button1 = ttk.Button(self, text="Get Image to Click Carcass",
                            command=lambda: self.get_imagedir())
                            #command=lambda: controller.show_frame(Show_Image_Page))
        button1.pack()
        button2 = ttk.Button(self, text="Write Carcass Info to File",
                            command=lambda: self.save_carcass_info())
                            #command=lambda: controller.show_frame(Show_Image_Page))
        button2.pack()
        button3 = ttk.Button(self, text="Quit",
                            command=quit)
        button3.pack()
  
        
    def get_imagedir(self):
        global imagedir
        global imagefile
        global savefile
        global gl_x
        global gl_y


        filename = ""
        print("file dialog for directory: ", savefile)

        filename =  filedialog.askopenfilename(initialdir=os.path.basename(savefile), 
                                               initialfile=os.path.basename(savefile),
                                               #parent = get_parent_window(self),
                                               title = "Select file",
                                               filetypes = (("jpeg files","*.jpg"), ("all files", "*.*")))
        if not filename :
            return

        savefile=filename
        print ("file name: ", filename) 
        imagedir, imagefile = os.path.split(filename)

        print('in get_imagedir, global imagedir: ', imagedir, "  global imagefile: ", imagefile)
        img=np.array(Image.open(filename))    # Read in gray scale image
        img_shape = img.shape
        print("image shape: ", img_shape)
        gl_x[:] = []
        gl_y[:] = []
        #ax_list = f.axes
        #print("axes list: ", f.axes)
        #f = Figure()
        #a = f.add_subplot(111)
        #ax = f.subplots(111)
        #fig = plt.gcf()


        #fig = plt.subplots(1,1, figsize=(14, 14))   # preferred alternative to gcf()
        #ax.set_title('Click on the carcass, then close image:', fontsize=25)
        #ax = plt.gca()
        #ax.set_title('Click on the carcass, then close image:', fontsize=25)
# attempts to get second selected image to display over first image

        a = save_axis
        a.clear()

        canvas = FigureCanvasTkAgg(f, self)
        #canvas.get_tk_widget().delete("all") 
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        cid = canvas.mpl_connect('button_press_event', self.onclick)
        implot = a.imshow(img, picker=True)
        #print(matplotlib.artist.getp(a))
        canvas.show()
        #f.show()


    def onclick(self, event):
        global gl_x
        global gl_y
        #gl_x = 0
        #gl_y = 0
        a = save_axis
        #print("in onclick")
        if event.xdata != None and event.ydata != None:
            gl_x.append(event.xdata)
            gl_y.append(event.ydata)
            print(gl_x[-1], gl_y[-1]) 
            a.plot(gl_x[0:-1], gl_y[0:-1], 'bx', markersize = 10)
            a.plot(gl_x[-1], gl_y[-1], 'r.', markersize = 10)
            #a.show
            
        return

class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Page One!!!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()




class Show_Image_Page(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Image Page", font=LARGE_FONT)
        label.pack()  #pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Enter Directory",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()


        canvas = FigureCanvasTkAgg(f, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


#app = ClickCarcass()
app = StartPage()

#app.geometry("1280x720")
app.geometry("700x800")

#ani = animation.FuncAnimation(f, animate, interval=5000)
app.mainloop()

