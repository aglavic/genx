'''
A utillity that exports all png files in the scripts directory
to a python file using img2py
'''

import wx.tools.img2py as impy
import os

files = [file for file in os.listdir('.') if file[-4:] == '.png']

appending = False

for file in files:
    impy.img2py(file, 'images.py', append = appending, imgName = file[:-4],
                functionCompatibile=True, functionCompatible=True)
    appending = True
