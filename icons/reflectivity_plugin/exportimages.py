''' 
A utillity that exports all png files in the scripts directory 
to a python file using img2py
'''

import wx.tools.img2py as impy
import os

pat = '/usr/lib/python2.6/dist-packages/wx-2.6-gtk2-unicode/wx/tools/'

files = [file for file in os.listdir('.') if file[-4:] == '.png']

appending = False

for file in files:
    if appending:
        os.system('python %simg2py.py -n %s -a %s images.py'%(pat, file[:-4], file))
    else:
        os.system('python %simg2py.py -n %s %s images.py'%(pat, file[:-4], file))
    #impy.img2py(file, 'images.py', append = appending, imgName = file[:-4])
    appending = True

