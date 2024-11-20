#!/usr/bin/env python

import os

from glob import glob

from wx.tools.img2py import img2py

HEADER = """# Icon images for GenX GUI

from wx.lib.embeddedimage import PyEmbeddedImage

"""


def main():
    output_file = os.path.join("..", "genx", "genx", "gui", "images.py")
    with open(output_file, "w") as f:
        f.write(HEADER)
    for icon in glob("main_gui/*.png") + glob("reflectivity_plugin/*.png"):
        name = os.path.split(icon)[-1][:-4]
        img2py(icon, output_file, append=True, imgName=name, functionCompatible=True)


if __name__ == "__main__":
    main()
