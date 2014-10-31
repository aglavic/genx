.. _install:

************
Installation
************

Windows
=======

Download the windows installer GenX-2.X.X-win.exe from the home page and follow the instructions in the installation
guide.

Mac OS-X
========

Download the disk image GenX-2.X.X-osx.dmg, open it and drag the GenX app to the application folder.

From source
===========

Download the source distribution GenX-2.X.X-source.zip and unzip it. Run the file genx.py directly with the
command ``python genx.py``.

Requirements
------------

The needed dependencies are:

* Python newer than 2.3.5
* wxPython version > 3.0
* Numpy version > 1.0
* Scipy version > 0.5
* Matplotlib version > 0.9
* appdirs version > 1.2
* h5py

The non-mandotary packages are

* mpi4py (with an MPI installation)

On a Linux system these packages can usually be installed through the package manager. On a windows system the
python (x,y) distribution contains all packages except appdirs which can be installed from the python package index.
On OS-X all the package has to be installed separately.
