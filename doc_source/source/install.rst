.. _install:

************
Installation
************

Windows
=======

Download the windows installer GenX-3.X.X_win64_setup.exe from the home page and follow the instructions in the installation
guide.

Mac OS-X
========

Install the required python 3 packages, especially wxPython. I would advice using a new Anaconda environment.
Afterwards you can install GenX from source.

Linux
=====

Install the requirements, at least wxPython, from your package manager (Ubuntu ``python3-wxgtk4.0``).
Then either install from source or, if you are using Ubuntu or a derivative, you can use the pre build .deb packages
for your system python version.

Clusters
--------

GenX can make use of MPI to run models on cluster systems. In many cases the user does not have the rights
to install libraries and there are various configuration that can be configured and make installation
of own libraries pretty complicated.
On the other hand, fitting with GenX from command line does not require the wx or matplotlib libraries to be present.

Making use of the Miniconda distribution, all required software can be installed as a user without too much
background knowladge of Linux configurations.

    * Install Miniconda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
    * Create genx environment and install packages:
        * ``conda create -n genx python=3.9``
        * ``conda activate genx``
        * ``conda install pip appdirs h5py scipy psutil numba``
        * Depending on configuration you might need to install other libraries like glib if the installed
          libraries are too old.
        * ``pip install orsopy bumps``
        * I don't recommend to use the mpi version of anaconda but instead follow the instructions on how to install
          mpi4py for the local mpi library using pip: https://mpi4py.readthedocs.io/en/stable/install.html
    * Tip: You can configure conda environments to update environment variables when they are activated.
      This can become handy if you need to selec specific library versions, PATH or LD_LIBRARY_PATH.
      ``conda env config vars set NAME=value``.


From source
===========

Download the source distribution GenX-3.X.X.tar.gz and unpack it. Run the file scripts/genx directly:

.. code-block:: bash

    tar -xvzf GenX-3.X.X.tar.gz
    cd GenX-3.X.X
    python3 scripts/genx

You can also install it in your python 3 environment as user ``pip3 install --user genx3`` or
system wide ``sudo pip3 install genx3`` and run:

.. code-block:: bash

    pip3 install --user genx3
    genx

Anaconda
--------

You can create a suitable anaconda environment using the following commands, i:

.. code-block:: bash

    conda create --name genx python=3.9 matplotlib appdirs h5py scipy numba psutil pymysql
    conda activate genx
    conda install wxpython # you might need a different channel, e.g. conda-forge
    pip install genx3
    genx
    # if the command is not recognized you can try instead
    python -m genx.run

You can also try :download:`this <_attachments/conda.yml>` environment file with ``conda env create --file conda.yml``.

Requirements
------------

The needed dependencies are:

* Python >= 3.6
* wxPython version > 4.0
* Numpy version > 1.0
* Scipy version > 0.5
* Matplotlib version > 0.9
* appdirs version > 1.2
* h5py

The non-mandotary packages are

* mpi4py (with an MPI installation)
* numba (calculation speedup by Just In Time compiler)
* vtk (graphical display of unit cells)

On a Linux system these packages can usually be installed through the package manager. On a windows and OSX systems the
anaconda distribution contains all packages.
