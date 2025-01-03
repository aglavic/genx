.. _install:

************
Installation
************

Windows
=======

Download the windows installer GenX-3.X.X_win64_setup.exe from the
`home page <https://github.com/aglavic/genx/releases/latest>`_ and follow the instructions in the installation
guide.

Mac OS
======

Binary packages for Mac OS are `provided <https://github.com/aglavic/genx/releases/latest>`_ as
GenX-3.X.X_Installer.dmg images that can directly be installed. If you are having
trouble with this distribution you can try installing from source. (And create a trouble ticket, please.)

Install the required python 3 packages, especially wxPython. I would advice using a new Anaconda environment.
Afterwards you can install GenX from source. The anaconda environment packages that are known to work can be found in
`conda_build.yml <https://raw.githubusercontent.com/aglavic/genx/v3.6.14/genx/mac_build/conda_build.yml>`_

For the latest verion of wxPython the PyPI installation is possible, too. So system python3 with pip should be
sufficent to install all requirements.

Linux
=====

Install the requirements, at least wxPython, from your package manager (Ubuntu ``python3-wxgtk4.0``).
Then either install from source or, if you are using Ubuntu or a derivative, you can use the pre build .deb packages
for your system python version.

.. note::
    For compatibility with Ubuntu 24.04 the python3-numba package will no longer be installed automatically.
    I highly recommend installing it manually as it has significant impact on simulation performance.

As an example, installation in Ubuntu 24.04 could look like this:

.. code-block:: bash

    sudo apt update
    wget https://github.com/aglavic/genx/releases/download/v3.6.26/GenX-3.6.26_py312.deb
    sudo dpkg -i GenX-3.6.26_py312.deb
    sudo apt -f install
    sudo apt install python3-pip
    python3 -m pip install --break-system-packages numba pint orsopy svgwrite pymysql bumps

Snap
----

The most convenient way to install GenX on Linux is the `snap package <https://snapcraft.io/genx>`_.
It ships all requirements and should work on any distribution where the snap package management tool is installed.
(e.g. all Ubuntu derivatives have it pre-installed)
See https://snapcraft.io/docs/installing-snapd for instructions how to install snapd on your distribution.

To install via snap use:

.. code-block:: bash

    sudo snap install genx

The encapsulation of snap packages means, that they are more compatible over various Linux distros but sometimes
limit functionality. For the most part this could be circumvented in my tests. There is currently one know limitation
when using GenX installed through snap via X11-forwarding over SSH. In this case, it is possible to work around the
display accesss error by creating a manual link via:

.. code-block:: bash

    ln -s ~/.Xauthority ~/snap/genx/current/.Xauthority

.. _install_cluster:

Clusters
--------

GenX can make use of MPI to run models on cluster systems. In many cases the user does not have the rights
to install libraries and there are various configurations that can be configured and make installation
of own libraries pretty complicated.
On the other hand, fitting with GenX from command line does not require the wx or matplotlib libraries to be present.

In case the cluster does not provide a python installation that is new enough (>=3.6), you can try to
make use of the Miniconda distribution, all required software can be installed as a user without too much
background knowladge of Linux configurations.

Using system python
...................

* Create python virtual environment
    .. code-block:: bash

        python -m venv /path/to/new/virtual/environment
        source /path/to/new/virtual/environment/bin/activate # script depends on your used shell

* Install via pip with you local python, which should install all requirements automatically
    .. code-block:: bash

        python -m pip install genx3server


Using Minconda
..............

* Install Miniconda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
* Prepare anaconda environment and required packages
    .. code-block:: bash

        conda create -n genx python=3.9
        conda activate genx
        conda install pip platformdirs h5py scipy psutil numba
        pip install orsopy bumps

    * Depending on configuration you might need to install other libraries like glib if the installed
      libraries are too old.

    * I don't recommend to use the mpi version of anaconda but instead follow the instructions on how to install
      mpi4py for the local mpi library using pip:

      https://mpi4py.readthedocs.io/en/stable/install.html
* Finally install the server package for GenX:
    .. code-block:: bash

        pip install genx3server

* Tip: You can configure conda environments to update environment variables when they are activated.
  This can become handy if you need to selec specific library versions, PATH or LD_LIBRARY_PATH.
  ``conda env config vars set NAME=value``.





From source
===========

`Download <https://github.com/aglavic/genx/releases/latest>`_ the source distribution GenX-3.X.X.tar.gz
and unpack it. Run the file scripts/genx directly:

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

    conda create --name genx python=3.9 matplotlib platformdirs h5py scipy numba psutil pymysql
    conda activate genx
    conda install wxpython # you might need a different channel, e.g. conda-forge
    pip install genx3
    genx
    # if the command is not recognized you can try instead
    python -m genx.run

You can also try `download this <_attachments/conda.yml>`_ environment file with ``conda env create --file conda.yml``.

Requirements
------------

The needed dependencies are:

* Python >= 3.6 (recommend >= 3.8)
* wxPython version > 4.0  (recommend >= 4.1)
* Numpy version > 1.0
* Scipy version > 0.5
* Matplotlib version > 0.9
* platformdirs
* h5py
* orsopy >= 1.2.0

The non-mandotary packages are

* mpi4py (with an MPI installation)
* numba (calculation speedup by Just In Time compiler)
* vtk (graphical display of unit cells)
* svgwrite (for graphical image showing the layring - LayerGraphics plugin)
* pint (support in orsopy conversion of units)
* pymysql (access of crystallography open database for SLD - SimpleLayer plugin
* bumps (statistical analysis and alternative refinement method)

On a Linux system these packages can usually be installed through the package manager. On a windows and OSX systems the
anaconda distribution contains all packages.
