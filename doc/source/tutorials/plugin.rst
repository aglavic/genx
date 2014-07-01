.. _tutorial-plugin:

************************
Plugins: an introduction
************************

If you start reading these tutorials that are housed here in this wiki you will hear a lot about plugins.
This page will briefly explain what plugins means in GenX and why they will help you.

What they are
=============
A plugin is an extension of GenX which can dynamically (that is when the program is running)
be loaded and unloaded. A plugin framework exist in most programs in order to make them extensible.
It also helps the developers to have reusable code. GenX have been designed so that it should work for
many problems. For GenX there are in principal three different plugins.

  1. **Models:** This is the models that simulate a certain physical effect, for example x-ray reflectivity.
     The boundary conditions is basically non-existing since the user will create the "model script" in
     GenX which glues the model to GenX.

  2. **Data loaders:** The first "real" plugin. This plugin allows for different ways
     to load data. See below.

  3. **Plugins:** This is what most people think about when talking about plugins. These plugins will mostly
     extend the graphical user interface with new functionality. See below.

How to use them
===============
First of all, there are some basic help built in in GenX already. Go to the menu Misc
and you will see help alternatives for both the models, data loaders and plugins.
This is mostly intended as a quick reference with most the information ready at hand.

Data Loader
-----------
The Data loaders are a small but powerful plugin feature. It allows user to load data from different file
formats. Now , as of 2009-04-25, only two different file formats are implemented. One for column
ASCII files and one for surface x-ray diffraction data. If one would like to change between different data
loaders go to the menu :menuselection:`Settings-->Data loader`. A dialog box appears that prompts the user
to choose a data loader. It should be noted that also the Data loading settings dialog can change when a different
data loader has been loaded.

As a more advanced comment: Some data loaders might insert extra data into the data structure.
This can accessed in the ``Sim`` function by
::

    data.extra_data["string"]


where string should be replaced to an identifier (see the documentation for that data loader in
:menuselection:`Misc-->Data loaders help`) for the sls_sxrd data loader there will be additional
information for the `h` and `k` positions. `data.extra_data` is an
`dictionary <http://docs.python.org/tutorial/datastructures.html#dictionaries>`_
These would be accessed by::

    data.extra_data["h"]


If you would like to know about how to write your own data loaders gor to :ref:`development-write-data-loader`.

Plugins
-------
As said above these plugins extends the graphical user interface to implement new functionality.
To load a plugin use the menu :menuselection:`Settings-->Plugins-->Load` and choose the plugin you want.
To remove them from them from GenX use the menu :menuselection:`Settings-->Plugins-->Unload` - and choose the
plugin you want to remove.

Reflectivity
^^^^^^^^^^^^
The reflectivity plugin is first plugin for defining a complete fitting model. ^
It does so by providing controls to define a sample structure. See the tutorials: :ref:`tutorial-xrr-fitting` and
:ref:`tutorial-neutron-sim`.

Shell
^^^^^
This will open a new folder in the lower input notebook. This allows for introspection of the
different parts of GenX. It should be used only by expert users for debugging. It can also be used to
debug script if you are proficient enough in python. See the help for more information.

UserFuncs
^^^^^^^^^
This little model creates a new menu item and searches the model for functions that takes no
input arguments and makes it possible to call them from that menu. This can be used to create output ^
from a physical model. For example export parameters or data. It is mainly for expert use, but very handy to have.
See the help in :menuselection:`Misc-->Plugin` help.

Test
^^^
This is just a test plugin which tests all the standard functionality provided in the plugin framework.
It will open a new folder in each notebook in the window and in addition it creates a new menu item. No practical use.

Models
------
Have a look at the other :ref:`tutorials`. Most of the information presented here is about the different models.
