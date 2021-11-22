.. _tutorial-plugin:

************************
Plugins: an introduction
************************

If you start reading the tutorials on this website you will hear a lot about plugins.
This page will briefly explain what plugins mean in GenX and why they will help you.

What they are
=============
A plugin is an extension of GenX which can dynamically (that is while the program is running)
be loaded and unloaded. A plugin framework exist in most programs in order to make them extensible.
It also helps the developers to have reusable code. GenX has been designed so that it should work for
many problems, while the problem specific interfaces are available thorugh such plugins.
For GenX there are in principal three different plugins.

  1. **Models:** This is the model that simulates a certain physical effect, for example x-ray reflectivity.
     They are not listed directly in the GUI since they are loaded by the "model script" in GenX.
     While it is in priciple possible to write a model fully inside one script, it is recommanded to
     build on existing model files to separate the simulation script from the model.

  2. **Data loaders:** The first "real" plugin. This plugin allows for different sources
     to load data. See below.

  3. **Plugins:** This is what most people think about when talking about plugins. These plugins will mostly
     extend the graphical user interface with new functionality. See below.

How to use them
===============
First of all, there is some basic help built into GenX already. Go to the menu :menuselection:`Help-->Plugins Help`
and you will see help pages for all plugins. This is mostly intended as a quick reference with
the important information ready at hand.

Data Loader
-----------
The Data loaders are a small but powerful plugin feature. It allows user to load data from different file
formats. Now , as of 2021-11-22, 11 different file formats are implemented as well as an *auto* loader
that selects a suitable file format from a list of supported extensions. There are two generic loaders
*default* and *resolution* that load ASCII data of x/y/error or x/y/error/resolution from user selectable
columns. These columns can be specified in the :menuselection:`Settings-->Import` menu.

If one would like to change between different data loaders go to the menu :menuselection:`Settings-->Data loader`.
A dialog box appears that prompts the user to choose a data loader.
It should be noted that the Data loader settings dialog can change after a different data loader has been selected.

.. note::
    As advanced comment: Some data loaders might insert extra data columns into the data structure as well
    as metadata from the file header like sample and instrument information (form ORSO files).
    The extra columns can accessed in the ``Sim`` function by::

        data.extra_data["string"]


    where string should be replaced by an identifier (see the documentation for that data loader in
    :menuselection:`Help-->Data loaders Help`) for the sls_sxrd data loader there will be additional
    information for the `h` and `k` positions. `data.extra_data` is a
    `dictionary <http://docs.python.org/tutorial/datastructures.html#dictionaries>`_
    These would be accessed by::

        data.extra_data["h"]


If you would like to know about how to write your own data loaders gor to :ref:`development-write-data-loader`.

Plugins
-------
As said above these plugins extend the graphical user interface to implement new functionality.
To load a plugin use the menu :menuselection:`Settings-->Plugins-->Load` and choose the plugin you want.
To remove them from GenX use the menu :menuselection:`Settings-->Plugins-->Unload` - and choose the
plugin you want to remove.

SimpleReflectivity
^^^^^^^^^^^^^^^^^^
This plugin is a simplified model builder for reflectivity measurements. It is based on *Reflectviity*
but has a much simpler interface and *hides* the more complex aspects of the GenX interface from
the user.
See the tutorials: :ref:`tutorial-simple-reflectivity`.

Reflectivity
^^^^^^^^^^^^
The reflectivity plugin was the first plugin for defining a complete fitting model.
It does so by providing controls to define a sample structure. See the tutorials: :ref:`tutorial-xrr-fitting` and
:ref:`tutorial-neutron-sim`.

SimpleLayer
^^^^^^^^^^^
A small interface to quickly define SLD layer parameters from structure or density and chemical composition.
It stores the defined materials for quicker use in future models. In the add material dialog there
are some options to query only databases for a given chemical formula.
If materials are definde, the SimpleReflectivity plugin will automatically select their density
for newly provided chemical formulas.

Exporter
^^^^^^^^
This is a tool to convert reflectometry models into formats for different software. Right now
only BornAgain is supported.

MagSLD
^^^^^^
Modifies the SLD plot from the reflectivity plugins to display magnetization units and layer
integrated magnetic moment.

ParameterVault
~~~~~~~~~~~~~~
This plugin can be used to store model parameter value sets from the grid to compare with
modified versions.

SXRD
~~~~
Similar model builder as in the Reflectivity plugin but for surface x-ray diffraction models.

SpinAsymmetry
~~~~~~~~~~~~~
Adds an extra graph that displays the neutron spin-asymmetry of magnetic models calculated
for the data points and model.

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
^^^^
This is just a test plugin which tests all the standard functionality provided in the plugin framework.
It will open a new folder in each notebook in the window and in addition it creates a new menu item. No practical use.

Models
------
Have a look at the other :ref:`tutorials`. Most of the information presented here is about the different models.
