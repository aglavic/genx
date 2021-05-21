.. _development-data:

************
Data classes
************
This page will describe how the data is stored in the classes that handles the data. It will not be a complete manual.
It will only deal with basic functionality which expert users might find handy to know.
For more information the reader is refereed to data.py in the source tree.

For storing the data two classes are implemented: The ``DataSet`` class which contains one data
set and the ``DataList`` class which contains several ``DataSet``\ s.

DataList
========
The major use of the ``DataList`` is to make ordinary list methods available and at the same time making it possible
to apply commands to entire data set or subset when working with the data from the GUI. Some of the list
functionality include:

* Extracting one element by list-like access
  ::

        dataset = datalist[2]

* Using slicing to extract a subset of the ``datalist`` such as::


        subdatalist = datalist[0:5]

* Iterations over the `DataSet`s in the `DataList` such as::

        for dataset in datalist:
            ....

  * Using the `len` function such as `len(datalist)`

The rest of the functions for the `DataList` is mainly of interest for GUI programmers and should not
be necessary to dwell upon here, if you need to know use the source.

DataSet
=======
The ``DataSet`` class contains all the information about a certain data set.

General data handling
---------------------
When data has been loaded into the data class it is loaded into the member variables
``x_raw``, ``y_raw``, ``error_raw``, ``extra_data_raw`` then the commands as defined
in ``x_command``, ``y_command``, ``error_command``, ``extra_data_command`` is evaluated and stored in
the variable names ``x``, ``y``, ``e`` and the keys in the ``extra_data`` identical to the raw data
variables. This makes the evaluations of the data calculations/transformations *independent* of previous
evaluations. The commands are always on the raw data. The result of these evaluations is then set to the
member variables ``x``, ``y``, ``error``, ``extra_data``.

Important members
-----------------
The members of the class that could be of interest are:

``x``
     The x-values after the commands has been executed on the raw data.
``y``
     The y-values after the commands has been executed on the raw data.
``error``
     The error on the y-values after the commands has been executed on the raw data.
``sim``
     The simulation of the data as calculated from the `Sim` function in the model.
``extra_data``
     A `dictionary <http://docs.python.org/tutorial/datastructures.html#dictionaries>`_ of the extra data as defined b
     y the data loader plugin. This is also after the commands has been executed on the extra data if it is defined
     to be accessible from the data loader plugin.
``show``
     A flag, boolean, that defines if the ``DataSet`` should be visible.
``use``
     A flag, boolean, that defines if the ``DataSet`` should be used the FOM calculation.
``use_error``
     A flag, boolean, that defines if the ``DataSet`` has errorbars that should be used.
``name``
     The name of the ``DataSet``. This is a string which can be non-unique.

There are also a number of member variables that defines plotting::

    data_color = (0.0, 0.0, 1.0)
    sim_color = (1.0, 0.0, 0.0)
    data_symbol = 'o'
    data_symbolsize = 4
    data_linetype = '-'
    data_linethickness = 2
    sim_symbol = ''
    sim_symbolsize = 1
    sim_linetype = '-'
    sim_linethickness = 2


And also the member variables that contains the raw data values as loaded from file and the commands applied to them.
::

    x_raw = array([])
    y_raw = array([])
    error_raw = array([])
    extra_data_raw = {}

    extra_commands = {}
    x_command = 'x'
    y_command = 'y'
    error_command = 'e'


Extra data
----------
In order to operate on loaded extra_data as ordinary data it has to be added as an item to the
dictionary extra_command. In doing so it will also be subjected to the same rigorous constraint as the
``x``, ``y`` and ``error`` values. It has to an array of the same length as all the other and it has
to possible to use it the commands for the data. Otherwise the data will just be present in
the extra_data array to use. This can be handy for external conditions that is stored in the data file,
for example magnetic field, temperature or pressure.

To create a new ``extra_data`` instance use the method ``set_extra_data(self, name, value, command = None)`` for
example to add a temperature variable ``T`` to the ``DataSet`` ``dataset``::

    dataset.set_extra_data('T', 10)


If you on the other hand want to make it as an additional independent variable
::

    dataset.set_extra_data('T', array([0, 1, 2, 3]), command = 'T')


To get extra data you can use the method ``get_extra_data(self, name)``, for example::

    dataset.get_extra_data('T')


In the future there might be an implementation so that the extra dat can be directly accessed as `dataset.T` but that
will not be implemented right now.
