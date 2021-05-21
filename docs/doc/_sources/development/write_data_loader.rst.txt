.. _development-write-data-loader:

**************************
How to write a Data Loader
**************************
The data loader are one of the two different plug in types that is used to customize GenX for different jobs.
The easiest plug in to write is probably a data loader since it requires very little coding. Most of the work is
usually to understand the data format one wants to use. The rest is easy. In principal it
usually consist of three steps:

1. Create a class according to the `plugins.data_loader_framework.Template`
2. Write the data loading code.
3. Create a data settings dialog box to get user input.

Finally the python file is added to [genx-path]/plugins/data_loaders and now it should appear in GenX
(you might need to restart the program). Below there is a more detailed description of the process.

The template
============
The implementation of the template class can be found in ``data_loader_framework.py``. The following
is a brief description of the methods in the ``Template`` class:

``__init__(self, parent)``
     The init function for the class should be overridden. Remember to user the ``Register`` function
     to tell the parent about the existence of the plug in.

``Register(self)``
     Register the function with the parent frame, i.e. the main frame of the program so it is possible
     to call it from in the general gui callbacks.

``SetData(data)``
     Sets the data structure, ``self.data`` of the plug in, used by external classes.

``UpdateDataList(self)``
     Forces the data list to update, which updates the gui with new data sets in the data list view.
     This is only necessary if new data sets have been added when the data has been loaded.

``SetStatusText(self, text)``
     Sets the status text in the main window. Should be used as output to show the user what is
     going on. Also for error messages to remind the user what has happened.

``LoadDataFile(self, selected_items)``
     Selected items is the selected_items in the current ``DataList`` into which data from file(s) should be loaded.
     The default function then shows a file dialog and then calls the `LoadData` data function with this file. Note
     that the default implementation only allows the loading of a single file! Overriding this function in subclasses
     can of course change this behavior. This function calls the `LoadData` function which implements the io function
     by it self. The ``LoadData`` has to be overloaded in order to have a working plugin.

``LoadData(self, data_item, file_path)``
     This function has to overridden as default it does nothing.

``SettingsDialog(self)``
     Override this function to implement a settings dialog so that the current import settings can be changed.
     Preferably it should be a dialog which is totally controlled from this function.

``SendUpdateDataEvent(self)``
     Sends an update event to the gui that new that data has been loaded and plots and such should be updated.

``Remove(self)``
     Removes the link between the plugin and its parent. Should be left as it is. Called by external classes.

So this is basically all you need to write your own data loader. In module ``plugins.utils``, there are some
utility functions that will display dialogs:

``ShowErrorDialog(frame, message)``
     Shows an error dialog using frame as parent with message message, a string.

``ShowWarningDialog(frame, message)``
     Same as above but an Warning dialog box

``ShowInfoDialog(frame, message)``
     Same as above but with just information.

As ``frame`` a class deriving from Template can use ``self.parent``.

The default as example
======================
Here we will display the current default data loader as an example, as of 2009-04-25, for
the mose current version look at ``genx/plugins/data_loaders/default.py``.
::

    import numpy as np
    import wx
    from wx.lib.masked import NumCtrl

    from plugins.data_loader_framework import Template
    from plugins.utils import ShowErrorDialog, ShowWarningDialog, ShowInfoDialog

    class Plugin(Template):
        def __init__(self, parent):
            Template.__init__(self, parent)
            self.x_col = 0
            self.y_col = 1
            self.e_col = 1
            self.comment = '#'
            self.skip_rows = 0
            self.delimiter = None

        def LoadData(self, data_item_number, filename):
            '''LoadData(self, data_item_number, filename) --> none

            Loads the data from filename into the data_item_number.
            '''
            try:
                load_array = np.loadtxt(filename, delimiter = self.delimiter,
                    comments = self.comment, skiprows = self.skip_rows)
            except Exception, e:
                ShowWarningDialog(self.parent, 'Could not load the file: ' +\
                        filename + ' \nPlease check the format.\n\n numpy.loadtxt'\
                        + ' gave the following error:\n'  +  str(e))
            else:
                # Check so we have enough columns
                if load_array.shape[1]-1 < max(self.x_col, self.y_col, self.e_col):
                    ShowWarningDialog(self.parent, 'The data file does not contain'\
                            + 'enough number of columns. It has ' + str(load_array[1])\
                            + ' columns. Rember that the column index start at zero!')
                    # Okay now we have showed a dialog lets bail out ...
                    return
                # The data is set by the default Template.__init__ function, neat hu
                # Know the loaded data goes into *_raw so that they are not
                # changed by the transforms
                self.data[data_item_number].x_raw = load_array[:, self.x_col]
                self.data[data_item_number].y_raw = load_array[:, self.y_col]
                self.data[data_item_number].error_raw = load_array[:, self.e_col]
                # Run the commands on the data - this also sets the x,y, error memebers
                # of that data item.
                self.data[data_item_number].run_command()

                # Send an update that new data has been loaded
                self.SendUpdateDataEvent()

        def SettingsDialog(self):
            '''SettingsDialog(self) --> None

            This function should - if necessary implement a dialog box
            that allows the user set import settings for example.
            '''
            col_values = {'y': self.y_col,'x': self.x_col,'y error': self.e_col}
            misc_values = {'Comment': str(self.comment), 'Skip rows': self.skip_rows,\
                    'Delimiter': str(self.delimiter)}
            dlg = SettingsDialog(self.parent, col_values, misc_values)
            if dlg.ShowModal() == wx.ID_OK:
                col_values = dlg.GetColumnValues()
                misc_values = dlg.GetMiscValues()
                self.y_col = col_values['y']
                self.x_col = col_values['x']
                self.e_col = col_values['y error']
                self.comment = misc_values['Comment']
                self.skip_rows = misc_values['Skip rows']
                self.delimiter = misc_values['Delimiter']
            dlg.Destroy()


As can be seen the creation process is quite easy. First we import the necessary packages from the plugin package.
Then we subclass the ``Template`` class to create a ``Plugin`` class. Note that the name here is important the class
has to be names ``Plugin``. The ``__init__`` function should be straight forward, note that the parent class's
``__init__`` function is first called to bind the parent and doing the default setup. Next some default values is
set for data import.

The ``LoadData`` method is also easy. In order to understand it fully the reader should have a look at the Data
class ``genx/data.py`` and the doc page that discusses it, :ref:`development-data`.
The functions only loads the data as an 2D array and cuts out the right columns and do some simple
error handling in order to catch errors and notice the user about them.

The `SettingDialog` is also simple, however, one needs to know a bit about wxPython programming with dialogs.
If you are new to wxPython you might want to look at the `wxPython tutorial <http://wiki.wxpython.org/AnotherTutorial>`_
or at the excellent demos/examples that are part of the
`wxPython distribution <http://downloads.sourceforge.net/wxpython/wxPython-demo-2.8.9.2.tar.bz2>`_ if they are
not part of your installation.

In addition it also possible to load extra data into the data sets by using the ``DataSet`` methods
``DataSet.set_extra_data(name, value, command = None)``. For more information about this see
:ref:`development-data` and the implementation in `genx/plugins/data_loaders/sls_sxrd.py` where this is used for
the `h` and `k` coordinates of the crystal truncation rods.

I hope this information makes it possible for you to get started with writing your own data loaders.
If you find your implementation useful make sure that they are included in the GenX distribution!
