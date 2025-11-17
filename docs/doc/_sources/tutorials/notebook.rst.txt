.. _tutorial-notebook:

*******************************************
Running GenX Models in Jupyter Notebook/Lab
*******************************************

Introduction
============

Jupyter notebooks have become quite popular in experimental and data science as they allow
to work on data analysis and write code interactively while being able to reproduce the results
and display results nicely.

While the main strength of GenX is the interactive development of models using a graphical user
interface while keeping the felxibility of a modifyable model script, GenX also supports to
use the essential features via an API interface that includes special jupyter notebook support.

While it is possible to build models via this API it is mostly intended to run and anlyze already
existing model, e.g. to perform statistical analysis of the results.

Examples
========

The genx installation includes some example notebooks that can be used as a reference.
You can find them in:

::

    {genx installation}/genx/examples/*.ipynb

Quick Start
===========

To use the GenX API, the package needs to be found by your python environment use in the notebook.
I you installed GenX via pip in the same environment, that should already be the case.

The following shows you how to load a GenX file, inspect the model and run a fit:

.. code-block:: python

    %pytlab inline # %matplotlib inline
    from genx import api

    model,optimizer=api.load('D17_TOF_SiO.hgx')
    # save the model script to a file for later inspection
    with open('genx_model.py', 'w') as fh:
        fh.write(model.script)

    ########## new cell ############
    model.parameters
    # ==> output a list of fit parameters from the "grid" tab of the GUI
    ########## new cell ############
    optimizer
    # ==> output the configuration options for the fit, can be modified as attributes of this object
    ########## new cell ############
    api.fit_notebook(model, optimizer)
    # ==> performs the fit and shows live update of the progress
    ########## new cell ############
    api.fit_update(model, optimizer)
    model
    # ==> update model with fit results and show the overview widget for the model
    ########## new cell ############
    # potentially save the fit result to a new file
    #api.save(r'testoutput.hgx', model, optimizer)

Widgets
=======

There are some experimental notebook widgets implemented, that allow interactive modification
of GenX models similar to the GUI. This can be useful for testing or if only partial code based
analysis is desired.
The widgets can be accessed as attributes of some objects:


