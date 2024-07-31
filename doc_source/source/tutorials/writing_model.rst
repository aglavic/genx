.. highlight:: default
    :linenothreshold: 10

.. _tutorial-writing-model:

**********************
Writing a custom model
**********************

Writing a model is quite easy in GenX if you are familiar with using the Python language.
There is a couple of things one needs to keep in mind in order to successfully build such a model,
which will be covered in this tutorial.

The only mandatory thing the model file has to contain is a function called
``Sim`` taking a member of the class Data as input parameter and return a list of intensity arrays
with the same length as the number of datasets and their data points.
However, to make the model useful, functions for setting the model parameter values have to be incorporated, too.

.. note::
    GenX uses function to set the parameters during fitting, this is why we need to provide them.
    Most models that are provided by GenX include automatic generation of the relevant functions for each parameter
    which are called *setParameter* for a model attribute called *parameter*.

Programming Python
==================
Since writing a model actually involves writing a script in Python it is good to have some basic knowledge of the syntax.
However, if you have some basic knowledge about programming it should be fairly easy to just look at the examples
and write your own models without having to learn Python specifically.
On the other hand, there exists a number of free introductory books as well as tutorials on the
internet for the interested reader, see below.

* `Python’s homepage <http://www.python.org>`_ contain most of the available tutorials online.
* `A Byte of Python <https://python.swaroopch.com>`_ is an introductory text for the absolute beginner.
* `How to Think Like a Computer Scientist: Learning with Python <http://openbookproject.net/thinkcs/python/english3e/>`_
  is a textbook for the beginner written for computer science students.
* `Dive Into Python <https://diveintopython3.net/>`_ is an introduction to Python book for the more experienced programmer.

In addition there are a number of tutorials on `SciPy's homepage <https://scipy.org>`_ which deal
with numerical computations. There is also a
`migration guide for those who are familiar with MatLab <https://www.enthought.com/wp-content/uploads/Enthought-MATLAB-to-Python-White-Paper.pdf>`_.

The Data class
==============
In order to write the Sim class it is necessary to know the structure of the class ``Data`` which is taken as a
parameter. The variables which could be useful in the ``Sim`` function are:

* ``x`` A list of 1-D arrays (vectors) containing the x-values of the processed data
* ``y`` A list of 1-D arrays (vectors) containing the y-values of the processed data
* ``e`` A list of 1-D arrays (vectors) containing the error-values of the processed data
* ``extra_data["key"]`` Same shape arrays for each of the extra data provided from a data reader (like "res" for resolution)
* ``x_raw`` A list of 1-D arrays (vectors) containing the raw x-values (the data loaded from the data file)
* ``y_raw`` A list of 1-D arrays (vectors) containing the raw y-values (the data loaded from the data file)
* ``e_raw`` A list of 1-D arrays (vectors) containing the raw error-values (the data loaded from the data file)
* ``use`` A list of booleans (True or False) denoting if the data should be fitted

Simple example
==============
Knowing what the Data class contains we will start with a simple example, making a model that fits one
Gaussian to the first data set. The free parameters of the Gaussian are; the center of the peak, Xc, the peak width,
W, and the amplitude of the peak, A. Writing a model for it would produce a code as shown below. Note that a
# produce a comment.::

    # import the numeric package and the UserVars class from GenX for parameters
    from numpy import *
    from models.utils import UserVars
    # Create a class for user variables
    MyVar=UserVars()
    # Create your variables + set the initial values
    MyVar.newVar('A',1.0)
    MyVar.newVar('W',2.0)
    MyVar.newVar('Xc',0.0)

    # Define the function for a Gaussian
    # i.e. definition of the model
    def Gaussian(x):
       return MyVar.A*exp((x-MyVar.Xc)**2/MyVar.W**2)

    # Define the function Sim
    def Sim(data):
       # Calculate the Gaussian
       I=Gauss(data[0].x)
       # The returned value has to be a list
       return [I]

First an object of the class *UserVars* is created. This object is used to store user defined fit parameters.
Then the parameters are initialized (created) with their names given as strings.
After that a function for calculating a Gaussian variable is defined. The function takes an array
of x values as input parameters and returns the calculated y-values. At last the Sim function is defined. The function
Gauss is called to calculate the y-values with the x-data as the input argument. The x-values of the first data set
are extracted as ``data[0].x``, and those of the second data set would be extracted by
``data[1].x``.

.. note::

    A list is returned by taking the array (vector) *I* inside square brackets, pythons list syntax.
    This example requires that only one data set has been loaded. In order to fit the parameters created in
    by ``MyVar`` the user only has to right click on a cell in the grid of the Parameter Window and choose the
    ``MyVar.set[Name]`` function, i.e. ``MyVar.setA``.

Making a class
==============
The code above is usually sufficient for prototyping and simple problems. For more complex models it is
recommended to write a model library (python module). This is what has been done for the simulation of x-ray reflectivity data.
Instead of writing a lot of functions for each model, a class, or several, can be written to make the model
simple to use. As a more elaborate solution for the previous simple example we can define a class::

    from numpy import *
    from dataclasses import dataclass
    from models.utils import UserVars
    from models.lib.base import ModelParamBase


    # Define user axes labels for plotting in the GUI
    __xlabel__ = 'x-axis'
    __ylabel__ = 'y-axis'

    # Definition of the class used for each peak
    @dataclass
    class Gauss(ModelParamBase):
        A: float = 1.0
        w: float = 1.0
        xc: float = 0.0

        def Simulate(self, x):
            return self.A*exp(-(x - self.xc)**2/2/self.w**2)

    # Definition of background parameter
    cp = UserVars()
    cp.new_var('bkg', 100)


    # Make a Gaussian:
    Peak1=Gauss(w=2.0, xc=1.5, A=2.0)

    def Sim(data):
        # Calculate the Gaussian
        I=Peak1.Simulate(data[0].x)
        # The returned value has to be a list
        return [I]


This code is quite similar to the first version but encapsulates all necessary information in one class.
It starts with the definition of the class ``Gauss``.
The use of the ``@dataclass`` decorator and derivation from ``ModelParamBase`` ensures that the parameters are
automatically set and validated during instanciation and that GenX recognizes the class for setting parameters
in the Grid. The class also contains a method to calculate a Gaussian with the these attributes.
After the class definition an instance (object), ``Peak1``, of the Gauss class is created. Then the ``Sim`` function
is defined as in the previous example but with the function call exchanged to ``Peak1.Simulate(data[0].x)`` in order
to simulate the object ``Peak1``. The function names that should go into the parameter column in the
parameter window will be: ``Peak1.setW``, ``Peak1.setXc`` and ``Peak1.setA`` and should be selecteble from the menu
after running the first simulation.

The main data plot we use the axes labels defined with the ``__xlabel__`` and ``__ylabel__`` variables.
For pre-defined models these are set automatically but can always be overwritten by the user in the script.

Multiple Gaussians
==================
Making the model based on a class makes it easier to extend. For example if two peaks should be fitted
the class does not have to be changed. Instead an additional object of the class ``Gauss``, for example called
``Peak2``, can be created and the two contributions are then added in the ``Sim`` function. The end of the script
above would then be modified to

.. code-block::
    :lineno-start: 27

    # Make Gaussians:
    Peak1=Gauss(w=2.0,xc=1.5,A=2.0)
    Peak2=Gauss(w=2.0,xc=1.5,A=2.0)

    def Sim(data):
        # Calculate the Gaussian
        I=Peak1.Simulate(data[0].x)+Peak2.Simulate(data[0].x)
        # The returned value has to be a list
        return [I]


Thus, for fitting the parameters for the second Gaussian the functions used should
be ``Peak2.setW``, ``Peak2.setXc`` and ``Peak2.setA``.

Parameter coupling
==================
When the base class is created it can be extended with more problem oriented constraints by using
functions as in the first example. For example, in some cases it might be known that the width of the two
Gaussians should be the same. This can be solved by defining a new variable

.. code-block::
    :lineno-start: 31

    def Sim(data):
        Peak2.setW(Peak1.w)
        # Calculate the Gaussian
        I=Peak1.Simulate(data[0].x)+Peak2.Simulate(data[0].x)
        # The returned value has to be a list
        return [I]


Instead of using both the ``Peak?.setW`` methods ``Peak1.setW`` can be used to set both peak width
at the same time.

In summary, it is recommended that the models implemented in libraries are defined as
classes and that these are as general as possible with respect to the parameters. The specific parameter
couplings can be included as functions in the model file, using any of the defined class or UserVars parameters.
The methods shown with the examples in this section also apply to the libraries included for x-ray reflectivity.
The classes are different but the general use is the same.

An application of multi-peak fitting with a similar implementation of a gaussian is in cluded in the GenX
distribution or on `github <https://github.com/aglavic/genx/tree/master/genx/genx/examples>`_ under *genx/examples/Peakfit_Gauss.hgx*.
