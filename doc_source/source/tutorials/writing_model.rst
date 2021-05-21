.. _tutorial-writing-model:

**********************
Writing a custom model
**********************

Writing a model is quite easy in GenX. There is a couple of things one need to keep in mind in order to successful
which will be covered in this tutorial. The only mandatory thing the model file has to contain is a function called
``Sim`` taking a member of the class Data as input parameter. However, to make the model useful, functions for
setting the values have to be incorporated. Note that GenX uses function to set the parameters during fitting,
this is why we need to have them.

Programming Python
==================
Since writing a model actually involves writing a script in Python it is good to have some basic knowledge of the syntax.
However, if you have some basic knowledge about programming it should be fairly easy to just look at the examples
and write your own models without having to learn to program in Python. On the other hand, there exists a number of
free introductory books as well as tutorials on the internet for the interested reader, see below.

* `Python’s homepage <http://www.python.org>`_ contain most of the available tutorials online.
* `A Byte of Python <http://www.byteofpython.info:8123>`_ is an introductory text for the absolute beginner.
* `How to Think Like a Computer Scientist: Learning with Python <http://www.greenteapress.com/thinkpython>`_ is a textbook
  for the beginner written for computer science students.
* `Dive Into Python <http://diveintopython.org>`_ is an introduction to Python book for the more experienced programmer.

In addition there are a number of tutorials on `SciPy's homepage <http://www.scipy.org>`_ which deal w
ith numerical computations. There is also a migration guide for those who are familiar with MatLab.

The Data class
==============
In order to write the Sim class it is necessary to know the structure of the class ``Data`` which is taken as a
parameter. The variables which could be useful in the ``Sim`` function are:

* ``x`` A list of 1-D arrays (vectors) containing the x-values of the processed data
* ``y`` A list of 1-D arrays (vectors) containing the y-values of the processed data
* ``xraw`` A list of 1-D arrays (vectors) containing the raw x-values (the data loaded from the data file)
* ``yraw`` A list of 1-D arrays (vectors) containing the raw y-values (the data loaded from the data file)
* ``use`` A list of booleans (True or False) denoting if the data should be fitted

Simple example
==============
Knowing what the Data class contains we will start with a simple example, making a model that fits one
Gaussian to the first data set. The free parameters of the Gaussian are; the center of the peak, Xc, the peak width,
W, and the amplitude of the peak, A. Writing a model for it would produce a code as shown below. Note that a
# produce a comment.
::

    # Create a class for user variables
    MyVar=UserVars()
    # Create your variables + set the initial values
    MyVar.newVar(’A’,1.0)
    MyVar.newVar(’W’,2.0)
    MyVar.newVar(’Xc’,0.0)

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


The following is a brief description of the code above. First an object of the class UserVars? is created.
This object is used to store user defined variables. Then the variables are initialized (created) with their names
given as strings. After that a function for calculating a Gaussian variable is created. The function takes an array
of x values as input parameters and returns the calculated y-values. At last the Sim function is defined. The function
Gauss is called to calculate the y-values with the x-data as the input argument. The x-values of the first data set
are extracted as ``data.x[0]``, and those of the second data set would be extracted by
``data.x[1]``. Note that a list is returned by taking the array (vector) I and making a list with one
element. Note that this requires that only one data set has been loaded. In order to fit the parameters created in
by ``MyVar`` the user only has to right click on a cell in the grid of the Parameter Window and choose the
``MyVar.set[Name]`` function, i.e. ``MyVar.setA``.

Making a class
==============
The code above is usually sufficient for prototyping and simple problems. For more complex models it is
recommended to write a library. This is what has been done for the simulation of x-ray reflectivity data.
Also, instead of writing a lot of functions for each model, a class, or several, can be written to make the model
simple to use. As a more elaborate example the previous simple example can be transformed into a class::

    # Definition of the class
    class Gauss:
        # A class for a Gaussian
        # The creator of the class
        def __init__(self,w=1.0,xc=0.0,A=1.0):
            self.w=w
            self.xc=xc
            self.A=A

        # The set functions used in the parameters column
        def setW(w):
        self.w=w

        def setXc(xc):
            self.xc=xc

        def setA(A):
            self.A=A

        # The function to calculate the model (A Gaussian)
        def Simulate(x):
            return A*exp((x-self.xc)**2/self.w**2)

    # Make a Gaussian:
    Peak1=Gauss(w=2.0,xc=1.5,A=2.0)

    def Sim(data):
        # Calculate the Gaussian
        I=Peak1.Simulate(data[0].x)
        # The returned value has to be a list
        return [I]


This code is quite similar to the first version with only functions. It starts with the definition of the class
``Gauss``. This class has a constructor, init, to initialize the parameters of the object and functions to set the
member variables, denoted as ``self.*``. It also contains a member function to calculate a Gaussian with the member
variables. After the class definition an object, ``Peak1``, of the Gauss class is created. Then the ``Sim`` function
is defined as in the previous example but with the function call exchanged to ``Peak1.Simulate(data.x[0])`` in order
to simulate the object ``Peak1``. The function names that should go into the parameter column in the
parameter window will be: ``Peak1.setW``, ``Peak1.setXc`` and ``Peak1.setA``.

Multiple Gaussians
==================

Making the model based on a class makes it easier to extend. For example if two peaks should be fitted
the class does not have to be changed. Instead an additional object of the class ``Gauss``, for example called
``Peak2``, can be created and the two contributions are then added in the ``Sim`` function. The code would then be
modified to (omitting the class definition)::

    # Insert the class definition from above
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
Gaussians should be the same. This can be solved by defining a new variable::

    #Insert the class definition from above
    # Make Gaussians:
    Peak1=Gauss(w=2.0,xc=1.5,A=2.0)
    Peak2=Gauss(w=2.0,xc=1.5,A=2.0)
    # Create a class for user variables
    MyVar=UserVars()
    # Create your variables + set the initial values
    MyVar.newVar(’BothW’,1.0)

    def Sim(data):
        Peak1.setW(MyVar.BothW)
        Peak2.setW(MyVar.BothW)
        # Calculate the Gaussian
        I=Peak1.Simulate(data[0].x)+Peak2.Simulate(data[0].x)
        # The returned value has to be a list
        return [I]


Instead of using the ``*.setW`` methods the ``MyVar.setBothW`` can be used, which is automatically created
by the MyVar class. In summary it is recommended that the models implemented in libraries are defined as
classes and that these are as general as possible with respect to the parameters. The specific parameter
couplings can be included as functions in the model file. The methods shown with the examples in this section also
apply to the libraries included for x-ray reflectivity. The classes are different but the general use is the same.
