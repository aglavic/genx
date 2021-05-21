.. _development-write-fom:

***********************
Writing a FOM function
***********************
The figure of merit (FOM) function is one of the most important things when fitting a model to
measured data. The FOM defines what a good fit is and distinguishes it from a bad fit.
Every fitting problem has its own quirks and twists and may therefore benefit from having a custom FOM
function written for that specific problem. Here, we will briefly go through the steps you need to take to
extend GenX with your very own FOM function.

Making a custom FOM available in GenX
=====================================
The file that defines the different built-in FOM functions is called ``fom_funcs.py``.
As of svn release 145, there is a simple and transparent way to add your own FOM functions in a separate file,
which must be called ``fom_funcs_custom.py`` and reside in the same directory as ``fom_funcs.py``. This file is
read by ``fom_funcs.py`` if it exists, otherwise it is ignored. The ``fom_funcs_custom.py`` file is not part of the
GenX distribution, but must be created by the user. This ensures that custom-build FOM functions are not
overwritten when updating the GenX distribution to the latest version. All custom FOM function definitions can
be included in this one file, or you may choose to read in several other files from ``fom_funcs_custom.py``,
just as ``fom_funcs_custom.py``is read in ``fom_funcs.py``
(Look at the code of ``fom_funcs.py`` to see how that can be achieved).

Once you have defined a new FOM function in ``fom_funcs_custom.py``, you just need to restart GenX and it
should become available under :menuselection:`Settings-->Optimizer` as usual. What you might want to do is to
add some documentation to the html string which is used as a help function within GenX. And finally,
when it works, why not send it to me for inclusion in the main distribution.

Example of a FOM function
=========================
It's always easiest to start with an example to get ones head around things. Let us start with an absolute
logarithmic difference figure of merit. The code below is a slow but easy to understand variant::

    def log(simulations, data):
        ''' The absolute logarithmic difference'''
        N = 0 # Total number of data points
        fom = 0 # The total fom to calculate
        for dataset, sim in zip(data, simulation):
            if dataset.use:
                fom = fom + np.sum(np.abs(np.log10(dataset.y) - np.log10(sim)))
                N = N + len(dataset.x)
        fom = fom/N
        return fom


For each data set which is active, i.e. which has its ``use`` attribute set to ``True``, we add the summed logarithmic
difference to ``fom`` and the number of data points to ``N``. The current status of ``active`` is displayed in
the data list seen in the left panel in GenX.

The most tricky statement for a non-python programmer is probably the ``zip`` function in the first loop.
This will, just like a zipper, create a long list of tuple pairs for each element in data and simulation.
`See <http://docs.python.org/library/functions.html>`_. The statement ``dataset, sim`` will "unbundle" them again,
just as if two items where returned from a function. Next up is the calculation of each data point in the current
data set and adding it to the total fom. We also keep track on the number of data points for later normalization.
Note that the fom is only calculated and added if the flag ``dataset.use`` is ``True``.

An example how this can be programmed in a more compact and computationally efficient way is
seen below, where a faster type of for loop is used.
::

    def log(simulations, data):
        ''' The absolute logarithmic difference'''
        N = np.sum([len(dataset.y)*dataset.use for dataset in data])
        return 1.0/(N - 1)*np.sum([np.sum(np.abs(np.log10(dataset.y) - np.log10(sim)))
            for (dataset, sim) in zip(data, simulations) if dataset.use])


It looks a bit more complicated since it uses list comprehension, which is faster and more compact than
ordinary for loops. Also note the if statement last inside the brackets: ``if dataset.use`` this will only
append an item to the list if the condition is true. That is if the use flag is set. To learn more about
list comprehensions go to `the python docs <http://docs.python.org/tutorial/datastructures.html#list-comprehensions>`_.
This is the syntax that you will find most in the built-in FOM functions
provided.
