.. _tutorial-mpi:

*******************************************
Using GenX from the command line (with mpi)
*******************************************

Introduction
============
Using GenX from the command line lets you in the simplest case start up the gui. As of version 2.3 you can also
run fits without starting up the gui at all. This opens possibilities to make batch script of multiple GenX runs and
in addition you can run GenX on machines without a desktop environment. As GenX also since 2.3 supports mpi for fitting
in parallel opens up the possibility to use it on clusters. The mpi implementation was contributed by Canrong Qiu.
Note that currently the command line is only fully implemented in the source version.

Dependencies
============
If you only intend to run GenX from the command line you do not need an installation of wxPython, appdirs or matplotlib.
Thus the packages you need are:

* Python newer than 2.3.5
* Numpy version > 1.0
* Scipy version > 0.5
* h5py version > ?
* If using mpi you will need mpi4py as well as an mpi installation.


Command line arguments
======================
The arguments to GenX can be viewed by executing the program with the ``--help`` option.

::

    $ python genx.py --help
    usage: genx.py [-h] [-r | --mpi] [--pr PR] [--cs CS] [--mgen MGEN]
                   [--pops POPS] [--asi ASI] [--km KM] [--kr KR] [-s] [-e]
                   [infile] [outfile]

    GenX 2.3.5, fits data to a model.

    positional arguments:
      infile       The .gx or .hgx file to load
      outfile      The .gx or hgx file to save into

    optional arguments:
      -h, --help   show this help message and exit
      -r, --run    run GenX fit (no gui)
      --mpi        run GenX fit with mpi (no gui)

    optimization arguments:
      --pr PR      Number of processes used in parallel fitting.
      --cs CS      Chunk size used for parallel processing.
      --mgen MGEN  Maximum number of generations that is used in a fit
      --pops POPS  Population size - number of individuals.
      --asi ASI    Auto save interval (generations).
      --km KM      Mutation constant (float 0 < km < 1)
      --kr KR      Cross over constant (float 0 < kr < 1)
      -s, --esave  Force save evals to gx file.
      -e, --error  Calculate error bars before saving to file.

    For support, manuals and bug reporting see http://genx.sf.net

To run a fit using the multiprocessing module (forking different processes) which is the same code as in the gui
the following command can be executed.

::

    $ python genx.py --run --mgen=10 ./examples/X-ray_Reflectivity.gx test.gx
    Loading model /Users/GenX/Desktop/v2.3.5/examples/X-ray_Reflectivity.gx...
    Simulating model...
    Setting up the optimizer...
    DE initilized
    Setting up a pool of workers ...
    Starting a pool with 2 workers ...
    Saving the initial model to /Users/GenX/Desktop/v2.3.5/test.gx
    Fitting starting...
    Calculating start FOM ...
    Going into optimization ...
    FOM: 0.277 Generation: 1 Speed: 541.6
    FOM: 0.277 Generation: 2 Speed: 550.4
    FOM: 0.268 Generation: 3 Speed: 528.7
    FOM: 0.268 Generation: 4 Speed: 544.2
    FOM: 0.243 Generation: 5 Speed: 546.8
    FOM: 0.243 Generation: 6 Speed: 544.7
    FOM: 0.243 Generation: 7 Speed: 549.8
    FOM: 0.243 Generation: 8 Speed: 544.1
    FOM: 0.218 Generation: 9 Speed: 546.9
    FOM: 0.215 Generation: 10 Speed: 550.1
    Stopped at Generation: 10 after 500 fom evaluations...
    Fitting finished!
    Time to fit:  0.0183591683706  min
    Updating the parameters
    Saving the fit to /Users/GenX/Desktop/v2.3.5/test.gx
    Fitting successfully completed

As can be seen this loads the file ``./examples/X-ray_Reflectivity.gx`` sets the maximum number of generation to run
to 10 and then runs the fit. The result is saved to ``test.gx``. Note that to be able to analyse the fits (calculate error bars
for example) the option ``--esave`` should be used. If the fits take a long time to run it is advisable to save them
every now and then with the ``--asi`` command that specifies how often the current result should be written to file.
It can also be good idea to directly calculate the errorbars before saving to file with the ``-e`` command.
Another point to see is that there is a significant speed-up when only using the command line. This is probably due to
that the GUI does not have to be updated.

Using MPI
=========
If MPI and mpi4py is installed on the system the ``--mpi`` switch will be activated. Note that the description for
``--mpi`` in the help will not appear until the mpi4py can loaded correctly. In order to use mpi the command ``mpirun``
or ``mpiexec`` has to be used. The argument ``-np`` defines how many processes to use. An example can be seen below.

::

    $ mpirun -np 2 python genx.py --mpi --mgen=10 ./examples/X-ray_Reflectivity.gx test.gx
    Loading model /Users/GenX/Desktop/v2.3.5/examples/X-ray_Reflectivity.gx...
    Simulating model...
    Setting up the optimizer...
    DE initilized
    Inits mpi with 2 processes ...
    Saving the initial model to /Users/GenX/Desktop/v2.3.5/test.gx
    Fitting starting...
    Calculating start FOM ...
    Going into optimization ...
    FOM: 0.288 Generation: 1 Speed: 549.5
    FOM: 0.288 Generation: 2 Speed: 550.3
    FOM: 0.288 Generation: 3 Speed: 561.3
    FOM: 0.240 Generation: 4 Speed: 563.7
    FOM: 0.240 Generation: 5 Speed: 566.1
    FOM: 0.240 Generation: 6 Speed: 560.2
    FOM: 0.209 Generation: 7 Speed: 563.9
    FOM: 0.209 Generation: 8 Speed: 559.6
    FOM: 0.209 Generation: 9 Speed: 564.2
    FOM: 0.190 Generation: 10 Speed: 559.5
    Stopped at Generation: 10 after 500 fom evaluations...
    Fitting finished!
    Time to fit:  0.0177068511645  min
    Updating the parameters
    Saving the fit to /Users/GenX/Desktop/v2.3.5/test.gx
    Fitting successfully completed

As MPI defines its process externally and the code calculates the chunk size automatically the arguments ``-pr`` and
``--cr`` will not be used in this case. This should be the only changes compared to using it from the command line as
usual.