.. _tutorial-mpi:

*******************************************
Using GenX from the command line (with mpi)
*******************************************

Introduction
============
Using GenX from the command line lets you, in the simplest case, start up the gui. You can also
run fits without starting up the gui at all. This opens possibilities to make a batch script of multiple GenX runs and,
in addition, you can run GenX on machines without a desktop environment. GenX also supports mpi for fitting
in parallel opens up the possibility to use it on clusters. The mpi implementation was contributed by Canrong Qiu.
Note that, currently, the command line is only fully implemented in the source/pip versions.

Dependencies
============
If you only intend to run GenX from the command line you do not need an installation of wxPython or matplotlib.

See section :ref:`install_cluster` for installation instructions.


Command line arguments
======================
The arguments to GenX can be viewed by executing the program with the ``--help`` option.

.. note::
    All commands should work from the source folder without installation by using ``python scripts/genx`` as execuable.

::

    $ genx --help
    usage: genx [-h] [-r | --mpi | -g | --pars | --mod]
                [--pr PR] [--cs CS] [--mgen MGEN] [--pops POPS] [--asi ASI] [--km KM] [--kr KR]
                [-s] [-e] [--var VAR] [--bumps]
                [-d DATA_SET] [--load DATAFILE] [--export SAVE_DATAFILE]
                [-l LOGFILE] [--debug] [--dpi-scale DPI_OVERWRITE] [--no-curses] [--disable-nb] [--nb1]
                [infile] [outfile]

    GenX 3.6.0, fits data to a model.

    positional arguments:
      infile                The .gx or .hgx file to load or .ort file to use as basis for model
      outfile               The .gx or hgx file to save into

    optional arguments:
      -h, --help            show this help message and exit
      -r, --run             run GenX fit (no gui)
      --mpi                 run GenX fit with mpi (no gui)
      -g, --gen             generate data.y with poisson noise added (no gui)
      --pars                extract the parameters from the infile (no gui)
      --mod                 modify the GenX file (no gui)

    optimization arguments:
      --pr PR               Number of processes used in parallel fitting.
      --cs CS               Chunk size used for parallel processing.
      --mgen MGEN           Maximum number of generations that is used in a fit
      --pops POPS           Population size - number of individuals.
      --asi ASI             Auto save interval (generations).
      --km KM               Mutation constant (float 0 < km < 1)
      --kr KR               Cross over constant (float 0 < kr < 1)
      -s, --esave           Force save evals to gx file.
      -e, --error           Calculate error bars before saving to file.
      --var VAR             Minimum relative parameter variation to stop the fit (%)
      --bumps               Use Bumps DREAM optimizer instead of GenX Differential Evolution

    data arguments:
      -d DATA_SET           Active data set to act upon. Index starting at 0.
      --load DATAFILE       Load file into active data set. Index starting at 0.
      --export SAVE_DATAFILE
                            Save active data set to file. Index starting at 0.

    startup options:
      -l LOGFILE, --logfile LOGFILE
                            Output debug information to logfile.
      --debug               Show additional debug information on console/logfile
      --dpi-scale DPI_OVERWRITE
                            Overwrite the detection of screen dpi scaling factor (=72/dpi)
      --no-curses           Disable Curses interactive console interface for command line fitting on UNIX systems.
      --disable-nb          Disable the use of numba JIT compiler
      --nb1                 Compile numba JIT functions without parallel computing support (use one core only).
                            This does disable caching to prevent parallel versions from being loaded.

    For support, manuals and bug reporting see http://genx.sf.net

To run a fit using the multiprocessing module (forking different processes) which is the same code as in the gui
the following command can be executed.

::

   $ python ./scripts/genx --run --mgen=10 --pr 8 ./genx/examples/X-ray_Reflectivity.hgx test.hgx
   INFO: *** GenX 3.6.0 Logging started ***
   INFO: Loading model C:\Users\Artur\genx\genx\genx\examples\X-ray_Reflectivity.hgx...
   INFO: Simulating model...
   INFO: Setting up the optimizer...
   INFO: DiffEv Optimizer:
    Fitting:
        use_start_guess=True    use_boundaries=True
        use_autosave=False      autosave_interval=10
        save_all_evals=False    max_log_elements=100000
    Differential Evolution:
        km                             0.6
        kr                             0.6
        create_trial                   best_1_bin
        use_pop_mult=False      pop_mult=3      pop_size=50
        use_max_generations=True        max_generations=10      max_generation_mult=6
        min_parameter_spread           0.0
    Parallel processing:
        use_parallel_processing        True
        parallel_processes             8
        parallel_chunksize             1

   INFO: Saving the initial model to C:\Users\Artur\genx\genx\test.hgx
   INFO: Fitting starting...
   INFO: DE initilized
   INFO: Setting up a pool of workers ...
   INFO: Starting the fit...
   INFO: Starting a pool with 8 workers ...
   INFO: Calculating start FOM ...
   INFO: Going into optimization ...
   INFO: FOM: 0.321 Generation: 1 Speed: 2777.7
   INFO: FOM: 0.293 Generation: 2 Speed: 2500.0
   INFO: FOM: 0.254 Generation: 3 Speed: 2500.2
   INFO: FOM: 0.217 Generation: 4 Speed: 2499.9
   INFO: FOM: 0.217 Generation: 5 Speed: 2777.7
   INFO: FOM: 0.217 Generation: 6 Speed: 2941.2
   INFO: FOM: 0.217 Generation: 7 Speed: 2941.2
   INFO: FOM: 0.206 Generation: 8 Speed: 2941.3
   INFO: FOM: 0.206 Generation: 9 Speed: 3124.8
   INFO: FOM: 0.206 Generation: 10 Speed: 2941.3
   INFO: Stopped at Generation: 10 after 500 fom evaluations...
   INFO: Fitting finished!
   INFO: Time to fit:  0.05453455845514933  min
   INFO: Updating the parameters
   INFO: Saving the fit to C:\Users\Artur\genx\genx\test.hgx
   INFO: Fitting successfully completed
   INFO: *** GenX 3.6.0 Logging ended ***

As can be seen this loads the file ``.genx/examples/X-ray_Reflectivity.hgx`` sets the maximum number of generation to run
to 10 and then runs the fit. The result is saved to ``test.hgx``. Note that to be able to analyse the fits (calculate error bars
for example) the option ``--esave`` should be used. If the fits take a long time to run it is advisable to save them
every now and then with the ``--asi`` command that specifies how often the current result should be written to file.
It can also be good idea to directly calculate the errorbars before saving to file with the ``-e`` command.
Another point to see is that there is a significant speed-up when only using the command line. This is probably due to
that the GUI does not have to be updated.

For UNIX systems the default command line output uses the curses library to better visualize the progress,
the output during refinement will look something like this:

::

        FOM: 0.051 Generation: 25 Speed: 2162.7
        FOM: 0.046 Generation: 26 Speed: 2141.1
        FOM: 0.046 Generation: 27 Speed: 2123.3
        FOM: 0.046 Generation: 28 Speed: 2120.4
        FOM: 0.046 Generation: 29 Speed: 1865.8
        FOM: 0.046 Generation: 30 Speed: 2185.8
        FOM: 0.046 Generation: 31 Speed: 2176.6
        FOM: 0.046 Generation: 32 Speed: 2227.9

                                    Relative value and spread of fit parameters:                     best/width
     Parameter 00: [                                        ==#                                     ] 0.53/0.03
     Parameter 01: [       ===================#====================                                 ] 0.34/0.51
     Parameter 02: [                                 ==========================================#=== ] 0.94/0.58
     Parameter 03: [                      =============================#===================         ] 0.64/0.62
     Parameter 04: [                                                    =======================#==  ] 0.94/0.33
     Parameter 05: [ =========================#=====================                                ] 0.33/0.59
     Parameter 06: [                    =============#==========                                    ] 0.42/0.31
     Parameter 07: [                                              =================#======          ] 0.79/0.31
     Parameter 08: [ ============#=================                                                 ] 0.17/0.38

.. note::
    The fit can be stopped before the breaking conditions using ``q``. To deactivate the interactive
    view use the ``--no-curses`` option.

    Stopping with q only works on UNIX without curses if ``<enter>`` is pressed afterwords. This can
    also be used to stop a MPI refinement at any time.

Using MPI
=========
If MPI and mpi4py is installed on the system the ``--mpi`` switch will be activated. Note that the description for
``--mpi`` in the help will not appear until the mpi4py can be loaded correctly. In order to use mpi the command ``mpirun``
or ``mpiexec`` has to be used. The argument ``-np`` defines how many processes to use. An example can be seen below.

::

   $ mpirun -np 2 python -m genx.run --mpi --mgen=10 ./genx/examples/X-ray_Reflectivity.hgx test.hgx
   INFO: *** GenX 3.6.0 Logging started ***
   INFO: Loading model /mnt/c/Users/Artur/genx/genx/genx/examples/X-ray_Reflectivity.hgx...
   INFO: Simulating model...
   INFO: Setting up the optimizer...
   INFO: DiffEv Optimizer:
    Fitting:
        use_start_guess=True    use_boundaries=True
        use_autosave=False      autosave_interval=10
        save_all_evals=False    max_log_elements=100000
    Differential Evolution:
        km                             0.6
        kr                             0.6
        create_trial                   best_1_bin
        use_pop_mult=False      pop_mult=3      pop_size=50
        use_max_generations=True        max_generations=10      max_generation_mult=6
        min_parameter_spread           0.0
    Parallel processing:
        use_parallel_processing        False
        parallel_processes             2
        parallel_chunksize             1

   INFO: Saving the initial model to /mnt/c/Users/Artur/genx/genx/test.hgx
   INFO: Fitting starting...
   INFO: DE initilized
   INFO: Inits mpi with 2 processes ...
   INFO: Starting the fit...
   INFO: Calculating start FOM ...
   INFO: Going into optimization ...
   INFO: FOM: 0.301 Generation: 1 Speed: 1244.8
   INFO: FOM: 0.234 Generation: 2 Speed: 1262.8
   INFO: FOM: 0.234 Generation: 3 Speed: 1225.5
   INFO: FOM: 0.234 Generation: 4 Speed: 1229.7
   INFO: FOM: 0.234 Generation: 5 Speed: 1148.9
   INFO: FOM: 0.234 Generation: 6 Speed: 1226.7
   INFO: FOM: 0.234 Generation: 7 Speed: 1112.0
   INFO: FOM: 0.234 Generation: 8 Speed: 1214.3
   INFO: FOM: 0.234 Generation: 9 Speed: 1200.5
   INFO: FOM: 0.234 Generation: 10 Speed: 1000.2
   INFO: Stopped at Generation: 10 after 500 fom evaluations...
   INFO: Fitting finished!
   INFO: Time to fit:  0.011236679553985596  min
   INFO: Updating the parameters
   INFO: Saving the fit to /mnt/c/Users/Artur/genx/genx/test.hgx
   INFO: Fitting successfully completed
   INFO: *** GenX 3.6.0 Logging ended ***

As MPI defines its process externally and the code calculates the chunk size automatically the arguments ``-pr`` and
``--cr`` will not be used in this case. This should be the only changes compared to using it from the command line as
usual.
If a logfile is written with the ``-l`` option the MPI process number will be added to the file name with the
primary process starting with number ``00``.


Using remote refinement server
==============================

To have the advantage of high performance computing and interactive refinement GenX has a server script that
can be started on the cluster and a desktop client within the same network can use this as worker for
refinement from a GUI client.

To start the server with the standard parameters run the genx_server command or execute with python directly:

::

   $ genx_server
   INFO: *** GenX 3.6.0 Logging started ***
   INFO: Importing numba based modules to pre-compile JIT functions, this can take some time
   INFO: Modules imported successfully
   INFO: Starting RemoteController
   INFO: Starting listening on localhost with port=3000

The fitting is then started from the GUI client selecting the "Remote DiffEv" optimizer. The configuration is done
the same way as for the standard optimizer with additional options for the server configuration.
From the client side the fit should look like a local run refinement and the server outputs a short information
on the console (if --debug is not set).

::

   INFO: Setting a new model
   INFO: Start fit was triggered
   INFO: Stop fit was triggered

It is also possible to use MPI on the server by starting it using ``mpiexec`` or ``mpirun``:

::

    mpiexec -np 32 python -m genx.server

The client optimizer settings will determine if multiprocessing or MPI will be used.

Connection settings
-------------------

The genx_server script takes two optional arguments ``address`` and ``port``. By default the sever listens only to
connections from **localhost** on port **3000**.
You can choose to listen on any incoming network interfaces by supplying **0.0.0.0** as ``address`` but this is
not very secure as anyone on the local network would be able to connect to this client.
The communication protocol does use a simple password authentication but communication is not encrypted so
it is adviced to keep the port open only locally and using ssh tunnel (-L option) to connect from you machine.

::

    $ ssh -L 3000:localhost:3000 {server_with_genx}
    $ mpiexec -np 32 genx_server
