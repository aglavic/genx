.. _tutorial-performance:

*******************************
Optimizer and model performance
*******************************

.. note::
    This page is still under development. Questions and comments are welcome to improve usability.

Introduction
============
Computing performance is a complex topic and in a modeling set like GenX,
where the user has a large amount of flexibilit within the model definition,
on generic one-size-fits-all solution is possible.

In this chapter I will therefore explain what has been implemented in GenX to try to
optimize the general model performance (mostly for reflectivity) and the different
options in the optimizer settings you can use to tune the calculation to your
sepcific model.

Main factors determining performance
====================================

Single model execution
----------------------
When simulating a single model, no generic solution for parallel processing can be applied.
Especially in Python with the so-called "Global Interpreter Lock" (GIL) there is no shared
state between threads running on different cores.

While GenX tries to optimize the time consuming parts of a model execution using
algorithms optimized for numpy and scipy functions, the resuling speed of models
are typically 5-10x slower than implmentation in compiled languages like C++.

Since version 3.0.2, the core functions for reflectivity have been ported to the
just-in-time (JIT) compiler package numba, which lead to single thread speedups
of 2-5x. In addition, it allows parallelization of these functions, which can
circumvent the GIL. Depending on the used CPU, complex models can gain another
2-10x speed improvement.

For the single calculations used during simulation, GenX does not provide any
further flexibility to alter the computation. The only exception is the
use of CUDA (NVidia GPU computation framework) that can be activated
in the GUI "Fit" menu.
The impact of this setting is strongly model dependant. With many
datapoints (resolution convolution included) and large amount of layers
the speed can be comparible with strong multicore CPUs while requiring less system resources.
Because of several caveats and the need to re-compile the JIT code every time GenX
is started I would not recommand this in most cases, at the moment.

Fitting the model
-----------------
One advantage of the Differential Evolution algorithm is,
that a large number of parameter sets are being calculated for every
generation without any interdependance. This allows a relatively simple
way of parallelizing computations as a pool of processes can be used
with model parameters being passed to them every generation.

Any parallel computing solutions have overhead involved in setting up
and communication between parallel threads. The optimizal settings will
therefore depend on the complexity of the model.

A general rule of thumb is that the more complex a computation within a
thread the lower the influence of the overhead of setting it up. At the
same time, the more data is needed for a computation the more overhead
is produced.

In GenX this means, that the parallel computation provided by the
numba JIT functions is less effective is the model is a small number of
layers and if the number of datapoints is small. So in the case of
simple and fast models the multiprocessing of the differential evolution
optimizer can lead to much higher preformance.
GenX does automatically reduce the number of cores used by numba functions
when the process number in the optimizer settings is increased (simulations
still use the maximum available cores). Transfer to the processes also has
its overhead, that can be influenced by the "items/chunk" parameter as well
as the population size.

Tips to optimize your model performance
=======================================
    * Always use the "Speed" indication in the status bar at the bottom of the window. It shows, how many
      function evaluations per second are being calculated during the fit
    * If you do not need it, you can disable "ignore fom nan" and "ignore fom inf", which can slightly improve performance
    * Try out different settings of population size, items/chunk, parallel on/off and number of processes

Simple models
-------------
    * Models with <20 layers and a few 100 datapoints
    * Expected computation speed >300 (can reach >10000 if optimized)
    * Use parallel processing with process=cores/threads.
    * Use large population size 100-1000
    * Adapt chunk/item to be =(population size)/(processes)

Complex models
--------------
    * These can have 100 or more layers and resolution convolution that leads to >1000 datapoints
    * Expected computation speed <150
    * Try without parallel processing or small number of parallel threads (2-4)
    * If CUDA is available, especially for neutron spin-flip calculations, try using CUDA in conjunction with
      2-8 parallel threads. In this case one of the threads will run on GPU. In tests this could lead
      to 1.5x to 2.0x improvement, even on a system with powerful 16-core CPU.
