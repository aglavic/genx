#!/usr/bin/env python
import logging
import multiprocessing
import threading

import appdirs
import argparse
import os
import os.path
import sys
from logging import debug

from . import version
from .core import custom_logging
from .core.custom_logging import activate_excepthook, activate_logging, iprint, setup_system


def start_interactive(args):
    ''' Start genx in interactive mode (with the gui)

    :param args:
    :return:
    '''
    debug('enter start_interactive')
    activate_excepthook()
    # Fix blurry text on Windows 10
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(True)
    except:
        pass

    from .gui import main_window
    if args.infile!='':
        debug('start GUI setup with file to load')
        filename=args.infile
    else:
        filename=None
    debug('start GUI setup')
    app= main_window.GenxApp(filename=filename)
    debug('setup complete, start WX MainLoop')
    app.MainLoop()
    debug('leave start_interactive')

def calc_errorbars(config, mod, opt):
    error_values=[]
    fom_error_bars_level=config.getfloat('solver', 'errorbar level')
    n_elements=len(opt.start_guess)
    for index in range(n_elements):
        # calculate the error
        # TODO: Check the error bar buisness again and how to treat Chi2
        (error_low, error_high)=opt.calc_error_bar(index, fom_error_bars_level)
        error_values.append('(%.3e, %.3e,)'%(error_low, error_high))
    mod.parameters.set_error_pars(error_values)

def create_simulated_data(args):
    """Function to create simulated data from the model and add it to data.y"""
    from .diffev import DiffEv
    from .model_control import ModelController
    from .core import config as io

    from scipy.stats import poisson

    io.config.load_default(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profiles', 'default.profile'))
    ctrl=ModelController(DiffEv())
    mod=ctrl.model

    iprint("Loading file: %s "%args.infile)
    ctrl.load_file(args.infile)
    # io.load_opt_config(opt, config)
    iprint("File loaded")

    iprint("Simualting..")
    mod.simulate()
    iprint("Storing simualted data")
    for data_set in mod.data:
        data_set.y_raw=poisson.rvs(data_set.y_sim)
        data_set.y_command='y'
        data_set.run_y_command()
        data_set.error_command='sqrt(where(y > 0, y, 1.0))'
        data_set.run_error_command()

    iprint('Saving the model to %s'%args.outfile)
    ctrl.save_file(args.outfile)

def extract_parameters(args):
    """Extracts the parameters to outfile"""

    from .diffev import DiffEv
    from .model_control import ModelController
    from .core import config as io

    # Open the genx file
    io.config.load_default(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profiles', 'default.profile'))
    ctrl=ModelController(DiffEv())
    mod=ctrl.model

    iprint("Loading file: %s "%args.infile)
    ctrl.load_file(args.infile)
    iprint("File loaded")

    names, values=mod.parameters.get_sim_pars()

    if args.outfile=='':
        fout=sys.stdout
    else:
        outfile=args.outfile
        if os.path.isfile(outfile):
            fout=open(outfile, 'a')
        else:
            fout=open(outfile, 'w')
            # Add header
            fout.write('\t'.join([name for name in names])+'\n')
    fout.write('\t'.join(['%f'%val for val in values])+'\n')
    fout.close()

def modify_file(args):
    """Modify a GenX file given command line arguments"""
    from .diffev import DiffEv
    from .model_control import ModelController
    from .core import config as io

    # Open the genx file
    io.config.load_default(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profiles', 'default.profile'))
    ctrl=ModelController(DiffEv())
    mod=ctrl.model

    iprint("Loading file: %s "%args.infile)
    ctrl.load_file(args.infile)
    iprint("File loaded")

    if args.datafile:
        datafile=os.path.abspath(args.datafile)
        if 0>args.data_set or args.data_set>=len(mod.data):
            iprint("The selected data set does not exist - select one between 0 and %d"%(len(mod.data)-1))
            return
        iprint('Loading dataset %s into data set %d'%(datafile, args.data_set))
        mod.data[args.data_set].loadfile(datafile)

        if args.outfile:
            iprint('Saving the fit to %s'%args.outfile)
            ctrl.save_file(args.outfile)

    elif args.save_datafile:
        save_datafile=os.path.abspath(args.save_datafile)
        iprint("Simualting..")
        mod.simulate()
        if 0>args.data_set or args.data_set>=len(mod.data):
            iprint("The selected data set does not exist - select one between 0 and %d"%(len(mod.data)-1))
            return
        iprint('Exporting data set %d into ASCII file %s'%(args.data_set, save_datafile))
        mod.data[args.data_set].save_file(save_datafile)

def start_fitting(args, rank=0):
    """ Function to start fitting from the command line.
    """
    # TODO: fix implementation of this
    import time
    from .diffev import DiffEv, DiffEvDefaultCallbacks
    from .model_control import ModelController
    from .core import config as io

    # Open the genx file
    io.config.load_default(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profiles', 'default.profile'))
    ctrl=ModelController(DiffEv())
    mod=ctrl.model
    opt:DiffEv=ctrl.optimizer
    config=io.config

    if rank==0:
        class CB(DiffEvDefaultCallbacks):
            def autosave(self):
                # print 'Updating the parameters'
                mod.parameters.set_value_pars(opt.best_vec)
                if args.error:
                    iprint("Calculating error bars")
                    calc_errorbars(io.config, mod, ctrl.optimizer)
                if args.outfile:
                    iprint("Saving to %s"%args.outfile)
                    ctrl.save_file(args.outfile)

        ctrl.set_callbacks(CB())

    if rank==0:
        iprint('Loading model %s...'%args.infile)
    ctrl.load_file(args.infile)
    # has to be used in order to save everything....
    if args.esave:
        io.config.set('solver', 'save all evals', True)
    # Simulate, this will also compile the model script
    if rank==0:
        iprint('Simulating model...')
    mod.simulate()

    # Sets up the fitting ...
    if rank==0:
        iprint('Setting up the optimizer...')
        iprint(opt)
    set_optimiser_pars(opt, args)

    if args.outfile and rank==0:
        iprint('Saving the initial model to %s'%args.outfile)
        ctrl.save_file(args.outfile)

    # To start the fitting
    if rank==0:
        iprint('Fitting starting...')
        t1=time.time()
    # print opt.use_mpi, opt.use_parallel_processing
    opt.start_fit(mod)
    while opt.is_running():
        try:
            time.sleep(0.1)
        except KeyboardInterrupt:
            iprint('KeyboardInterrupt, trying to stop fit.')
            opt.stop=True
    if rank==0:
        t2=time.time()
        iprint('Fitting finished!')
        iprint('Time to fit: ', (t2-t1)/60., ' min')

    if rank==0:
        iprint('Updating the parameters')
        mod.parameters.set_value_pars(opt.best_vec)

    if args.outfile and rank==0:
        if args.error:
            iprint('Calculating errorbars')
            calc_errorbars(config, mod, opt)
        iprint('Saving the fit to %s'%args.outfile)
        opt.set_use_mpi(False)
        ctrl.save_file(args.outfile)

    if rank==0:
        iprint('Fitting successfully completed')

def set_optimiser_pars(optimiser, args):
    """ Sets the optimiser parameters from args
    """
    if args.pr:
        optimiser.set_processes(args.pr)
        optimiser.set_use_parallel_processing(True)

    if args.mpi:
        optimiser.set_use_mpi(True)

    if args.cs:
        optimiser.set_chunksize(args.cs)

    if args.mgen:
        optimiser.set_max_generations(args.mgen)
        optimiser.set_use_max_generations(True)

    if args.pops:
        optimiser.set_pop_size(args.pops)
        optimiser.set_use_pop_mult(False)

    if args.asi:
        optimiser.set_autosave_interval(args.asi)
        optimiser.set_use_autosave(True)

    if args.km>=0:
        optimiser.set_km(args.km)
    # else:
    #    print "km not set has to be bigger than 0"

    if args.kr>=0:
        optimiser.set_kr(args.kr)
    # else:
    #    print "kr not set has to be bigger than 0"

def main():
    multiprocessing.freeze_support()
    # Attempt to load mpi:
    __mpi__=False
    try:
        from mpi4py import MPI
    except ImportError:
        pass
    else:
        __mpi__=True
        rank=MPI.COMM_WORLD.Get_rank()

    parser=argparse.ArgumentParser(description="GenX %s, fits data to a model."%version.__version__,
                                   epilog="For support, manuals and bug reporting see http://genx.sf.net"
                                   )
    run_group=parser.add_mutually_exclusive_group()
    run_group.add_argument('-r', '--run', action='store_true', help='run GenX fit (no gui)')
    if __mpi__:
        run_group.add_argument('--mpi', action='store_true', help='run GenX fit with mpi (no gui)')
    run_group.add_argument('-g', '--gen', action='store_true', help='generate data.y with poisson noise added')
    run_group.add_argument('--pars', action='store_true', help='extract the parameters from the infile')
    run_group.add_argument('--mod', action='store_true', help='modify the GenX file')
    opt_group=parser.add_argument_group('optimization arguments')
    opt_group.add_argument('--pr', type=int, default=0, help='Number of processes used in parallel fitting.')
    opt_group.add_argument('--cs', type=int, default=0, help='Chunk size used for parallel processing.')
    opt_group.add_argument('--mgen', type=int, default=0, help='Maximum number of generations that is used in a fit')
    opt_group.add_argument('--pops', type=int, default=0, help='Population size - number of individuals.')
    opt_group.add_argument('--asi', type=int, default=0, help='Auto save interval (generations).')
    opt_group.add_argument('--km', type=float, default=-1, help='Mutation constant (float 0 < km < 1)')
    opt_group.add_argument('--kr', type=float, default=-1, help='Cross over constant (float 0 < kr < 1)')
    opt_group.add_argument('-s', '--esave', action='store_true', help='Force save evals to gx file.')
    opt_group.add_argument('-e', '--error', action='store_true', help='Calculate error bars before saving to file.')
    data_group=parser.add_argument_group('data arguments')
    data_group.add_argument('-d', dest='data_set', type=int, default=0,
                            help='Active data set to act upon. Index starting at 0.')
    data_group.add_argument('--load', dest='datafile',
                            help='Load file into active data set. Index starting at 0.')
    data_group.add_argument('--export', dest='save_datafile',
                            help='Save active data set to file. Index starting at 0.')
    data_group.add_argument('-l', '--logfile', dest='logfile', default=None, type=str,
                            help='Output debug information to logfile.')
    data_group.add_argument('--debug', dest='debug', default=False, action="store_true",
                            help='Show additional debug information on console/logfile')

    parser.add_argument('infile', nargs='?', default='', help='The .gx or .hgx file to load')
    parser.add_argument('outfile', nargs='?', default='', help='The .gx  or hgx file to save into')

    args=parser.parse_args()
    if not __mpi__:
        args.mpi=False

    if args.run or args.mpi:
        custom_logging.CONSOLE_LEVEL=logging.INFO
    setup_system()

    if args.logfile:
        activate_logging(args.logfile)
    debug("Arguments from parser: %s"%args)
    if not args.outfile:
        args.outfile=args.infile
    args.outfile=os.path.abspath(args.outfile)
    if args.infile:
        args.infile=os.path.abspath(args.infile)
    path=os.path.split(__file__)[0]
    if os.path.abspath(path).endswith('.zip'):
        os.chdir(os.path.split(path)[0])
    else:
        os.chdir(path)

    if args.run:
        start_fitting(args)
    elif args.mpi:
        start_fitting(args, rank)
    elif args.gen:
        create_simulated_data(args)
    elif args.pars:
        extract_parameters(args)
    elif args.mod:
        modify_file(args)
    elif not args.run and not args.mpi:
        # Check if the application has been frozen
        if hasattr(sys, "frozen") and True:
            # Redirect all the output to log files
            log_file_path=appdirs.user_log_dir('GenX3', 'ArturGlavic')
            # Create dir if not found
            if not os.path.exists(log_file_path):
                os.makedirs(log_file_path)
            # print log_file_path
            # log_file_path = genx_gui._path + 'app_data/'
            sys.stdout=open(log_file_path+'/genx.log', 'w')
            sys.stderr=open(log_file_path+'/genx.log', 'w')
        start_interactive(args)

if __name__=="__main__":
    main()