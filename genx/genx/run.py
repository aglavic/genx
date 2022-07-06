#!/usr/bin/env python
import logging
import multiprocessing
from threading import Thread

import appdirs
import argparse
import os
import os.path
import sys
from logging import debug, getLogger

from . import version
from .core import custom_logging
from .core.custom_logging import activate_excepthook, activate_logging, iprint, setup_system


def start_interactive(args):
    '''
    Start genx in interactive mode (with the gui)

    :param args: command line arguments evaluated with argparse.
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
        filename = args.infile
    else:
        filename = None
    debug('start GUI setup')
    app = main_window.GenxApp(filename=filename, dpi_overwrite=args.dpi_overwrite)
    debug('setup complete, start WX MainLoop')
    app.MainLoop()
    debug('leave start_interactive')


def create_simulated_data(args):
    """Function to create simulated data from the model and add it to data.y"""
    from .diffev import DiffEv
    from .model_control import ModelController
    from .core import config as io

    from scipy.stats import poisson

    io.config.load_default(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profiles', 'default.profile'))
    ctrl = ModelController(DiffEv())
    mod = ctrl.model

    iprint("Loading file: %s "%args.infile)
    ctrl.load_file(args.infile)
    # io.load_opt_config(opt, config)
    iprint("File loaded")

    iprint("Simualting..")
    mod.simulate()
    iprint("Storing simualted data")
    for data_set in mod.data:
        data_set.y_raw = poisson.rvs(data_set.y_sim)
        data_set.y_command = 'y'
        data_set.run_y_command()
        data_set.error_command = 'sqrt(where(y > 0, y, 1.0))'
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
    ctrl = ModelController(DiffEv())
    mod = ctrl.model

    iprint("Loading file: %s "%args.infile)
    ctrl.load_file(args.infile)
    iprint("File loaded")

    names, values = mod.parameters.get_sim_pars()
    index_names = mod.parameters.get_names()
    errors = [mod.parameters[index_names.index(ni)].error if mod.parameters.get_fit_state_by_name(ni)==1
              else '-' for ni in names]
    names = [ni if mod.parameters.get_fit_state_by_name(ni)!=1 else ni+'*' for ni in names]

    paramstr = '\t'.join([name for name in names])+'\n'+ \
               '\t'.join(['%f'%val for val in values])+'\n'
    if args.error:
        paramstr += '\t'.join([err for err in errors])+'\n'
    if args.outfile!='':
        iprint(f'Appending parameter values to file {args.outfile}')
        with open(args.outfile, 'a') as fout:
            # Add header
            fout.write(paramstr)
    iprint('\n'+paramstr)


def modify_file(args):
    """Modify a GenX file given command line arguments"""
    from .diffev import DiffEv
    from .model_control import ModelController
    from .core import config as io

    # Open the genx file
    io.config.load_default(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profiles', 'default.profile'))
    ctrl = ModelController(DiffEv())
    mod = ctrl.model

    iprint("Loading file: %s "%args.infile)
    ctrl.load_file(args.infile)
    iprint("File loaded")

    if args.datafile:
        datafile = os.path.abspath(args.datafile)
        if 0>args.data_set or args.data_set>=len(mod.data):
            iprint("The selected data set does not exist - select one between 0 and %d"%(len(mod.data)-1))
            return
        iprint('Loading dataset %s into data set %d'%(datafile, args.data_set))
        mod.data[args.data_set].loadfile(datafile)

        if args.outfile:
            iprint('Saving the fit to %s'%args.outfile)
            ctrl.save_file(args.outfile)

    elif args.save_datafile:
        save_datafile = os.path.abspath(args.save_datafile)
        iprint("Simualting..")
        mod.simulate()
        if 0>args.data_set or args.data_set>=len(mod.data):
            iprint("The selected data set does not exist - select one between 0 and %d"%(len(mod.data)-1))
            return
        iprint('Exporting data set %d into ASCII file %s'%(args.data_set, save_datafile))
        mod.data[args.data_set].save_file(save_datafile)


def set_numba_single():
    config_path = os.path.abspath(appdirs.user_data_dir('GenX3', 'ArturGlavic'))
    cache_dir = os.path.join(config_path, 'numba_cache_single_cpu')

    debug('Setting numba JIT compilation to single CPU')
    import numba
    numba.config.CACHE_DIR = cache_dir
    old_jit = numba.jit

    def jit(*args, **opts):
        opts['parallel'] = False
        return old_jit(*args, **opts)

    numba.jit = jit
    numba.GENX_OVERWRITE_SINGLE = True
    try:
        numba.set_num_threads(1)
    except AttributeError:
        pass


class InputThread(Thread):
    def __init__(self):
        Thread.__init__(self, daemon=True)
        self.stop_fit = False

    def run(self):
        # concole input has to be queried differently for non-UNIX systems
        if sys.platform.startswith('win'):
            self.run_windows()
        else:
            self.run_unix()

    def run_windows(self):
        import msvcrt, time
        while not self.stop_fit:
            if msvcrt.kbhit():
                debug('Key event received')
                key = msvcrt.getwch()
                if 'q' in key:
                    self.stop_fit = True
            time.sleep(0.1)

    def run_unix(self):
        import fcntl, selectors, time
        # set sys.stdin non-blocking
        orig_fl = fcntl.fcntl(sys.stdin, fcntl.F_GETFL)
        fcntl.fcntl(sys.stdin, fcntl.F_SETFL, orig_fl | os.O_NONBLOCK)

        m_selector = selectors.DefaultSelector()
        m_selector.register(sys.stdin, selectors.EVENT_READ)
        while True:
            (k, evt)=m_selector.select()[0]
            res = k.fileobj.read()
            if 'q' in res:
                self.stop_fit = True
                break
        m_selector.close()

def start_fitting(args, rank=0):
    """
    Function to start fitting from the command line.
    """
    import time
    from .model_control import ModelController
    from .core import config as io
    from .core.console import calc_errorbars, setup_console

    # Open the genx file
    io.config.load_default(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profiles', 'default.profile'))
    if not args.bumps:
        from .diffev import DiffEv
        ctrl = ModelController(DiffEv())
        mod = ctrl.model
        opt: DiffEv = ctrl.optimizer
        set_optimiser_pars = set_diffev_pars
    else:
        from .bumps_optimizer import BumpsOptimizer
        ctrl = ModelController(BumpsOptimizer())
        mod = ctrl.model
        opt: BumpsOptimizer = ctrl.optimizer
        set_optimiser_pars = set_bumps_pars

    if rank==0:
        setup_console(ctrl, args.error, args.outfile, use_curses=(args.use_curses and not args.mpi))

    if (args.mpi or args.pr>0) and not args.disable_numba:
        try:
            set_numba_single()
        except ImportError:
            pass

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
    set_optimiser_pars(opt, args)
    if rank==0:
        iprint('Setting up the optimizer...')
        iprint(opt)

    if args.outfile and rank==0:
        iprint('Saving the initial model to %s'%args.outfile)
        ctrl.save_file(args.outfile)

    # To start the fitting
    if rank==0:
        iprint('Fitting starting...')
        t1 = time.time()
    # print opt.use_mpi, opt.use_parallel_processing
    opt.start_fit(mod)
    if rank==0:
        inp = InputThread()
        inp.start()
    else:
        class Inp(): stop_fit = False
        inp = Inp()
    while opt.is_running():
        try:
            time.sleep(0.1)
            if inp.stop_fit:
                opt.stop = True
        except KeyboardInterrupt:
            iprint('KeyboardInterrupt, trying to stop fit.')
            opt.stop = True

    if rank==0:
        t2 = time.time()
        iprint('Fitting finished!')
        iprint('Time to fit: ', (t2-t1)/60., ' min')

    if rank==0:
        iprint('Updating the parameters')
        mod.parameters.set_value_pars(opt.best_vec)

    if args.outfile and rank==0:
        if args.error:
            iprint('Calculating errorbars')
            calc_errorbars(mod, opt)
        iprint('Saving the fit to %s'%args.outfile)
        # opt.set_use_mpi(False)
        ctrl.save_file(args.outfile)

    if rank==0:
        iprint('Fitting successfully completed')


def set_diffev_pars(optimiser, args):
    """
    Sets the optimiser parameters from args
    """
    if args.pr:
        optimiser.set_processes(args.pr)
        optimiser.set_use_parallel_processing(True)

    if args.mpi:
        optimiser.set_use_mpi(True)
    else:
        optimiser.set_use_mpi(False)

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

    if args.var>=0:
        optimiser.opt.min_parameter_spread = args.var

    if args.km>=0:
        optimiser.set_km(args.km)

    if args.kr>=0:
        optimiser.set_kr(args.kr)


def set_bumps_pars(optimiser, args):
    if args.pr:
        optimiser.opt.use_parallel_processing = True
        optimiser.opt.parallel_processes = args.pr

def compile_numba(cache_dir=None):
    try:
        # perform a compilation of numba functions with console feedback
        import numba
        if cache_dir:
            numba.config.CACHE_DIR = cache_dir
        elif hasattr(numba.config, 'CACHE_DIR'):
            import appdirs
            config_path = os.path.abspath(appdirs.user_data_dir('GenX3', 'ArturGlavic'))
            # make sure to use a user directory for numba cache
            numba.config.CACHE_DIR = os.path.join(config_path, 'numba_cache')

        real_jit = numba.jit

        class UpdateJit:
            update_counter = 1

            def __call__(self, *args, **opts):
                print(f'compiling numba functions {self.update_counter}/21')
                self.update_counter += 1
                return real_jit(*args, **opts)

        print('Starting to compile numba functions..')
        numba.jit = UpdateJit()
        from .models.lib import paratt_numba, neutron_numba, instrument_numba, offspec, surface_scattering
        numba.jit = real_jit
    except Exception as e:
        print('An exception occured when trying to compile the numba functions:')
        print(e)
        return 1
    return 0

def main():
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn')
    # Attempt to load mpi:
    try:
        from mpi4py import MPI
    except ImportError:
        rank = 0
        __mpi__ = False
    else:
        __mpi__ = bool(MPI.COMM_WORLD.Get_size()>1)
        rank = MPI.COMM_WORLD.Get_rank()

    parser = argparse.ArgumentParser(description="GenX %s, fits data to a model."%version.__version__,
                                     epilog="For support, manuals and bug reporting see http://genx.sf.net"
                                     )
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument('-r', '--run', action='store_true', help='run GenX fit (no gui)')
    if __mpi__:
        run_group.add_argument('--mpi', action='store_true', help='run GenX fit with mpi (no gui)')
    run_group.add_argument('-g', '--gen', action='store_true', help='generate data.y with poisson noise added')
    run_group.add_argument('--pars', action='store_true', help='extract the parameters from the infile')
    run_group.add_argument('--mod', action='store_true', help='modify the GenX file')
    opt_group = parser.add_argument_group('optimization arguments')
    opt_group.add_argument('--pr', type=int, default=0, help='Number of processes used in parallel fitting.')
    opt_group.add_argument('--cs', type=int, default=0, help='Chunk size used for parallel processing.')
    opt_group.add_argument('--mgen', type=int, default=0, help='Maximum number of generations that is used in a fit')
    opt_group.add_argument('--pops', type=int, default=0, help='Population size - number of individuals.')
    opt_group.add_argument('--asi', type=int, default=0, help='Auto save interval (generations).')
    opt_group.add_argument('--km', type=float, default=-1, help='Mutation constant (float 0 < km < 1)')
    opt_group.add_argument('--kr', type=float, default=-1, help='Cross over constant (float 0 < kr < 1)')
    opt_group.add_argument('-s', '--esave', action='store_true', help='Force save evals to gx file.')
    opt_group.add_argument('-e', '--error', action='store_true', help='Calculate/export error bars.')
    opt_group.add_argument('--var', type=float, default=-1,
                           help='Minimum relative parameter variation to stop the fit (%%)')
    opt_group.add_argument('--bumps', action='store_true',
                           help='Use Bumps DREAM optimizer instead of GenX Differential Evolution')
    data_group = parser.add_argument_group('data arguments')
    data_group.add_argument('-d', dest='data_set', type=int, default=0,
                            help='Active data set to act upon. Index starting at 0.')
    data_group.add_argument('--load', dest='datafile',
                            help='Load file into active data set. Index starting at 0.')
    data_group.add_argument('--export', dest='save_datafile',
                            help='Save active data set to file. Index starting at 0.')
    data_group = parser.add_argument_group('startup options')
    data_group.add_argument('-l', '--logfile', dest='logfile', default=None, type=str,
                            help='Output debug information to logfile.')
    data_group.add_argument('--debug', dest='debug', default=False, action="store_true",
                            help='Show additional debug information on console/logfile')
    data_group.add_argument('--dpi-scale', dest='dpi_overwrite', default=None, type=float,
                            help='Overwrite the detection of screen dpi scaling factor (=72/dpi)')
    data_group.add_argument('--no-curses', dest='use_curses', default=True, action="store_false",
                            help='Disable Curses interactive console interface for command line '
                                 'fitting on UNIX systems.')
    data_group.add_argument('--disable-nb', dest='disable_numba', default=False, action="store_true",
                            help='Disable the use of numba JIT compiler')
    data_group.add_argument('--nb1', dest='numba_single', default=False, action="store_true",
                            help='Compile numba JIT functions without parallel computing support (use one core only). '
                                 'Caching in this case is done in a different user directory.')
    data_group.add_argument('--compile-nb', dest='compile_nb', default=False, action="store_true",
                            help='Perform a first-/recompilation of the numba modules and exit')

    parser.add_argument('infile', nargs='?', default='',
                        help='The .gx or .hgx file to load or .ort file to use as basis for model')
    parser.add_argument('outfile', nargs='?', default='', help='The .gx  or hgx file to save into')

    args = parser.parse_args()
    if not __mpi__:
        args.mpi = False

    if args.compile_nb:
        sys.exit(compile_numba())

    if args.run or args.mpi or args.pars or args.mod:
        # make sure at least info-messages are shown (default is warning)
        custom_logging.CONSOLE_LEVEL = min(logging.INFO, custom_logging.CONSOLE_LEVEL)
    if rank>0:
        custom_logging.CONSOLE_LEVEL = logging.WARNING
    setup_system()

    if args.logfile:
        if __mpi__:
            lfbase, lfend = args.logfile.rsplit('.', 1)
            activate_logging(f'{lfbase}_{rank:02}.{lfend}')
        else:
            activate_logging(args.logfile)
    debug("Arguments from parser: %s"%args)
    if not args.outfile:
        args.outfile = args.infile
    args.outfile = os.path.abspath(args.outfile)
    if args.infile:
        args.infile = os.path.abspath(args.infile)

    if args.disable_numba:
        debug('disable numba')
        # set numba flag
        from genx.models import lib as modellib
        modellib.USE_NUMBA = False
    elif args.numba_single:
        set_numba_single()

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
            log_file_path = appdirs.user_log_dir('GenX3', 'ArturGlavic')
            # Create dir if not found
            if not os.path.exists(log_file_path):
                os.makedirs(log_file_path)
            # print log_file_path
            # log_file_path = genx_gui._path + 'app_data/'
            sys.stdout = open(log_file_path+'/genx.log', 'w')
            sys.stderr = open(log_file_path+'/genx.log', 'w')
        start_interactive(args)


if __name__=="__main__":
    main()
