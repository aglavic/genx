#!/usr/bin/env python

import sys, os, appdirs, argparse
import os.path
from logging import debug

from . import version, model
from .gui_logging import setup_system, iprint, activate_logging, activate_excepthook

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

    from genx import genx_gui
    if args.infile.endswith('.gx') or args.infile.endswith('.hgx'):
        debug('start GUI setup with file to load')
        app=genx_gui.MyApp(False, 0)
        # Create the window
        app.Yield()
        frame=app.TopWindow
        # load a model on start
        from genx import filehandling as io
        import traceback
        from io import StringIO
        from genx.event_handlers import ShowModelErrorDialog, ShowErrorDialog, get_pages, \
            _post_new_model_event, set_title
        from genx import model as modellib
        path=os.path.abspath(args.infile)
        try:
            io.load_file(path, frame.model, frame.solver_control.optimizer, frame.config)
        except modellib.IOError as e:
            ShowModelErrorDialog(frame, e.__str__())
        except Exception as e:
            outp=StringIO()
            traceback.print_exc(200, outp)
            val=outp.getvalue()
            outp.close()
            ShowErrorDialog(frame, 'Could not open the file. Python Error:' \
                                   '\n%s'%(val,))
        else:
            app.Yield()
            try:
                [p.ReadConfig() for p in get_pages(frame)]
            except Exception as e:
                outp=StringIO()
                traceback.print_exc(200, outp)
                val=outp.getvalue()
                outp.close()
                ShowErrorDialog(frame, 'Could not read the config for the'
                                       ' plots. Python Error:\n%s'%(val,))
        # Letting the plugin do their stuff...
        try:
            frame.plugin_control.OnOpenModel(None)
        except Exception as e:
            outp=StringIO()
            traceback.print_exc(200, outp)
            val=outp.getvalue()
            outp.close()
            ShowErrorDialog(frame, 'Problems when plugins processed model.' \
                                   ' Python Error:\n%s'%(val,))
        frame.main_frame_statusbar.SetStatusText('Model loaded from file',
                                                 1)
        app.Yield()
        # Post an event to update everything else
        _post_new_model_event(frame, frame.model)
        # Needs to put it to saved since all the widgets will have
        # been updated
        frame.model.saved=True
        set_title(frame)
        app.Yield()
        # Just a force update of the data_list
        frame.data_list.list_ctrl.SetItemCount(frame.data_list.list_ctrl.data_cont.get_count())
        # Updating the imagelist as well
        frame.data_list.list_ctrl._UpdateImageList()
        frame.plot_data.plot_data(frame.model.data)
        frame.paramter_grid.SetParameters(frame.model.parameters)
        debug('setup complete, start WX MainLoop')
        app.MainLoop()
    elif args.infile=='':
        debug('start GUI setup')
        app=genx_gui.MyApp(True, 0)
        debug('setup complete, start WX MainLoop')
        app.MainLoop()
    else:
        iprint('Wrong file ending on infile, should be .gx or .hgx. Exiting.')
    debug('leave start_interactive')

def calc_errorbars(config, mod, opt):
    error_values=[]
    fom_error_bars_level=config.get_float('solver', 'errorbar level')
    n_elements=len(opt.start_guess)
    for index in range(n_elements):
        # calculate the error
        # TODO: Check the error bar buisness again and how to treat Chi2
        (error_low, error_high)=opt.calc_error_bar(index, fom_error_bars_level)
        error_values.append('(%.3e, %.3e,)'%(error_low, error_high))
    mod.parameters.set_error_pars(error_values)

def create_simulated_data(args):
    """Function to create simulated data from the model and add it to data.y"""

    from genx import model
    from genx import diffev
    from genx import filehandling as io

    from scipy.stats import poisson

    mod=model.Model()
    config=io.Config()
    config.load_default(os.path.split(os.path.abspath(__file__))[0]+'genx.conf')
    opt=diffev.DiffEv()

    iprint("Loading file: %s "%args.infile)
    io.load_file(args.infile, mod, opt, config)
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
    io.save_file(args.outfile, mod, opt, config)

def extract_parameters(args):
    """Extracts the parameters to outfile"""

    from genx import model
    from genx import diffev
    from genx import filehandling as io

    # Open the genx file
    mod=model.Model()
    config=io.Config()
    config.load_default(os.path.split(os.path.abspath(__file__))[0]+'genx.conf')
    opt=diffev.DiffEv()

    iprint("Loading file: %s "%args.infile)
    io.load_file(args.infile, mod, opt, config)
    iprint("File loaded")

    names, values=mod.parameters.get_sim_pars()

    if args.outfile=='':
        outfile=sys.stdout
        fout=open(outfile, 'w')
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
    from genx import model
    from genx import diffev
    from genx import filehandling as io

    # Open the genx file
    mod=model.Model()
    config=io.Config()
    config.load_default(os.path.split(os.path.abspath(__file__))[0]+'genx.conf')
    opt=diffev.DiffEv()

    iprint("Loading file: %s "%args.infile)
    io.load_file(args.infile, mod, opt, config)
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
            io.save_file(args.outfile, mod, opt, config)

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

    :param args:
    :return:
    """
    import time
    from genx import model
    from genx import diffev
    from genx import filehandling as io

    mod=model.Model()
    config=io.Config()
    config.load_default(os.path.split(os.path.abspath(__file__))[0]+'genx.conf')
    opt=diffev.DiffEv()

    if rank==0:
        def autosave():
            # print 'Updating the parameters'
            mod.parameters.set_value_pars(opt.best_vec)
            if args.error:
                iprint("Calculating error bars")
                calc_errorbars(config, mod, opt)
            if args.outfile:
                iprint("Saving to %s"%args.outfile)
                io.save_file(args.outfile, mod, opt, config)

        opt.set_autosave_func(autosave)

    if rank==0:
        iprint('Loading model %s...'%args.infile)
    io.load_file(args.infile, mod, opt, config)
    io.load_opt_config(opt, config)
    # has to be used in order to save everything....
    if args.esave:
        config.set('solver', 'save all evals', True)
    # Simulate, this will also compile the model script
    if rank==0:
        iprint('Simulating model...')
    mod.simulate()

    # Sets up the fitting ...
    if rank==0:
        iprint('Setting up the optimizer...')
    set_optimiser_pars(opt, args)
    opt.reset()
    opt.init_fitting(mod)
    opt.init_fom_eval()
    opt.set_sleep_time(0.0)

    if args.outfile and rank==0:
        iprint('Saving the initial model to %s'%args.outfile)
        io.save_file(args.outfile, mod, opt, config)

    # To start the fitting
    if rank==0:
        iprint('Fitting starting...')
        t1=time.time()
    # print opt.use_mpi, opt.use_parallel_processing
    opt.optimize()
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
        io.save_file(args.outfile, mod, opt, config)

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
    setup_system()
    import multiprocessing

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
    if args.logfile:
        activate_logging(args.logfile)
    debug("Arguments from parser: %s"%args)
    if not args.outfile:
        args.outfile=args.infile
    args.outfile=os.path.abspath(args.outfile)
    if args.infile:
        args.infile=os.path.abspath(args.infile)
    path=os.path.split(model.__file__)[0]
    if os.path.abspath(path).endswith('.zip'):
        os.chdir(os.path.split(path)[0])
    else:
        os.chdir(path)

    if not __mpi__:
        args.mpi=False

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