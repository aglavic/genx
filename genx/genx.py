#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys, os, appdirs, argparse



def start_interactive(args):
    ''' Start genx in interactive mode (with the gui)

    :param args:
    :return:
    '''
    try:
        import genx_gui
    except ImportError:
        import genx
        genx_path = os.path.split(os.path.abspath(genx.__file__))[0]
        sys.path.insert(0, genx_path)
        import genx_gui
    if args.infile.endswith('.gx'):
        app = genx_gui.MyApp(False, 0)
        # Create the window
        app.Yield()
        frame=app.TopWindow
        # load a model on start
        import filehandling as io
        import StringIO, traceback
        from event_handlers import ShowModelErrorDialog, ShowErrorDialog, get_pages, \
                                      _post_new_model_event, set_title
        import model as modellib
        path = os.path.abspath(sys.argv[1])
        try:
            io.load_gx(path, frame.model,\
                              frame.solver_control.optimizer,\
                              frame.config)
        except modellib.IOError, e:
            ShowModelErrorDialog(frame, e.__str__())
        except Exception, e:
            outp = StringIO.StringIO()
            traceback.print_exc(200, outp)
            val = outp.getvalue()
            outp.close()
            ShowErrorDialog(frame, 'Could not open the file. Python Error:'\
                      '\n%s'%(val,))
        else:
            app.Yield()
            try:
                [p.ReadConfig() for p in get_pages(frame)]
            except Exception, e:
                outp = StringIO.StringIO()
                traceback.print_exc(200, outp)
                val = outp.getvalue()
                outp.close()
                ShowErrorDialog(frame, 'Could not read the config for the'
                                ' plots. Python Error:\n%s'%(val,))
        # Letting the plugin do their stuff...
        try:
            frame.plugin_control.OnOpenModel(None)
        except Exception, e:
            outp = StringIO.StringIO()
            traceback.print_exc(200, outp)
            val = outp.getvalue()
            outp.close()
            ShowErrorDialog(frame, 'Problems when plugins processed model.'\
                        ' Python Error:\n%s'%(val,))
        frame.main_frame_statusbar.SetStatusText('Model loaded from file',\
                                                1)
        app.Yield()
        # Post an event to update everything else
        _post_new_model_event(frame, frame.model)
        # Needs to put it to saved since all the widgets will have
        # been updated
        frame.model.saved = True
        set_title(frame)
        app.Yield()
        # Just a force update of the data_list
        frame.data_list.list_ctrl.SetItemCount(frame.data_list.list_ctrl.data_cont.get_count())
        # Updating the imagelist as well
        frame.data_list.list_ctrl._UpdateImageList()
        frame.plot_data.plot_data(frame.model.data)
        frame.paramter_grid.SetParameters(frame.model.parameters)
        app.MainLoop()
    elif args.infile == '':
      app = genx_gui.MyApp(True, 0)
      app.MainLoop()

def start_fitting(args):
    """ Function to start fitting from the command line.

    :param args:
    :return:
    """
    import time
    import model
    import diffev
    import filehandling as io

    mod = model.Model()
    config = io.Config()
    config.load_default(os.path.split(os.path.abspath(__file__))[0] + 'genx.conf')
    opt = diffev.DiffEv()

    def autosave():
        #print 'Updating the parameters'
        mod.parameters.set_value_pars(opt.best_vec)
        if args.outfile:
            io.save_gx(args.outfile, mod, opt, config)

    opt.set_autosave_func(autosave)

    print 'Loading model %s...'%args.infile
    io.load_gx(args.infile, mod, opt, config)
    io.load_opt_config(opt, config)
    # has to be used in order to save everything....
    #config.set('solver', 'save all evals', True)
    # Simulate, this will also compile the model script
    print 'Simulating model...'
    mod.simulate()

    # Sets up the fitting ...
    print 'Setting up the optimizer...'
    opt.reset()
    opt.init_fitting(mod)
    opt.init_fom_eval()

    if args.outfile:
        print 'Saving the initial model to %s'%args.outfile
        io.save_gx(args.outfile, mod, opt, config)

    # To start the fitting
    print 'Fitting starting...'
    t1 = time.time()
    opt.optimize()
    t2 = time.time()
    print 'Fitting finished!'
    print 'Time to fit: ', (t2-t1)/60., ' min'

    print 'Updating the parameters'
    mod.parameters.set_value_pars(opt.best_vec)

    if args.outfile:
        print 'Saving the fit to %s'%args.outfile
        io.save_gx(args.outfile, mod, opt, config)

    print 'Fitting successfully completed'

if __name__ == "__main__":
    # Check if the application has been frozen
    if hasattr(sys,"frozen") and True:
        # Redirect all the output to log files
        log_file_path = appdirs.user_log_dir('GenX', 'MattsBjorck')
        # Create dir if not found
        if not os.path.exists(log_file_path):
            os.makedirs(log_file_path)
        print log_file_path
        #log_file_path = genx_gui._path + 'app_data/'
        sys.stdout = open(log_file_path + '/genx.log', 'w')
        sys.stderr = open(log_file_path + '/genx.log', 'w')
    
    # py2exe multiprocessing support
    try:
      from multiprocessing import freeze_support
      freeze_support()
    except ImportError:
      pass

    parser = argparse.ArgumentParser(prog='genx')
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument('-r', '--run', action='store_true', help='run GenX fit (no gui)')
    run_group.add_argument('--mpi', action='store_true', help='run GenX fit with mpi (no gui)')
    parser.add_argument('infile', nargs='?', default='', help='The .gx file to load')
    parser.add_argument('outfile', nargs='?', default='', help='The .gx file to save into')

    args = parser.parse_args()

    if args.run:
        start_fitting(args)
    elif not args.run and not args.mpi:
        start_interactive(args)


