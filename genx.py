#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys, os
try:
  import genx_gui
except ImportError:
  import genx
  genx_path=os.path.split(os.path.abspath(genx.__file__))[0]
  sys.path.insert(0, genx_path)
  import genx_gui

if __name__ == "__main__":
    # py2exe multiprocessing support
    try:
      from multiprocessing import freeze_support
      freeze_support()
    except ImportError:
      pass
    app = genx_gui.MyApp(0)
    if len(sys.argv)==2 and sys.argv[1].endswith('.gx'):
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
        frame.data_list.list_ctrl.SetItemCount(\
            frame.data_list.list_ctrl.data_cont.get_count())
        # Updating the imagelist as well
        frame.data_list.list_ctrl._UpdateImageList()    
        frame.plot_data.plot_data(frame.model.data)
        frame.paramter_grid.SetParameters(frame.model.parameters)
        app.MainLoop()
    elif len(sys.argv)>2:
      print "Usage: genx {model.gx}"
    else:
      app.MainLoop()
