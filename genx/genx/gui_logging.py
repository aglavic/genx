'''
  Module used to setup the default GUI logging and messaging system.
  The system contains on a python logging based approach with logfile,
  console output and GUI output dependent on startup options and
  message logLevel.
'''

import sys
import atexit
import logging
import traceback
import inspect
# from numpy import seterr, seterrcall
from .version import version as str_version

# default options used if nothing is set in the configuration
CONSOLE_LEVEL, FILE_LEVEL, GUI_LEVEL=logging.WARNING, logging.INFO, logging.INFO

# set log levels according to options
if 'pdb' in list(sys.modules.keys()) or 'pydevd' in list(sys.modules.keys()):
  # if common debugger modules have been loaded, assume a debug run
  CONSOLE_LEVEL, FILE_LEVEL, GUI_LEVEL=logging.INFO, logging.DEBUG, logging.INFO
elif '--debug' in sys.argv:
  sys.argv.remove('--debug')
  CONSOLE_LEVEL, FILE_LEVEL, GUI_LEVEL=logging.DEBUG, logging.DEBUG, logging.INFO

def excepthook_overwrite(*exc_info):
  logging.critical('python error', exc_info=exc_info)

def ip_excepthook_overwrite(self, etype, value, tb, tb_offset=None):
  logging.critical('python error', exc_info=(etype, value, tb))

def goodby():
  logging.info('*** GenX %s Logging ended ***'%str_version)

def iprint(*objects, sep=None, end=None, file=None, flush=False):
  """
  A logging function that behaves like print but uses logging.info.
  """
  if sep is None:
    sep=' '
  if end is None:
    end='\n'
  logging.info(sep.join(map(str, objects))+end)
  

class NumpyLogger(logging.getLoggerClass()):
  '''
    A logger that makes sure the actual function definition filename, lineno and function name
    is used for logging numpy floating point errors, not the numpy_logger function.
  '''

  if sys.version_info[0:2]>=(3, 2): #sinfo was introduced in python 3.2
    def makeRecord(self, name, lvl, fn, lno, msg, args, exc_info, func=None, extra=None, sinfo=None):
      curframe=inspect.currentframe()
      calframes=inspect.getouterframes(curframe, 2)
      # stack starts with:
      # (this method, debug call, debug call rootlogger, numpy_logger, actual function, ...)
      ignore, fname, lineno, func, ignore, ignore=calframes[4]
      return logging.getLoggerClass().makeRecord(self, name, lvl, fname, lineno,
                                   msg, args, exc_info, func=func, extra=extra, sinfo=sinfo)
  else:
    def makeRecord(self, name, lvl, fn, lno, msg, args, exc_info, func=None, extra=None):
      curframe=inspect.currentframe()
      calframes=inspect.getouterframes(curframe, 2)
      # stack starts with:
      # (this method, debug call, debug call rootlogger, numpy_logger, actual function, ...)
      ignore, fname, lineno, func, ignore, ignore=calframes[4]
      return logging.getLoggerClass().makeRecord(self, name, lvl, fname, lineno,
                                                 msg, args, exc_info, func=func, extra=extra)

nplogger=None
def numpy_logger(err, flag):
  nplogger.debug('numpy floating point error encountered (%s)'%err)

def setup_system():
  logger=logging.getLogger()#logging.getLogger('genx')
  logger.setLevel(min(FILE_LEVEL, CONSOLE_LEVEL, GUI_LEVEL))
  if not sys.platform.startswith('win'):
    # no console logger for windows (win32gui)
    console=logging.StreamHandler(sys.__stdout__)
    formatter=logging.Formatter('%(levelname) 7s: %(message)s')
    console.setFormatter(formatter)
    console.setLevel(CONSOLE_LEVEL)
    logger.addHandler(console)

  logfile=logging.FileHandler('genx.log', 'w')
  formatter=logging.Formatter('[%(levelname)s] - %(asctime)s - %(filename)s:%(lineno)i:%(funcName)s %(message)s', '')
  logfile.setFormatter(formatter)
  logfile.setLevel(FILE_LEVEL)
  logger.addHandler(logfile)

  logging.info('*** GenX %s Logging started ***'%str_version)

  # define numpy warning behavior
  # global nplogger
  # old_class=logging.getLoggerClass()
  # logging.setLoggerClass(NumpyLogger)
  # nplogger=logging.getLogger('numpy')
  # nplogger.setLevel(logging.DEBUG)
  # null_handler=logging.StreamHandler(StringIO())
  # null_handler.setLevel(logging.CRITICAL)
  # nplogger.addHandler(null_handler)
  # logging.setLoggerClass(old_class)
  # seterr(divide='call', over='call', under='ignore', invalid='call')
  # seterrcall(numpy_logger)

  # write information on program exit
  # sys.excepthook=excepthook_overwrite
  atexit.register(goodby)

class QtHandler(logging.Handler):
  '''
  A logging Handler to be used by a GUI widget to show the data.
  '''
  max_items=1e5
  info_limit=logging.INFO
  warn_limit=logging.WARNING

  def __init__(self, main_window):
    logging.Handler.__init__(self, level=GUI_LEVEL)
    self.logged_items=[]
    self.reported_bugs=[]
    self.main_window=main_window

  def emit(self, record):
    self.logged_items.append(record)
    # make sure the buffer doesn't get infinitly large
    if len(self.logged_items)>self.max_items:
      self.logged_items.pop(0)
    if record.levelno<=self.info_limit:
      self.show_info(record)
    elif record.levelno<=self.warn_limit:
      self.show_warning(record)
    else:
      self.show_error(record)

  def show_info(self, record):
    msg=record.msg
    if record.levelno!=logging.INFO:
      msg=record.levelname+': '+msg
    self.main_window.ui.statusbar.showMessage(msg, 5000.)
    # make sure the message gets displayed during method executions
    self.main_window.ui.statusbar.update()

  def show_warning(self, record):
    '''
      Warning messages display a dialog to the user.
    '''
    from PyQt4.QtGui import QMessageBox
    if record.exc_info:
      msg='%s\nError Message:\n%s:   %s'%(record.msg, record.exc_info[0].__name__, record.exc_info[1])
      QMessageBox.warning(self.main_window, 'QuickNXS '+record.levelname, msg)
    else:
      QMessageBox.warning(self.main_window, 'QuickNXS '+record.levelname, record.msg)

  def show_error(self, record):
    '''
      More urgent error messages allow to send a bug report.
    '''
    from PyQt4.QtGui import QMessageBox
    from PyQt4.QtCore import Qt
    mbox=QMessageBox(self.main_window)
    mbox.setIcon(QMessageBox.Critical)
    mbox.setTextFormat(Qt.RichText)
    mbox.setInformativeText('Do you want to send the logfile to software support?')
    mbox.setStandardButtons(QMessageBox.Yes|QMessageBox.No)
    mbox.setDefaultButton(QMessageBox.No)
    mbox.setWindowTitle('QuickNXS Error')

    if record.exc_info:
      tb=traceback.format_exception(*record.exc_info)
      message='\n'.join(tb)
      mbox.setDetailedText(message)
      if message in self.reported_bugs:
        mbox.setText('An unexpected error has occurred: <b>%s</b><br />&nbsp;&nbsp;&nbsp;&nbsp;<i>%s</i>: %s'%(
                                    record.msg,
                                    record.exc_info[0].__name__,
                                    record.exc_info[1]))
        mbox.setStandardButtons(QMessageBox.Close)
        mbox.setInformativeText('This error has already been reported to the support team.')
      else:
        mbox.setText('An unexpected error has occurred: <b>%s</b><br />&nbsp;&nbsp;&nbsp;&nbsp;<i>%s</i>: %s'%(
                                    record.msg,
                                    record.exc_info[0].__name__,
                                    record.exc_info[1])+
                   '<br /><br />If you know what triggered the exception please select "No"'+
                   ' and activate full logging from the "Advanced->Debug" menu, trigger the error again'+
                   ' and send a full report.')
    else:
      message=''
      mbox.setText('An unexpected error has occurred: <br />&nbsp;&nbsp;<b>%s</b>'%record.msg)
    result=mbox.exec_()
    if result==QMessageBox.Yes:
      logging.info('Sending mail')
      try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from getpass import getuser

        msg=MIMEMultipart()
        msg['Subject']='QuickNXS error report'
        msg['From']='%s@ornl.gov'%getuser()
        msg['To']=misc.ADMIN_EMAIL
        text='This is an automatic bugreport from QuickNXS\n\n%s'%record.msg
        if record.exc_info:
          text+='\n\n'+message
        text+='\n'
        msg.preamble=text
        msg.attach(MIMEText(text))

        mitem=MIMEText(open(paths.LOG_FILE, 'r').read(), 'log')
        mitem.add_header('Content-Disposition', 'attachment', filename='debug.log')
        msg.attach(mitem)

        smtp=smtplib.SMTP(misc.SMTP_SERVER)
        smtp.sendmail(msg['From'], msg['To'].split(','), msg.as_string())
        smtp.quit()
        logging.info('Mail sent')
      except:
        logging.warning('problem sending the mail', exc_info=True)
      else:
        # after successful email notification the same error is not reported twice
        self.reported_bugs.append(message)

def install_gui_handler(main_window):
  logging.root.addHandler(QtHandler(main_window))
  # config errors will be raised before GUI is initialized,
  # make sure they are displayed to the user
  for text, exc_info in proxy._PARSE_ERRORS:
    logging.warning(text, exc_info=exc_info)
