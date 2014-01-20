# -*- encoding: utf-8 -*-
'''
  Script used for setup and installation purpose. 
  If all works the right way this should test the system environment for all dependencies and create source and binary distributions.
  
  The script can create exe stand alone programs under windows, but py2app doesn't word until now.
'''

import sys, os

try:
  # Use easy setup to ensure dependencies
  import ez_setup
  ez_setup.use_setuptools()
except ImportError:
  pass

from glob import glob
import subprocess
import version

__name__='GenX'
__author__ = "Artur Glavic"
__copyright__ = "Copyright 2008-2011"
__license__ = "GPL v3"
__version__ = version.version.split()[0]
__email__ = "a.glavic@fz-juelich.de"
__author_email__ = __email__
__url__ = "http://genx.sourceforge.net/"
__description__='''X-ray and Neutron reflectivity simulation and fitting.'''

def rec_glob(path):
  # recursive list of files
  items=glob(path+'/*')
  output=[]
  for name in items:
    if os.path.isdir(name):
      output+=rec_glob(name)
    else:
      output.append(name)
  return output

__scripts__=['scripts/genx']
__py_modules__=[]
__package_dir__={'genx': '.'}
__packages__=['genx', 
    'genx.plugins', 'genx.plugins.add_ons',  'genx.plugins.add_ons.help_modules', 
                'genx.plugins.data_loaders', 'genx.plugins.data_loaders.help_modules', 
                'genx.models', 'genx.models.lib', 'genx.lib']
__package_data__={
                  'genx': ['genx.conf', 'examples/*.*', 'LICENSE.txt', 'changelog.txt', 'profiles/*.*'], 
                  'genx.models': ['databases/*.*', 'databases/f1f2_cxro/*.*'], 
                  }
__data_files__=[]

if "py2exe" in sys.argv:
  import py2exe
  import matplotlib  
  __data_files__+=matplotlib.get_py2exe_datafiles()
  __options__={ 
                #"setup_requires": ['py2exe'], 
                #"console": [ "__init__.py"], # set the executable for py2exe
                "windows": [ {
                            "script": "genx.py",
                            "icon_resources": [(1, "windows_build/genx.ico"), (2, "windows_build/genx_file.ico")]
                            } ], # executable for py2exe is windows application            
                "options": {  "py2exe": {
                              "includes": "numpy, matplotlib, StringIO, traceback, thread, multiprocessing",
                              "optimize": 1, # Keep docstring (e.g. Shell usage)
                              "skip_archive": True, # setting not to move compiled code into library.zip file
                              'packages':'plugins, models, wx, matplotlib',
                              "dll_excludes": ["MSVCP90.dll", 'libglade-2.0-0.dll'], 
                              'excludes': ['_gtkagg', '_tkagg', 'gtk', 'glib', 'gobject'], 
                             }, 
                           }
              }
elif"py2app" in sys.argv:
  import py2app
  import matplotlib  
  __data_files__+=matplotlib.get_py2exe_datafiles()
  
  __options__ = dict(
         setup_requires=['py2app'],
         app=["genx.py"],
         # Cross-platform applications generally expect sys.argv to
         # be used for opening files.
         options=dict(py2app=dict(argv_emulation = True, 
                                  packages = ['matplotlib', 'numpy', 'plugins', 'models','wx',],
                                  includes = ['genx_gui'], 
                                  #resources = ['genx.conf','profiles'],
                                  excludes = ['_gtkagg', '_tkagg', 'gtk', 'glib', 'gobject'],
                                  iconfile = 'mac_build/genx.icns',
                                  plist = 'mac_build/Info.plist',
                                  ),
                    )
     )  

else:
  __options__={#"setup_requires":[], 
                }

__requires__=['numpy', 'matplotlib']
from distutils.core import setup, Extension

# extensions modules written in C
__extensions_modules__=[]

if 'install' not in sys.argv:
  # Remove MANIFEST befor distributing to be sure no file is missed
  if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

#### Run the setup command with the selected parameters ####
setup(name=__name__,
      version=__version__,
      description=__description__,
      author=__author__,
      author_email=__email__,
      url=__url__,
      scripts=__scripts__, 
      py_modules=__py_modules__, 
      ext_modules=__extensions_modules__, 
      packages=__packages__, 
      package_dir=__package_dir__, 
      package_data=__package_data__,
      data_files=__data_files__, 
      requires=__requires__, #does not do anything
      **__options__
     )

# If binary distribution has been created rename it and create .deb package, too.
# The .deb only works on similar systems so we use python2.6 and python2.7 folders
# as these are the versions used in the latest ubuntu releases
if ('bdist' in sys.argv):
  print "Moving distribution files..."
  from glob import glob
  os.chdir('dist')
  os.rename(__name__+'-'+__version__+'-1.noarch.rpm', __name__+'-'+__version__+'.rpm')
  os.remove(__name__+'-'+__version__+'-1.src.rpm')
  print "Creating debian folder..."
  subprocess.Popen(['fakeroot', 'alien', '-k', '-g', __name__+'-'+__version__+'.rpm'], shell=False, 
                   stderr=subprocess.PIPE,stdout=subprocess.PIPE).communicate()
  # creating menu entries
  os.mkdir(__name__+'-'+__version__+'/usr/share/')
  os.mkdir(__name__+'-'+__version__+'.orig/usr/share/')
  os.mkdir(__name__+'-'+__version__+'/usr/share/applications/')
  os.mkdir(__name__+'-'+__version__+'.orig/usr/share/applications/')
  subprocess.Popen(['cp']+ glob('../debian_build/*.desktop')+[__name__+'-'+__version__+'/usr/share/applications/'], 
                   shell=False, stderr=subprocess.PIPE,stdout=subprocess.PIPE).communicate()
  subprocess.Popen(['cp']+glob('../debian_build/*.desktop')+[__name__+'-'+__version__+'.orig/usr/share/applications/'], 
                   shell=False, stderr=subprocess.PIPE,stdout=subprocess.PIPE).communicate()
  # Icons
  #   menu
  os.mkdir(__name__+'-'+__version__+'/usr/share/pixmaps')
  os.mkdir(__name__+'-'+__version__+'.orig/usr/share/pixmaps')
  subprocess.Popen(['cp']+ glob('../debian_build/*.xpm')+[__name__+'-'+__version__+'/usr/share/pixmaps/'], 
                   shell=False, stderr=subprocess.PIPE,stdout=subprocess.PIPE).communicate()
  subprocess.Popen(['cp']+glob('../debian_build/*.xpm')+[__name__+'-'+__version__+'.orig/usr/share/pixmaps/'], 
                   shell=False, stderr=subprocess.PIPE,stdout=subprocess.PIPE).communicate()
  #   mime
  os.mkdir(__name__+'-'+__version__+'/tmp')
  os.mkdir(__name__+'-'+__version__+'.orig/tmp')
  os.mkdir(__name__+'-'+__version__+'/tmp/genx_icons')
  os.mkdir(__name__+'-'+__version__+'.orig/tmp/genx_icons')
  for icon in glob('../debian_build/genx_*.png'):
    subprocess.Popen(['cp', icon, __name__+'-'+__version__+'/tmp/genx_icons/'], 
                     shell=False, stderr=subprocess.PIPE,stdout=subprocess.PIPE).communicate()
    subprocess.Popen(['cp', icon, __name__+'-'+__version__+'.orig/tmp/genx_icons/'], 
                     shell=False, stderr=subprocess.PIPE,stdout=subprocess.PIPE).communicate()
  # creating mime types
  os.mkdir(__name__+'-'+__version__+'/usr/share/mime/')
  os.mkdir(__name__+'-'+__version__+'/usr/share/mime/packages/')
  os.mkdir(__name__+'-'+__version__+'.orig/usr/share/mime/')
  os.mkdir(__name__+'-'+__version__+'.orig/usr/share/mime/packages/')
  subprocess.Popen(['cp']+ glob('../debian_build/*.xml')+[__name__+'-'+__version__+'/usr/share/mime/packages/'], 
                   shell=False, stderr=subprocess.PIPE,stdout=subprocess.PIPE).communicate()
  subprocess.Popen(['cp']+glob('../debian_build/*.xml')+[__name__+'-'+__version__+'.orig/usr/share/mime/packages/'], 
                   shell=False, stderr=subprocess.PIPE,stdout=subprocess.PIPE).communicate()
  os.chdir(__name__+'-'+__version__)
  # debian control file
  deb_con=open('debian/control', 'w')
  deb_con.write(open('../../debian_build/control', 'r').read())
  deb_con.close()
  # post install and remove scripts (e.g. adding mime types)
  deb_tmp=open('debian/postinst', 'w')
  deb_tmp.write(open('../../debian_build/postinst', 'r').read())
  deb_tmp.close()
  deb_tmp=open('debian/postrm', 'w')
  deb_tmp.write(open('../../debian_build/postrm', 'r').read())
  deb_tmp.close()
  # python 2.7
  print "Packaging for debian (python2.7)..."
  subprocess.Popen(['dpkg-buildpackage', '-i.*', '-I', '-rfakeroot', '-us', '-uc'], shell=False, 
                   stderr=subprocess.STDOUT, stdout=open('../last_package.log', 'w')
                   ).communicate()
  os.chdir('..')
  os.rename((__name__+'_'+__version__).lower()+'-1_all.deb', __name__+'-'+__version__+'_natty.deb')
  # python 2.6
  subprocess.Popen(['cp']+ glob('../debian_build/*.desktop')+[__name__+'-'+__version__+'/usr/share/applications/'], 
                   shell=False, stderr=subprocess.PIPE,stdout=subprocess.PIPE).communicate()
  subprocess.Popen(['cp']+glob('../debian_build/*.desktop')+[__name__+'-'+__version__+'.orig/usr/share/applications/'], 
                   shell=False, stderr=subprocess.PIPE,stdout=subprocess.PIPE).communicate()
  subprocess.Popen(['mv', __name__+'-'+__version__+'/usr/local/lib/python2.7', 
                    __name__+'-'+__version__+'/usr/local/lib/python2.6'], 
                   shell=False, stderr=subprocess.PIPE,stdout=subprocess.PIPE).communicate()
  subprocess.Popen(['mv', __name__+'-'+__version__+'.orig/usr/local/lib/python2.7', 
                    __name__+'-'+__version__+'.orig/usr/local/lib/python2.6'], 
                   shell=False, stderr=subprocess.PIPE,stdout=subprocess.PIPE).communicate()
  os.chdir(__name__+'-'+__version__)
  print "Packaging for debian (python2.6)..."
  subprocess.Popen(['dpkg-buildpackage', '-i.*', '-I', '-rfakeroot', '-us', '-uc'], shell=False, 
                   stderr=subprocess.STDOUT, stdout=open('../last_package_2.log', 'w')).communicate()
  os.chdir('..')
  os.rename((__name__+'_'+__version__).lower()+'-1_all.deb', __name__+'-'+__version__+'_maverick.deb')
  print "Removing debian folder..."
  os.popen('rm '+__name__+'-'+__version__+' -r')
  os.popen('rm '+(__name__+'_'+__version__).lower()+'-1*')
  os.popen('rm *.rpm')
  os.popen('rm '+(__name__+'_'+__version__).lower()+'.orig.tar.gz')
  print "Removing build folder..."
  os.chdir('..')
  os.popen('rm build -r')

#if ('install' in sys.argv) and len(sys.argv)==2:
  #if ('win' in sys.platform):
    ## In windows the scriptpath is not in the path by default
    #win_script_path=os.path.join(sys.prefix.lower(), 'scripts')
    #win_path=os.path.expandvars('$PATH').lower().split(';')
    #if not win_script_path in win_path:
      #print "Could not verify path!\nPlease be sure that '" + win_script_path + "' is in your path."
  #else:
    ## Linux/OS-X installation
    #pass

  # If not installing to python default path change a line in the script to add the program location
  # to pythons module search path when executing.
  # TODO: Try to make this work with all setup parameters not only --install-scripts + --prefix
if ('--install-scripts' in sys.argv) and ('--prefix' in sys.argv):
  print "Adding module directory to python path in plot.py script."
  script=open(os.path.join(sys.argv[sys.argv.index('--install-scripts')+1], 'plot.py'), 'r')
  text=script.read().replace('##---add_python_path_here---##','sys.path.append("'+\
                    glob(os.path.join(sys.argv[sys.argv.index('--prefix')+1], 'lib/python2.?/site-packages'))[-1]\
                    +'")')
  script.close()
  script=open(os.path.join(sys.argv[sys.argv.index('--install-scripts')+1], 'plot.py'), 'w')
  script.write(text)
  script.close()


if "py2exe" in sys.argv:
  def xcopy_to_folder(from_folder, to_folder):
    dest=os.path.join('dist', to_folder)
    if getattr(from_folder, '__iter__', False):
      src=os.path.join(*from_folder)
    else:
      src=from_folder
    print "Copy %s to %s..." % (src, dest)
    try:
      os.mkdir(os.path.join('dist', to_folder))
    except OSError:
      print "\tDirectory %s already exists." % dest
    try:
      handle=os.popen('xcopy %s %s /y /e' % (src, dest))
      files=len(handle.read().splitlines())
      print "\t%i Files" % files
    except:
      print "\tSkipped because of errors!" % src
  print "\n*** Copying source and datafiles ***"  
# py2exe specific stuff to make it work:
  for src, dest in [
                    ('plugins', 'plugins'), 
                    ('models', 'models'), 
                    ('profiles', 'profiles'),
                    ('examples', 'examples'),
                    ]:
    xcopy_to_folder(src, dest)
  os.popen('xcopy genx.conf dist')

if 'py2app' in sys.argv:
  def xcopy_to_folder(from_folder, to_folder):
    dest=os.path.join('dist', to_folder)
    if getattr(from_folder, '__iter__', False):
      src=os.path.join(*from_folder)
    else:
      src=from_folder
    print "Copy %s to %s..." % (src, dest)
    try:
      os.mkdir(os.path.join('dist', to_folder))
    except OSError:
      print "\tDirectory %s already exists." % dest
    try:
      handle=os.popen('cp -R %s %s' % (src, dest))
      files=len(handle.read().splitlines())
      print "\t%i Files" % files
    except:
      print "\tSkipped because of errors!" % src
  print "\n*** Copying source and datafiles ***"  
# py2exe specific stuff to make it work:
  base = 'genx.app/Contents/Resources/lib/python2.7'
  for src, dest in [
                    ('profiles', ''),
                    ('examples', ''),
                    ]:
    xcopy_to_folder(src, base + dest)
  os.popen('cp genx.conf ' + 'dist/' + base)
  