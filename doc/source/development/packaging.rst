.. _development-distribution:

*******************
How to package GenX
*******************

#. Update README.txt
#. Tag a release into the tags folder.
#. Change the content in the ``version.py`` file

OS-X
====

#. ``cd models/lib; rm *.cpp; rm *.so; rm *.pyd; python build_ext.py; cd ../..``
#. ``cd dist; rm -R *; cd ..``
#. ``run python setup.py py2app``
#. ``cd mac_build; mv ../dist/GenX.app .``
#. ``appdmg appdmg.json GenX-2.X.X-OSX.dmg``

Win 7/Win 8
===========
#. copy the tagged version to desktop
#. Go to vX.Y.Z/models/lib; del *.so; del *.cpp; del *.pyc
#. python build_ext.py
#. Go to vX.Y.Z
#. python -m compileall .
#. python setup.py py2exe
#. Open vX.Y.Z/windows_build/genx.iss with Inno Script Studio change AppVerName
#. In Inno Script Studio press Ctrl+F9 to compile script.

Source
======
#. Copy the current release and rename the folder to genx
#. Zip it.
#. Rename the archive to GenX-2.X.X-source.zip

Testing
=======
#. Remove AppData/MattsBjorck on Win7 & Win8
#. Install GenX on the machines
#. Run the examples

Distributing
============
#. Create a folder GenX2.X.X on sourceforge
#. Upload the files GenX-2.X.X-OSX.dmg, GenX-2.X.X-win.exe and GenX-2.X.X-source.zip to the folder.
#. Set the default downlodd for each file.
#. Update the homepage by editing download_box.html (changing link and header)
#. run websync to update the server.