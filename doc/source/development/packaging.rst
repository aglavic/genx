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
