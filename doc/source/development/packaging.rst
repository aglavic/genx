.. _development-distribution:

*******************
How to package GenX
*******************

1. Tag a release into the tags folder.
2. Change the content in the ``__version__.py`` file

OS-X
====

#. ``cd models/lib; rm *.cpp; rm *.so; rm *.pyd; python build_ext.py; cd ../..``
#. ``cd dist; rm -R *; cd ..``
#. ``run python setup.py py2app``
#. ``cd mac_build; mv ../dist/GenX.app .``
#. ``appdmg appdmg.json GenX-2.X.X-OSX.dmg``

Win 7/Win 8
===========

?

Binary dist
===========

1. python setup.py dist
