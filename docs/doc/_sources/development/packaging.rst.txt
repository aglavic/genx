.. _development-distribution:

*******************
How to package GenX
*******************

.. note::
    This is outdated and no more valid for GenX 3. An automatic packaging
    script is placed into the git repository to be run on github automatically.
    Only the PyPI distribution and upload to sourceforge is manual.

#. Update README.txt
#. Change the content in the ``version.py`` file
#. Create commit on git.
#. Tag a release into the tags folder.

Right now I could not get a binary OSX distribution to work, please let me know if you find a solution.

PyPI distribution
-----------------

The steps needed for PyPI are:

.. code::

    python -m build
    python -m twine upload build/*
    user: __token__
    password: {secret token}

