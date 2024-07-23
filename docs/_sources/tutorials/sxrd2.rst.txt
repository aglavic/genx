.. _tutorial-sxrd2:

*****************************
Surface X-diffraction Model 2
*****************************

GenX has also an SXRD plugin using the updated model sxrd2 that makes building the model easier in a way
similar to the reflectivity plugin.

I'm not an expert on this techynique and how it was initially implemented and used in GenX.
So this section is just to help users with a few tips. Please feel free to send me your input
about this chapter.

An example of the usage of this model and plugin can be found in the examples folder with the model "SXRD2.hgx",
which implements the same sample as "SXRD.hgx" but with the new facility.

To convert existing models from sxrd to sxrd2 you need to make some changes to the script and then load the
SXRD plugin. (If the script is not correct, the plugin won't recognize the model correctly.)
Best you use a text editor and copy the example script from "SXRD2.hgx" to one file and your model to a second file.

The following changes will be needed:
    1. Change the import statements in the top of the script

    2. Transfer Instrument, UnitCell and Slab objects to the blocks marked with "# BEGIN"/"# END".
       Make sure that they are defined with keywords (e.g. wavel=...) as in the example.
       (slab.add_atom is also recognized without any keyword)

    3. Put the slabs into Domain objects and these into the Sample object. Different from sxrd the sxrd2
       model defines the slabs in the domain and is only one object.

    4. Make sure there are "DataSet" blocks for all your imported data in the "Sim" function.

    5. If you need to run any user-defined code on the model, place it in the "Sim" function before the first dataset.
