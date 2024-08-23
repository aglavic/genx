This package contains GenX 3.7 a program to refine x-ray and neutron reflectivity as well as
surface x-ray diffraction using differential evolution. It can also serve as a general fitting program.

Support
=======
Tutorials can be found at: http://genx.sourceforge.net/doc/
Examples can be found in the Menu Help->Examples.
If you need more support send an e-mail to artur.glavic@psi.ch.

References
==========
If you use the program please give reference to the following publication:
A. Glavic and M. Björck J. Appl. Cryst. 55, 1063-1071 (2022).

Changes 3.7.0
=============
 * Add the FrequencyAnalysis plugin that allows to analyze the reflectivity using various corrections to
   extract approximate layer thicknesses.
 * Add advanced footprint and resolution classes that can even be replaced by user defined functions. See
   the trapezoidal beam profile example in "SuperAdam_SiO_advanced_fp_res.hgx".
 * Add Zeeman-effect correction for neutron polarization analysis with spin-flip in elevated external field
   to the spec_adaptive model. Can be activated using the instrument parameter "zeeman" and "mag_field".
 * Model help button in all parameter dialogs to quickly look up the meaning of parameters. 
 * Implement new python dataclass based model parameterization. The GUI will detect any parameter in the model
   based on the base class which allows more flexibility in model modification and improves general maintainability.
 * Add code signature to Mac OS distribution to remove need for user to ignore security warnings on installation/run.
   (Thanks to the international scattering alliance for support in creating the certificate.)
 * Change parameteriazation of interdiff model to use sigma+sigmar instead of sigmai+sigmar to make it
   equivalent to reflectivity models that only use sigma = sqrt(sigmai**2+sigmar**2). To fit sigmai one
   should create a user parameter or set proper limits in sigma+sigmar fit.
 * Increase test coverage, especially for code inside of models. This lead to several bug fixes and
   will improve stability of future releases.

Changes 3.6.28
==============
 * Fix bug when running mag_refl that lead to an error due to missing import in model.

Changes 3.6.27
==============
 * Fix bug when using log function within column calulation that prohibited use of D17 dataloader.

Changes 3.6.26
==============
 * Add documentation tutorial about ORSO file integration.
 * Update of SNAP builde system, should allow use with Waylend and fix some other minor issues.
 * Update of Windows build libraries for additional functionality.
 * Add debian build for newer Ubuntu versions (22.04 / 24.04). See documentation for installation details.
 * Add a GUI dialog when critical python errors occure that required console/logging to be noticed before.
 * Fix incompatibility with numpy 2.x due to bool/numpy.bool confusion.

Changes 3.6.25
==============
 * Fix bug in MagSLD where magnetization was reported 10x too high in graph (see Ticket #205).
 * Fix inconsistent behavor for x-values <=0 (see Ticket #201).

Changes 3.6.24
==============
 * Add compatibility to ORSO binary format.
 * Export ORSO simple model language description of GenX simulation in ORT export.
 * Accept ORSO datasets for new models using drag-n-drop.
 * Fix ORSO export for current orsopy version.

Changes 3.6.23
==============
 * Fix plot style dialog not working on newer version of WX.
 * Fix handling of some chemical formulae.
 * Fix issue when closing the GUI through the menu.

Changes 3.6.22
==============
 * Fix a bug with the update code for newer urllib3 versions (see PR #5, thanks to azelcer)
 * Upgrade windows build to python 3.11 and recent libraries.

Changes 3.6.21
==============
 * Add data loader for nja XRR file format.
 * Add pint and latest orsopy to binary distributions to allow for better parsing of .ort metadata.
 * Fix the Bumps error dialog filling the wrong error ranges into the parameter grid.
 * Fix a multiprocessing logger related bug that crashes the program under certain circumstances.

Changes 3.6.20
==============
 * Fix Rigaku data loader to include attenuation factors.

Changes 3.6.19
==============
 * Introduce crop_sigma sample option to spec_adaptive model that allows to limit the
   influence of the interface transition function within the adjacent layers.
   Thanks to Rico Ehrler for the suggestion.

Changes 3.6.18
==============
 * Update gsecars_ctr data loader to detect additional columns by first header line
 * Some minor fixes for wxPython 4.2.0 and newer numba

Changes 3.6.17
==============
 * Use single numba cache directory for any GenX executable, speeding up program start 
 * Fix multiprocessing fit stuck in Windows binary
 * Better logging and error reporting in multiprocessing fit

Changes 3.6.16
==============
 * Improve error handling and allow forcefull termination of multiprocessing fits
 * Add full logging support when running fit with multiprocessing
 * Add caching of GPU kernels for newver versions of numba
 * Correctly count the number of functions to be compiled with numba
 * Fix error when trying to use multiprocessing fit without numba installed

Changes 3.6.15
==============
 * Add new LayerGraphics plugin that creates a simple sketch drawing for reflectometry models
   to use in presentations etc.
 * Update the Mac build system to Mac OS 12 and system python 3.10 using new wxPython 4.2 PyPI package

Changes 3.6.14
==============
 * Fix re-compilation of numba code when opening project filed directly on Windows
 * Add some NeXus file attributes to the .hgx file format to allow plotting of the data e.g. with nexpy
 * Small change to the MacOS configuration that should support file type filtering in open dialog

Changes 3.6.13
==============
 * Fix a bug where exporting the script with special characters raised an error under windows (ticket #197)
 * Fix some bugs in export and parsing of .ort files
 * Some refactoring

Changes 3.6.12
==============
 * Fix a bug where fitting from console with autosave and --error options stopped the fit after first autosave
 * Improve the meta data editing capability

Changes 3.6.11
==============
 * Update the ORSO file definition to version 1.0.0 released recently
 * Modify the metadata dialog to allow adding and editing values
 * Add a new data loader for the Rigaku .ras format
 * Fix default and resolution loader to ignore non utf-8 encoded values

Changes 3.6.10
==============
 * Implement a tech-preview using alternative plotting backend with improved performance
   (selected in Settings -> Startup Profile... menu.)
 * Automatically restart the window when switching from legacy to widescreen layout.

Changes 3.6.9
=============
 * First version of MacOS binary distribution
 * Add new script "genx_mac" to PyPI package to start with framework build (pythonw)
 * Allow file names with upper case endings (.GX/.HGX)
 * Try to fix some plot drawing issues on some Linux systems with Wayland backend.
 * Open GenX model files on drag&drop to the window (if not above data list)
 * Fix GUI not remembering a model is unchanged after loading from a file
 * Fix bug where the parametr grid could be wrong after loading a model while value editor was active

Changes 3.6.8
=============
 * Fix a bug where values for the instrument parameters where parsed by int type if the script used integer values
 * Fix a compatibility issue with older wxPython/wxWidgets that would prevent genx from starting on fedora 35
 * Fix issues when running numba together with multiprocessing on UNIX bases systems due to fork method

Changes 3.6.7
=============
 * Fix compatibility with python 3.6-3.7

Changes 3.6.6
=============
 * Fix wx dialog issue where instrument editor in advanced reflectivity would not work (thanks to Leon Lohse)

Changes 3.6.5
=============
 * Fix parameter grid value cell out of bounds coloring lost after loading a new model

Changes 3.6.4
=============
 * Add simple syntax completion, object help and undo/redo to script editor. To use
   try ctrl+enter, shift+ctrl+enter, ctrl+alt+Z or shift+ctrl+alt+Z.
 * Do not raise an error when starting a fit with parameters outside of min/max boundaries
   if the optimizer does not use them. (ticket #175)
 * Fix compatibility issue with python 3.10, tested with wxPython 3.1.1 and 3.1.2a

Changes 3.6.3
=============
 * Fix a bug that could lead to a strange error messages when editing items in the Simulations tab.
 * Fix a crash on Linux when running the bumps dialog depending on wx version
 * Fix an issue where genx would not start on macOS environments with python >=3.9 and anaconda

Changes 3.6.2
=============
 * Add finite polarization effects for neutron reflectivity to spec_nx, spec_adaptive and spec_inhom models.
   To use you have to select instrument probe as "neutron pol spin-flip" and change the simulation function
   from "Specular" to "PolSpecular". This function has 4 additional parameters; p1, p2, F1, F2 for 
   polarizer, analyzer and filpper efficiencies. For definition see https://doi.org/10.1063/1.1150060
 * Update UserFuncs pluging to work with type-annotated functions to generate user dialogs automatically.
   The SXRD.hgx example shows a usage for storing XYZ files.
 * Add entry to the **Help** menu to open example files, directly jumping to the right directory.
   About dialog now shows the path where configuration files are stored.
 * Fix a bug where editing the script in some circumstances would loose lines.

Changes 3.6.1
=============
 * Add a batch processing interface to the GUI. This can be accessed through the File dialog. See
   new **Batch Fitting** section of the documentation.
 * Add generic definition for plot x- and y-labels. Build-in models define the values depending on last simulated
   scans and user can always overwrite in script with **__xlabel__** and **__ylabel__** special variables.
 * Add detailed documentation about SLD plot configuration and batch processing
 * Add more unit tests for models and loading/saving
 * Fix remote fit crashing server when ending normally instead of being stopped by the user

Changes 3.6.0
=============
 * Add new genx_server (python -m genx.server) script that allows to run a remote service
   on a cluster that can be used to fit from a GUI on a different machine.
   See: https://aglavic.github.io/genx/doc/tutorials/mpi.html for more information.
 * Implement asymmetric errors from bumps statistics, fix some bugs and add option to normalize parameter
   uncertainties by sqrt(chi2) to eliminate scaling factors on error bars. (see ticket #190)
 * New command line parameters for better control of refinement and performance
 * Improve console and logging output on MPI runs, q+<enter> can now stop a fit started with MPI
 * Fix some command line options
 * Allow changing of plot scales with mouse scroll wheel and ctrl-/alt-/shift-modifier, 
   always reset zoom with middle mouse button
 * Improve SLD plot context menu, allowing to show only first dataset, 
   external legend or coloring associated with datasets
 * Option to generate a SLD uncertainty graph based on a user-defined reference interface
 * Do not show separate mag_x SLD for neutron magnetic reflectivity, if there is no mag_y component
 * Slight improvement or SXRD model performance
 * Add a genx3server PyPI package without GUI package requirements
 * Updates on documentation concerning use from command line
 * Startup script to automatically select pythonw when run on Mac OS (untested)
 * Fix some more minor bugs

Changes 3.5.11
==============
 * Fix export of Table
 * Fix bumps statistics setting the error column of the parameter table
 * Add documentation for Norm FOM

Changes 3.5.10
==============
 * Add command line option to set the relative parameter variation beak condition
 * Fix an error that can happen when a numpy floating point error is raised in the windows version
 * Fix parameter addition in python API module

Changes 3.5.9
=============
 * Update sns_mr data loader to changes in reduced data format.

Changes 3.5.8
=============
 * Fix crash on Linux systems when automatically simulating, bug #189
 * Fix some with the snap that prevented loading SXRD plugin with 3D view
 * Snap now stable enough to use, but does not support multiprocessing due to access issues with confinement

Changes 3.5.7
=============
 * Online check for new GenX versions and option to download new setup file/in-place pip update
 * First version of snap binary distribution for Linux systems other than Ubuntu 18.04/20.04
 * Better copy to clipboard of selected parameters in error statistics dialog 

Changes 3.5.6
=============
 * Add option to edit the script in an external editor
 * Fix issue of GenX crashing if incompatible locals are specified in system configuration #187
 * Fix bug that caused script editing issues on some platforms #188

Changes 3.5.5
=============
 * Fix an issue with Reflectivity plugin instrument dialog causing
   silent fails to update an read values from after running Sim function.

Changes 3.5.4
=============
 * Add data loader for SINQ six text file format

Changes 3.5.3
=============
 * Fix bugs in handling of insertion/deletion of parameters
 * Fix bug in printing of plots and table
 * Fix query of ORSO database in SimpleReflectivity in some circumstances

Changes 3.5.2
=============
 * Add a new modeling option to spec_inhom model that allows automatic generation
   of neutron super-mirror structures from user defined Stack parameters.
 * Captuer model errors during fit that did not occure on first evaluation.

Changes 3.5.1
=============
 * Fix some issues with deleting and moving parameters in the grid that
   were caused by changes for undo/redo functionality.
 * Add (beta) support for ORSO SLD database for SimpleLayer and SimpleReflectivity
 * Some fixes in SimpleLayer plugin

Changes 3.5.0
=============
 * Add undo/redo functionality for most user actions as for changing the script or parameter values
 * History dialog that shows the undo actions and allows removal of previous steps while keeping later ones
 * Reorganize menus to make it more accessible
 * Improved sorting of parameters by object or parameter with grouping
 * Start logfile from GUI and show dialog with logged messages (Help menu)
 * Load multiple datasets from suitable data loaders (orso+xrdml)
 * Configure new reflectivity model from metadata read from .ort files (radiation, resolution etc.)
 * New option to automatically stop a fit when relative parameter spreads a reduced below a threashold value.
   Setting the parameter to e.g. 1% will stop once the parameter that varies the largest fraction of its fit
   range has a spread of less than 1% within the population. Seems very stable and is helpful for long-running
   fits with MPI that can't be stopped manually, well. (Thanks to Larry Anovitz for the idea.)
 * Major updates to the main tutorails in the documentation
 * Update orso .ort file format to use the new orsopy package with the updated specification
 * Fix update of Pars plot during fit when SimpleReflectivity plugin is loaded
 * Fix bumps fitting functionality and add update of Pars plot for this solver
 * Fix bug where fitting with multiprocessing without numba would fail
 * Fix in plotting of error bars to be below simulation
 * Fix incompatibility with matplotlib >=3.5.0
 * General refactoring for the GUI code to allow undo/redo functionality. May have introduced new bugs. As always
   All feedback is welcome.
 
Changes 3.4.12
==============
 * Limit matplotlib version for PyPI to <3.5.0 as this breaks some code within GenX

Changes 3.4.11
==============
 * Fix missing library in windows distribution
 * Fix xrdml loader for newer version where tag has changed
 * Add xrdml file format to auto data loader

Changes 3.4.10
==============
 * Fix bug with missing os import in genx/data.py to allow export

Changes 3.4.9
=============
 * Add --dpi-scale option to overwrite the automatic detection.
   Use --dpi-scale 1.0 on OS X if you encounter large icons overrunning the toolbar.

Changes 3.4.8
=============
 * Fitting from command line on unix systems now displays the parameter values and spread on the console

Changes 3.4.7
=============
 * Add export as XYZ file option to sxrd/sxrd2 models ( sample.export_xyz / domain.export_xyz )
 * SXRD plugin option to hide bulk and display arrows for the dx,dy,dz movement of atoms

Changes 3.4.6
=============
 * Some fixes to the sxrd2 model
 * Fix backwards compatibility issues with wx and python 3.6. (The latter needs to "pip install dataclasses".)

Changes 3.4.5
=============
 * Fix sls_sxrd plugin to work with additional LB/dL columns as explained in the documentation example

Changes 3.4.4
=============
 * Fix fitting using MPI (command line on cluster)
 * Fix stopping a command line fit with multiprocessing using ctrl+c
 * Improvements to the publication graph dialog
 * Some small bug fixes for fitting from command line

Changes 3.4.3
=============
 * Fix backward compatibility issue with older numpy and numba libraries

Changes 3.4.2
=============
 * Fix bug #185 of broken import settings dialog in windows .exe
 * Add config file option "gui/solver update time" that can be used on slower computers
   to reduce GUI load during fitting

Changes 3.4.1
=============
 * First preview of a publication graph dialog that allows precise definition of plot attributes through
   small script. A user defined graph size can be chosen and the plot exported to an image file.
 * Fix bugs #183 and  #184 causing crashes on new installs due to configuration default and path issues.
 * Fix bug in parameter scan using wrong configuration option (#182) and project fom with non-DE fits.

Changes 3.4.0
=============
 * Add additional optimizers to be used for refinement (fast Levenberg-Marquardt or Bumps library)
 * Improved simulation performance for simple models and better stability of GUI for fast updates.
   If CUDA calculation and parallel is selected, one process will run on GPU and the rest on CPU.
 * Add option to automatically color datasets according to pre-defined cycle (2, 4 states or rainbow)
 * Allow drag&drop of data files onto GenX data table
 * Show a startup splash screen to give user feedback, especially when delayed by JIT compilation during first run
 * Major refactoring of core code and configuration system for better maintenance and expandibility
   (Be aware that this may lead to new Bugs compred to 3.3.x versions. Please submit bug-reports
   if you find any!)
 * Reduce number of threads used by numba when running in multiprocessing mode, large increase in performance.
 * Some minor bug fixes

Changes 3.3.6
=============
 * Fix bug in hgx save/load models with non-ascii characters

Changes 3.3.5
=============
 * Fix bug in file export that could lead to missing lines at the end of the file due to caching
 * Expand unit testing and remove unused code to support maintenance

Changes 3.3.4
=============
 * Allow neutron matrix calculations with non-zero ambient layer SLD
 * Fix but in fast neutron matrix calculation where roughnesses were used from wrong layer index
 * Updating of bumps statistical analysis example notebook
 * Fix residual High DPI issue in SimpleReflectivity wizard
 * Fix a bug when loading datasets lost options e.g. for plotting
 * Added link to new video tutorial do documentation.
 * Replace some physical constants by more precise values

Changes 3.3.3
=============
 * Fix issues with SimpleReflectivity Wizard in High DPI environments
 * Fix dataset options being lost when loading new data
 * Prevent closing error statistics dialog if thread still runs in background

Changes 3.3.2
=============
 * Reintroduce wait time per iteration as the GUI can crash without it. Now it can be changed from the optimizer dialog.

Changes 3.3.1
=============
 * Fix column type in ORSO reader to be ndarray and not derived class

Changes 3.3.0
=============
 * Updated the documentation website to include the SimpleReflectivity interface
 * Reimplementation of the off-specular and x-ray surface diffraction models (sxrd, sxrd2, interdiff)
 * In Reflectometry plugin, automatically update the GUI when the script is changed manually
 * Add an alpha version of ORSO text format data reader
 * Make auto data loader the default, this includes the following loaders:
   (default, resolution, sns_mr, amor, d17_cosmos, orso)
   Please send me your instrument data files as examples if you want your own data loader that can include meta data, too.
 * Fix crashes in Linux systems when changing parameters in the grid (especially when automatic update is active)
 * Fix incompatibility with h5py version 3 when loading models
 * Fix the d17_cosmos data loader and add d17_legacy for old style files
 * Fix issues in windows binary that prohibited opening of Help dialogs
 * New type os user parameter intended for systematic errors that influence all datapoints. It has
   a sigma parameter and biases the FOM with (x0-x)²/sigma² to take the systematic error uncertainty
   into account.
 * The column calculation now supports a rms(sigma1, sigma2, ...) function to combine different error contributions
 * Example columns showing how to include systematic errors from motor position and/or beam distribution uncertainty
 * Remove unnecessary sleep per iteration when fitting in single thread mode. Please report if you notice issues like
   crashes
 * Some additional improvements for simulation performance

Changes 3.2.3
=============
 * Fix a bug in footprint correction introduced in 3.2.0
 * Improve parameter grid interface with parameter relative value indicator and slider controls
 * Allow copy & paste in parameter grid
 * Can now use space to start edit on selected parameter and to accept parameter changes
 * Fix a DPI display bug for toolbar icons aside grid
 * Can not toggle negative value with "-" at any edit location in the value editor
 * Don't automatically open context menu on single click of first column, allows to select and edit manually easier

Changes 3.2.2
=============
 * Update windows build to python 3.9 and wxPython 4.1.1 to better support High DPI displays
 * Improve value entry in parameter grid (ENTER/TAB key, mouse scrolling)
 * Prevent parameter grid entry resizing to prevent non-intentional layout issues
 * Automatize PyPI releases

Changes 3.2.1
=============
 * Fix error in new Numba functions that calculate the resolution vector (ORSO validation failed)

Changes 3.2.0
=============
 * Add simple API for use in python scripts and Jupyter notebooks. Can read, write, modify and fit models
 * Add some examples of Jupyter notebooks to show usage of API
 * Integration of GenX models into bumps library (see https://bumps.readthedocs.io/en/latest/index.html )
 * Dialog for statistical error analysis with bumps MCMC to evaluate cross-correlations of parameters
 * New export function (alpha) for ORSO text format with detailed header containing analysis information
 * Improvements to script editor behavior concerning indentation
 * Reflectivity plugin now re-analyses a manually changed script after it has been run once
 * SimpleReflectivity now shows an error summary dialog when errorbars are calculated
 * New 'auto' data loader that chooses the method by file type, supports AMOR, SNS MR, default and resolution loaders
 * Improvements in reflectivity model performance, possibility to use CUDA with multiprocessing
 * Improvements in plot performance for data and SLD graphs
 * Some refactoring of code started
 * Fix SimpleLayer plugin to allow multiple materials with same chemical formula
 * Fix some bugs where plot updates could crash the GUI or freeze the SLD graph

Changes 3.1.4
=============
 * Fix bug in mag_ref (issue #178)
 * Update GenX documentation website

Changes 3.1.3
=============
 * Fix some GUI crashes on wxPython >=4.1
 * Fix GUI issue/crash when auto update SLD is active (issue #177)
 * Fix about dialog
 * Use new DPI scaling function for better cross-platfrom high DPI handling, if available (wxPython >=4.1)

Changes 3.1.2
=============
 * Small fix of build system and contact email.

Changes 3.1.1
=============
 * Update build system to be compatible with PyPI, thanks to Leon Lohse
 * Include vtk module in windows distribution for SXRD plugin

Changes 3.1.0
=============
 * Implement numba JIT compiler for significant x-ray and neutron reflectivity calculation performance gain
 * Implement GPU accelerated version with CUDA (NVidia graphics cards, menu option to activate)
 * SpinAsymmetry plugin to plot the SA for data and model
 * Exporter plugin for reflectivity models, up to now supports BornAgain python scripts
 * Several bug fixes in various modules

Changes 3.0.8
=============
 * Fix some issues with newer wxPython versions (4.1.1)
 * Fix an error in the unit for neutron SLD display (10^-6 AA^-1)
 * Automatic build process on github

Changes 3.0.7
=============
 * Fix bug in spec_nx when trying to use spin-flip model
 * Fix bug #160 in spin-flip that would not recognize a changed model correctly
 * Add button to SimpleReflectivity for switching to Reflecivity plugin for more complex models

Changes 3.0.6
=============
 * Fix GUI bugs reported in tickets #172 and #173

Changes 3.0.5
=============
 * Fix some handling of fomulas and material in SimpleLayer and SimpleReflectivity plugin

Changes 3.0.4
=============
 * Fix bugs #171 and #169

Changes 3.0.3
=============
 * Fix bug in spin-flip model that failed simulation with an error
 * Try to make SXRD work again

Changes 3.0.2
=============
 * Fix plotting error when loading new dataset with different shape
 * Fix sample parameter dialog not evaluating input type correctly in spec_adaptive model (#167)

Changes 3.0.1
=============
 * Fix issue with model table when creating new model in SimpleReflectivity
 * Fix unicode error in sns_mr data loader
 * Handle footpring and tth offset parameter correctly when ToF neutron is selected
 * Update windows installer to run with user privileges
 * Fix evaluation of extra data columns like "res"

Changes 3.0.0
=============
 * Convert to python 3
 * Convert to wxPython 4 (Phoenix)
 * Add new SimpleReflectivity plugin for simple structures and beginner users
 * Updated icons with dpi awareness
 * New optional wide screen optimized GUI that allows to see Data and SLD side-by-side
 * Improved SimpleLayer materials database with query to Materials Project and Open Crystallography databases
 * Fix windows binary to work with Windows 10 without compatibility mode
 * Improved plot layout that uses full space, provides correct axes labes and can be copied with white background


Changes 2.4.9
=============
 * Fixed bug in SimpleLayer plugin - could not load cif files under OSX.

Changes 2.4.8
=============
 * Fixed bug that delete and backspace did not work in the parameter grid under Windows.
 * Fixed so that data can be loaded with the resolution data loader.
 * Fixed bug in the SimpleLayer plugin.
 * Small bug fixes in parameter and data model classes

Changes 2.4.7
=============
 * Fixed bug, parallel fitting with mag_refl stopped in "going into optimisation".
 * Fixed bug with adding data sets into a new reflectivity plugin model.
 * Fixed wrong spin state calculations in soft_nx

Changes 2.4.6
=============
 * Fixed bug that the SLD for neutrons were scaled with wl**2/2/pi.

Changes in 2.4.5
================
 * Fixed bug that the SLD for neutrons were scaled with wl**2/2/pi.
 * Problem with the precision in some neutron calculations solved.
 * Numbers in the grid can be given in scientific/exponential notation, for example 1e5.
 * Problems with fractional numbers using "." on systems with defualt deciaml seprator as "," solved.
 * Scan FOM not always functioning with blank rows in the grid solved.

Changes in 2.4.2
================
 * Minor bug fixes in the gui
 * Fixed that the models ignored negative b's (spec_nx and mag_refl)

Changes 2.4.0
=============
 * Added sliders and spin controls to change the parameter values, updated dynamically.
 * A new reflectivity aimed to soft matter called soft_nx.
 * Added the possibility to have a logarithmic x scale.
 * Data points causing nan and inf in the FOM can be ignored (see the Options dialog).
 * A resolution type for constant dq/q have been added to spec_nx and soft_nx.
 * Simulation data sets can be created through a wizard.
 * Added data loader for SNS BL4A (programmer: Artur Glavic)
 * Added plugin to more easily define layers (programmer: Artur Glavic)
 * Various bug fixes.

Changes 2.3.6
=============
 * Fixed bug regarding the definition of instruments (not working) in the Reflectivity plugin.
 * Fixed bug that caused an error when trying to fit the number of repetitions.
 * Fixed bug regardgin q=0 simualtions - the models now throws an error for q = 0.
 * Fixed bug in the buffering of spin flip calculations (caused an error when trying to simulate data sets with differing number of x-values).
 * Fixed not working choice boxes in the Calculation dialog.
 * Added an data loader for four column data which also includes the resolution.
 * Included so that 'du' works in spec_nx for calculating spin flip and the same thing in mag_refl.


Changes 2.3.5
=============
 * Fixed bug that GenX does not start after installation on Windows machine.
 * Fixed bug so that command line execution works better on frozen versions.
 * Fixed bugs regarding the c extensions in the frozen version.

Changes 2.3.0
=============
 * Changed the x-ray scattering length data tables to use the ffast nist, which
   is more accurate at low energies, database:
   http://www.nist.gov/pml/data/ffast/index.cfm
 * Refurbished the table of fitting parameters with new functionality and a new toolbar.
 * The reflectivity plugin has been improved:
   - Which parameter to fit can be set in the sample definition dialogs.
   - The Sample tab shows the current value of the fitted parameters and also indicates which are fitted.
 * Command line fitting has been added. Possible to run fit without the GUI.
 * A new file format based on hdf5 has been implemented (more platform independent).
 * MPI support has been added, thanks to Canrong Qiu (University of Alaska).
 * The model mag_refl can now:
   - Simulate energy scans.
   - Simulate "normal" x-ray reflectivity.
   - Simulate scans with polarisation analysis.
   - Use negative values of mag.
 * spec_nx and mag_refl can now simulate the asymmetry signal in neutron reflectivity.
 * Refactoring of the Reflectivity base models.
 * Numerous reported bugs fixed. (See http://sourceforge.net/p/genx/code/commit_browser
   for detailed changes).
