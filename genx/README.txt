This package contains GenX 3.4 a program to refine x-ray and neutron reflectivity as well as surface x-ray diffraction using differential evolution. It can also serve as a general fitting program.

Support
=======
Tutorials can be found at: http://genx.sourceforge.net/doc/
Examples can be found in the Menu Help->Examples.
If you need more support send an e-mail to artur.glavic@psi.ch.

References
==========
If you use the program please give reference to the following publication:
M. Bjorck and G. Andersson J. Appl. Cryst. 40, 1174 (2007).

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
