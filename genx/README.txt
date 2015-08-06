This package contain GenX 2.4b2 a program to refine x-ray and neutron reflectivity as well as surface x-ray diffraction using differential evolution. It can also serve as a general fitting program.

Support
=======
Tutorials can be found at: http://genx.sourceforge.net/doc/
Examples can be found in the Menu Help->Examples.
If you need more support send an e-mail to Matts.Bjorck@gmail.com.

References
==========
If you use the program please give reference to the following publication:
M. Bjorck and G. Andersson J. Appl. Cryst. 40, 1174 (2007).

Changes 2.5.0 (Trunk)
=====================
 * Added an SXRD plugin


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