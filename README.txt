This package contain GenX 2.3.3 a program to refine x-ray and neutron reflectivity as well as surface x-ray diffraction using differential evolution. It can also serve as a general fitting program.

Support
=======
Tutorials can be found at: http://genx.sourceforge.net/doc/
Examples can be found in the Menu Help->Examples.
If you need more support send an e-mail to Matts.Bjorck@gmail.com.

References
==========
If you use the program please give reference to the following publication:
M. Björck and G. Andersson J. Appl. Cryst. 40, 1174 (2007).

Changes
=======
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