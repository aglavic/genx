This package contain GenX 2.2 a program to refine x-ray and neutron reflectivity as well 
as surface x-ray diffraction using differential evolution. It can also serve as a general
 fitting program.

Support
=======
Tutorials can be found at: http://sourceforge.net/p/genx/wiki/IntroGuides/
Examples can be found in the Menu Help->Examples.
If you need more support send an e-mail to Matts.Bjorck@gmail.com.

References
==========
If you use the program please give reference to the following publication:
M. Björck and G. Andersson J. Appl. Cryst. 40, 1174 (2007).

Changes
=======
 * The model mag_refl has been fully implemented and ca be considered to in beta state.
   The model has also been evaluated against Sergey Stephanov’s x-ray server.
 * The Reflectivity plugin has been extended with:
    - SLD profiles for each simulated data set
    - Possible to define multiple instrument instances in the GUI
    - Possible to choose from different simulation functions (Specular, OffSpecular 
      and SLD).
    - Storage of configuration files is done in the correct system folder 
      (thanks to the module appdirs).
    - Bundled versions (no need to have python installed) with installers for OS-X and 
      Windows.
 * Numerous reported bugs fixed. (See http://sourceforge.net/p/genx/code/commit_browser 
   for detailed changes).


