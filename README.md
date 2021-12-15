# GenX <img src="icons/main_gui/genx.png" width="50" alt="GenX 3" align="left" />
## What is Genx?
<img src="https://aglavic.github.io/genx/Screenshot.png" width="50%" alt="GenX Screenshot" align="right" />
GenX is a versatile program using the differential evolution algorithm for fitting, primarily, X-ray and neutron reflectivity data, 
lately also surface x-ray diffraction data. The differential evolution algorithm is a robust optimization method which avoids local minima 
but at same is a highly effective. GenX is written in python and uses the wxpython package for the Graphical User Interface (GUI) Screenshot. 
A model to fit is defined either through a GUI plug-in or via a python script. The possibility to script everything makes it easy to develop completely new fitting model. 
Clearly, GenX is extremely modular, making it possible to extend the program with models and plug-ins for most fitting problems. 

At the present GenX is shipped with models for x-ray and neutron specular reflectivity, off-specular x-ray reflectivity and surface x-ray diffraction. 
A detailed description, of a older version, has been published in [J. Appl. Cryst. 40, 1174 (2007)](https://www.iucr.org/cgi-bin/paper?aj5091).

## Feedback and Help
You can find a manual with examples at https://aglavic.github.io/genx/doc/, for reflectometry a good start is with the 
[Simple Reflectivity Model](https://aglavic.github.io/genx/doc/tutorials/simple_reflectivity.html) guide and our 
[Tutorial Videos](https://aglavic.github.io/genx/doc/tutorials/neutron_fit.html).

Naturally a program does not become better if I do not get feedback from you! 
If you found a bug or want to have something included in the program submit a [ticket on SourceForge](https://sourceforge.net/p/genx/tickets/) or [drop me an e-mail](mailto:artur.glavic@psi.ch).

## Releases
The project is released as binary packages for Ubuntu and Windows as well as source distribution here on GitHub. 
Additionally a package is available on PyPI and can thus be installed via "pip".
For other linux distributions try the pre-build snap [package](https://snapcraft.io/genx).
See the package [project readme](genx/README.txt) for details on changes for the released versions.
