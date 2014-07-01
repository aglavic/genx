.. _model-mag_refl:

*********************************
A model for magnetic reflectivity
*********************************

The model described here is  work in progress and this is only intended as a first draft of the model.
It is not intended for general use.

Changes necessary
=================

* Implement faster calculation of multilayers - make use of repetition unit

Changes done
============

* Perhaps the angle theta_m should be relative the quantization axis of the probe.
  That is as described above for the photon case but an offset of 90 deg for the neutron case.
  Then we do not have to "turn" the magnetization when refining.
* There should probably be two dd_m parameters one dd_ml and one dd_mu so that also depletion in
  both the lower and upper layer can be modeled.
* Neutron reflectivity calcs should be included.
* Remove the slicing parameter from the instrument. - Done
* Should perhaps define the magnetic roughness as the convolution between the chemical
  roughness and the magnetic roughness. This is to my opinion more physically reasonable.
  So sigma_m_real = sqrt(sigma_c2\+ sigma_m2) - Done
* magdens should be renamed to resdens.
* A new parameter called resmag should be included to say how much of the total magnetic
  moment comes from the resonant magnetic species. - Done
* Make a more ordered way to present the parameters in the reflectivity plugin. - Done
* Include a separate slicing criterion for the off-diagonal components - Done.
* Implement inline c coding of the reflectivity calculations to improve the speed. - Done it is about 3.5
  faster than the previous python implementation.

Layer Parameters
================

Optical constants
-----------------

f
     The ordinary (tabualted) scattering length. For example ``fp.Fe`` for pure Fe, ``0.5*fp.Fe + fp.O*0.5`` for
     FeO (this should be in atomic percent).
b
     The neutron scattering length, inserted as for f
xs_ai
     The absorption cross section for neutrons.
fr
     The resonant part of the scattering length (corrections to f close to an absorption edge). Note that this constant is scaled with magdens before added to f. Note that ftot=f+magdens*fr. This is the opposite as to the definition by Stephanov and Shina, thus fr = -(F11+F1-1)
fm1
     The XMCD scattering term. Responsible for circular dichroism. fm1=F11-F1-1.
fm2
     The XMLD scattering term. Responsible for linear dichroism. fm2=2F10-(F11+F1-1)

Densities
---------

dens
     The atomic density of the material in questions in units of atoms/AA3. The density of Fe is calculated as 2/2.8663
resmag
     The density of the magnetic species relative to the resonant species.
resdens
     The relative density of the resonant atomic species. In FeO (and a scattering length as defined above)
     it would correspond to 0.5.

Non-magnetic layer parameters
-----------------------------
d
     The thickness of the layer.
sigma_c
     The chemical/structural roughness.

Magnetic parameters
-------------------

sigma_m
     The magnitude of the magnetic roughness.
mag
     The magnetic moment of the resonant species. This is a scaling factor of the fm1 and fm2 parameters.
phi_m
     The in-plane angle of the magnetization. An angle of zero is along the photon beam directions and 90
     perpendicular to it.
theta_m
     Angle of the magnetic moment relative to the surface of the sample. An angel of 0 corresponds to a
     in-plane magnetized sample and an angle of 90 deg means an perpendicular magnetized sample.
dmag_l
     The relative change of the magnetization of the lower interface.
dmag_u
     The relative enhancement of the magnetization of the upper interface.
dd_l
     The shift of the magnetic moment profile for the lower interface.
dd_u
     The shift of the magnetic moment profile for the upper interface.
sigma_ml
     The magnetic part of the roughness of the lower interface layer.
sigma_mu
     The magnetic part of the roughness of the upper interface layer.


Stack Parameters
================

Sample Parameters
=================

Control flags
-------------
slicing
     If yes the model will use roughness values and slice the sld profile up as given by the layer parameters.
     Otherwise the layers will only be boxes compress: If yes and slicing also is yes the model will merge layers
     with similar optical densities.

Slicing Parameters
------------------

slice_depth
     This is the size of each layer that the model slices up the model into.
sld_buffer
     An extra buffer added below ...
sld_delta
     To come..
sld_mult
     To Come...

Compression parameters
----------------------

dsld_max
     Steps smaller than this parameter will be merged into thicker layers. This applies to the diagonal parts of
     the susceptibility matrix. The units are electrons.
dsld_offdiag_max
     Same as dsld_max but this one applies to the off diagonal components.
dang_max
     Not used and should be removed.
