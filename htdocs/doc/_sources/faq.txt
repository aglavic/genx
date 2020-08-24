.. _faq:

**************************
Frequently Asked Questions
**************************

General
=======

How should I cite GenX?
-----------------------
Give a reference to the paper describing GenX:
M. Björck and G. Andersson J. Appl. Cryst. 40, 1174 (2007)
And if you want to give a link to the homepage: <http://genx.sf.net>

I can not run GenX on a 64-bit Mac
----------------------------------
Some distributions (EPD) does not provide wxPython compiled for 64-bit machines. You then have to tell python
to use the 32-bit version instead by setting an environment variable.

    export PYTHON_PREFER_32_BIT=yes


If you want this to have permanent effect you can edit your ``/etc/bashrc`` file alt. edit your ``.bashrc`` file
in the home directory. If you do not have a ``.bashrc`` file in home you can copy your ``/etc/bashrc`` file to
home and then edit it.

Fitting
=======

Can't fit and FOM shows inf or Nan
----------------------------------
If you are using a figure of merit that uses log, for example the log fom (figure of merit)
which is the **standard**, you probably have a negative or zero data point. Remove the data point or change
the fom to a more suitable one. The same can happen if you fit with a fom that uses errors as weight and you
have an error of zero somewhere.

An error message appears when I start fitting or the FOM plot and the FOM message in the toolbar differ
-------------------------------------------------------------------------------------------------------


The following message appears: ERROR The disagreement between two subsequent evaluations is larger
than [Number(1e-10)]. Check the model for circular assignments. In older versions this showed up as different
FOM's in the toolbar and the FOM plot.

Then you have somehow made a logical error in your model script. This means that the value of the FOM depends
on the **history** of the parameters. This is because when fitting the fom is evaluated for the entire population and
then at the end of each generation the program will do one additional simulation to calculate the data shown in the
gui. The plotted fom shown in the FOM folder comes from the population but the FOM that comes from the simulation
is shown in the tool bar! This could occur, for example, if you set a value of a variable that you think that you
will fit but then leave unchanged. For example (``cp`` is a custom parameter)
::

    def Sim(data):
       ....
       ....
       cp.test = cp.test - 1.0
       ...
       ...
       # Simulate something that uses cp.test



This will work all fine if you simulate your model it will behave as expected, even if you just evaluate
it. However if you fit it and leave it as a non-fitable parameter or remove it from the grid strange things will
happen!. For each evaluation the parameter `cp.test` will be decreased by 1.0 since it will not be given a value
from the fitting algorithm anymore. So write the ``Sim`` function like this instead::

    def Sim(data):
       ....
       ....
       new_test = cp.test - 1.0
       ...
       ...
       # Simulate something that uses new_test


.. warning::
    And remember **never ever** set a parameter so you create a circular behavior as shown above.

Note that this could also occur in a perfectly right model if the underlying computer/code has a too low
precision compared to the threshold value. You can then change the threshold value in ``genx.conf``, section ``solver``
parameter ``allowed fom discrepancy`` to a larger value.

Reflectivity
============

How do I add an 2Theta offset
-----------------------------


Sometimes during fitting one needs to add a small 2Theta offset due to misalignment. To do this you start by
adding a custom variable in the reflectivity plugin. This is the blue nut button in the simulation tab. Lets call
the parameter ``TthOff``.

.. note::
    This way is deprecated since version 2.2.0 as it can now been done through the gui with the reflectivity plugin
    in the simulation tab. That is the `Sim` function does not have to be changed manually.

Then the ``Sim`` function have to be changed to something like (located in the Script tab)::

    def Sim(data):
        I = []
        # Things that should be inserted begin:
        tmp0 = data[0].x*1.0
        data[0].x = tmp0 + cp.TthOff
        # End insertion

        # BEGIN Dataset 0 DO NOT CHANGE
        I.append(sample.SimSpecular(data[0].x, inst))
        # END Dataset 0
        # Things that should be inserted begin:
        data[0].x = tmp0
        # End insertion


Note that it is important to place the code outside the part marked with `DO NOT CHANGE`. The offset variable
has been implemented in the spec_nx model.

What about different scattering length databases?
-------------------------------------------------
GenX has currently three different scattering length databases fp, fw, bc and f. The databases beginning with f
relates to x-ray scattering. These are based on the Henke tables and can be found at the
`CXRO's homepage <http://www.cxro.lbl.gov/>`_. The difference between fp and fw is the scaling fp has
units of electrons (or rather Thompson scattering lengths), this means that any densities in inserted in a
layer using fp has to be in atoms/AA3. If one uses fw the scattering length of an atom has been scaled by its
atomic weight consequently the density can be inserted in units of g/cm3. Note that the composition has to be given
in weight percent of the different constituents an a compound. Both fp and fw are the scattering factors at :math:`Q=0`
(forward scattering) if one wants to use the Q varying atomic scattering factor the f library should be used which
returns f as a function of Q. The data is collected from ESRF's DABAX library. bc is the neutron
coherent scattering length used for neutron reflectivity calculations. It has a corresponding
bw which works as for fw. Note that bc is given in fm (fermi meters).

What happens if I fit the scattering length?
--------------------------------------------
As GenX only can fit real numbers the complex part of the scattering len    gth will be ignored.
Thus the absorption is put to zero. If you want to fit both parameters you have to make a custom variable
for at least the complex part. Note that in newer versions > 2.0b6.2 the real and imaginary part of the
scattering lengths can be fitted separately.

Could you explain how the density is defined/works with an example?
-------------------------------------------------------------------
The key thing to understand is that the calculations use the scattering length density, the scattering length
multiplied with the density. Thus, how you define and scale your scattering length must be taken into account
when you define the density.

Example 1: Pure Fe. You define the scattering length as ``fp.Fe`` Then the density becomes (1 formula unit = 1 Fe atom,
Fe bcc 2 atoms/unit cell, a = 2.866): ``2/2.866**3`` You could also define the scattering length as one
unit cell of Fe ``2*fp.Fe`` Then the density becomes: ``1/2.866**3``

Example 2: SrTiO3. You define the scattering length as ``fp.Sr + fp.Ti + fp.O*3`` Then the density becomes
(1 formula unit = 1 unitcell of SrTiO:sub:`3`, a = 3.9045): ``1/3.9045**3``

How do I transform from g/cm\ :sup:`3` to formula unit/Å\ :sup:`3` ?
--------------------------------------------------------------------
I have the density of my material in g/cm\ :sup:`3` or kg/m\ :sup:`3`, how do I use it in GenX? There are two ways:
The first involves only to work with compositions in weight percent and use the fw scattering factors which are
scattering lengths per weight and use densities in g/cm\ :sup:`3`. The other is recalculate the density in g/cm\ :sup:`3`
to formula units per Å\ :sup:`3`. This goes as follows:

The relative atomic mass, u, is defined as :math:`1 u = 1.66054\times 10-27\, \mathrm{kg}`. :math:`1 A = 10^-10 m`.
This means that

.. math:: \rho \mathrm{[kg/m^3]} = 1.66054 \times 10^3 \times (\rho \mathrm{[u/A^3]}).

Thus, the density (scatterers per unit volume (density in GenX)) becomes:

.. math:: \mathrm{dens} = (\rho \mathrm{[kg/m^3]})/(1.66054 \times 10^3 \times uscatt),

where

.. math:: uscatt = \sum_i u_i \times x_i

and the scattering length is written as

.. math:: f = \sum_i f_i \times x_i.

Let us use SrTiO3 as example, it has a density :math:`\rho = 5.12 g/cm^3`. The scattering length is
defined as: ``f = 1*fp.Sr + 1*fp.Ti + 3*fp.O``. The weight of one "formula unit" becomes
:math:`uscatt = 1 \times 87.62 + 1 \times 47.87 + 3 \times 16.00 = 183.24`.
The density becomes: :math:`dens = \frac{5.12 \times 10^3}{(1.66054 \times 10^3 \times 183.24} = 0.017`

Error when simulating spin flip reflectivity
--------------------------------------------
I get an error when the program tries to calculate neutron spin flip reflectivity. The error is following::

    It was not possible to evaluate the model script.

    Check the Sim function.
    ... LOTS OF TEXT ...
    LinAlgError: Singular matrix


This is because there is a singular matrix calculation for an incident angle of 0 or Q = 0. Try to
remove the first data points and it should work.

Is it possible to automatically normalise the fitted function to the data?
--------------------------------------------------------------------------
Yes. Add the parameter I0 from the Instrument sub-menu to scale the fit; fitting this parameter will
autoscale the fit to the data.

I want to use different instruments instances to fit data sets collected of different instruments
-------------------------------------------------------------------------------------------------

.. note::
    This functionality has been included in the reflectivity plugin since version 2.2.0. Multiple instrument can defined
    in the instrument definition dialog. By double clicking on the simulation function definition in the simulation tab
    the instrument to use for that particular simulation can be chosen.

Assuming that you want to use the reflectivity plugin the following hack has to be done to the script.
1. Define your new instruments manually in the code. This has to be done outside the the BEGIN and END sections of the instrument definition
2. Inside the Sim function - store the the original inst parameters
3. Just before the dataset that should use a different instrument redefine the instrument and change the wavelength of the scattering length table, if needed.
4. Copy over the original settings to the inst object again.
::

    import models.spec_nx as model
    from models.utils import UserVars, fp, fw, bc, bw

    # BEGIN Instrument DO NOT CHANGE
    inst = model.Instrument(footype = 'gauss beam',probe = 'neutron',beamw = 0.2,resintrange = 2,tthoff = 0.0,pol = 'uu',wavelength = 4.4,respoints = 5,Ibkg = 0.0,I0 = 2,samplelen = 50.0,restype = 'full conv and varying res.',coords = 'tth',res = 0.001,incangle = 0.0)
    fp.set_wavelength(inst.wavelength)
    #Compability issues for pre-fw created gx files
    try:
        fw
    except:
        pass
    else:
        fw.set_wavelength(inst.wavelength)
    # END Instrument
    # Step 1. Defining instruments
    xraydiff = model.Instrument(footype = 'gauss beam',probe = 'x-ray',beamw = 0.2,resintrange = 2,tthoff = 0.0,wavelength = 1.54,respoints = 5,Ibkg = 0.0,I0 = 2,samplelen = 50.0,restype = 'full conv and varying res.',coords = 'tth',res = 0.01)
    fp.set_wavelength(xraydiff.wavelength)

    # BEGIN Sample DO NOT CHANGE
    Amb = model.Layer(b = 0, d = 0.0, f = (1e-20+1e-20j), dens = 1.0, magn_ang = 0.0, sigma = 0.0, xs_ai = 0.0, magn = 0.0)
    SiO = model.Layer(b = bc.Si.real + bc.O.real*2, d = 1205, f = fp.Si + fp.O*2, dens = 0.026, magn_ang = 0.0, sigma = 2, xs_ai = 0.0, magn = 0.0)
    Sub = model.Layer(b = bc.Si.real, d = 0.0, f = fp.Si, dens = 8/5.443**3, magn_ang = 0.0, sigma = 2, xs_ai = 0.0, magn = 0.0)

    surf = model.Stack(Layers=[SiO], Repetitions = 1)

    sample = model.Sample(Stacks = [surf], Ambient = Amb, Substrate = Sub)
    # END Sample

    # BEGIN Parameters DO NOT CHANGE
    cp = UserVars()
    # END Parameters


    def Sim(data):
        # Step 2. store the original inst parameters
        default_pars = inst._todict()
        I = []
        # You only need the line below if you work with two different x-ray instruments.
        #fp.set_wavelength(inst.wavelength)
        # BEGIN Dataset 0 DO NOT CHANGE
        I.append(sample.SimSpecular(data[0].x, inst))
        # END Dataset 0
        # Step 3. Copying all parameters from instrument xraydiff to inst
        inst._fromdict(xraydiff._todict())
        # You only need the line below if you work with two different x-ray instruments.
        fp.set_wavelength(xraydiff.wavelength)
        # BEGIN Dataset 1 DO NOT CHANGE
        I.append(sample.SimSpecular(data[1].x, inst))
        # END Dataset 1
        # Step 4. Copying all instrument parameter from the original inst to inst.
        inst._fromdict(default_pars)
        return I

Hopefully, this will become automated in the future so that it can be done from within the plugin.