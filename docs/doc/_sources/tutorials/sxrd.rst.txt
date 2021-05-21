.. _tutorial-sxrd:

******************************
Surface X-diffraction tutorial
******************************

At the present GenX implements two different facilities to handle sxrd data.

  * SXRD data loader
  * SXRD model to calculate sxrd data

Loading SXRD data
=================
First the data loader has to be changed to load files for sxrd. Open the menu
:menuselection:`Settings-->Data Loaders` (or press Shift-Ctrl-D) and an dialog pops up.
Choose the sls_sxrd data loader in the drop-down box that appear. The data is loaded by marking the data set
where the rods should be inserted and choose File-Import-Import Data (It is also possible to click on the small open
icon above the data sets). A dialog appears and the data file to load is chosen. An example for an valid data file
can be found in :download:`rods.dat<_attachments/surfdiff/rods.dat>` or see below.
::

    # Test file created: 2009-04-14
    # Headers:
    #h        k         l         I         Ie
    0.000e+00 0.000e+00 1.000e-01 0.000e+00 0.00e+00
    0.000e+00 0.000e+00 1.100e-01 0.000e+00 0.00e+00
    0.000e+00 0.000e+00 1.200e-01 0.000e+00 0.00e+00
    0.000e+00 0.000e+00 1.300e-01 0.000e+00 0.00e+00
    0.000e+00 0.000e+00 1.400e-01 0.000e+00 0.00e+00
    0.000e+00 0.000e+00 1.500e-01 0.000e+00 0.00e+00
    0.000e+00 0.000e+00 1.600e-01 0.000e+00 0.00e+00


The data format that this data loader wants is a 5-column consisting of h, k, l, Intensity and Intensity error
separated with a whitespace (comma, tabs..). The default token for a comment is a #. The settings for the data
loaders are accessed through Setting-Import Settings (or Shift-Ctrl-I). If you would happen to need to create
your own test files you can use the python script :download:`make_sxrd_data.py <_attachments/surfdiff/make_sxrd_data.py>`.
.. image::_attachments/surfdiff/data_loaded.png

When the data is loaded **new** data sets are appended automatically. Each data set is given the name of the type '(h, k)'. If you started from the beginning you should also remove the data set 'Data0' since the data sets are appended to the previous sets.

Creating an SXRD model
======================
Okay this is the tricky part first off we will create a simple model consisting of and 2 unit cell thick L
aAlO3 film on top of a SrTiO3 substrate. The script will look something like this:
::

    # 1
    import models.sxrd as model

    # 2 Defining the unit cell parameters
    unitcell = model.UnitCell(3.945, 3.945, 3.945, 90, 90, 90)
    # 3 Define the instrument
    inst = model.Instrument(wavel = 1.0, alpha = 1.0)

    # 4 Defining the bulk
    bulk = model.Slab()
    # 4.a Define the atoms
    bulk.add_atom('Sr', 'sr', 0.0, 0.0, 0.0, 0.08, 1.0)
    bulk.add_atom('Ti', 'ti', 0.5, 0.5, 0.5, 0.08, 1.0)
    bulk.add_atom('O1', 'o', 0.5, 0.0, 0.5, 0.08, 1.0)
    bulk.add_atom('O2', 'o', 0.0, 0.5, 0.5, 0.08, 1.0)
    bulk.add_atom('O3', 'o', 0.5, 0.5, 0.0, 0.08, 1.0)

    # 5 Creating an LaAlO3 unit cell - note asymmetric unit cell
    laouc =  model.Slab(c = 1.05)
    # 5.a Define the atoms
    laouc.add_atom('La', 'la', 0.0, 0.0, 0.0, 0.08, 1.0, 1)
    laouc.add_atom('Al', 'al', 0.5, 0.5, 0.5, 0.08, 1.0, 1)
    laouc.add_atom('O1', 'o', 0.5, 0.5, 0.0, 0.08, 1.0, 1)
    laouc.add_atom('O2', 'o', 0.5, 0.0, 0.5, 0.08, 1.0, 2)

    # 6 Symmetry operations
    p4 = [model.SymTrans([[1, 0],[0, 1]]), model.SymTrans([[-1, 0],[0, -1]]),
               model.SymTrans([[0, -1],[1, 0]]), model.SymTrans([[0, 1],[-1, 0]]) ]

    # 7 Creating the sample with one unit cell
    sample = model.Sample(inst, bulk, [laouc], unitcell)

    #8 Put the right symmetry in the surface
    sample.set_surface_sym(p4)

    # 9 Define the Sim function
    def Sim(data):
        I = []
        #9.a loop through the data sets
        for data_set in data:
            # 9.b create all the h,k,l values for the rod (data_set)
            h = data_set.extra_data['h']
            k = data_set.extra_data['k']
            l = data_set.x
            # 9.c. Calculate the struct factor
            f = sample.calc_f(h, k, l)
            # 9.d Calculate the intensity
            i = abs(f)**2
            # 9.e Append the calculated intensity to the list I
            I.append(i)
        return I


So to create a model script the following actions has to be made:

1. The first row imports the necessary model ``import models.sxrd as model``.
2. Create a ``UnitCell(a, b, c, alpha, beta, gamma)`` this object contains all the lattice parameters of the
   substrate. This also relates the (h, k, l) values in the measurement to the real world.
3. Create an ``Instrument(wavel, alpha)``. This contains all the parameters related to the measurement. Wavelength
   in Angstroms and incidence angle in degrees.
4. So now we define the bulk ``Slab``, this takes the instrument as input.

   a. Add all the atoms to the bulk (here we use a P1 symmetry), the command has this syntax
      ``bulk.add_atom(id, element,x, y, z, u, occ)``. ``id`` denotes a unique id for the atom added. ``element``
      is the element in question which has to defined in the f and rho table. ``x``, ``y``, ``z`` is the
      base position of the atom (movements of atoms is made through the variables ``dx``, ``dy``, ``dz``}) . ``u``
      is the Debye-Waller parameter (isotropic) and `occ` is the occupancy of the atom.

5. Next up is the same procedure for the film atoms. Here we use the same class ``Slab`` but we create it
   with a twist. As an optional arguments we insert ``c = 1.05``. This will make the c-axis
   constant of this ``Slab`` 5% longer than the one defined in the ``unitcell``.

   a. Now we add all the symmetry in-equivalent atoms to this Slab, ``laouc``, assuming that we have in plane
      p4 symmetry (more about the symmetry later on). Also there is then an extra argument to the `add_atom` method.
      The call is now on the form: `laouv.add_atom(id, element, x, y, z, u, occ, mult)`. The optional parameter ``mult``
      is then the multiplicity of the site, i.e. how many atoms there are of this atomic site in the unitcell
      (it is tabulated in the international tables of crystallography <http://it.iucr.org/>).
      For a non-special position (general) this will be 4. For the (0.5,0.5) position this will
      then be 1 as it will occur only one time.

6. Now both the bulk unit cell and the surface is defined as slabs. What we have to do now is to define the
   symmetry operations for the top layer. This is done by creating `SymTrans` objects and placing them in a list so we
   build up all allowed symmetry operations of the plane group. The call is:
   ``SymTrans(P = [[1, 0],[0, 1], t = [0, 0])`` where also the default values are shown. ``P`` is the rotation
   matrix of the transformation and `t` is the displacement vector. The default value is creating the identity
   matrix for ``P`` and the no displacement at all for ``t``.
7. A sample is creating by combining the instrument, bulk slab, surface slabs and the unitcell. Note that several
   surface slabs can be stacked on top of each other to create more complex structures. Call:
   ``Sample(inst, bulk_slab, surface_slabs, unit_cell)`` .
8. To be added
9. Now we need to define the `Sim` function this is a mandatory function that has to included
   in **all** model scripts. It takes a data structure as input. First we create an empty list which will contain
   the calculated data. This is the function that will be executed when you press on the simulate and when
   you fit your data.

   a. Then we loop through all data sets that we have loaded.
   b. For each ``data_set`` we extract the h,k and l values from the data file. Note that h,k is loaded as extra
      data but the l values which is the independent variable in the plots is retrieved by ``data_set.x``.
      (As a side note ``data.extra_data`` is an dictionary of data which is created by the sls_sxrd data loader)
   c. Then we calculate the _complex_ structure factor for the ``sample``. The syntax should be obvious.
   d. The intensity the square of the absolute of the structure factor is calculated and is appended to our
      result list ``I``
   e. The result list (which has to have exactly the same length as the number of data sets),
      ``I``, is returned from the ``Sim`` function.

Making a simulation
===================
To simulate the data click on the yellow lightning in the main tool bar. The following screen should appear.

![Screenshot - simulation, linear scale](/apps/trac/genx/raw-attachment/wiki/IntroGuides/SurfDiff/simulation_lin.png)

In order to see anything the scaling has to change from linear to logarithmic. Right click on the plot and choose
:menuselection:`y-scale-->log`.

.. image::_attachments/surfdiff/simulation_log.png

This should make the plot a bit clearer. However, we still have a bunch of lines on top of each other.

Lets make som of the data sets non-active. Mark all data sets in the right data list except the first one,
right click and choose Toggle show. This turn on and off the plotting of the selected data sets. Toggle Active
sets a flag whether or not to the data should be simulated. Naturally this flag must be read and used in the simulation
function to have an effect.

.. image::_attachments/surfdiff/selecting_data_sets.png


Adding more layers
==================
Usually I am more interesting in fitting and simulating films that are a couple of unitcells thick.
So here a couple of tricks to make that that easy. If you just want to add identical unitcells on top of each
other you can just expand the list, to simulate lets say 3 UC:
::

    # 7 Creating the sample with one unit cell
    sample = model.Sample(inst, bulk, [laouc, laouc, laouc], unitcell)


Another possiblity is to use the smart list functionality that python offers, which makes the following
code equivalent to the top one::

    # 7 Creating the sample with one unit cell
    sample = model.Sample(inst, bulk, 3*[laouc], unitcell)


If you have several non-identical layers lets say another ``Slab`` called ``stouc`` we can make an 2 UC over
layer on top of the ``loauc``.
::

    # 7 Creating the sample with one unit cell
    sample = model.Sample(inst, bulk, [laouc, laouc, laouc, stouc, stouc], unitcell)


which can also be written as::

    # 7 Creating the sample with one unit cell
    sample = model.Sample(inst, bulk, 3*[laouc] + 2*[stouc], unitcell)


An important consequence of reusing the defined unitcells is that **all** the parameters in the repeated unitcells is
the same. For example if you would like to fit identical displacement parameters for several layers this is the way
to go. Eventually you might want to decouple the different layers and fit them separately. Then you have to make
a copy each slab into a separate object and place into a list. The following code copies one slab into a new one with
another name.
::

    sto_other = stouc.copy()


The object ``sto_other`` will then contain atoms at the same position as ``stouc`` but when a property of ``stouc``
is changed it will not affect the copy, ``sto_other``. Note that if you would write
::

    sto_other = stouc


a change in ``stouc`` or ``sto_other`` would affect the other! Naturally, in this object has also to be
included in the sample in order to be simulated.
::

    # 7 Creating the sample with one unit cell
    sample = model.Sample(inst, bulk, 3*[laouc] + [sto_other, stouc], unitcell)


Defining parameters to fit
==========================
Before we proceed with the more advanced topics we will briefly review how to define which parameters to fit.
As a user you should be aware that in GenX _everything_ can be fitted. However, it is no guarantee that the parameters
that you have chosen to fit make any sense at all! So you should use your judgment and think about the physical model
before you decide to fit (or freeze) a parameter. Fitting is also a little bit like craftsmanship, so in order to be
good at it you have to practice in using the tools.

The definition of which parameters the program fits is done in the tab Grid (lower panel). There are 6 columns.
The first, ``Parameter``,is the name of the _function_ that sets the parameter, next, `Value}},
the default/refined value to set the parameter to. Column number three, {{{Fit` selects if the parameter in this row
should be refined. Next the boundaries for the refinement is given by the `Min` and `Max` columns. The last column
`Errors` display the calculated result of the error calculation that can be conducted after a fit.

No to find the name of the parameter right click on the a box in the parameter column (Note the box is not allowed
to be in edit mode with a cursor blinking). This will cause a pop-up menu to appear with all possible set
functions for the objects you have defined in the script. All function that appear in this menu and as parameters
will appear on the form ``object_name.set_parameter`` for example to fit the slab-global c-axis of the ``stouc``
::

    stouc.set_c


would be chosen. Most parameter will also define its current value in the ``Value`` column and define
a +/- 25% span between the ``Min`` and ``Max`` columns. So, care must be taken that the boundaries is
physical and relevant to the parameter you have chosen.

The grid can also be used as a quick way to play around with the parameters in the model.
Something to keep in made is the following:

1. When you press simulate the script will be compiled.
2. The Sim function will **not** be evaluated yet.
3. The values in the Grid will be set to their values in the ``Value`` column.
4. The Sim function will be evaluated.

When you build your model you think about that when fitting the following procedure will be followed:

1. Set the parameter values.
2. Evaluate the Sim function.
3. Calculate the figure of merit.

So if you want to do something special to your parameters, more about that later on, you can not do it
outside the ``Sim`` function as that code will only be evaluated as you simulate and recompile the model.

If you would like to try a quick fit just press the green arrow in the tool bar and of you go.
Keep an eye on the status bar at the bottom of the window. This will display important information
about how the it proceeds and what the program does. To stop the fit press the red stop sign in the toolbar and
wait until a dialog appear which asks you if you want to keep the Values the program has fitted. If you
would like to resume the fit, if you, for example, stopped it to early, press the round green arrow
to the right of the stop button.

During fitting you should keep an eye on the FOM tab and the Pars tab which shows the progress of the algorithm.
For a more detailed discussion about this see the x-ray tutorial :ref:_tutorials-xrr-fitting.

Scaling the simulation
======================
When fitting and simulatiing one very important thing is to also fit/change the scale factor. One can fit the
parameter ``inst.inten`` which is the "incoming intensity". This parameter multiplies with the structure
factor when using the ``sample.calc_f`` function. A more effective way of doing it when fitting data, to my opinion,
is to scale each simulation separately. The sxrd model contains two function that accomplish this task:
``model.scale_sim(data, I)`` and ``model.scale_sqrt_sim(data, I)``. Both of these functions will return a
new intensity array that are the least-squared fitted simulation to the data with respect to the scale factor.
The minimization is done by analytically by solving :math:`\mathrm{min}_s \sum_i \left(D_i - sI_i\right)^2` or
:math:`\mathrm{min}_s \sum_i \left(\sqrt{D_i} - s\sqrt{I_i}\right)^2`.
This function should be placed right before returning the value in the ``Sim`` function.
::

            ...
            # 9.e Append the calculated intensity to the list I
            I.append(i)
        I = model.scale_sqrt_sim(data, I)
        return I


Grouping atoms
==============
A common sense approach can be that several atoms should move together and not independently of each other.
For example, strain field will compress/expand an epitaxial film and consequently the lattice parameter will change.
We have already briefly dealt with this as we talked about Slab`s and its parameter ``c`` which can scale the
thickness of the slab. This should be the first attempt to use when fitting the data. Other, more elaborate,
models might be to move atoms of the same element/site in the same manner. One obvious example of this if an alloyed
site is simulated. This is where atoms group can come in handy. The class that handles this is called ``AtomGroup``.

Basic usage
-----------
Naturally there are different ways of creating and working with these groups. There are two ways to create a group:

1. The `Slab.add_atom` returns an `AtomGroup` object. Easy to use when only a small number of atoms has to chosen.
   For example::

        la_atom = laouc.add_atom('La', 'la', 0.0, 0.0, 0.0, 0.08, 1.0, 4)

2. Fetch one atom from a Slab with its unique identity (name), ``id``. This is done by typing the ``id`` between
   brackets (cmp. dictionary lookup)::

        la_atom = laouc['La']

3. Atoms can be searched by the ``Slab.find_atoms`` method. This lets you use a logical expression to locate
   the atoms. To locate an atom all at a site (0,0,0) one would write::

        la_atom = laouc.find_atoms('x == 0 and y == 0 and z == 0')



If we would have included two atoms at (0,0,0), perhaps interdiffusion between Sr and La, these two
would now be moved together if the functions ``la_atom.setdx``, ``la_atom.setdy``,``la_atom.setdz`` would be called.
The same is valid for the occupation ``oc`` and the Debye-Waller parameter, ``u``.

There are more to this thing with groups as they stand now they are handy but groups can also be added
together to form super groups. For example I want to have the same Debye-Waller parameter of all oxygen
in both the ``stouc`` slab and the ``laouc`` slab. This could be done with::

    all_ox = loauc.find_atoms('el == "o"') + stouc.find_atoms('el == "o"')


As a parameter in the grid we would choose the function ``all_ox.setu`` as a parameter.

Finally as a practical note. It is preferable to use version 2 with brackets(``[]``) if you would like to group
together atoms by adding them since you will not clutter up the name space with a lot of parameters you will never
use by assigning variables to each atom.

Composition coupling
--------------------
One common task for coupling atoms together is for fitting compositions instead of occupancies. This is
especially true for solid-solid interfaces where the total occupancy usually can be considered to be 1.0.
Lets assume we have a LaAlO/SrTiO interface and we want to fit the composition of the interface layer. First we define
a interface ``Slab`` consisting of a mix the two elements.
::

    slato =  model.Slab(c = 1.05)
    # 5.a Define the atoms
    slato.add_atom('La', 'La', 0.0, 0.0, 0.0, 0.08, 1.0, 1)
    slato.add_atom('Sr', 'Sr', 0.0, 0.0, 0.0, 0.08, 1.0, 1)
    slato.add_atom('Al', 'Al', 0.5, 0.5, 0.5, 0.08, 1.0, 1)
    slato.add_atom('Ti', 'Ti', 0.5, 0.5, 0.5, 0.08, 1.0, 1)
    slato.add_atom('O1', 'O', 0.5, 0.5, 0.0, 0.08, 1.0, 1)
    slato.add_atom('O2', 'O', 0.5, 0.0, 0.5, 0.08, 1.0, 2)


So for creating a composition pair of the Sr and La we type::

    sl = slato['La']|slato['Sr']


The ``AtomGroup`` will now have member functions ``sl.setcomp`` to set the La composition and ``sl.setoc`` to set
the occupancy. Note that the occupancy is calculated as: ``oc_Sr_ = (1 - comp)*oc, oc_La_ = comp*oc``. Per default
the positions and debye waller parameters are **not** coupled (In this case they will refer to the La atom).
To do this use the exclusive or operator instead
::

    sl = slato['La']^slato['Sr']


, which couples all parameter through ``sl``. The distinction becomes more important if one has more than two
atoms and do nesting of the operators (assume we have an Y atom as well)::

    sl = slato['La']^slato['Sr']
    ysl = slato['Y']^sl


will connect all positional and Debye-Waller parameters together but...
::

    sl = slato['La']^slato['Sr']
    ysl = slato['Y']|sl


this will couple the positions of La and Sr but not to the Y. These operators offer a large degree of flexibility
but they also probably create hard to track error - so they should be used wisely.

General coupling
================
As you probably understand the concept with AtomGroup`s cant take you all the way to complete generality.
This naturally more complicated but incredibly powerful as _anything_ can be implemented. The class which is
used a container class for this is called `UserVars`. To use it you have to import it by inserting the following text
(preferably in the top of the script)
::

    from models.utils import UserVars


Next you create a new object of the class::

    cp = UserVars()


You can as many of the objects as possible, a good way to structure your parameters, with as many parameters as
possible inside. A new parameter is created by calling the method ``new_var``::

    cp.new_var('comp', 0.5)


Now there is a new variable with the name ``cp.comp`` and it is initialized to have a default value of 0.5. It can
also be accessed by right clicking on the grid and choose it as a fitting parameter. However, now we have to connect
the parameter to our sample. Lets assume out model has been extended to include an interdiffused slab between
the ``laouc`` and ``stouc``.
::

    # 5 Creating an LaAlO3 unit cell - note asymmetric unit cell
    laouc =  model.Slab(c = 1.05)
    # 5.a Define the atoms
    laouc.add_atom('La', 'la', 0.0, 0.0, 0.0, 0.08, 1.0, 1)
    laouc.add_atom('Al', 'al', 0.5, 0.5, 0.5, 0.08, 1.0, 1)
    laouc.add_atom('O1', 'o', 0.5, 0.5, 0.0, 0.08, 1.0, 1)
    laouc.add_atom('O2', 'o', 0.5, 0.0, 0.5, 0.08, 1.0, 2)
    # Create an SrTiO3 unit cell
    stouc =  model.Slab(c = 1.05)
    # 5.a Define the atoms
    stouc.add_atom('Sr', 'sr', 0.0, 0.0, 0.0, 0.08, 1.0, 1)
    stouc.add_atom('Al', 'al', 0.5, 0.5, 0.5, 0.08, 1.0, 1)
    stouc.add_atom('O1', 'o', 0.5, 0.5, 0.0, 0.08, 1.0, 1)
    stouc.add_atom('O2', 'o', 0.5, 0.0, 0.5, 0.08, 1.0, 2)
    # Create an interface layer
    lstouc =  model.Slab(c = 1.05)
    # 5.a Define the atoms
    lstouc.add_atom('Sr', 'sr', 0.0, 0.0, 0.0, 0.08, 1.0, 1)
    lstouc.add_atom('La', 'la', 0.0, 0.0, 0.0, 0.08, 1.0, 1)
    lstouc.add_atom('Al', 'al', 0.5, 0.5, 0.5, 0.08, 1.0, 1)
    lstouc.add_atom('O1', 'o', 0.5, 0.5, 0.0, 0.08, 1.0, 1)
    lstouc.add_atom('O2', 'o', 0.5, 0.0, 0.5, 0.08, 1.0, 2)

    # 6 Symmetry operations
    p4 = [model.SymTrans([[1, 0],[0, 1]]), model.SymTrans([[-1, 0],[0, -1]]),
               model.SymTrans([[0, -1],[1, 0]]), model.SymTrans([[0, 1],[-1, 0]]) ]

    # 7 Creating the sample with one unit cell
    sample = model.Sample(inst, bulk, [stouc, lstouc, laouc], unitcell)

    # Couple the parameter!
    # Create the UserVar
    cp = UserVars()
    cp.new_var('comp', 0.5)


So the explicit coupling has to be done *within* the ``Sim`` function otherwise the coupling will not work
when you fit it at a later stage. So the ``Sim`` function would change to
::

    # 9 Define the Sim function
    def Sim(data):
        # Parameter coupling goes before simulation!
        lstouc.setSroc(1.0 - cp.comp)
        lstouc.setLaoc(cp.comp)
        I = []
        #9.a loop through the data sets
        for data_set in data:
            # 9.b create all the h,k,l values for the rod (data_set)
            h = data_set.extra_data['h']
            k = data_set.extra_data['k']
            l = data_set.x
            # 9.c. Calculate the struct factor
            f = sample.calc_f(h, k, l)
            # 9.d Calculate the intensity
            i = abs(f)**2
            # 9.e Append the calculated intensity to the list I
            I.append(i)
        return I


As seen above the different occupancies are set with the help of the cp.comp. Note that the parameter coupling
is *within* the ``Sim`` function and *before* we calculate the rods. Also a good code of conduct is never assign a
user variable a value from within the ``Sim`` function. This can cause a very erratic behavior when fitting
(the evaluation depends on its history).

I hope you have understood that with ``UserVars`` you can do just about anything. Unfortunately, this will
also involve a lot of typing and long scripts. If you want to know more about the general philosophy of the
layout with ``UserVars`` and the simulation the tutorial about writing models might be good next step
:ref:`tutorial-writing-model`.
