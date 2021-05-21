.. _beta-rough-model:

********************
Beta Roughness Model
********************
This tutorial describes how to incorporate the general roughness model for crystal truncation rod data
after Robinson [ROB86]_.

To use this roughness model you will need the following:

1. Data file that includes L-value of nearest Bragg peak (LB) and distance in reciprocal lattice units between adjacent
   Bragg peaks (dL) for each data point. If dL is the same for all Bragg peaks on a given rod,
   you may use the same LB for all.
   ::

      # Sample data file with Bragg peak position and spacing
      # H	K	L	I	Ierr	LB	dL
      0	0	1.0	0	0	3	3
      0	0	1.1	0	0	3	3
      0	0	1.2	0	0	3	3
      0	0	1.3	0	0	3	3
      0	0	1.4	0	0	3	3
      0	0	1.5	0	0	3	3
      0	0	1.6	0	0	3	3

2. SXRD model script modified to include the beta parameter as a user variable, and
   roughness in the structure factor calculation. Add/replace the following code in the model script
   shown at :ref:`tutorial-sxrd`
   ::

      # 3.a Define beta for roughness model
      rgh=UserVars()
      rgh.new_var('beta', 0.0)


      # 9 Define the Sim function
      def Sim(data):
         I = []
         beta = rgh.beta
         #9.a loop through the data sets
         for data_set in data:
            # 9.b create all the h,k,l,LB,dL values for the rod (data_set)
            h = data_set.extra_data['h']
            k = data_set.extra_data['k']
            l = data_set.x
            LB = data_set.extra_data['LB']
            dL = data_set.extra_data['dL']
            # 9.c. calculate roughness using beta model
            rough = (1-beta)/((1-beta)**2 + 4*beta*np.sin(np.pi*(l - LB)/dL)**2)**0.5
            # 9.d. Calculate the structure factor
            f = rough*sample.calc_f(h, k, l)
            # 9.e Calculate |F|
            i = abs(f)**2
            # 9.f Append the calculated intensity to the list I
            I.append(i)
         return I


3. In your parameter grid, select an empty row, right click, and select :menuselection:`UserVars-->rgh.setBeta`

References
==========

.. [ROB86] ROBINSON, I., 1986. CRYSTAL TRUNCATION RODS AND SURFACE-ROUGHNESS. Physical Review B 33, 3830-3836.
