.. _tutorial-diffev:

***********************************
Fitting with Differential Evolution
***********************************

A review of using the Differential Evolution algorithm to fit reflectivity data has been published in J
ournal of Applied Crystallography [Bjorck11]_. If you do not have access to
the journal just send me an e-mail artur.glavic@psi.ch.

In summary, one can say that choosing a minimum population size of 30-50 and k_r = k_m = 0.5-0.9 provides a rather
robust algorithm for reflectivity refinements. See the summary figure below for good parameter ranges.
If you need more stability the population size can be increased and/or
the k_r be lowered. For more details see the paper.

.. figure:: _attachments/diffev/diffev_parameters.jpg
    :width: 60%

    Good parameter ranges for the relevant DE methods and their influence on convergence (Figure 9 from [Bjorck11]_)

References
==========

.. [Bjorck11] `M. Björck J. Appl. Cryst. (2011) vol. 44, p. 1198-1204. <http://dx.doi.org/10.1107/S0021889811041446>`_