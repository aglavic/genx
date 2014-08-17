#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
.. module:: xmcd_moment
   :synopsis: A physical model for the 2p edges in x-ray magnetic resonant reflectivity.

.. moduleauthor:: Matts Björck <matts.bjorck@gmail.com>

This module provides he framework to model the spectroscopic response of the scattering lengths
around an 2p absorption edge. The approach was first presented by van der Laan [Laan97]_ for modelling the line shape
of the XMCD (x-ray magnetic circular dichrosim) spectrum of the 3d metals and later used experimentally by
Gold, Bayer and Goering [Gold04]_ for fitting XMCD spectra and extracting moments.

The ground state moment model has also been justified by Dörfler and Fähnle [Dörfler06]_.

Theory
======
All the equation shown in this section are reproduced from [Laan97]_ if not otherwise stated.

The total intensity, :math:`I_{j,m}^{(r)}`, from an :math:`j,m` level  can be written as

.. math:: I_{j,m}^{(a), \mathrm{tot}} = \sum_{x,y,z}  \left<\underline{w}^{x,y,z}\right> \sum_r C_j^{xyzar}
          u_{jm}^r = \sum_{x,y,z}  \left<\underline{w}^{x,y,z}\right> h_{jm}^{xyza},

where :math:`C_j^{xyzar}` is the probability of creating a core hole :math:`j` with a multipole moment :math:`r` using
a polarised light that belongs to the ground state moment :math:`\left<\underline{w}^{x,y,z}\right>`. The underscore
signifies that the properties belong to the holes. The physical meaning of the different ground state moments can be
found in the table below.

+-------------------------------------------+----------------------------------------------------+
| Ground state moment                       | Meaning                                            |
+===========================================+====================================================+
| :math:`\left<\underline{w}^{000}\right>`  | The number of holes, :math:`n_h`                   |
+-------------------------------------------+----------------------------------------------------+
| :math:`\left<\underline{w}^{110}\right>`  | Spin orbit coupling :math:`-\sum_i{l_i \dot s_i}`  |
+-------------------------------------------+----------------------------------------------------+
| :math:`\left<\underline{w}^{101}\right>`  | The orbital magnetic moment :math:`L_z/2`          |
+-------------------------------------------+----------------------------------------------------+
| :math:`\left<\underline{w}^{011}\right>`  | The spin magnetic moment :math:`2S_z`              |
+-------------------------------------------+----------------------------------------------------+
| :math:`\left<\underline{w}^{211}\right>`  | The "magnetic dipole" operator :math:`7T_z/2`      |
+-------------------------------------------+----------------------------------------------------+
| :math:`\left<\underline{w}^{202}\right>`  | The quadrupole moment :math:`-Q_{zz}/2`            |
+-------------------------------------------+----------------------------------------------------+
| :math:`\left<\underline{w}^{112}\right>`  | ??                                                 |
+-------------------------------------------+----------------------------------------------------+
| :math:`\left<\underline{w}^{312}\right>`  | ??                                                 |
+-------------------------------------------+----------------------------------------------------+
| :math:`\left<\underline{w}^{303}\right>`  | ??                                                 |
+-------------------------------------------+----------------------------------------------------+
| :math:`\left<\underline{w}^{213}\right>`  | ??                                                 |
+-------------------------------------------+----------------------------------------------------+
| :math:`\left<\underline{w}^{413}\right>`  | ??                                                 |
+-------------------------------------------+----------------------------------------------------+
| :math:`\left<\underline{w}^{404}\right>`  | ??                                                 |
+-------------------------------------------+----------------------------------------------------+
| :math:`\left<\underline{w}^{314}\right>`  | ??                                                 |
+-------------------------------------------+----------------------------------------------------+
| :math:`\left<\underline{w}^{415}\right>`  | ??                                                 |
+-------------------------------------------+----------------------------------------------------+

Unpolarised light correspond to :math:`a = 0`, circular dichroism
correspond to :math:`a = 1` and linear dichroism to :math:`a = 2`. The values for :math:`h_{jm}^{xyza}` has been
tabulated in [Laan97]_ and can be found reproduced below.

.. tabularcolumns:: |l|c|c|c|c|c|c|

Values for :math:`h_{jm}^{xyza}` for each ground state moment and j and light polarisation :math:`a`.

+-------------------------------------------+-------------------------------------------------------------+--------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+
|                                           |                         a = 0                               |                         a = 1                                                        |                                                     a = 2                            |
+-------------------------------------------+------------------------------+------------------------------+-----------------------------------------------------+--------------------------------+------------------------------------------------------+-------------------------------+
| :math:`\left<\underline{w}^{xyz}\right>`  |  :math:`j = \frac{3}{2}`     | :math:`j = \frac{1}{2}`      | :math:`j = \frac{3}{2}`                             |    :math:`j = \frac{1}{2}`     |    :math:`j = \frac{3}{2}`                           |   :math:`j = \frac{1}{2}`     |
+===========================================+==============================+==============================+=====================================================+================================+======================================================+===============================+
| :math:`\left<\underline{w}^{000}\right>`  |    :math:`2u_{jm}^0`         |   :math:`u_{jm}^0`           | :math:`\frac{5}{9}u_{jm}^1`                         |   :math:`\frac{1}{3}u_{jm}^1`  | :math:`\frac{1}{5}u_{jm}^2`                          |                               |
+-------------------------------------------+------------------------------+------------------------------+-----------------------------------------------------+--------------------------------+------------------------------------------------------+-------------------------------+
| :math:`\left<\underline{w}^{110}\right>`  |    :math:`u_{jm}^0`          |   :math:`-u_{jm}^0`          | :math:`\frac{4}{9}u_{jm}^1`                         |   :math:`-\frac{1}{3}u_{jm}^1` | :math:`\frac{2}{5}u_{jm}^2`                          |                               |
+-------------------------------------------+------------------------------+------------------------------+-----------------------------------------------------+--------------------------------+------------------------------------------------------+-------------------------------+
| :math:`\left<\underline{w}^{101}\right>`  | :math:`\frac{5}{3}u_{jm}^1`  |   :math:`-u_{jm}^1`          | :math:`2u_{jm}^0 + \frac{2}{5}u_{jm}^2`             |  :math:`u_{jm}^0`              | :math:`\frac{2}{3}u_{jm}^1`                          | :math:`\frac{2}{5}u_{jm}^1`   |
+-------------------------------------------+------------------------------+------------------------------+-----------------------------------------------------+--------------------------------+------------------------------------------------------+-------------------------------+
| :math:`\left<\underline{w}^{011}\right>`  | :math:`\frac{10}{9}u_{jm}^1` |   :math:`u_{jm}^1`           | :math:`\frac{1}{3}u_{jm}^0 + \frac{2}{3}u_{jm}^2`   |  :math:`-\frac{1}{3}u_{jm}^0`  | :math:`\frac{2}{45}u_{jm}^1 + \frac{3}{6}u_{jm}^3`   | :math:`-\frac{2}{15}u_{jm}^1` |
+-------------------------------------------+------------------------------+------------------------------+-----------------------------------------------------+--------------------------------+------------------------------------------------------+-------------------------------+
| :math:`\left<\underline{w}^{211}\right>`  | :math:`\frac{2}{9}u_{jm}^1`  | :math:`-\frac{1}{3}u_{jm}^1` | :math:`\frac{2}{3}u_{jm}^0 + \frac{2}{15}u_{jm}^2`  |  :math:`-\frac{2}{3}u_{jm}^0`  | :math:`\frac{22}{45}u_{jm}^1 + \frac{6}{35}u_{jm}^3` | :math:`-\frac{4}{15}u_{jm}^1` |
+-------------------------------------------+------------------------------+------------------------------+-----------------------------------------------------+--------------------------------+------------------------------------------------------+-------------------------------+
| :math:`\left<\underline{w}^{202}\right>`  | :math:`u_{jm}^2`             | :math:`-\frac{2}{3}u_{jm}^1` | :math:`\frac{10}{9}u_{jm}^1`                        |  :math:`\frac{2}{3}u_{jm}^1`   | :math:`2u_{jm}^0 + \frac{2}{7}u_{jm}^2`              | :math:`u_{jm}^0`              |
+-------------------------------------------+------------------------------+------------------------------+-----------------------------------------------------+--------------------------------+------------------------------------------------------+-------------------------------+
| :math:`\left<\underline{w}^{112}\right>`  | :math:`2u_{jm}^0`            |                              | :math:`\frac{34}{45}u_{jm}^1 + \frac{6}{5}u_{jm}^3` |  :math:`-\frac{4}{15}u_{jm}^1` | :math:`\frac{2}{5}u_{jm}^0 + \frac{2}{5}u_{jm}^2`    | :math:`-\frac{2}{5}u_{jm}^0`  |
+-------------------------------------------+------------------------------+------------------------------+-----------------------------------------------------+--------------------------------+------------------------------------------------------+-------------------------------+
| :math:`\left<\underline{w}^{312}\right>`  |                              |                              | :math:`\frac{2}{15}u_{jm}^1 + \frac{3}{35}u_{jm}^3` |  :math:`-\frac{2}{5}u_{jm}^1`  | :math:`\frac{3}{5}u_{jm}^0 + \frac{6}{35}u_{jm}^2`   | :math:`-\frac{3}{5}u_{jm}^0`  |
+-------------------------------------------+------------------------------+------------------------------+-----------------------------------------------------+--------------------------------+------------------------------------------------------+-------------------------------+
| :math:`\left<\underline{w}^{303}\right>`  |                              |                              | :math:`\frac{3}{5}u_{jm}^2`                         |                                | :math:`u_{jm}^1`                                     | :math:`\frac{3}{5}u_{jm}^1`   |
+-------------------------------------------+------------------------------+------------------------------+-----------------------------------------------------+--------------------------------+------------------------------------------------------+-------------------------------+
| :math:`\left<\underline{w}^{213}\right>`  |  :math:`3u_{jm}^3`           |                              | :math:`\frac{6}{5}u_{jm}^2`                         |                                | :math:`\frac{24}{35}u_{jm}^1 + \frac{24}{35}u_{jm}^3`| :math:`-\frac{9}{35}u_{jm}^1` |
+-------------------------------------------+------------------------------+------------------------------+-----------------------------------------------------+--------------------------------+------------------------------------------------------+-------------------------------+
| :math:`\left<\underline{w}^{413}\right>`  |                              |                              |                                                     |                                | :math:`\frac{4}{35}u_{jm}^1 + \frac{4}{35}u_{jm}^3`  | :math:`-\frac{12}{35}u_{jm}^1`|
+-------------------------------------------+------------------------------+------------------------------+-----------------------------------------------------+--------------------------------+------------------------------------------------------+-------------------------------+
| :math:`\left<\underline{w}^{404}\right>`  |                              |                              |                                                     |                                | :math:`\frac{18}{35}u_{jm}^2`                        |                               |
+-------------------------------------------+------------------------------+------------------------------+-----------------------------------------------------+--------------------------------+------------------------------------------------------+-------------------------------+
| :math:`\left<\underline{w}^{314}\right>`  |                              |                              | :math:`\frac{12}{7}u_{jm}^3`                        |                                | :math:`\frac{36}{35}u_{jm}^2`                        |                               |
+-------------------------------------------+------------------------------+------------------------------+-----------------------------------------------------+--------------------------------+------------------------------------------------------+-------------------------------+
| :math:`\left<\underline{w}^{415}\right>`  |                              |                              |                                                     |                                | :math:`\frac{10}{7}u_{jm}^3`                         |                               |
+-------------------------------------------+------------------------------+------------------------------+-----------------------------------------------------+--------------------------------+------------------------------------------------------+-------------------------------+

The :math:`h_{jm}^{xyza}` expressions contain unevaluated :math:`u_{jm}^r` values.

.. math:: u_{jm}^r = \left(2r + 1\right) n_{jr} (-)^{(j - m)} \left(\begin{matrix} j & r & j \\
          -m & 0 & m \end{matrix}\right),

where

.. math:: n_{jr} = \left(\begin{matrix} j & r & j \\ -j & 0 & j \end{matrix}\right),

is a normalisation factor. The symbols written out as matrices are Wigner 3j symbols. These have been evaluated using
RREF [Stone80]_.

=====================  ==============================  ===========================================  =============================================  =============================================
:math:`j`              :math:`n_{j0}`                  :math:`n_{j1}`                               :math:`n_{j2}`                                 :math:`n_{j3}`
=====================  ==============================  ===========================================  =============================================  =============================================
:math:`\frac{3}{2}`     :math:`\frac{1}{2}`             :math:`\frac{1}{2}\sqrt{\frac{3}{5}}`        :math:`\frac{1}{2}\sqrt{\frac{1}{5}}`          :math:`\frac{1}{2}\sqrt{\frac{1}{35}}`
:math:`\frac{1}{2}`     :math:`\frac{1}{\sqrt{2}}`      :math:`\frac{1}{\sqrt{6}}`                   :math:`0`                                      :math:`0`
=====================  ==============================  ===========================================  =============================================  =============================================

Calculating the :math:`u_{jm}^r` for the different combinations of :math:`j`, :math:`m` and :math:`r` yields the
following table.

===================================  =======================  ========================  =======================  =======================
:math:`j, m`                          :math:`u_{jm}^0`         :math:`u_{jm}^1`         :math:`u_{jm}^2`         :math:`u_{jm}^3`
===================================  =======================  ========================  =======================  =======================
:math:`\frac{3}{2}, -\frac{3}{2}`     :math:`\frac{1}{4}`      :math:`-\frac{9}{20}`     :math:`\frac{1}{4}`      :math:`-\frac{1}{20}`
:math:`\frac{3}{2}, -\frac{1}{2}`     :math:`\frac{1}{4}`      :math:`-\frac{3}{20}`     :math:`-\frac{1}{4}`     :math:`\frac{3}{20}`
:math:`\frac{3}{2}, \frac{1}{2}`      :math:`\frac{1}{4}`      :math:`\frac{3}{20}`      :math:`-\frac{1}{4}`     :math:`-\frac{3}{20}`
:math:`\frac{3}{2}, \frac{3}{2}`      :math:`\frac{1}{4}`      :math:`\frac{9}{20}`      :math:`\frac{1}{4}`      :math:`\frac{1}{20}`
:math:`\frac{1}{2}, -\frac{1}{2}`     :math:`\frac{1}{2}`      :math:`-\frac{1}{2}`      :math:`0`                :math:`0`
:math:`\frac{1}{2}, \frac{1}{2}`      :math:`\frac{1}{2}`      :math:`\frac{1}{2}`       :math:`0`                :math:`0`
===================================  =======================  ========================  =======================  =======================

:math:`I_{j,m}^{(a), \mathrm{tot}}` yields the total intensity of a certain state. In order to calculate the line shape
the energy position of each line has to be known as well as its shape function. The energy splitting, :math:`E_s`,
between different :math:`m` is given by exchange field, :math:`H_s` as

.. math:: E_s = H_s m \frac{j(j+1) + s(s+1) - l(l+1)}{2j(j+1)}.

Also, the edge energies of the :math:`L_2` and :math:`L_3` has to be known. This can be implemented as defining the
energy of the :math:`L_3` edge, :math:`E_{L_3}`, and the 2p-SOC which yields the splitting between the levels,
:math:`E_\mathrm{SOC}`.

The shape of a atomic transition will depend on the lifetime of the excitation and consequently it will have an
lorentz shape. However, other broadening mechanisms exists as well. For example, the band structure will yield a
broadening
that corresponds to the band structure. This has successfully been modelled as a linear superposition of different
gaussian contributions [Gold04]. Instrumental resolution is also usually considered to have a gaussian distribution.
The Voigt function which is the convolution between a lorentzian function and a gaussian function is therefore suitable
to use.

References:
===========
.. [Laan97] Laan, G. Van Der. (1997). Line shape of 2p magnetic-x-ray-dichroism spectra in 3d metallic systems.
       Physical Review B, 55(13), 8086–8089.

       Laan, G. (1997). The role of the spin polarization in x-ray magnetic circular dichroism spectra of
       itinerant magnets. Journal of Physics: Condensed Matter, 9, L259–L265.

.. [Gold04] Gold, S., Bayer, a., & Goering, E. (2004). Ground-State-Moment-Analysis: A quantitative tool for X-ray
       magnetic circular dichroism analysis for 3d transition metals. Applied Physics A: Materials Science & Processing,
       78(6), 855–865. doi:10.1007/s00339-003-2442-8

.. [Dörfler06] Dörfler, Fabian (2006). Contributions to the theory of x-ray magnetic dichroism. Max-Planck-Institut für
       Metallforschung Stuttgart and Universität Stuttgart. PhD thesis.

       Dörfler, F., & Fähnle, M. (2006). Theoretical justification of ground-state moment analysis of magnetic
       dichroic x-ray absorption spectra for 3d transition metals. Phys. Rev. B, 74(22),
       224424. doi:10.1103/PhysRevB.74.224424

.. [Stone80] Stone, A and Wood, (1980). C. Comp. Phys. Comm., 21 , 195.

"""

import numpy as np
from scipy import special

from refl import ReflFunction

import refl

def voigt(x, x_0, gamma, sigma):
    """ The complex Voigt function (a convolution of a Lorentzian and a Gaussian function).

    Parameters:
       x (array or float): Independent value where to evaluate the Voigt function.
       x_0 (float or array): Centre of the peak.
       gamma (float or array): Scale parameter (HWHM) of the Lorentzian part.
       sigma (float or array): The standard deviation of the Gaussian part.

    Returns:
       V(complex array): The complex Voigt function evaluated at x

    The complex Voigt function is defined as

    .. math::
        V(x,\sigma,\gamma) = w(z)/(\sigma*sqrt(2*pi)),

    where

    .. math::
        z = (x+i*\gamma)/(\sigma*sqrt(2)).

    :math:`\gamma` is the half width at half max of the Lorentizan component and :math:`sigma` is the standard deviation
    of the Gaussian component.

    References:
       `http://en.wikipedia.org/wiki/Voigt_profile <http://en.wikipedia.org/wiki/Voigt_profile>`_
    """
    z = ((x - x_0) + 1.0J * gamma) / (sigma * np.sqrt(2.))
    return special.wofz(z) / (sigma * np.sqrt(2. * np.pi)) * 1.0J


def calc_h(xyz, j2, m2, a):
    """ Calculate the h values for the given parameters.

    For a full description see the Theory section in module documentation.

    Args:
       xyz (tuple, 3 int): A tuple indicating the ground state moment.
       j2 (int): The j quantum number multiplied by 2 (value 3 or 1).
       m2 (int): The m quantum number multiplied by 2 (value [-j2 .. j2]).
       a (int): The polarisation of the exciting light.

    Returns:
       h (float): Transition probability of the given state with ground state moment 1.
    """

    u0 = {(3, -3): 1./4., (3, -1): 1./4., (3, 1): 1./4., (3, 3): 1./4., (1, -1): 1./2., (1, 1): 1./2.}
    u1 = {(3, -3): -9./20., (3, -1): -3./20., (3, 1): 3./20., (3, 3): 9./20., (1, -1): -1./2., (1, 1): 1./2.}
    u2 = {(3, -3): 1./4., (3, -1): -1./4., (3, 1): -1./4., (3, 3): 1./4., (1, -1): 0., (1, 1): 0.}
    u3 = {(3, -3): -1./20., (3, -1): 3./20., (3, 1): -3./20., (3, 3): 1./20., (1, -1): 0., (1, 1): 0.}

    if (j2, m2) not in u0:
        raise ValueError('The value of j2 (%s) or m2 (%s) is invalid'%(repr(j2), repr(m2)))

    if a not in [0, 1, 2]:
        raise ValueError('The value of a (%s) is invalid'%(repr(a)))

    if xyz == (0, 0, 0):
        if a == 0:
            h_val = u0[j2, m2] if j2 == 3 else u0[j2, m2]
        elif a == 1:
            h_val = 5./9.*u1[j2, m2] if j2 == 3 else 1./3.*u1[j2, m2]
        elif a == 2:
            h_val = 1./5.*u2[j2, m2] if j2 == 3 else 0.
    elif xyz == (1, 1, 0):
        if a == 0:
            h_val = 2.*u0[j2, m2] if j2 == 3 else u0[j2, m2]
        elif a == 1:
            h_val = 4./9.*u1[j2, m2] if j2 == 3 else -1./3.*u1[j2, m2]
        elif a == 2:
            h_val = 2./5.*u2[j2, m2] if j2 == 3 else 0.
    elif xyz == (1, 0, 1):
        if a == 0:
            h_val = 5./3.*u1[j2, m2] if j2 == 3 else -u1[j2, m2]
        elif a == 1:
            h_val = 2.*u0[j2, m2] + 2./5.*u2[j2, m2] if j2 == 3 else u0[j2, m2]
        elif a == 2:
            h_val = 2./3.*u1[j2, m2] if j2 == 3 else 2./5.*u1[j2, m2]
    elif xyz == (0, 1, 1):
        if a == 0:
            h_val = 10./9.*u1[j2, m2] if j2 == 3 else u1[j2, m2]
        elif a == 1:
            h_val = 1./3*u0[j2, m2] + 2./3.*u2[j2, m2] if j2 == 3 else -1./3.*u0[j2, m2]
        elif a == 2:
            h_val = 2./45.*u1[j2, m2] + 3./6.*u3[j2, m2] if j2 == 3 else -2./15.*u1[j2, m2]
    elif xyz == (2, 1, 1):
        if a == 0:
            h_val = 2./9.*u1[j2, m2] if j2 == 3 else -1./3.*u1[j2, m2]
        elif a == 1:
            h_val = 2./3*u0[j2, m2] + 2./15.*u2[j2, m2] if j2 == 3 else -2./3.*u0[j2, m2]
        elif a == 2:
            h_val = 22./45.*u1[j2, m2] + 6./35.*u3[j2, m2] if j2 == 3 else -4./15.*u1[j2, m2]
    elif xyz == (2, 0, 2):
        if a == 0:
            h_val = u2[j2, m2] if j2 == 3 else 0
        elif a == 1:
            h_val = 10./9*u1[j2, m2] if j2 == 3 else 2./3.*u1[j2, m2]
        elif a == 2:
            h_val = 2.*u0[j2, m2] + 2./7.*u2[j2, m2] if j2 == 3 else u0[j2, m2]
    elif xyz == (1, 1, 2):
        if a == 0:
            h_val = 2*u2[j2, m2] if j2 == 3 else 0
        elif a == 1:
            h_val = 34./45*u1[j2, m2] + 6./5*u3[j2, m2] if j2 == 3 else -4./15.*u1[j2, m2]
        elif a == 2:
            h_val = 2./5.*u0[j2, m2] + 2./5.*u2[j2, m2] if j2 == 3 else -2./5.*u0[j2, m2]
    elif xyz == (3, 1, 2):
        if a == 0:
            h_val = 0 if j2 == 3 else 0
        elif a == 1:
            h_val = 2./15*u1[j2, m2] + 3./35*u3[j2, m2] if j2 == 3 else -2./15.*u1[j2, m2]
        elif a == 2:
            h_val = 3./5.*u0[j2, m2] + 6./35.*u2[j2, m2] if j2 == 3 else -3./5.*u0[j2, m2]
    elif xyz == (3, 0, 3):
        if a == 0:
            h_val = 0. if j2 == 3 else 0.
        elif a == 1:
            h_val = 3./15*u2[j2, m2] if j2 == 3 else 0.
        elif a == 2:
            h_val = u1[j2, m2] if j2 == 3 else 3./5.*u1[j2, m2]
    elif xyz == (2, 1, 3):
        if a == 0:
            h_val = 3*u3[j2, m2] if j2 == 3 else 0.
        elif a == 1:
            h_val = 6./5*u2[j2, m2] if j2 == 3 else 0.
        elif a == 2:
            h_val = 24./35.*u1[j2, m2] + 24./35.*u3[j2, m2] if j2 == 3 else -9./35.*u1[j2, m2]
    elif xyz == (4, 1, 3):
        if a == 0:
            h_val = 0. if j2 == 3 else 0.
        elif a == 1:
            h_val = 0. if j2 == 3 else 0.
        elif a == 2:
            h_val = 4./35.*u1[j2, m2] + 4./35.*u3[j2, m2] if j2 == 3 else -12./35.*u1[j2, m2]
    elif xyz == (4, 0, 4):
        if a == 0:
            h_val = 0. if j2 == 3 else 0.
        elif a == 1:
            h_val = 0. if j2 == 3 else 0.
        elif a == 2:
            h_val = 18./35.*u2[j2, m2] if j2 == 3 else 0.
    elif xyz == (3, 1, 4):
        if a == 0:
            h_val = 0. if j2 == 3 else 0.
        elif a == 1:
            h_val = 12./7.*u3[j2, m2] if j2 == 3 else 0.
        elif a == 2:
            h_val = 36./35.*u2[j2, m2] if j2 == 3 else 0.
    elif xyz == (4, 1, 5):
        if a == 0:
            h_val = 0. if j2 == 3 else 0.
        elif a == 1:
            h_val = 0. if j2 == 3 else 0.
        elif a == 2:
            h_val = 10./7.*u3[j2, m2] if j2 == 3 else 0.
    else:
        raise ValueError('The value of argument xyz = %s is not valid'%(repr(xyz)))

    return h_val


def test_h():
    """ Compare the calculations with previously tabulated values.

    Validate the calculation of the u values by comparing the h values calculated by this module
    with the ones calculated from [Gold04]_

    .. [Gold04] Gold, S., Bayer, a., & Goering, E. (2004). Ground-State-Moment-Analysis: A quantitative tool for X-ray
       magnetic circular dichroism analysis for 3d transition metals. Applied Physics A: Materials Science & Processing,
       78(6), 855–865. doi:10.1007/s00339-003-2442-8
    """
    xyz_table = [(0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 1, 1), (2, 0, 2)]

    C_table_L3 = np.array([[-1./4., -1./5., 3./5., 1./4., 1./5., -1./2.],
                           [-1./12., -1/15., 2./5., -1./12., 2./15., -1./6.],
                           [1./12., 1./15., 2./5., -1./12., 2./15., 1./6.],
                           [1./4., 1./5., 3/5., 1./4., 1/5., 1./2.]
                           ])
    m_L3 = np.array([-3./2., -1./2., 1./2., 3./2.])
    j_L3 = 3./2.

    C_table_L2 = np.array([[1./6., -1./6., 1./2., -1./6., -1./3., 1./3.],
                           [-1./6., 1./6., 1./2., -1./6., -1./3., -1./3.]
                           ])
    m_L2 = np.array([-1./2., 1./2.])
    j_L2 = 1./2.

    print "Validating the h values at the L3 edge..."
    failed = False
    i = 0
    for xyz in xyz_table:
        j = 0
        for m in m_L3:
            if not np.isclose(C_table_L3[j, i], calc_h(xyz, int(2*j_L3), int(2*m), 1)):
                print 'Failed for xyz = ', xyz, 'j = ', j_L3, 'm = ', m
                print 'Values: ', C_table_L3[j, i], calc_h(xyz, int(2*j_L3), int(2*m), 1)
                failed = True
            j += 1
        i += 1

    print "Validating the h values at the L2 edge..."
    print " Note that this assumes a typo in Gold's paper..."
    i = 0
    for xyz in xyz_table:
        j = 0
        for m in m_L2:
            #TODO: Seems as there is an error in gold's paper CHECK!
            m = -m
            if not np.isclose(C_table_L2[j, i], calc_h(xyz, int(2*j_L2), int(2*m), 1)):
                print 'Failed for xyz = ', xyz, 'j = ', j_L2, 'm = ', m
                print 'Values: ', C_table_L2[j, i], calc_h(xyz, int(2*j_L2), int(2*m), 1)
                failed = True
            j += 1
        i += 1

    if failed:
        print 'The result is not in agreement with previously published!'
    else:
        print 'All tests passed!'

def create_h_table(j2, xyz_values, a):
    """ Create an h values for the given j level's m subleves for the different xyz values

    Parameters:
        j2(int): 2*j where j is the j quatum number for the ground state.
        xyz_values(list): A list of three int tuples denoting the ground state moment.

    Returns:
        h_table(array):
    """
    m2s = np.arange(-j2, j2+1, 2)
    h_table = []
    for xyz in xyz_values:
        h_table.append([])
        for m2 in m2s:
            h_table[-1].append(calc_h(xyz, j2, m2, a))
    return np.array(h_table).transpose()


def calc_de(hs, j, l, s):
    """ Calculates the energy spacing between different m levels due to the exchange field hs.

    Parameters:
        hs(float): The exchange filed in eV
        j(float): The j quantum number of the core level
        l(float): The l quantum number of the core level
        s(float): The s quatum number of the core level

    Returns:
       de(float): The energy spacing
    """

    return hs*(j*(j + 1.) + s*(s + 1) - l*(l + 1.))/(2*j*(j + 1.))

class Spectrum2p(refl.ReflBase):
    """ Class to model the lineshape of a 2p level including dichroism
    """
    _parameters = {'hs': 0., 'soc': 0.}

    def __init__(self, hs=0., soc=0., gsm=((0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 1, 1), (2, 0, 2)),
                 valid_e=700.):
        self.gsm = gsm
        for par in self._parameters:
            self._make_set_func(par)
            self._make_get_func(par)
        self.hs = hs
        self.soc = soc

        self.h_l3 = [create_h_table(3, gsm, 0), create_h_table(3, gsm, 1), create_h_table(3, gsm, 2)]
        self.h_l2 = [create_h_table(1, gsm, 0), create_h_table(1, gsm, 1), create_h_table(1, gsm, 2)]

        self.components = []

        self.j_l3, self.l_l3, self.s_l3 = 3./2., 1., 1./2.
        self.j_l2, self.l_l2, self.s_l2 = 1./2., 1., -1./2.
        self.m_l3 = np.arange(-self.j_l3, self.j_l3 + 1./2., 1.0)
        self.m_l2 = np.arange(-self.j_l2, self.j_l2 + 1./2., 1.0)

        self.fres = ReflFunction(self.calc_fres, (valid_e, ), (), id='f(E)')
        self.fm1 = ReflFunction(self.calc_fm1, (valid_e, ), (), id='f(E)')
        self.fm2 = ReflFunction(self.calc_fm2, (valid_e, ), (), id='f(E)')

    def add_component(self, **kwargs):
        comp = SpectrumComponent(self, self.gsm, **kwargs)
        self.components.append(comp)
        return comp

    def calc_spectra_comp(self, energy, comp, a):
        """ Calculates the spectra from one component

        Parameters:
            e (float array): The energy values where the spectra are to be calcualted.
            comp (SpectralComponent): The spectral component.
            a (int): The polarization of the exciting light.

        Returns:
           spectra (complex np.array):
        """
        e_jm_l3 = self.m_l3*calc_de(self.hs, self.j_l3, self.l_l3, self.s_l3)
        e_jm_l2 = self.m_l2*calc_de(self.hs, self.j_l2, self.l_l2, self.s_l2)

        peak_l3 = voigt(energy, comp.el3 + e_jm_l3[:, np.newaxis, np.newaxis], comp.gamma, comp.sigma)
        peak_l2 = voigt(energy, comp.el3 + e_jm_l2[:, np.newaxis, np.newaxis] + self.soc, comp.gamma, comp.sigma)

        spectra = ((self.h_l3[a][:, :, np.newaxis]*comp.w[np.newaxis, :, np.newaxis]*peak_l3).sum(0).sum(0) +
                   (self.h_l2[a][:, :, np.newaxis]*comp.w[np.newaxis, :, np.newaxis]*peak_l2).sum(0).sum(0))

        return spectra

    def calc_spectra(self, energy, a):
        """ Calculates the total spectra.

        Sums up all contributions from the different components.

        Parameters:
            energy (float or array): The energy where to evaluate the spectra.
            a(int): Number representing the exciting polarization.

        Returns:
            spectra(complex array or complex):
        """
        if energy is float:
            energy = np.array([energy])

        spectra = np.zeros_like(energy, dtype=np.complex128)
        for comp in self.components:
            spectra = spectra + self.calc_spectra_comp(energy, comp, a)

        if len(spectra) == 1:
            spectra = spectra[0]

        # TODO: Check if this is correct!
        return spectra.conj()

    def calc_fres(self, energy):
        """ Calculate the isotropic resonant scattering length. """
        return self.calc_spectra(energy, 0)

    def calc_fm1(self, energy):
        """ Calculate the circular dirchroic compenet. """
        return self.calc_spectra(energy, 1)

    def calc_fm2(self, energy):
        """ Calculate the linear dichroic component."""
        return self.calc_spectra(energy, 2)

class SpectrumComponent(refl.ReflBase):
    """ Class to keep the variables of a spectral component.
    """
    _parameters = {'el3': 700., 'gamma': 1.0, 'sigma': 1.0}

    def __init__(self, spectra, gsm, **kwargs):
        self.spectra = spectra
        self.gsm = gsm

        self.w = np.zeros(len(gsm))
        for i in range(len(self.gsm)):
            self._make_w_set_func(i)
            self._make_w_get_func(i)

        for par in self._parameters:
            setattr(self, par, self._parameters[par])
            self._make_set_func(par)
            self._make_get_func(par)

        # Set all parameters given as keyword arguments
        for k in kwargs:
            try:
                func = getattr(self, 'set' + k.capitalize())
            except AttributeError:
                raise AttributeError('%s is not an parameter in %s' %
                                     (k, self.__class__))
            else:
                func(kwargs[k])

    def _make_w_set_func(self, w_index):
        """ Creates a set function for a ground state moment and binds it to the object
            """
        xyz = self.gsm[w_index]
        par_name = 'w%d%d%d'%xyz
        def set_func(val):
            self.w[w_index] = val

        set_func.__name__ = 'set' + par_name.capitalize()
        setattr(self, set_func.__name__, set_func)

    def _make_w_get_func(self, w_index):
        """ Creates a get function for parameter par and binds it to the object
        """
        xyz = self.gsm[w_index]
        par_name = 'w%d%d%d'%xyz
        def get_func():
            return self.w[w_index]

        get_func.__name__ = 'get' + par_name.capitalize()
        setattr(self, get_func.__name__, get_func)


class ModelGoering:
    """ A fit model to simulate one xmcd spectral moment of a p_3/2, p_1/2 absorption edge."""

    C_table_L3 = np.array([[-1./4., -1./5., 3./5., 1./4., 1./5., -1./2.],
                           [-1./12., -1/15., 2./5., -1./12., 2./15., -1./6.],
                           [1./12., 1./15., 2./5., -1./12., 2./15., 1./6.],
                           [1./4., 1./5., 3/5., 1./4., 1/5., 1./2.]
                           ])
    j_L3 = 3./2.
    l_L3 = 1.0  # Needs to be checked
    s_L3 = 1./2. # Needs to be checked!
    m_L3 = np.array([-3./2., -1./2., 1./2., 3./2.])
    #C_table_L2 = np.array([[1./6., -1./6., 1./2., -1./6., -1./3., 1./3.],
    #                       [-1./6., 1./6., 1./2., -1./6., -1./3., -1./3.]
    #                       ])
    # TODO: Probably an error in original table.
    C_table_L2 = np.array([[-1./6., 1./6., 1./2., -1./6., -1./3., -1./3.],
                           [1./6., -1./6., 1./2., -1./6., -1./3., 1./3.]
                           ])

    j_L2 = 1./2.
    l_L2 = 1.0  # Needs to be checked
    s_L2 = -1./2. # Needs to be checked!
    m_L2 = np.array([-1./2., 1./2.])

    def __init__(self, w000=0., w110=0., w101=0., w011=0., w211=0., w202=0., gamma=0.5, sigma=0.1,
                 soc=0.0, hs=0., e_L3=0., valid_e=710.):
        ''' A fit model to simulate one xmcd spectral moment of a p_3/2, p_1/2 absorption edge.

        :param w000: proportional to :math:`n_h`
        :param w110: proportional to :math:`-l \dot s`
        :param w101: proportional to :math:`L_z/2`
        :param w011: proportional to :math:`2 S_z`
        :param w211: proportional to :math:`7 T_z/2`
        :param w202: proportional to :math:`-Q_{zz}/2`
        :param gamma: Scale parameter (HWHM) of the Lorentzian part.
        :param sigma: The standard deviation of the Gaussian part.
        :param soc: The spin orbit coupling (eV).
        :param hs: The effective pd-exchange (eV).
        :param e_L3: The centre of the :math:`L_3` white line.
        :param valid_e: The energy where the validation, testing, of the function is conducted.

        :return: An object of the class
        '''
        self.w = np.array([w000, w110, w101, w011, w211, w202])
        self.gamma = gamma
        self.sigma = sigma
        self.soc = soc
        self.hs = hs
        self.e_L3 = e_L3

        self.fm1 = ReflFunction(self.calc_fm1, (valid_e, ), (), id='f(E)')

    def set_w000(self, val):
        self.w[0] = val

    def set_w110(self, val):
        self.w[1] = val

    def set_w101(self, val):
        self.w[2] = val

    def set_w011(self, val):
        self.w[3] = val

    def set_w211(self, val):
        self.w[4] = val

    def set_w202(self, val):
        self.w[5] = val

    def set_gamma(self, val):
        self.gamma = val

    def set_sigma(self, val):
        self.sigma = val

    def set_soc(self, val):
        self.soc = val

    def set_hs(self, val):
        self.hs = val

    def set_eL3(self, val):
        self.e_L3 = val

    def xmcd(self, e):
        """ Calculates the xmcd absorption signal.

        :param e: The energy where the xmcd is calculated.
        :return: xmcd signal
        """
        return self.calc_fm1(e).imag

    def calc_fm1(self, e):
        """ Calculates the fm1 structure factor

        :param e: The energy points where fm1 is evaluated.
        :return: fm1 scattering factor
        """
        E_jm_L3 = self.hs*self.m_L3*(self.j_L3*(self.j_L3 + 1) + self.s_L3*(self.s_L3 + 1) -
                                                self.l_L3*(self.l_L3 + 1))/(2*self.j_L3*(self.j_L3 + 1))
        E_jm_L2 = self.hs*self.m_L2*(self.j_L2*(self.j_L2 + 1) + self.s_L2*(self.s_L2 + 1) -
                                                self.l_L2*(self.l_L2 + 1))/(2*self.j_L2*(self.j_L2 + 1))

        peak_L3 = voigt(e, self.e_L3 + E_jm_L3[:, np.newaxis, np.newaxis], self.gamma, self.sigma)
        peak_L2 = voigt(e, self.e_L3 + E_jm_L2[:, np.newaxis, np.newaxis] + self.soc, self.gamma, self.sigma)

        spectra = ((self.C_table_L3[:, :, np.newaxis]*self.w[np.newaxis, :, np.newaxis]*peak_L3).sum(0).sum(0) +
                   (self.C_table_L2[:, :, np.newaxis]*self.w[np.newaxis, :, np.newaxis]*peak_L2).sum(0).sum(0))

        if len(spectra) == 1:
            spectra = spectra[0]

        # TODO: Need to check the sign?
        # Accoroding to XOP's handbook f = f1 - 1.0J*f2
        spectra = spectra.real - 1.0J*spectra.imag

        return spectra

    def get_fm1(self):
        return self.fm1

if __name__ == '__main__':
    import pylab as pl
    mod = ModelGoering(e_L3=709., soc = 13.0, hs=0.8,
                       w000=0.0, w110=-1.0, w101=-0.07, w011=-1.0,
                       sigma=0.5, gamma=0.4)
    e = np.arange(690, 750, 0.1)
    mod2 = Spectrum2p(hs=0.8, soc=13.0)
    cmp1 = mod2.add_component(el3=709.,
                              w000=0.0, w110=-1.0, w101=-0.07, w011=-1.0,
                              sigma=0.5, gamma=0.4)
    print mod2.calc_spectra(710, 1)
    pl.plot(e, mod.xmcd(e), 'x')
    pl.plot(e, -mod2.calc_spectra(e, 1).imag)
    pl.show()
