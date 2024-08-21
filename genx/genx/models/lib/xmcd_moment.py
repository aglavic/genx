#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
.. module:: xmcd_moment
   :synopsis: A physical model for the 2p edges in x-ray magnetic resonant reflectivity.

This module provides the framework to model the spectroscopic response of the scattering lengths
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

.. [Dörfler06] Dörfler, Fabian (2006). Contributions to the theory of x-ray magnetic dichroism. Max-Planck-Institut fur
       Metallforschung Stuttgart and Universität Stuttgart. PhD thesis.

       Dörfler, F., & Fähnle, M. (2006). Theoretical justification of ground-state moment analysis of magnetic
       dichroic x-ray absorption spectra for 3d transition metals. Phys. Rev. B, 74(22),
       224424. doi:10.1103/PhysRevB.74.224424

.. [Stone80] Stone, A and Wood, (1980). C. Comp. Phys. Comm., 21 , 195.

"""

import os

from dataclasses import dataclass, field

import numpy as np

from scipy import integrate, special

from genx.core.custom_logging import iprint

from . import refl_base as refl

_head, _tail = os.path.split(__file__)
# Look only after the file name and not the ending since
# the file ending can be pyc if compiled...
__FILENAME__ = _tail.split(".")[0]
# This assumes that plugin is under the current dir may need
# changing
__MODULE_DIR__ = _head
if __MODULE_DIR__ == "":
    __MODULE_DIR__ = "."

__F_DB_DIR__ = os.path.join(__MODULE_DIR__, "../databases/f1f2_nist/")


def voigt(x, x_0, gamma, sigma):
    """The complex Voigt function (a convolution of a Lorentzian and a Gaussian function).

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
    z = ((x - x_0) + 1.0j * gamma) / (sigma * np.sqrt(2.0))
    return special.wofz(z) / (sigma * np.sqrt(2.0 * np.pi)) * 1.0j


def calc_h(xyz, j2, m2, a):
    """Calculate the h values for the given parameters.

    For a full description see the Theory section in module documentation.

    Args:
       xyz (tuple, 3 int): A tuple indicating the ground state moment.
       j2 (int): The j quantum number multiplied by 2 (value 3 or 1).
       m2 (int): The m quantum number multiplied by 2 (value [-j2 .. j2]).
       a (int): The polarisation of the exciting light.

    Returns:
       h (float): Transition probability of the given state with ground state moment 1.
    """

    u0 = {
        (3, -3): 1.0 / 4.0,
        (3, -1): 1.0 / 4.0,
        (3, 1): 1.0 / 4.0,
        (3, 3): 1.0 / 4.0,
        (1, -1): 1.0 / 2.0,
        (1, 1): 1.0 / 2.0,
    }
    u1 = {
        (3, -3): -9.0 / 20.0,
        (3, -1): -3.0 / 20.0,
        (3, 1): 3.0 / 20.0,
        (3, 3): 9.0 / 20.0,
        (1, -1): -1.0 / 2.0,
        (1, 1): 1.0 / 2.0,
    }
    u2 = {(3, -3): 1.0 / 4.0, (3, -1): -1.0 / 4.0, (3, 1): -1.0 / 4.0, (3, 3): 1.0 / 4.0, (1, -1): 0.0, (1, 1): 0.0}
    u3 = {(3, -3): -1.0 / 20.0, (3, -1): 3.0 / 20.0, (3, 1): -3.0 / 20.0, (3, 3): 1.0 / 20.0, (1, -1): 0.0, (1, 1): 0.0}

    if (j2, m2) not in u0:
        raise ValueError("The value of j2 (%s) or m2 (%s) is invalid" % (repr(j2), repr(m2)))

    if a not in [0, 1, 2]:
        raise ValueError("The value of a (%s) is invalid" % (repr(a)))

    if xyz == (0, 0, 0):
        if a == 0:
            h_val = u0[j2, m2] if j2 == 3 else u0[j2, m2]
        elif a == 1:
            h_val = 5.0 / 9.0 * u1[j2, m2] if j2 == 3 else 1.0 / 3.0 * u1[j2, m2]
        elif a == 2:
            h_val = 1.0 / 5.0 * u2[j2, m2] if j2 == 3 else 0.0
    elif xyz == (1, 1, 0):
        if a == 0:
            h_val = 2.0 * u0[j2, m2] if j2 == 3 else u0[j2, m2]
        elif a == 1:
            h_val = 4.0 / 9.0 * u1[j2, m2] if j2 == 3 else -1.0 / 3.0 * u1[j2, m2]
        elif a == 2:
            h_val = 2.0 / 5.0 * u2[j2, m2] if j2 == 3 else 0.0
    elif xyz == (1, 0, 1):
        if a == 0:
            h_val = 5.0 / 3.0 * u1[j2, m2] if j2 == 3 else -u1[j2, m2]
        elif a == 1:
            h_val = 2.0 * u0[j2, m2] + 2.0 / 5.0 * u2[j2, m2] if j2 == 3 else u0[j2, m2]
        elif a == 2:
            h_val = 2.0 / 3.0 * u1[j2, m2] if j2 == 3 else 2.0 / 5.0 * u1[j2, m2]
    elif xyz == (0, 1, 1):
        if a == 0:
            h_val = 10.0 / 9.0 * u1[j2, m2] if j2 == 3 else u1[j2, m2]
        elif a == 1:
            h_val = 1.0 / 3 * u0[j2, m2] + 2.0 / 3.0 * u2[j2, m2] if j2 == 3 else -1.0 / 3.0 * u0[j2, m2]
        elif a == 2:
            h_val = 2.0 / 45.0 * u1[j2, m2] + 3.0 / 6.0 * u3[j2, m2] if j2 == 3 else -2.0 / 15.0 * u1[j2, m2]
    elif xyz == (2, 1, 1):
        if a == 0:
            h_val = 2.0 / 9.0 * u1[j2, m2] if j2 == 3 else -1.0 / 3.0 * u1[j2, m2]
        elif a == 1:
            h_val = 2.0 / 3 * u0[j2, m2] + 2.0 / 15.0 * u2[j2, m2] if j2 == 3 else -2.0 / 3.0 * u0[j2, m2]
        elif a == 2:
            h_val = 22.0 / 45.0 * u1[j2, m2] + 6.0 / 35.0 * u3[j2, m2] if j2 == 3 else -4.0 / 15.0 * u1[j2, m2]
    elif xyz == (2, 0, 2):
        if a == 0:
            h_val = u2[j2, m2] if j2 == 3 else 0
        elif a == 1:
            h_val = 10.0 / 9 * u1[j2, m2] if j2 == 3 else 2.0 / 3.0 * u1[j2, m2]
        elif a == 2:
            h_val = 2.0 * u0[j2, m2] + 2.0 / 7.0 * u2[j2, m2] if j2 == 3 else u0[j2, m2]
    elif xyz == (1, 1, 2):
        if a == 0:
            h_val = 2 * u2[j2, m2] if j2 == 3 else 0
        elif a == 1:
            h_val = 34.0 / 45 * u1[j2, m2] + 6.0 / 5 * u3[j2, m2] if j2 == 3 else -4.0 / 15.0 * u1[j2, m2]
        elif a == 2:
            h_val = 2.0 / 5.0 * u0[j2, m2] + 2.0 / 5.0 * u2[j2, m2] if j2 == 3 else -2.0 / 5.0 * u0[j2, m2]
    elif xyz == (3, 1, 2):
        if a == 0:
            h_val = 0 if j2 == 3 else 0
        elif a == 1:
            h_val = 2.0 / 15 * u1[j2, m2] + 3.0 / 35 * u3[j2, m2] if j2 == 3 else -2.0 / 15.0 * u1[j2, m2]
        elif a == 2:
            h_val = 3.0 / 5.0 * u0[j2, m2] + 6.0 / 35.0 * u2[j2, m2] if j2 == 3 else -3.0 / 5.0 * u0[j2, m2]
    elif xyz == (3, 0, 3):
        if a == 0:
            h_val = 0.0 if j2 == 3 else 0.0
        elif a == 1:
            h_val = 3.0 / 15 * u2[j2, m2] if j2 == 3 else 0.0
        elif a == 2:
            h_val = u1[j2, m2] if j2 == 3 else 3.0 / 5.0 * u1[j2, m2]
    elif xyz == (2, 1, 3):
        if a == 0:
            h_val = 3 * u3[j2, m2] if j2 == 3 else 0.0
        elif a == 1:
            h_val = 6.0 / 5 * u2[j2, m2] if j2 == 3 else 0.0
        elif a == 2:
            h_val = 24.0 / 35.0 * u1[j2, m2] + 24.0 / 35.0 * u3[j2, m2] if j2 == 3 else -9.0 / 35.0 * u1[j2, m2]
    elif xyz == (4, 1, 3):
        if a == 0:
            h_val = 0.0 if j2 == 3 else 0.0
        elif a == 1:
            h_val = 0.0 if j2 == 3 else 0.0
        elif a == 2:
            h_val = 4.0 / 35.0 * u1[j2, m2] + 4.0 / 35.0 * u3[j2, m2] if j2 == 3 else -12.0 / 35.0 * u1[j2, m2]
    elif xyz == (4, 0, 4):
        if a == 0:
            h_val = 0.0 if j2 == 3 else 0.0
        elif a == 1:
            h_val = 0.0 if j2 == 3 else 0.0
        elif a == 2:
            h_val = 18.0 / 35.0 * u2[j2, m2] if j2 == 3 else 0.0
    elif xyz == (3, 1, 4):
        if a == 0:
            h_val = 0.0 if j2 == 3 else 0.0
        elif a == 1:
            h_val = 12.0 / 7.0 * u3[j2, m2] if j2 == 3 else 0.0
        elif a == 2:
            h_val = 36.0 / 35.0 * u2[j2, m2] if j2 == 3 else 0.0
    elif xyz == (4, 1, 5):
        if a == 0:
            h_val = 0.0 if j2 == 3 else 0.0
        elif a == 1:
            h_val = 0.0 if j2 == 3 else 0.0
        elif a == 2:
            h_val = 10.0 / 7.0 * u3[j2, m2] if j2 == 3 else 0.0
    else:
        raise ValueError("The value of argument xyz = %s is not valid" % (repr(xyz)))

    return h_val


def test_h():
    """Compare the calculations with previously tabulated values.

    Validate the calculation of the u values by comparing the h values calculated by this module
    with the ones calculated from the (Gold, 2004)

    References:
        Gold, S., Bayer, a., & Goering, E. (2004). Ground-State-Moment-Analysis: A quantitative tool for X-ray
        magnetic circular dichroism analysis for 3d transition metals. Applied Physics A: Materials Science & Processing,
        78(6), 855–865. doi:10.1007/s00339-003-2442-8
    """
    xyz_table = [(0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 1, 1), (2, 0, 2)]

    C_table_L3 = np.array(
        [
            [-1.0 / 4.0, -1.0 / 5.0, 3.0 / 5.0, 1.0 / 4.0, 1.0 / 5.0, -1.0 / 2.0],
            [-1.0 / 12.0, -1 / 15.0, 2.0 / 5.0, -1.0 / 12.0, 2.0 / 15.0, -1.0 / 6.0],
            [1.0 / 12.0, 1.0 / 15.0, 2.0 / 5.0, -1.0 / 12.0, 2.0 / 15.0, 1.0 / 6.0],
            [1.0 / 4.0, 1.0 / 5.0, 3 / 5.0, 1.0 / 4.0, 1 / 5.0, 1.0 / 2.0],
        ]
    )
    m_L3 = np.array([-3.0 / 2.0, -1.0 / 2.0, 1.0 / 2.0, 3.0 / 2.0])
    j_L3 = 3.0 / 2.0

    C_table_L2 = np.array(
        [
            [1.0 / 6.0, -1.0 / 6.0, 1.0 / 2.0, -1.0 / 6.0, -1.0 / 3.0, 1.0 / 3.0],
            [-1.0 / 6.0, 1.0 / 6.0, 1.0 / 2.0, -1.0 / 6.0, -1.0 / 3.0, -1.0 / 3.0],
        ]
    )
    m_L2 = np.array([-1.0 / 2.0, 1.0 / 2.0])
    j_L2 = 1.0 / 2.0

    iprint("Validating the h values at the L3 edge...")
    failed = False
    i = 0
    for xyz in xyz_table:
        j = 0
        for m in m_L3:
            if not np.isclose(C_table_L3[j, i], calc_h(xyz, int(2 * j_L3), int(2 * m), 1)):
                iprint("Failed for xyz = ", xyz, "j = ", j_L3, "m = ", m)
                iprint("Values: ", C_table_L3[j, i], calc_h(xyz, int(2 * j_L3), int(2 * m), 1))
                failed = True
            j += 1
        i += 1

    iprint("Validating the h values at the L2 edge...")
    iprint(" Note that this assumes a typo in Gold's paper...")
    i = 0
    for xyz in xyz_table:
        j = 0
        for m in m_L2:
            # TODO: Seems as there is an error in gold's paper CHECK!
            m = -m
            if not np.isclose(C_table_L2[j, i], calc_h(xyz, int(2 * j_L2), int(2 * m), 1)):
                iprint("Failed for xyz = ", xyz, "j = ", j_L2, "m = ", m)
                iprint("Values: ", C_table_L2[j, i], calc_h(xyz, int(2 * j_L2), int(2 * m), 1))
                failed = True
            j += 1
        i += 1

    if failed:
        iprint("The result is not in agreement with previously published!")
    else:
        iprint("All tests passed!")


def create_h_table(j2, xyz_values, a):
    """Create an h values for the given j level's m subleves for the different xyz values

    Parameters:
        j2(int): 2*j where j is the j quatum number for the ground state.
        xyz_values(list): A list of three int tuples denoting the ground state moment.

    Returns:
        h_table(array):
    """
    m2s = np.arange(-j2, j2 + 1, 2)
    h_table = []
    for xyz in xyz_values:
        h_table.append([])
        for m2 in m2s:
            h_table[-1].append(calc_h(xyz, j2, m2, a))
    return np.array(h_table).transpose()


def calc_de(hs, j, l, s):
    """Calculates the energy spacing between different m levels due to the exchange field hs.

    Parameters:
        hs(float): The exchange filed in eV
        j(float): The j quantum number of the core level
        l(float): The l quantum number of the core level
        s(float): The s quatum number of the core level

    Returns:
       de(float): The energy spacing
    """

    return hs * (j * (j + 1.0) + s * (s + 1) - l * (l + 1.0)) / (2 * j * (j + 1.0))


@dataclass
class Spectrum2p(refl.ReflBase):
    """Class to model the lineshape of a 2p level including dichroism.

    Parameters:
            hs(float): The exchange field (eV).
            soc(float):  The spin orbit coupling (eV).
            gsm(list of tuples 3xint):  The ground state moments used in the spectra, a list of 3 int tuples.
            norm_denom(list of floats):  The normalisation denominators for the ground
                                         state moments (same order as gsm).
            valid_e(float): Energy (eV) where a validation (type check) of the function is made when the scattering
                            lengths are inserted in a validator (used in the Reflectivity plugin).


    This class contains spectra-wide parameters for the model. It also contains the
    scattering lengths that can be used in a reflectivity model. The scattering lengths functions are

    * ``fres`` - The resonant (unpolarised) part of the scattering length.
    * ``fm1`` - The dicroic part of the scattering length. The imaginary part correspond to the XMCD signal.
    * ``fm2`` - The linear dichroic signal. The imaginary part correspond to the XMLD signal.

    The wXYZs from the SpectrumComponent class is scaled according to the following formula

    .. math::
        w^{xyxz}_\mathrm{eff} = w^{xyz}_\mathrm{SpectrumComponent} w^{xyz}_\mathrm{Spectrum2p}/\mathrm{norm\_denom}.


    The subscripts on the right hand side denote which class the w parameter belongs to. The use for these normalisation
    could be to define normalisation factors (norm_denom) from reference spectra with known values of number of holes,
    spin magnetic moment and orbital magnetic moment. While keeping the spectral shape constant, constant
    :math:`w^{xyz}_\mathrm{SpectrumComponent}` the overall spectra can be scaled to change the different values with the
    parameters :math:`w^{xyz}_\mathrm{Spectrum2p}`.

    Fitting Parameters
        * hs(float) - The exchange field that causes the levels to split (eV).
        * soc(float) - The spin orbit coupling (distance between L_2 and L_3 edge (eV).
        * wXYZ(float) - The different global ground state moments (X, Y and Z are integers.

    Member functions for simulation
       * ``add_component`` - Add an component (ground state moment set) to the spectra.
       * ``calc_spectra`` - Calculates a spectra (``fm1``, ``fres`` and ``fm2`` can be used as well).

    """

    gsm: tuple = field(default_factory=lambda: ((0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 1, 1), (2, 0, 2)))
    norm_denom: tuple = field(default_factory=lambda: (1.0, 1.0, 1, 1.0, 1.0, 1.0))
    hs: float = 0.0
    soc: float = 0.0
    valid_e: float = 700.0

    def __post_init__(self):
        super().__post_init__()
        self.norm_denom = np.array(self.norm_denom, dtype=np.float64)
        self.w_glob = np.ones_like(self.norm_denom)

        for i in range(len(self.gsm)):
            self._make_w_set_func(i)
            self._make_w_get_func(i)

        self.h_l3 = [create_h_table(3, self.gsm, 0), create_h_table(3, self.gsm, 1), create_h_table(3, self.gsm, 2)]
        self.h_l2 = [create_h_table(1, self.gsm, 0), create_h_table(1, self.gsm, 1), create_h_table(1, self.gsm, 2)]

        self.components = []
        self.bkg = None

        self.j_l3, self.l_l3, self.s_l3 = 3.0 / 2.0, 1.0, 1.0 / 2.0
        self.j_l2, self.l_l2, self.s_l2 = 1.0 / 2.0, 1.0, -1.0 / 2.0
        self.m_l3 = np.arange(-self.j_l3, self.j_l3 + 1.0 / 2.0, 1.0)
        self.m_l2 = np.arange(-self.j_l2, self.j_l2 + 1.0 / 2.0, 1.0)

        self.fres = refl.ReflFunction(self.calc_fres, (self.valid_e,), {}, id="f(E)")
        self.fm1 = refl.ReflFunction(self.calc_fm1, (self.valid_e,), {}, id="f(E)")
        self.fm2 = refl.ReflFunction(self.calc_fm2, (self.valid_e,), {}, id="f(E)")

    def _make_w_set_func(self, w_index):
        """Creates a set function for a ground state moment and binds it to the object"""
        xyz = self.gsm[w_index]
        par_name = "w%d%d%d" % xyz

        def set_func(val):
            self.w_glob[w_index] = val

        set_func.__name__ = "set" + par_name.capitalize()
        setattr(self, set_func.__name__, set_func)

    def _make_w_get_func(self, w_index):
        """Creates a get function for parameter par and binds it to the object"""
        xyz = self.gsm[w_index]
        par_name = "w%d%d%d" % xyz

        def get_func():
            return self.w_glob[w_index]

        get_func.__name__ = "get" + par_name.capitalize()
        setattr(self, get_func.__name__, get_func)

    def add_component(self, **kwargs):
        comp = SpectrumComponent(self, self.gsm, **kwargs)
        self.components.append(comp)
        return comp

    def add_background(self, *args, **kwargs):
        self.bkg = Background2p(self, *args, **kwargs)
        return self.bkg

    def calc_spectra_comp(self, energy, comp, a):
        """Calculates the spectra from one component

        Parameters:
            e (float array): The energy values where the spectra are to be calcualted.
            comp (SpectralComponent): The spectral component.
            a (int): The polarization of the exciting light.

        Returns:
           spectra (complex np.array):
        """
        e_jm_l3 = self.m_l3 * calc_de(self.hs, self.j_l3, self.l_l3, self.s_l3)
        e_jm_l2 = self.m_l2 * calc_de(self.hs, self.j_l2, self.l_l2, self.s_l2)

        peak_l3 = voigt(energy, comp.el3 + e_jm_l3[:, np.newaxis, np.newaxis], comp.gamma, comp.sigma)
        peak_l2 = voigt(energy, comp.el3 + e_jm_l2[:, np.newaxis, np.newaxis] + self.soc, comp.gamma, comp.sigma)

        eff_w = self.w_glob * comp.w / self.norm_denom

        spectra = (self.h_l3[a][:, :, np.newaxis] * eff_w[np.newaxis, :, np.newaxis] * peak_l3).sum(0).sum(0) + (
            self.h_l2[a][:, :, np.newaxis] * eff_w[np.newaxis, :, np.newaxis] * peak_l2
        ).sum(0).sum(0)

        return spectra

    def calc_spectra(self, energy, a):
        """Calculates the total spectra.

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

        return spectra

    def calc_fres(self, energy):
        """Calculate the isotropic resonant scattering length."""
        res = self.calc_spectra(energy, 0)
        if self.bkg is not None:
            res += self.bkg.calc_f(energy)

        return res

    def calc_fm1(self, energy):
        """Calculate the circular dirchroic compenet."""
        return self.calc_spectra(energy, 1)

    def calc_fm2(self, energy):
        """Calculate the linear dichroic component."""
        return self.calc_spectra(energy, 2)


@dataclass
class SpectrumComponent(refl.ReflBase):
    """Class to keep the variables of a spectral component."""

    spectra: Spectrum2p = field(default_factory=Spectrum2p)
    gsm: tuple = field(default_factory=lambda: ((0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 1, 1), (2, 0, 2)))
    el3: float = 700.0
    gamma: float = 1.0
    sigma: float = 1.0

    def __new__(cls, spectra, gsm, **kwargs):
        self = super().__new__(cls)
        self.__init__(
            spectra=spectra,
            gsm=gsm,
            el3=kwargs.get("el3", cls.el3),
            gamma=kwargs.get("gamma", cls.gamma),
            sigma=kwargs.get("sigma", cls.sigma),
        )
        self.spectra = spectra
        self.gsm = gsm

        self.w = np.zeros(len(gsm))
        for i in range(len(self.gsm)):
            self._make_w_fields(i)

        # Set all parameters given as keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

        return self

    def _make_w_fields(self, w_index):
        new_field = field(default=0.0)
        xyz = self.gsm[w_index]
        new_field.name = "w%d%d%d" % tuple(xyz)
        new_field.type = float
        from dataclasses import _FIELD, _FIELDS  # noqa

        new_field._field_type = _FIELD
        updt_fields = getattr(self, _FIELDS) | {new_field.name: new_field}
        setattr(self, _FIELDS, updt_fields)
        setattr(self, new_field.name, 0.0)


@dataclass
class Background2p(refl.ReflBase):
    r"""A class to model the 2p background spectra, the non-resonant part, with a smoothed step function.


    Parameters:
        el3 (float): Energy position of the L3 edge.
        sigma (float): The width of the error function (std in eV).
        pre_edge (float): The width of the pre edge region (eV).
        post_edge (float): The width of the post edge region (eV).
        de (float): The sampling spacing for the recalcualted table (eV).
        kk_emax (float): The maximum energy values taken from the tables in eV. The value should be well above the K edge.
        element (string): The element to which the background applies. Used to find theoretical f values (from the nist database), position of the L2/L3 edge and Z (number of electrons).
        parent (Spectra2p): The parent spectra to which the Background applies.


    The background, B, is modelled with two error function according to

    .. math::
        B = a_0 + a_1 e + (a_2 + a_3 e) \Phi(e, e_l3) + (a_4 + a_5 e) \Phi(e, e_l2),

    where

    .. math::
        \Phi(e, e_0) = \frac{1}{2}\left( 1 + \mathrm{erf}\left( \frac{e - e_0}{\sqrt{2} \sigma} \right)\right).

    This function is used to up-sample the theoretical, tabulates, f2 values for the spectral region (the L2/L3 within
    the pre and post edge values) to a point spacing of de. This data then replaces the data in the tabulated f2 values
    and a Kramer-Kronig (KK) transform is conducted to get the f1 values. Note that the tabulated values in the class
    contains a moved L2/L3 edge since this removes the necessity of conducting a full KK transform when a new background
    is calculated. The KK transform fulfill the superposition principle. The KK transform conducted when the complex
    are calculated, which is the difference between the tabulated and the current, over the spectral region only.

    Note that this implementation does incorporates exchange split levels.

    The parameters relating to KK transform are:
        * sigma_tab (float): The width of the error function applied to the table (std in eV), defualt value 1 eV.
        * edge_offset_tab (float): The offset of the stored tabulated data to the read data (eV), defualt value 40 eV.

    Fitting parameters
        * el3
        * sigma
    """

    parent: Spectrum2p = field(default_factory=Spectrum2p)
    element: str = "fe"

    el3: float = 710.0
    sigma: float = 0.5
    pre_edge: float = 100.0
    post_edge: float = 100.0
    de: float = 0.2
    kk_emax: float = 15000.0

    _elements = {
        "ti": {"Z": 22, "El3": 455.5, "El2": 461.5},
        "v": {"Z": 23, "El3": 512.9, "El2": 520.5},
        "cr": {"Z": 24, "El3": 574.5, "El2": 583.7},
        "mn": {"Z": 25, "El3": 640.3, "El2": 651.4},
        "fe": {"Z": 26, "El3": 708.1, "El2": 721.1},
        "co": {"Z": 27, "El3": 778.6, "El2": 793.6},
        "ni": {"Z": 28, "El3": 854.7, "El2": 871.9},
        "cu": {"Z": 29, "El3": 931.1, "El2": 951.0},
    }

    def __new__(cls, parent, element, **kwargs):
        self = super().__new__(cls)
        self.__init__(
            parent=parent,
            el3=kwargs.get("el3", cls.el3),
            sigma=kwargs.get("sigma", cls.sigma),
            pre_edge=kwargs.get("pre_edge", cls.pre_edge),
            post_edge=kwargs.get("post_edge", cls.post_edge),
            de=kwargs.get("de", cls.de),
            kk_emax=kwargs.get("kk_emax", cls.kk_emax),
        )

        # Default numerical values for the step function width and its offset.
        self.sigma_tab = 0.5
        self.edge_offset_tab = 20.0

        # Setting element specific details
        if element.lower() not in self._elements:
            raise NotImplementedError("Element %s does not have a background that is implemented" % element)
        el_db = self._elements[element.lower()]
        self.Z = el_db["Z"]
        self.element = element.lower()

        self.load_nff_file(os.path.join(__F_DB_DIR__, "%s.nff" % element.lower()), el_db["El3"], el_db["El2"])

        return self

    def fit_table_data(self, e_tab, f2_tab, pre_slice, between_slice, post_slice):
        """Fits the L2 and L3 steps in tabular data to parametrise the spectra in the fitting range.


        Parameters:
            e_tab (array of floats): The scattering length tables energy values in eV.
            f2_tab (array of floats): The f2 values of the scattering length tables.
            pre_slice (slice): A slice object that contain the pre edge data.
            between_slice (slice): A slice object that indexes the region between the L2 and L3 edge.
            post_slice (slice): A slice object that indexes the region post edge region.

        Returns:
           f (function): A function of the form f(e, e_l3, e_l2, sigma).
        """
        e, f2 = e_tab, f2_tab

        pre_edge = pre_slice
        post_edge = post_slice
        between = between_slice

        p_pre = np.polyfit(e[pre_edge], f2[pre_edge], 1)
        a1, a0 = p_pre
        p_between = np.polyfit(e[between], (f2[between] - np.polyval(p_pre, e[between])), 1)
        a3, a2 = p_between
        p_post = np.polyfit(
            e[post_edge], (f2[post_edge] - np.polyval(p_between, e[post_edge]) - np.polyval(p_pre, e[post_edge])), 1
        )
        a5, a4 = p_post

        self.a0, self.a1, self.a2, self.a3, self.a4, self.a5 = a0, a1, a2, a3, a4, a5

        def fit_table_bkg(e, e_l3, e_l2, sigma):
            return a0 + a1 * e + (a2 + a3 * e) * step_func(e, e_l3, sigma) + (a4 + a5 * e) * step_func(e, e_l2, sigma)

        return fit_table_bkg

    def create_bkg_f1(self, e_l3, e_l2, e_tab, f2_tab):
        """Create the background f1 data to be used in later calculations.

        The method will fit the tabulated f2 value to analytical function and replace the step in the table with
        a error function (with a width (std) given by the parameter ``sigma_tab``)
        whose steps are offset ``edge_offset_tab`` eV. This function is then regridded with a step size of ``de``
        on a range of ``e_l3 + pre_edge``and ``e_l2 - pre_edge``. These values are then merged into the original
        f2 values in the ``f2_tab``array. This array is Kramer-Kronig transformed to yield the corresponding f1 values.
        The values are stored as ``f1_tab`` and ``f2_tab``.

        Parameters:
            e_l3 (float): Position of the L3 edge in eV.
            e_l2 (float): Position of the L2 edge in eV.
            e_tab (array of floats): The scattering length tables energy values in eV.
            f2_tab (array of floats): The f2 values of the scattering length tables.

        Returns:
            Nothing.
        """

        pre_edge = create_interval(e_tab, e_l3 - self.pre_edge, e_l3)
        post_edge = create_interval(e_tab, e_l2, e_l2 + self.post_edge)

        between = create_interval(e_tab, e_l3, e_l2)
        spectra = create_interval(e_tab, e_l3 - self.pre_edge, e_l2 + self.post_edge)

        e_fine = np.arange(e_tab[spectra].min(), e_tab[spectra].max(), self.de)

        self.table_bkg_func = self.fit_table_data(e_tab, f2_tab, pre_edge, between, post_edge)

        # We offset our calculated step values so that we don't get sharp peaks in our integrand which
        # can cause strange behaviours.
        f2_fine = self.table_bkg_func(e_fine, e_l3 + self.edge_offset_tab, e_l2 + self.edge_offset_tab, self.sigma_tab)

        # Create arrays for doint the KK integral
        e_new = np.r_[e_tab[: pre_edge.start], e_fine, e_tab[post_edge.stop :]]
        f2_new = np.r_[f2_tab[: pre_edge.start], f2_fine, f2_tab[post_edge.stop :]]

        # Grid the data so it becomes suitable for kk transforms.
        e_int = np.arange(e_tab.min(), self.kk_emax, self.de)
        f2_int = np.interp(e_int, e_new, f2_new)
        ekk, f1kk = kk_int(e_int, f2_int, self.Z, e_l3 - self.pre_edge, e_l2 + self.post_edge)
        self.e_tab = ekk
        # Remove the constant (which will be added later when the KK transform is applied to the calculated bkg)
        self.f1_tab = f1kk - (self.Z - (self.Z / 82.5) ** 2.37)
        self.f2_tab = self.table_bkg_func(
            self.e_tab, e_l3 + self.edge_offset_tab, e_l2 + self.edge_offset_tab, self.sigma_tab
        )

    def load_nff_file(self, file_path, e_l3_tab, e_l2_tab):
        """Load an nff (scattering length table) file.

        Parameters:
            file_path (string): Path to file.
            e_l3_tab (float): Position of the L3 edge in the table (eV).
            e_l2_tab (float): Position of the L2 edge in the table (eV).

        Returns:
            Nothing.
        """
        a = np.loadtxt(file_path, skiprows=1)
        e, f1, f2 = a[:, 0], a[:, 1], a[:, 2]
        self.create_bkg_f1(e_l3_tab, e_l2_tab, e, f2)

    def calc_f(self, energy):
        """Calculates the complex background scattering length for energy.

        Parameters:
            energy (array of floats): Energy points where to evaluate f

        Returns:
            f (array of complex): Scattering length.
        """
        f2 = self.calc_abs(energy)
        f2kk = self.calc_abs(self.e_tab) - self.f2_tab
        ekk, f1kk = kk_int(self.e_tab, f2kk, self.Z)
        f1kk += self.f1_tab
        f1 = np.interp(energy, self.e_tab, f1kk)
        return f1 - 1.0j * f2

    def calc_abs(self, energy):
        """Calculates the absorption background function for the given energies.

        Parameters:
            energy (array of floats): Energy points in eV.

        Returns:
           Abs (array of floats): Absorption, the complex part of the scattering length.
        """
        # TODO And the exchange splitting as well....
        # e_jm_l3 = self.parent.m_l3*calc_de(self.parent.hs, self.parent.j_l3, self.parent.l_l3, self.parent.s_l3)
        # e_jm_l2 = self.parent.m_l2*calc_de(self.parent.hs, self.parent.j_l2, self.parent.l_l2, self.parent.s_l2)

        # step_l3 = step_func(e, e_l3 + e_jm_l3[np.newaxis, :], sigma).mean(axis=1)
        # step_l2 = step_func(e, e_l2 + e_jm_l2[np.newaxis, :], sigma).mean(axis=1)

        # Hmm then we need to know the relative weights to excite them up to a continuum as well...

        # bkg = a0 + a1*e + (a2 + a3*e)*step_l3 + (a4 + a5*e)*step_l2

        bkg = self.table_bkg_func(energy, self.el3, self.el3 + self.parent.soc, self.sigma)

        return bkg


class ModelGoering:
    """A fit model to simulate one xmcd spectral moment of a p_3/2, p_1/2 absorption edge."""

    C_table_L3 = np.array(
        [
            [-1.0 / 4.0, -1.0 / 5.0, 3.0 / 5.0, 1.0 / 4.0, 1.0 / 5.0, -1.0 / 2.0],
            [-1.0 / 12.0, -1 / 15.0, 2.0 / 5.0, -1.0 / 12.0, 2.0 / 15.0, -1.0 / 6.0],
            [1.0 / 12.0, 1.0 / 15.0, 2.0 / 5.0, -1.0 / 12.0, 2.0 / 15.0, 1.0 / 6.0],
            [1.0 / 4.0, 1.0 / 5.0, 3 / 5.0, 1.0 / 4.0, 1 / 5.0, 1.0 / 2.0],
        ]
    )
    j_L3 = 3.0 / 2.0
    l_L3 = 1.0  # Needs to be checked
    s_L3 = 1.0 / 2.0  # Needs to be checked!
    m_L3 = np.array([-3.0 / 2.0, -1.0 / 2.0, 1.0 / 2.0, 3.0 / 2.0])
    # C_table_L2 = np.array([[1./6., -1./6., 1./2., -1./6., -1./3., 1./3.],
    #                       [-1./6., 1./6., 1./2., -1./6., -1./3., -1./3.]
    #                       ])
    # TODO: Probably an error in original table.
    C_table_L2 = np.array(
        [
            [-1.0 / 6.0, 1.0 / 6.0, 1.0 / 2.0, -1.0 / 6.0, -1.0 / 3.0, -1.0 / 3.0],
            [1.0 / 6.0, -1.0 / 6.0, 1.0 / 2.0, -1.0 / 6.0, -1.0 / 3.0, 1.0 / 3.0],
        ]
    )

    j_L2 = 1.0 / 2.0
    l_L2 = 1.0  # Needs to be checked
    s_L2 = -1.0 / 2.0  # Needs to be checked!
    m_L2 = np.array([-1.0 / 2.0, 1.0 / 2.0])

    def __init__(
        self,
        w000=0.0,
        w110=0.0,
        w101=0.0,
        w011=0.0,
        w211=0.0,
        w202=0.0,
        gamma=0.5,
        sigma=0.1,
        soc=0.0,
        hs=0.0,
        e_L3=0.0,
        valid_e=710.0,
    ):
        """A fit model to simulate one xmcd spectral moment of a p_3/2, p_1/2 absorption edge.

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
        """
        self.w = np.array([w000, w110, w101, w011, w211, w202])
        self.gamma = gamma
        self.sigma = sigma
        self.soc = soc
        self.hs = hs
        self.e_L3 = e_L3

        self.fm1 = refl.ReflFunction(self.calc_fm1, (valid_e,), (), id="f(E)")

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
        """Calculates the xmcd absorption signal.

        :param e: The energy where the xmcd is calculated.
        :return: xmcd signal
        """
        return self.calc_fm1(e).imag

    def calc_fm1(self, e):
        """Calculates the fm1 structure factor

        :param e: The energy points where fm1 is evaluated.
        :return: fm1 scattering factor
        """
        E_jm_L3 = (
            self.hs
            * self.m_L3
            * (self.j_L3 * (self.j_L3 + 1) + self.s_L3 * (self.s_L3 + 1) - self.l_L3 * (self.l_L3 + 1))
            / (2 * self.j_L3 * (self.j_L3 + 1))
        )
        E_jm_L2 = (
            self.hs
            * self.m_L2
            * (self.j_L2 * (self.j_L2 + 1) + self.s_L2 * (self.s_L2 + 1) - self.l_L2 * (self.l_L2 + 1))
            / (2 * self.j_L2 * (self.j_L2 + 1))
        )

        peak_L3 = voigt(e, self.e_L3 + E_jm_L3[:, np.newaxis, np.newaxis], self.gamma, self.sigma)
        peak_L2 = voigt(e, self.e_L3 + E_jm_L2[:, np.newaxis, np.newaxis] + self.soc, self.gamma, self.sigma)

        spectra = (self.C_table_L3[:, :, np.newaxis] * self.w[np.newaxis, :, np.newaxis] * peak_L3).sum(0).sum(0) + (
            self.C_table_L2[:, :, np.newaxis] * self.w[np.newaxis, :, np.newaxis] * peak_L2
        ).sum(0).sum(0)

        if len(spectra) == 1:
            spectra = spectra[0]

        spectra = spectra.real - 1.0j * spectra.imag

        return spectra

    def get_fm1(self):
        return self.fm1


def create_interval(x, x_min, x_max):
    """Creates a closed slice [x_min, x_max] where x_min < x[slice] < x_max.

    Returns:
        slice object
    """
    upper = np.argmin(np.abs(x - x_max))
    if x[upper - 1] >= x_max:
        upper -= 1
    lower = np.argmin(np.abs(x - x_min))
    if x[lower] <= x_min:
        lower += 1

    return slice(lower, upper)


def step_func(x, x0, sigma):
    """Defines the step funciton (erf) of the L2/L3 edges

        x (array of floats):
        x0 (float):
        sigma (float):

    Returns:
        step_func (array of floats):
    """
    return 0.5 * special.erf((x - x0) / np.sqrt(2) / sigma) + 0.5


def bkg_func(e, e_l3, e_l2, sigma, a0, a1, a2, a3, a4, a5):
    return a0 + a1 * e + (a2 + a3 * e) * step_func(e, e_l3, sigma) + (a4 + a5 * e) * step_func(e, e_l2, sigma)


def kk_int(e, f2, Z=0, e_min=None, e_max=None):
    """Do the Kramer Kronig transform from f2 to f1

    If the parameters e_min or e_max is given the evaluation range is reduced so that the final size of
    f1kk is reduced. The Kramer-Kronig transformation is conducted as suggested by the Newville in the DIFFKK manual.
    The integrand is defined as in the XOP data booklet.

    Parameters:
        e (array of floats): Energy values of the values found in f2 (eV).
        f2 (array of floats): The imaginary part of the scattering length (electrons)
        Z (int): The number of electrons of the element, (electrons).
        e_min (float): The minimum value of the returned real part of f.
        e_max (float): The maximum value of the returned real part of f.

    Returns:
        array of floats: The energy values for the calculated f1 (eV).
        array of floats: The real part of the scattering length, f1, (electrons).

    References:
        Newville, M. and Olmsted Cross J. DIFFKK Manual. Downloaded from http://cars9.uchicago.edu/dafs/diffkk/diffkk.pdf
        20140902.

        X-ray data booklet, CXRO, Lawrence Berkley National Laboratory, Downloaded from http://xdb.lbl.gov 20140902.

    """
    ekk = e
    if e_min is not None:
        min_val = np.argmin(np.abs(e_min - ekk))
        if min_val % 2 == 1:
            min_val += 1
        ekk = ekk[min_val:]
    if e_max is not None:
        ekk = ekk[: np.argmin(np.abs(e_max - ekk))]

    f1kk = np.empty_like(ekk)
    f1kk[1::2] = integrate.trapz(e[::2] * f2[::2] / (ekk[1::2][:, np.newaxis] ** 2 - e[::2] ** 2), e[::2])
    f1kk[::2] = integrate.trapz(e[1::2] * f2[1::2] / (ekk[::2][:, np.newaxis] ** 2 - e[1::2] ** 2), e[1::2])

    fkk = (
        Z
        - (Z / 82.5) ** 2.37
        + f1kk * 2 / np.pi
        + 1 / np.pi * f2[-1] * e[-1] ** 2 / ekk**2 * np.log(np.abs(e.max() ** 2 / (e.max() ** 2 + ekk**2)))
    )

    return ekk, fkk


def kk_int_old(e_new, e, f2, Z, offset=1e-1, offset_outer=1, e_step=0.1):
    """Original Kramer-Kronig transform - a slower implementation of kk_int"""

    def create_int_func(e, f2, epoint):
        def integ(e_val):
            # print e.shape, f2.shape, e_val.shape
            f2p = np.interp(e_val, e, f2)
            return e_val * f2p / (epoint**2 - e_val**2)

        return integ

    ekk = e_new

    f1kk_tmp = [
        integrate.trapz(
            create_int_func(e, f2, ep)(np.arange(0, ep - offset_outer + e_step, e_step)),
            np.arange(0, ep - offset_outer + e_step, e_step),
        )
        + integrate.quad(create_int_func(e, f2, ep), ep - offset_outer, ep - offset)[0]
        + integrate.quad(create_int_func(e, f2, ep), ep + offset, ep + offset_outer)[0]
        + integrate.trapz(
            create_int_func(e, f2, ep)(np.arange(ep + offset_outer, e.max() + e_step, e_step)),
            np.arange(ep + offset_outer, e.max() + e_step, e_step),
        )
        for ep in ekk
    ]
    f1kk = (
        Z
        - (Z / 82.5) ** 2.37
        + np.array(f1kk_tmp) * 2 / np.pi
        + 1 / np.pi * f2[-1] * e[-1] ** 2 / ekk**2 * np.log(np.abs(e.max() ** 2 / (e.max() ** 2 + ekk**2)))
    )
    return f1kk


if __name__ == "__main__":
    import pylab as pl

    mod = ModelGoering(e_L3=709.0, soc=13.0, hs=0.8, w000=0.0, w110=-1.0, w101=-0.07, w011=-1.0, sigma=0.5, gamma=0.4)
    e = np.arange(690, 750, 0.1)
    mod2 = Spectrum2p(hs=0.8, soc=13.0)
    cmp1 = mod2.add_component(el3=709.0, w000=0.0, w110=-1.0, w101=-0.07, w011=-1.0, sigma=0.5, gamma=0.4)

    a = np.loadtxt("../databases/f1f2_nist/fe.nff", skiprows=1)
    e, f1, f2 = a[:, 0], a[:, 1], a[:, 2]

    bkg = Background2p(mod2, "fe")
    el_tab = bkg._elements[bkg.element]
    bkg.setEl3(el_tab["El3"])
    mod2.setSoc(el_tab["El2"] - el_tab["El3"])
    e_test = np.arange(400, 1100, 0.5)
    f2_mod = bkg.calc_abs(e_test)
    f_mod = bkg.calc_f(e_test)

    pl.subplot(211)
    pl.plot(e_test, f_mod.imag)
    pl.plot(e, f2, ".")
    pl.xlim(e_test.min(), e_test.max())
    pl.subplot(212)
    pl.plot(e_test, f_mod.real)
    pl.plot(e, f1, ".")
    pl.xlim(e_test.min(), e_test.max())

    fm1 = mod2.calc_fm1(e_test)
    ekk, fm1kk_real = kk_int(e_test, fm1.imag)
    pl.figure()
    pl.plot(e_test, fm1kk_real)
    pl.plot(e_test, fm1.real, ".")

    pl.show()
