#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
.. module:: spec_nx2.py
   :synopsis: A model for simulating x-ray and neutron reflectivity

.. moduleauthor:: Matts Björck <matts.bjorck@gmail.com>

This model implements a Sample-Stack-Layer model for calculating x-ray and neutron reflectivity.
"""

from .lib import stratified_base
from .lib.parameters import Float, Complex, Enum
from .lib.materials import Material

from .lib import paratt

import numpy as np

z = stratified_base.z
rep = stratified_base.rep
e = stratified_base.e


class Instrument(stratified_base.Instrument):

    def __init__(self, **kwargs):
        self.wl = Float(1.54, unit="AA", help="Wavelength of radiation used")
        self.probe = Enum(allowed_values=['x-ray', 'neutron'], help="Probe used")
        self.i_0 = Float(1.0, unit="", help="Incident intensity, scale factor")
        self.i_bkg = Float(0.0, unit="", help="Background intensity")

        super(Instrument, self).__init__(**kwargs)


class Layer(stratified_base.Layer):

    def __init__(self, **kwargs):
        self.sigma = Float(0, unit="AA", help="Roughness of upper interface")
        self.sld_x = Complex(0, unit="1/AA^2", help="X-ray scattering length density")
        self.sld_n = Complex(0, unit="1/AA^2", help="Neutron scattering length density")
        self.mag = Float(0, unit="mu_B/AA^3", help="Magnetic moment density")
        self.mag_ang = Float(0, unit="deg.", help="Angle of magnetisation ")

        super(Layer, self).__init__(**kwargs)


class Stack(stratified_base.Stack):
    pass


class Sample(stratified_base.Sample):

    def __init__(self, **kwargs):
        self.ambient = Layer()
        self.substrate = Layer()

        super(Sample, self).__init__(**kwargs)

    def specular(self, tth, instrument):
        """Calculate the specular reflectivity"""
        kwargs = {}
        wavelength = instrument.wl(**kwargs)
        kwargs['wl'] = wavelength

        p = self.build_layer_list(['d', 'sigma', 'sld_x'], kwargs)
        r = paratt.Refl(tth/2.0, wavelength, 1.0 - np.array(p.sld_x), np.array(p.d), np.array(p.sigma))

        return r

if __name__ == "__main__":
    inst = Instrument()

    fe = Material('Fe', density=7.87)
    alloy = Material('Fe0.01Pd', density=11.9)
    si = Material('Si', density=2.3290)

    lamda = Float(30.0, unit="AA", help="Bilayer repetition length")

    fe_lay = Layer(sld_x=fe, d=20.0, sigma=3.0)
    pd_lay = Layer(sld_x=alloy, d=lamda - fe_lay.d + rep*0.1, sigma=3.0 + rep*1.0)
    sub = Layer(sld_x=si.sld_x, sigma=3.0)

    overlayer = Stack(layers=[fe_lay, pd_lay], reps=40)

    sample = Sample(stacks=[overlayer])

    alloy.Fe = 0.05

    tth = np.arange(0.01, 8.0, 0.01)
    r = sample.specular(tth, inst)

    import matplotlib.pylab as pl
    pl.semilogy(tth, r)
    pl.show()



