#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
.. module:: island_lattice.py
   :synopsis: Models to calcualte the reflectivity from a 2D island array

This module provide functions to calculate the reflectivity from gratings using the distorted wave Born approximation
(DWBA) and the kinematical theory.
"""

import numpy as np

from genx.core.custom_logging import iprint

from . import grating
from .materials import Material
from .parameters import ComplexArray, Float, Func, HasParameters, Int, List, Var

q = Var("q", np.array([0, 0, 0]), help="The scattering vector.")
e = Var("wl", 1.54, help="The wavelength of the radiation.")
z = Var("z", 0.0, help="The out-of-plane position in the sample.")
rep = Var("rep", 0, help="The repetition index in a Stack, starting with zero.")

validation_kwargs = {"q": np.array([0, 0, 0]), "z": 0, "rep": 0}


class Layer(HasParameters):
    validation_kwargs = validation_kwargs
    protected = True

    def __init__(self, **kwargs):
        self.d = Float(0, unit="AA", help="Layer thickness")
        self.sld = ComplexArray(
            0, unit="1/AA^2", help="The in-plane fourier transform of the scattering length density"
        )
        HasParameters.__init__(self, **kwargs)


class Stack(HasParameters):
    validation_kwargs = validation_kwargs
    protected = True

    def __init__(self, **kwargs):
        try:
            value = kwargs.pop("layers")
        except KeyError:
            value = []
        self.layers = List(Layer, value=value, help="The layers in the stack")
        self.reps = Int(1, help="Number of repetitions of the stack")
        HasParameters.__init__(self, **kwargs)

    def build_layer_list(self, parameters, kwargs):
        """Builds an array for layer parameters by looping trough the stacks and layers

        Parameters:
            parameters(list of string): name of a valid layer parameters
            kwargs(dict): Keyword argument to be passed to the parameter call
        """
        out = dict()
        [out.__setitem__(par, []) for par in parameters]
        kwargs["rep"] = -1
        for n in range(self.reps(**kwargs)):
            kwargs["rep"] += 1
            for layer in self.layers:
                # Evaluate the parameter value
                # print kwargs
                [out[par].append(layer.__getattribute__(par)(**kwargs)) for par in parameters]
                kwargs["z"] += layer.d(**kwargs)
        kwargs.pop("rep")
        return out


class Sample(HasParameters):
    validation_kwargs = validation_kwargs

    def __init__(self, **kwargs):
        try:
            value = kwargs.pop("stacks")
        except KeyError:
            value = []
        self.stacks = List(Stack, value=value, help="The stacks in the sample")
        self.substrate = Layer(d=0.0, sld=0 * FLayer())
        self.ambient = Layer(d=0.0, sld=0 * FLayer())

        self.a = Float(2 * np.pi, unit="AA", help="Unit cell size along the x-direction")
        self.b = Float(2 * np.pi, unit="AA", help="Unit cell size along the y-direction")

        # dictionary that holds the last evaluated layer parameters - used for checking the calcs
        self.layer_dic = {}

        HasParameters.__init__(self, **kwargs)

    def build_layer_list(self, parameters, kwargs):
        """Builds an array for layer parameters by looping through the stacks and layers

        Parameters:
            parameters(list of string): name of a valid layer parameter
            kwargs(dict): Keyword argument to be passed to the parameter call
        """
        kwargs["z"] = 0.0
        out = dict()
        [out.__setitem__(par, [self.substrate.__getattribute__(par)(**kwargs)]) for par in parameters]
        for stack in self.stacks:
            stack_dict = stack.build_layer_list(parameters, kwargs)
            [out[par].extend(stack_dict[par]) for par in parameters]
        kwargs["z"] = 0.0
        [out[par].append(self.ambient.__getattribute__(par)(**kwargs)) for par in parameters]
        return out

    def simulate(self, instrument, h, k, qz):
        self.layer_dic = {}
        # Add a q = 0 point for the mean scattering potential
        q = np.c_[h * 2 * np.pi / self.a(), k * 2 * np.pi / self.b(), -qz].T
        kwargs = {"z": 0.0, "q": q}
        wavelength = instrument.wl(**kwargs)
        kwargs["wl"] = wavelength

        area_uc = (self.a * self.b)(**kwargs)
        dic = self.build_layer_list(["d", "sld"], kwargs)
        vf, d = 4 * np.pi * np.array(dic["sld"]), np.array(dic["d"])
        kwargs["q"] = np.array([0, 0, 0])
        vmean = 4 * np.pi * np.array(self.build_layer_list(["sld"], kwargs)["sld"]) / area_uc
        self.layer_dic["sld_mean"] = vmean / 4 / np.pi
        self.layer_dic["d"] = d
        iprint(vmean.shape, vf.shape)
        k_in, k_out = instrument.reflectometer_3axis_kinout(q)
        R = grating.coherent_refl(k_in, k_out, wavelength, vf.T, vmean, d, area_uc, kin=False)
        return R


class Instrument(HasParameters):
    validation_kwargs = validation_kwargs

    def __init__(self, **kwargs):
        self.wl = Float(1.54, unit="AA", help="Wavelength of the used radiation")
        HasParameters.__init__(self, **kwargs)

    def reflectometer_3axis_kinout(self, q):
        """Tranforms q values to k_in and k_out vectors for a reflectometer setup with a sample rotation circle.

        Parameters:
            q(array): an array for qx, qy and qz, size (3xM)

        Returns:
            k_in (array): The incomming wavevector, size (3xM)
            k_out (array): The outgoing wavevector, size (3xM)
        """
        qx = q[0]
        qy = q[1]
        qz = q[2]
        k = 2 * np.pi / self.wl()
        qp = np.sqrt(qx * qx + qy * qy)
        q_abs = np.sqrt(qx * qx + qy * qy + qz * qz)
        phi = np.arctan2(qy, qx)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        apb = np.arcsin(q_abs / 2.0 / k)
        amb = np.arctan2(qp, -qz) % np.pi
        alpha = apb + amb
        beta = apb - amb
        cos_alpha = np.cos(alpha)
        cos_beta = np.cos(beta)
        k_in = k * np.c_[cos_alpha * cos_phi, cos_alpha * sin_phi, np.sin(alpha)].T
        k_out = k * np.c_[cos_beta * cos_phi, cos_beta * sin_phi, -np.sin(beta)].T
        return k_in, k_out


class FLayer(Func):
    """The in-plane fourier transform of an in-plane infinite layer"""

    def function(self, **kwargs):
        q = kwargs["q"]
        return np.where(np.bitwise_and(q[0] == 0, q[1] == 0), 1.0, 0.0)


class FCircle(Func):
    """The in-plane fourier transform of a circle"""

    def __init__(self, **kwargs):
        self.r = Float(1.0, help="The radius of the circle")
        Func.__init__(self, **kwargs)

    def function(self, **kwargs):
        q = kwargs["q"]
        r = self.r(**kwargs)
        q_r = np.sqrt(q[0] ** 2 + q[1] ** 2)
        return 2 * np.pi * r**2 * grating.jinc(q_r * r)


if __name__ == "__main__":
    radius = Float(1.0)
    slope = Float(1)
    mat_Fe = Material("Fe", density=7.87)
    Fe = Layer(d=rep * 0.1 + 10.0, sld=mat_Fe.sld_x * FCircle(r=radius + z * slope))
    Fe.d = 20.0
    q = np.array([0, 0, 0])
    iprint(Fe.sld(q=q))
    iprint(Fe.d)
    ML = Stack(
        reps=10,
        layers=[
            Fe,
        ],
    )

    s = Sample(
        stacks=[
            ML,
        ]
    )
    iprint(isinstance(s, HasParameters), s.__class__.__name__)
    inst = Instrument()

    iprint("D with a gradient: ", s.build_layer_list(["d"], {})["d"])

    # print s.stacks[0].layers[0]
    qz = np.arange(0, 0.1, 0.001)
    s.simulate(inst, qz * 0 + 0.0, qz * 0 + 0.01, qz)
