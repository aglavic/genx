#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
.. module:: stratified_base.py
   :synopsis: Base model for a stratified sample (Sample-Stack-Layer)

.. moduleauthor:: Matts Björck <matts.bjorck@gmail.com>

This module provides the base classes for a Sample-Stack-Layer classes. The classes implements utility functions
so that samples can be parametrised effectively.
"""

from collections import namedtuple

import numpy as np

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

        def inner_loop(layer):
            [out[par].append(layer.__getattribute__(par)(parameter=par, **kwargs)) for par in parameters]
            kwargs["z"] += layer.d(**kwargs)

        def outer_loop(rep):
            kwargs["rep"] += 1
            list(map(inner_loop, self.layers))

        list(map(outer_loop, list(range(self.reps(**kwargs)))))

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
        # self.substrate = Layer()
        # self.ambient = Layer()

        # dictionary that holds the last evaluated layer parameters - used for checking the calcs
        self.layer_dic = {}

        HasParameters.__init__(self, **kwargs)

    def build_layer_list(self, parameters, kwargs):
        """Builds an array for layer parameters by looping through the stacks and layers

        Parameters:
            parameters(list of string): name of a valid layer parameter
            kwargs(dict): Keyword argument to be passed to the parameter call

        Returns:
            ret: namedtuple with the parameters as members, for example ret.d
        """
        kwargs["z"] = 0.0

        out = dict()
        [out.__setitem__(par, [self.substrate.__getattribute__(par)(**kwargs)]) for par in parameters]
        for stack in self.stacks:
            stack_dict = stack.build_layer_list(parameters, kwargs)
            [out[par].extend(stack_dict[par]) for par in parameters]
        kwargs["z"] = 0.0
        [out[par].append(self.ambient.__getattribute__(par)(parameter=par, **kwargs)) for par in parameters]
        Collection = namedtuple("layer_parameters", list(out.keys()))
        return Collection(**out)


class Instrument(HasParameters):
    validation_kwargs = validation_kwargs

    def __init__(self, **kwargs):
        HasParameters.__init__(self, **kwargs)
