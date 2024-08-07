"""
Library that contains the the class Model. This
is a general class that store and binds togheter all the other
classes that is part of model parameters, such as data and parameters.
"""

import inspect
import os
import pickle as pickle
import traceback
import types
import typing
import zipfile

from copy import deepcopy
from dataclasses import dataclass, field
from io import StringIO
from logging import debug, info

import numpy as np

# GenX libraries
from . import fom_funcs
from .core.config import BaseConfig
from .core.custom_logging import iprint
from .core.h5_support import H5HintedExport
from .data import DataList
from .exceptions import FomError, GenxIOError, ModelError, ParameterError
from .models.lib.base import ModelParamBase
from .models.lib.parameters import NumericParameter, get_parameters
from .parameters import Parameters

try:
    from yaml import dumper as _dumper
except ImportError:
    pass
else:

    class DumpDict(dict):
        # Make compact layout for non-orso header items read from files
        def yaml_representer(self, dumper):
            return dumper.represent_mapping(dumper.DEFAULT_MAPPING_TAG, dict(self), flow_style=True)


@dataclass
class StartupScript(BaseConfig):
    section = "startup"
    script: str = '[""]'


@dataclass
class ModelParameters(BaseConfig):
    section = "parameters"
    registred_classes: list = field(default_factory=list)
    set_func: str = "set"


@dataclass
class SolverParameters(BaseConfig):
    section = "solver"
    figure_of_merit: str = BaseConfig.GChoice("log", selection=fom_funcs.func_names)
    ignore_fom_nan: bool = True
    ignore_fom_inf: bool = True

    limit_fit_range: bool = False
    fit_xmin: float = BaseConfig.GParam(0.0, pmin=-1000.0, pmax=1000.0)
    fit_xmax: float = BaseConfig.GParam(180.0, pmin=-1000.0, pmax=1000.0)

    groups = {
        "FOM": ["figure_of_merit", ["ignore_fom_nan", "ignore_fom_inf"], "limit_fit_range", ["fit_xmin", "fit_xmax"]]
    }


class GenxScriptModule(types.ModuleType):
    data: DataList
    _sim: bool

    def __init__(self, data: DataList):
        types.ModuleType.__init__(self, "genx_script_module")
        self.__package__ = fom_funcs.__package__
        self.data = data
        self._sim = False
        self.TextIO = typing.TextIO

    @staticmethod
    def Sim(data: DataList):
        # default implementation of Sim function, will be overwritten by user.
        return [di.y * 0 for di in data]


class Model(H5HintedExport):
    """A class that holds the model i.e. the script that defines
    the model and the data + various other attributes.
    """

    h5group_name = "current"
    _group_attr = {"NX_class": "NXentry", "default": "data"}

    saved = True
    fom = None
    fom_mask_func = None

    # parameters stored to file
    script: str
    fomfunction: str
    data: DataList
    parameters: Parameters
    sequence_value: float = 0.0

    @property
    def fomfunction(self):
        return self.fom_func.__name__

    @fomfunction.setter
    def fomfunction(self, value):
        if value in fom_funcs.func_names:
            self.set_fom_func(eval("fom_funcs." + value))
            self.solver_parameters.figure_of_merit = value
        else:
            iprint("Can not find fom function name %s" % value)

    def __init__(self):
        """
        Create a instance and init all the variables.
        """
        self.opt = ModelParameters()
        self.startup_script = StartupScript()
        self.solver_parameters = SolverParameters()
        self.ReadConfig()

        self.data = DataList()
        self.set_script("")
        # noinspection PyBroadException
        try:
            self.set_script("\n".join(eval(self.startup_script.script)))
        except Exception:
            debug("Issue when loading script from config:", exc_info=True)
        self.parameters = Parameters()
        self.fom_func = fom_funcs.log

        self._reset_module()

        # Temporary stuff that needs to keep track on
        self.filename = ""

        self.extra_analysis = {}

    def ReadConfig(self):
        self.opt.load_config()
        self.startup_script.load_config()
        self.solver_parameters.load_config()
        self.create_fom_mask_func()
        self.set_fom_from_config()

    def WriteConfig(self):
        self.opt.save_config()
        self.startup_script.save_config()
        self.solver_parameters.save_config()
        self.set_fom_from_config()

    def set_fom_from_config(self):
        self.fomfunction = self.solver_parameters.figure_of_merit

    def load(self, filename):
        """
        Function to load the necessary parameters from a model file.
        """
        try:
            loadfile = zipfile.ZipFile(filename, "r")
        except Exception as e:
            raise GenxIOError(f"Could not open file, python error {e!r}", filename)
        try:
            new_data = pickle.loads(loadfile.read("data"), encoding="latin1", errors="ignore")
            self.data.safe_copy(new_data)
        except Exception as e:
            iprint("Data section loading (gx file) error:\n ", e, "\n")
            raise GenxIOError("Could not locate the data section.", filename)
        try:
            self.set_script(pickle.loads(loadfile.read("script"), encoding="latin1", errors="ignore"))
        except Exception as e:
            iprint("Script section loading (gx file) error:\n ", e, "\n")
            raise GenxIOError("Could not locate the script.", filename)

        try:
            new_parameters = pickle.loads(loadfile.read("parameters"), encoding="latin1", errors="ignore")
            self.parameters.safe_copy(new_parameters)
        except Exception as e:
            iprint("Script section loading (gx file) error:\n ", e, "\n")
            raise GenxIOError("Could not locate the parameters section.", filename)
        try:
            self.fom_func = pickle.loads(loadfile.read("fomfunction"), encoding="latin1", errors="ignore")
        except Exception:
            raise GenxIOError("Could not locate the fomfunction section.", filename)

        loadfile.close()

        self.filename = os.path.abspath(filename)
        self.saved = True
        self.script_module = GenxScriptModule(self.data)

    def save(self, filename):
        """
        Function to save the model to file filename
        """
        try:
            savefile = zipfile.ZipFile(filename, "w")
        except Exception as e:
            raise GenxIOError(str(e), filename)

        # Save the data structures to file
        try:
            savefile.writestr("data", pickle.dumps(self.data))
        except Exception as e:
            raise GenxIOError("Error writing data: " + str(e), filename)
        try:
            savefile.writestr("script", pickle.dumps(self.script))
        except Exception as e:
            raise GenxIOError("Error writing script: " + str(e), filename)
        self.parameters.model = None
        try:
            savefile.writestr("parameters", pickle.dumps(self.parameters))
        except Exception as e:
            raise GenxIOError("Error writing parameters: " + str(e), filename)
        try:
            savefile.writestr("fomfunction", pickle.dumps(self.fom_func))
        except Exception as e:
            raise GenxIOError("Error writing fom_func:  " + str(e), filename)

        savefile.close()

        self.filename = os.path.abspath(filename)
        self.saved = True

    def __getstate__(self):
        # generate a pickleable object for thie model, it cannot contain dynamically generated functions
        state = self.__dict__.copy()
        if "fom_mask_func" in state:
            del state["fom_mask_func"]
        if "script_module" in state:
            del state["script_module"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.create_fom_mask_func()
        if self.compiled:  # if the model was compiled before pickling, do it now
            self.compile_script()

    def copy(self) -> "Model":
        state = self.__getstate__().copy()
        output = Model()
        output.__setstate__(state)
        return output

    def deepcopy(self) -> "Model":
        return deepcopy(self)

    def read_h5group(self, group):
        """
        Read the parameters from a hdf5 group
        """
        H5HintedExport.read_h5group(self, group)
        self.create_fom_mask_func()
        self.saved = True
        self.script_module = GenxScriptModule(self.data)
        self.compiled = False

    def save_addition(self, name, text):
        """
        Save additional text sub-file with name name/text to the current file.
        """
        if self.filename == "":
            raise GenxIOError("File must be saved before new information is added", "")
        try:
            savefile = zipfile.ZipFile(self.filename, "a")
        except Exception as e:
            raise GenxIOError(str(e), self.filename)

        # Check so the model data is not overwritten
        if name == "data" or name == "script" or name == "parameters":
            raise GenxIOError("It not alllowed to save a subfile with name: %s" % name)

        try:
            savefile.writestr(name, text)
        except Exception as e:
            raise GenxIOError(str(e), self.filename)
        savefile.close()

    def load_addition(self, name) -> bytes:
        """
        Load additional text from sub-file
        """
        if self.filename == "":
            raise GenxIOError("File must be loaded before additional information is read", "")
        try:
            loadfile = zipfile.ZipFile(self.filename, "r")
        except Exception:
            raise GenxIOError("Could not open the file", self.filename)

        try:
            raw_string = loadfile.read(name)
        except Exception:
            raise GenxIOError("Could not read the section named: %s" % name, self.filename)
        loadfile.close()
        return raw_string

    def reset(self):
        self._reset_module()

    def _reset_module(self):
        """
        Internal method for resetting the module before compilation
        """
        self.create_fom_mask_func()
        self.script_module = GenxScriptModule(self.data)
        self.compiled = False

    def compile_script(self):
        """
        compile the script in a seperate module.
        """
        self._reset_module()
        # Testing to see if this works under windows
        self.set_script("\n".join(self.script.splitlines()))
        try:
            exec(self.script, self.script_module.__dict__)
            for obj in self.script_module.__dict__.values():
                if isinstance(obj, ModelParamBase):
                    obj._extract_callpars(self.script)
        except Exception:
            outp = StringIO()
            traceback.print_exc(200, outp)
            val = outp.getvalue()
            outp.close()
            raise ModelError(str(val), 0)
        else:
            self.compiled = True

    def eval_in_model(self, codestring):
        """
        Excecute the code in codestring in the namespace of model module
        """
        result = eval(codestring, self.script_module.__dict__)
        return result

    def set_script_variable(self, name, value):
        self.script_module.__dict__[name] = value

    def unset_script_variable(self, name):
        del self.script_module.__dict__[name]

    @staticmethod
    def fom_mask_nan_inf(a):
        return np.where(np.isfinite(a), a, np.zeros_like(a))

    @staticmethod
    def fom_mask_nan(a):
        return np.where(np.isnan(a), a, np.zeros_like(a))

    @staticmethod
    def fom_mask_inf(a):
        return np.where(np.isfinite(a), a, np.zeros_like(a))

    @staticmethod
    def fom_mask_empty(a):
        return a

    def create_fom_mask_func(self):
        """
        Create a mask func for fom to take care of unallowed values.
        """
        if self.solver_parameters.ignore_fom_nan and self.solver_parameters.ignore_fom_inf:
            self.fom_mask_func = Model.fom_mask_nan_inf
        elif self.solver_parameters.ignore_fom_nan:
            self.fom_mask_func = Model.fom_mask_nan
        elif self.solver_parameters.ignore_fom_inf:
            self.fom_mask_func = Model.fom_mask_inf
        else:
            self.fom_mask_func = Model.fom_mask_empty

    def calc_fom(self, simulated_data):
        """
        Sums up the evaluation of the fom values calculated for each
        data point to form the overall fom function for all data sets.
        """
        fom_raw = self.fom_func(simulated_data, self.data)
        # limit the x-range of fitting
        if self.solver_parameters.limit_fit_range:
            for i, di in enumerate(self.data):
                fltr = (di.x < self.solver_parameters.fit_xmin) | (di.x > self.solver_parameters.fit_xmax)
                fom_raw[i][fltr] = 0.0
        # Sum up a unique fom for each data set in use
        fom_indiv = [np.sum(np.abs(self.fom_mask_func(fom_set))) for fom_set in fom_raw]
        fom = np.sum([f for f, d in zip(fom_indiv, self.data) if d.use])

        # Lets extract the number of data points as well:
        N = np.sum([len(self.fom_mask_func(fom_set)) for fom_set, d in zip(fom_raw, self.data) if d.use])
        # And the number of fit parameters
        p = self.parameters.get_len_fit_pars()
        # self.fom_dof = fom/((N-p)*1.0)
        try:
            use_dif = self.fom_func.__div_dof__
        except AttributeError:
            use_dif = False
        if use_dif:
            fom = fom / ((N - p) * 1.0)

        penalty_funcs = self.get_par_penalty()
        if len(penalty_funcs) > 0 and fom is not np.nan:
            fom += sum([pf() for pf in penalty_funcs])
        return fom_raw, fom_indiv, fom

    def evaluate_fit_func(self, get_elements=False):
        """
        Evalute the Simulation fucntion and returns the fom. Use this one
        for fitting. Use evaluate_sim_func(self) for updating of plots
        and such.
        """
        self.script_module._sim = False
        simulated_data = self.script_module.Sim(self.data)
        fom_raw, fom_inidv, fom = self.calc_fom(simulated_data)
        if get_elements:
            return np.hstack([self.fom_mask_func(fom_set) for fom_set in fom_raw])
        else:
            return fom

    def evaluate_sim_func(self):
        """
        Evalute the Simulation function and updates the data simulated data
        as well as the fom of the model. Use this one for calculating data to
        update plots, simulations and such.
        """
        self.script_module._sim = True
        try:
            simulated_data = self.script_module.Sim(self.data)
        except Exception:
            outp = StringIO()
            traceback.print_exc(200, outp)
            val = outp.getvalue()
            outp.close()
            raise ModelError(str(val), 1)

        # check so that the Sim function returns anything
        if not simulated_data:
            text = (
                "The Sim function does not return anything, it should"
                + " return a list of the same length as the number of data sets."
            )
            raise ModelError(text, 1)
        # Check so the number of data sets is correct
        if len(simulated_data) != len(self.data):
            text = (
                "The number of simulated data sets returned by the Sim function"
                + " has to be same as the number of loaded data sets.\n"
                + "Number of loaded data sets: "
                + str(len(self.data))
                + "\nNumber of simulated data sets: "
                + str(len(simulated_data))
            )
            raise ModelError(text, 1)

        self.data.set_simulated_data(simulated_data)

        try:
            fom_raw, fom_inidv, fom = self.calc_fom(simulated_data)
            self.fom = fom
        except Exception:
            outp = StringIO()
            traceback.print_exc(200, outp)
            val = outp.getvalue()
            outp.close()
            raise FomError(str(val))
        self.data.set_fom_data(fom_raw)

    def create_fit_func(self, identifier):
        """
        Creates a function from the string expression in string.
        If the string is a function in the model this function will be
        returned if string represents anything else a function that sets that
        object will be returned.
        """
        obj = self.eval_in_model(identifier)
        # Is it a function or a method!
        if callable(obj):
            return obj
        # Make a function to set the object
        elif isinstance(obj, NumericParameter):
            # We have a NumericParameter that should be set
            exec("def __tempfunc__(val):\n" "    %s.value = val" % identifier, self.script_module.__dict__)
            # noinspection PyUnresolvedReferences
            return self.script_module.__tempfunc__
        else:
            # The function must be created in the module in order to access
            # the different variables
            exec("def __tempfunc__(val):\n\t%s = val" % identifier, self.script_module.__dict__)
            # noinspection PyUnresolvedReferences
            return self.script_module.__tempfunc__

    def get_fit_pars(self, use_bounds=True):
        """
        Returns the parameters used with fitting. i.e. the function to
        set the paraemters, the guess value (values), minimum allowed values
        and the maximum allowed values
        """
        (row_numbers, sfuncs, vals, minvals, maxvals) = self.parameters.get_fit_pars()
        if len(sfuncs) == 0:
            raise ParameterError(sfuncs, 0, None, 4)
        if use_bounds:
            # Check for min and max on all the values
            for i in range(len(vals)):
                # parameter less than min
                if vals[i] < minvals[i]:
                    raise ParameterError(sfuncs[i], row_numbers[i], None, 3)
                # parameter larger than max
                if vals[i] > maxvals[i]:
                    raise ParameterError(sfuncs[i], row_numbers[i], None, 2)

        # Compile the strings to create the functions..
        funcs = []
        for func in sfuncs:
            try:
                funcs.append(self.create_fit_func(func))
            except Exception as e:
                raise ParameterError(func, row_numbers[len(funcs)], e, 0)
        return funcs, vals, minvals, maxvals

    def get_par_penalty(self):
        for var in self.script_module.__dict__.values():
            if var.__class__.__name__ == "UserVars":
                # noinspection PyProtectedMember
                return var._penalty_funcs
        return []

    def get_sim_pars(self):
        """
        Returns the parameters used with simulations. i.e. the function to
        set the parameters, the guess value (values). Used for simulation,
        for fitting see get_fit_pars(self).s
        """
        (sfuncs, vals) = self.parameters.get_sim_pars()
        # Compile the strings to create the functions..
        funcs = []
        for func in sfuncs:
            try:
                funcs.append(self.create_fit_func(func))
            except Exception as e:
                raise ParameterError(func, len(funcs), e, 0)

        return funcs, vals

    # noinspection PyShadowingBuiltins
    def simulate(self, compile=True):
        """
        Simulates the data sets using the values given in parameters...
        also compiles the script if asked for (default)
        """
        if compile:
            self.compile_script()
        (funcs, vals) = self.get_sim_pars()
        # Set the parameter values in the model
        i = 0
        for func, val in zip(funcs, vals):
            try:
                func(val)
            except Exception as e:
                (sfuncs_tmp, vals_tmp) = self.parameters.get_sim_pars()
                raise ParameterError(sfuncs_tmp[i], i, e, 1)
            i += 1

        self.evaluate_sim_func()

    def new_model(self):
        """
        Reinitilizes the model. Thus, removes all the traces of the
        previous model.
        """
        debug("create new model")
        self.data = DataList()
        self.set_script("")
        self.parameters = Parameters()

        self.fom_func = fom_funcs.log
        self._reset_module()

        # Temporary stuff that needs to keep track on
        self.filename = ""
        self.saved = False

        self.extra_analysis = {}

    def pickable_copy(self):
        """
        Creates a pickable object of the model. Can be used for saving or
        sending to other processes, i.e., parallel processing.
        """
        model_copy = Model()
        model_copy.data = self.data
        model_copy.script = self.script
        model_copy.parameters = self.parameters.copy()
        model_copy.fom_func = self.fom_func
        # The most important stuff - a module is not pickleable
        model_copy.script_module = None
        model_copy.filename = self.filename
        model_copy.compiled = self.compiled
        model_copy.fom = self.fom
        model_copy.solver_parameters = self.solver_parameters.copy()
        model_copy.startup_script = self.startup_script.copy()
        model_copy.opt = self.opt.copy()
        model_copy.saved = self.saved
        # Needs to reset the fom_mask_func since this fails under windows.
        model_copy.fom_mask_func = None
        #
        return model_copy

    def get_table_as_ascii(self):
        """
        Just a copy of the parameters class method get_ascii_output()
        """
        return self.parameters.get_ascii_output()

    def get_data_as_asciitable(self, indices=None):
        """
        Just a copy of the method defined in data with the same name.
        """
        return self.data.get_data_as_asciitable(indices)

    def export_table(self, filename):
        """
        Export the table to filename. ASCII output.
        """
        self._save_to_file(filename, self.parameters.get_ascii_output())

    def create_simple_model(self):
        """
        Create ORSO simple model representation of this model.
        """
        from orsopy.fileio import model_language as ml

        sm = self.script_module

        if sm.inst.probe == "x-ray":
            # use x-ray SLDs
            def get_material(li):
                sld = complex(li.f * li.dens) * 1e-6
                return ml.Material(sld=ml.ComplexValue(sld.real, sld.imag))

        else:
            # use neutron SLDs
            def get_material(li):
                sld = complex(li.b * li.dens) * 1e-6
                return ml.Material(sld=ml.ComplexValue(sld.real, sld.imag))

        def get_layer(li):
            return ml.Layer(thickness=li.d, material=get_material(li), roughness=li.sigma)

        names = list(sm.__dict__.keys())
        objects = list(sm.__dict__.values())

        defaults = ml.ModelParameters(length_unit="angstrom", sld_unit="1/angstrom^2")
        layers = {}
        materials = {}
        materials["ambient"] = get_material(sm.Amb)
        materials["substrate"] = get_material(sm.Sub)
        sub_stacks = {}
        stack_order = ["ambient"]
        for si in reversed(sm.sample.Stacks):
            if len(si.Layers) == 0:
                continue
            ni = names[objects.index(si)]
            stack_order.append(ni)

            layer_order = []
            for lj in reversed(si.Layers):
                nj = names[objects.index(lj)]
                layers[nj] = get_layer(lj)
                layer_order.append(nj)
            sub_stacks[ni] = ml.SubStack(repetitions=si.Repetitions, stack=" | ".join(layer_order))
        stack_order.append("substrate")
        stack_str = " | ".join(stack_order)

        return ml.SampleModel(
            stack=stack_str,
            origin="GenX model",
            sub_stacks=sub_stacks,
            layers=layers,
            materials=materials,
            globals=defaults,
        )

    def export_orso(self, basename):
        """
        Export the data to files with basename filename. ORT output.
        The fileending will be .ort
        """
        self.simulate(True)
        from orsopy.fileio import Column, ErrorColumn, Orso, OrsoDataset, save_orso

        from .version import __version__ as version

        para_list = [
            dict(
                [
                    (nj, tpj(pij))
                    for nj, pij, tpj in zip(self.parameters.data_labels, pi, [str, float, bool, float, float, str])
                ]
            )
            for pi in self.parameters.data
            if pi[0].strip() != ""
        ]
        add_header = {
            "analysis": {
                "software": {"name": "GenX", "version": version},
                "model": self.create_simple_model(),
                "script": self.script,
                "parameters": para_list,
            }
        }
        # add possible additional analysis data to the header
        add_header["analysis"].update(self.extra_analysis)
        ds = []
        for di in self.data:
            header = Orso.from_dict(di.meta).to_dict()
            try:
                import getpass

                # noinspection PyTypeChecker
                add_header["analysis"]["operator"] = {"name": getpass.getuser()}
            except Exception:
                pass
            import datetime

            # noinspection PyTypeChecker
            add_header["analysis"]["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            header.update(add_header)
            header["data_set"] = di.name
            columns = [di.x, di.y, di.error]
            if "columns" in header:
                prev_columns_list = [Column(**ci) if "name" in ci else ErrorColumn(**ci) for ci in header["columns"]]
                prev_columns = dict([(ci.name, ci) for ci in prev_columns_list])
            else:
                prev_columns_list = None
                prev_columns = {}
            if hasattr(self.script_module, "inst") and self.script_module.inst.coords in ["tth", "2θ"]:
                # this will probably need changing in the future as ORSO standard
                # will probably require Qz for first column
                column_names = [Column("TTh", "deg"), Column("R"), ErrorColumn("R")]
            else:
                column_names = [Column("Qz", "1/angstrom"), Column("R"), ErrorColumn("R")]
            if "res" in di.extra_data:
                columns.append(di.extra_data["res"])
                column_names.append(ErrorColumn(column_names[0].name))
            columns.append(di.y_sim)
            column_names.append(Column("Rsim"))
            columns.append(di.y_fom)
            column_names.append(Column("FOM"))
            for name, col in sorted(di.extra_data.items()):
                if name == "res":
                    continue
                columns.append(col)
                if name in prev_columns:
                    column_names.append(prev_columns[name])
                else:
                    column_names.append(Column(name))
            header["columns"] = column_names
            header_obj = Orso.from_dict(header)
            # for key, value in header_obj._user_data.items():
            #     if not key in ['analysis'] and isinstance(value, dict):
            #         header_obj._user_data[key] = DumpDict(value)
            ds.append(OrsoDataset(header_obj, np.array(columns).T))
        try:
            save_orso(ds, basename, data_separator="\n")
        except GenxIOError as e:
            raise GenxIOError(e.error_message, e.file)

    @staticmethod
    def update_orso_meta(datasets: DataList):
        from orsopy.fileio import Orso

        for di in datasets:
            defaults = Orso.empty().to_dict()
            Model.update_dictionary(defaults, di.meta)
            di.meta = defaults

    def validate_orso_meta(self):
        from orsopy.fileio import base

        base._validate_header_data([di.meta for di in self.data])

    @staticmethod
    def update_dictionary(to_update, to_include):
        """
        Updates the entries in one dictrionary recursively to keep existing sub-keys.
        """
        for key, value in to_include.items():
            if key in to_update and isinstance(to_update[key], dict):
                if not isinstance(value, dict):
                    info(f"Metadata for {key} according to specification should be dictionary, found {value}")
                else:
                    Model.update_dictionary(to_update[key], value)
            else:
                to_update[key] = value

    def export_data(self, basename):
        """
        Export the data to files with basename filename. ASCII output.
        The fileending will be .dat
        First column is the x-values.
        Second column is the data y-vales.
        Third column the error on the data y-values.
        Fourth column the calculated y-values.
        """
        try:
            self.data.export_data_to_files(basename)
        except GenxIOError as e:
            raise GenxIOError(e.error_message, e.file)

    def export_script(self, filename):
        """
        Export the script to filename. Will be a python script with ASCII
        output (naturally).
        """
        self._save_to_file(filename, self.script)

    def import_script(self, filename):
        """
        Imports the script from file filename
        """
        read_string = self._read_from_file(filename)
        self.set_script(read_string)
        self.compiled = False

    def import_table(self, filename):
        """
        import the table from filename. ASCII input. tab delimited
        """
        read_string = self._read_from_file(filename)
        self.parameters.set_ascii_input(read_string)

    def _save_to_file(self, filename, save_string):
        """
        Save the string to file with filename.
        """
        try:
            with open(filename, "w", encoding="utf-8") as savefile:
                savefile.write(save_string)
        except Exception as e:
            raise GenxIOError(e.__str__(), filename)

    def _read_from_file(self, filename):
        """
        Reads the entrie file into string and returns it.
        """
        try:
            with open(filename, "r", encoding="utf-8") as loadfile:
                read_string = loadfile.read()
        except Exception as e:
            raise GenxIOError(e.__str__(), filename)

        return read_string

    def get_parameters(self) -> Parameters:
        return self.parameters

    def get_data(self) -> DataList:
        return self.data

    def get_script(self) -> str:
        return self.script

    def get_filename(self) -> str:
        return self.filename

    def get_possible_parameters(self) -> dict:
        """
        Returns all the parameters that can be fitted. Is used by the parameter grid.
        """
        par_dict = get_parameters(self.script_module, numeric_types_only=True)
        if len(par_dict) == 0:
            par_dict = self.get_possible_set_functions()
        return par_dict

    def get_possible_set_functions(self) -> dict:
        """
        Returns all the parameters that can be fitted given by the old style of defining parameters GenX2.4.X
        """
        # Start by updating the config file
        self.ReadConfig()
        # First we should see if any of the
        # classes is defined in model.__pars__
        # or in __pars__
        pars = []
        try:
            # Check if the have a pars in module named model
            pars_tmp = self.eval_in_model("model.__pars__")
            pars_tmp = ["model.%s" % p for p in pars_tmp]
            pars += pars_tmp
        except Exception:
            pass

        # Check if we have a __pars__ in the main script
        try:
            pars_tmp = self.eval_in_model("__pars__")
            pars_tmp = ["%s" % p for p in pars_tmp]
            pars += pars_tmp
        except Exception:
            pass

        isstrings = sum([type(p) is str for p in pars]) == len(pars)

        if not isstrings:
            pars = []

        # First find the classes that exists..
        # and defined in self.registred_classes
        # In addition, any class derived from ModelParamBase is found.
        classes = [ModelParamBase]
        for c in self.opt.registred_classes + pars:
            try:
                ctemp = self.eval_in_model(c)
            except Exception:
                pass
            else:
                if inspect.isclass(ctemp):
                    classes.append(ctemp)
        # Check if there are any classes defined before we proceed
        if len(classes) > 0:
            # Get all the objects in the compiled module
            names = list(self.script_module.__dict__.keys())
            # Create a tuple of the classes we defined above
            tuple_of_classes = tuple(classes)
            # Creating a dictionary that holds the name of the classes
            # each item for a classes is a new dictionary that holds the
            # object name and then a list of the methods.
            par_dict = {}
            # [par_dict.__setitem__(clas.__name__, {}) for clas in classes]
            # find all the names of the objects that belongs to
            # one of the classes
            objs = [(name, self.eval_in_model(name)) for name in names]
            for name, obj in objs:
                if isinstance(obj, tuple_of_classes):
                    if obj.__class__.__name__ not in par_dict:
                        par_dict[obj.__class__.__name__] = {}
                    par_dict[obj.__class__.__name__].__setitem__(
                        name, [member for member in dir(obj) if member.startswith(self.opt.set_func)]
                    )
            return par_dict
        return {}

    def set_script(self, text: str):
        self.script = text
        self.compiled = False

    def set_fom_func(self, fom_func: callable):
        self.fom_func = fom_func

    def is_compiled(self) -> bool:
        return self.compiled

    def bumps_problem(self):
        """
        Return the fit function as a bumps problem to be used with the bumps.fitters.fit function.
        """
        if not self.compiled:
            self.compile_script()
        # indep_module = self.script_module
        # data = self.data
        # compile again to have a completely independent script module
        self.compile_script()

        from bumps.fitproblem import FitProblem

        return FitProblem(GenxCurve(self))

    def asym_stderr(self, dream_fit):
        """
        Approximate standard error as distance to lower and upper bound of
        the 68% interval for the sample.
        Based on the stderr estimation from dream fit in bumps.
        """
        from bumps.dream.stats import var_stats

        vstats = var_stats(dream_fit.state.draw(portion=dream_fit._trimmed))
        return np.array([(v.p68[0] - v.best, v.p68[1] - v.best) for v in vstats], "d")

    def bumps_fit(
        self,
        method="dream",
        pop=15,
        samples=1e5,
        burn=100,
        steps=0,
        thin=1,
        alpha=0,
        outliers="none",
        trim=False,
        monitors=None,
        problem=None,
        **options,
    ):
        # create a fitter similar to the bumps.fitters.fit function but with option for GUI monitoring
        if monitors is None:
            monitors = []
        from bumps.fitters import FIT_ACTIVE_IDS, FIT_AVAILABLE_IDS, FITTERS, FitDriver

        from genx.bumps_optimizer import BumpsResult

        options["pop"] = pop
        options["samples"] = samples
        options["burn"] = burn
        options["steps"] = steps
        options["thin"] = thin
        options["alpha"] = alpha
        options["outliers"] = outliers
        options["trim"] = trim

        if problem is None:
            problem = self.bumps_problem()
        problem.fitness.stop_fit = False
        options["abort_test"] = lambda: problem.fitness.stop_fit

        # verbose = True
        if method not in FIT_AVAILABLE_IDS:
            raise ValueError("unknown method %r not one of %s" % (method, ", ".join(sorted(FIT_ACTIVE_IDS))))
        for fitclass in FITTERS:
            if fitclass.id == method:
                break
        # noinspection PyUnboundLocalVariable
        driver = FitDriver(fitclass=fitclass, problem=problem, monitors=monitors, **options)
        driver.clip()  # make sure fit starts within domain

        x, fx = driver.fit()
        problem.setp(x)
        dx = driver.stderr()
        if method == "dream":
            dxpm = self.asym_stderr(driver.fitter)
        else:
            dxpm = None
        result = BumpsResult(x=x, dx=dx, dxpm=dxpm, cov=driver.cov(), chisq=driver.chisq(), bproblem=problem)
        if hasattr(driver.fitter, "state"):
            result.state = driver.fitter.state
        return result

    def build_dream(self, x_n_gen=25, x_n_chain=5, sigmas=None):
        """
        Compile an independent script model and function to include in the
        Dream MCMC model and return that model.
        """
        if not self.compiled:
            self.compile_script()
        indep_module = self.script_module

        from bumps.dream import Dream
        from bumps.dream.model import Simulation
        from numpy import hstack, random

        data = self.data
        # compile again to have a completely independent script module
        self.compile_script()

        # generate a function that takes a single parameter list and returns the stacked data
        mstring = "def model_data(p):\n    "
        fstring = ""
        pstart = []
        i = 0
        for p in self.parameters:
            if not p.fit:
                continue
            name = p.name.replace(".set", "_")
            fstring += "script_module.%s(p[%i])\n" % (p.name, i)
            pstart.append((name, p.value, p.min, p.max))
            i += 1
        fstring += "res=script_module.Sim(data)\nreturn hstack(res)"
        namespace = {"script_module": indep_module, "data": data, "model_data": None, "hstack": hstack}
        exec(mstring + fstring.replace("\n", "\n    "), namespace)

        n = len(pstart)
        if sigmas is None:
            # if no ranges are supplied, use fit range from model
            bounds = (tuple(p[2] for p in pstart), tuple(p[3] for p in pstart))
        else:
            bounds = (
                tuple(p[1] + 3 * s[0] for p, s in zip(pstart, sigmas)),
                tuple(p[1] + 3 * s[1] for p, s in zip(pstart, sigmas)),
            )
        bsim = Simulation(
            f=namespace["model_data"],
            data=hstack([di.y for di in data]),
            sigma=hstack([di.error for di in data]),
            bounds=bounds,
            labels=[p[0] for p in pstart],
        )
        bdream = Dream(
            model=bsim, draws=20000, population=random.randn(x_n_gen * n, x_n_chain * n, n)
        )  # n_gen, n_chain, n_var
        return bdream

    def bumps_update_parameters(self, res):
        """
        Update the GenX model from a bumps fit result.
        """
        x = res.x
        bproblem = res.bproblem
        names = list(bproblem.labels())
        if len(names) != len(x):
            raise ValueError("The number of parameters does not fit the model parameters, was the model changed?")

        for pi in self.parameters:
            if pi.fit:
                pi.value = x[names.index(pi.name.replace(".set", "_"))]

    def plot(self, data_labels=None, sim_labels=None):
        """
        Plot all datasets in the model wtih matplotlib similar to the display in the GUI.
        data_labels and sim_labels are lists of strings for each curve,
        if none is supplied the curve label is created from the dataset names and index.
        """
        self.simulate()
        return self.data.plot(data_labels=data_labels, sim_labels=sim_labels)

    def __eq__(self, other: "Model"):
        # compare relevant parts of model object agains another
        return self.data == other.data and self.script == other.script and self.parameters == other.parameters

    def __repr__(self):
        """
        Display information about the model.
        """
        output = "Genx Model"
        if self.compiled:
            output += " - compiled"
        else:
            output += " - not compiled yet"
        output += "\n"
        output += "File: %s\n" % self.filename
        output += self.parameters.__repr__()
        output += self.data.__repr__()
        return output

    def _repr_html_(self):
        """
        Display information about the model.
        """
        try:
            self.simulate()
        except Exception as error:
            print(error)

        output = "<h3>Genx Model"
        if self.compiled:
            output += " - compiled"
        else:
            output += " - not compiled yet"
        output += "</h3>\n"
        output += "<p>File: %s</p>\n" % self.filename

        output += '<div style="width: 100%;"><div style="width: 40%; float: left;">'
        output += self.data._repr_html_()
        output += "</div>"

        # generate a plot of the model
        import binascii

        from io import BytesIO

        from matplotlib import pyplot as plt

        sio = BytesIO()
        fig = plt.figure(figsize=(10, 8))
        self.data.plot()
        plt.xlabel("q/tth")
        plt.ylabel("Intensity")
        fig.canvas.print_png(sio)
        plt.close()
        img_data = binascii.b2a_base64(sio.getvalue()).decode("ascii")
        output += '<div style="margin-left: 40%;"><img src="data:image/png;base64,{}&#10;"></div></div>'.format(
            img_data
        )

        output += self.parameters._repr_html_()
        return output

    @property
    def widget(self):
        return self._repr_ipyw_()

    def _repr_ipyw_(self):
        import ipywidgets as ipw

        try:
            self.simulate()
        except Exception as error:
            print(error)

        graphw = ipw.Output()
        with graphw:
            from matplotlib import pyplot as plt

            fig = plt.figure(figsize=(10, 8))
            self.data.plot()
            plt.xlabel("q/tth")
            plt.ylabel("Intensity")
            from IPython.display import display

            display(fig)
            plt.close()

        dataw = self.data._repr_ipyw_()
        top = ipw.HBox([dataw, graphw])
        parameters = self.parameters._repr_ipyw_()

        replot = ipw.Button(description="simulate")
        replot.on_click(self._ipyw_replot)
        replot._plot_output = graphw

        script_area = ipw.Textarea(self.script, layout=ipw.Layout(width="100%"))
        script_area.observe(self._ipyw_script, names="value")
        tabs = ipw.Tab(children=[parameters, script_area])
        tabs.set_title(0, "Parameters")
        tabs.set_title(1, "Script")

        return ipw.VBox([replot, top, tabs])

    def _ipyw_replot(self, button):
        try:
            self.simulate()
        except Exception as error:
            print(error)
        with button._plot_output:
            from IPython.display import clear_output, display
            from matplotlib import pyplot as plt

            fig = plt.figure(figsize=(10, 8))
            self.data.plot()
            plt.xlabel("q/tth")
            plt.ylabel("Intensity")
            plt.draw()
            clear_output()
            display(fig)
            plt.close()

    def _ipyw_script(self, change):
        self.set_script(change.new)


class GenxCurve:
    """
    Bumps Curve object for a GenX model.
    """

    model: Model
    n_fev: int

    def __init__(self, model: Model):
        self._num_curves = 1

        self.labels = ["x", "I"]

        self.model = model

        if not self.model.compiled:
            self.model.compile_script()
        self.model_script = self.model.script_module

        self.name = "GenX model"
        self.plot_x = None

        pars, state, funcs = self._parse_pars()

        self._pnames = list(pars.keys())
        self._pars = pars
        self._state = state
        self._set_funcs = funcs
        self._cached_theory = None
        self.stop_fit = False
        self.n_fev = 0

    @property
    def x(self):
        return [self._pars[name].value for name in self._pnames]

    @property
    def y(self):
        return np.hstack([di.y for di in self.model.data if di.use])

    @property
    def dy(self):
        return np.hstack([di.error for di in self.model.data if di.use])

    def _parse_pars(self):
        from bumps.parameter import Parameter

        pars = {}
        state = {}
        funcs = {}
        for p in self.model.parameters:
            if p.name.strip() == "":
                continue
            name = p.name.replace(".set", "_")
            if p.fit:
                pars[name] = Parameter.default(p.value, name=name, bounds=(p.min, p.max))
            else:
                state[name] = p.value
            funcs[name] = self.model.create_fit_func(p.name)
        return pars, state, funcs

    def update(self):
        self._cached_theory = None
        self.n_fev = 0

    def parameters(self):
        return self._pars

    def numpoints(self):
        return np.sum([di.x.shape[0] for di in self.model.data if di.use])

    def theory(self, x=None):
        # Use cache if x is None, otherwise compute theory with x.
        if x is None:
            if self._cached_theory is None:
                self._cached_theory = self._compute_theory(self.x)
            return self._cached_theory
        return self._compute_theory(x)

    def _compute_theory(self, x):
        self._apply_par(x)
        sim = self.model_script.Sim(self.model.data)
        self.n_fev += 1
        return np.hstack([si for si, di in zip(sim, self.model.data) if di.use])

    def _apply_par(self, x):
        for ni, si in self._state.items():
            self._set_funcs[ni](si)
        for ni, xi in zip(self._pnames, x):
            self._set_funcs[ni](xi)

    def simulate_data(self, noise=None):
        theory = self.theory()
        if noise is not None:
            if noise == "data":
                pass
            elif noise < 0:
                self.dy = -0.01 * noise * theory
            else:
                self.dy = noise
        self.y = theory + np.random.randn(*theory.shape) * self.dy

    def residuals(self):
        return (self.theory() - self.y) / self.dy

    def nllf(self):
        r = self.residuals()
        fom = np.sum(r**2)
        penalty_funcs = self.model.get_par_penalty()
        if len(penalty_funcs) > 0 and fom is not np.nan:
            fom += sum([pf() for pf in penalty_funcs]) * (len(r) - len(self._pars))
        return 0.5 * fom

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["model_script"]
        del state["_set_funcs"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not self.model.compiled:
            self.model.compile_script()
        self.model_script = self.model.script_module

        pars, state, funcs = self._parse_pars()
        self._pnames = list(pars.keys())
        self._pars = pars
        self._state = state
        self._set_funcs = funcs
