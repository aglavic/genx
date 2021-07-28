'''
Library that contains the the class Model. This
is a general class that store and binds togheter all the other
classes that is part of model parameters, such as data and parameters.
'''

import inspect
import io
# Standard libraries
import os
import pickle as pickle
# assure compatibility with scripts that don't include genx in import
import sys
import traceback
import types
import zipfile
from dataclasses import dataclass, field
from typing import Dict
from logging import debug

import numpy as np

# GenX libraries
from . import fom_funcs
from .parameters import Parameters
from .data import DataList
from .exceptions import FomError, GenxIOError, ModelError, ParameterError
from .filehandling import BaseConfig
from .gui_logging import iprint
from .models.lib.parameters import get_parameters, NumericParameter
from .lib.h5_support import H5HintedExport


sys.modules['models'] = sys.modules['genx.models']


@dataclass
class StartupScript(BaseConfig):
    section = 'startup'
    script: str = '[""]'


@dataclass
class ModelParameters(BaseConfig):
    section = 'parameters'
    registred_classes: list = field(default_factory=list)
    set_func: str = 'set'


@dataclass
class SolverParameters(BaseConfig):
    section = 'solver'
    ignore_fom_nan: bool = True
    ignore_fom_inf: bool = True

    limit_fit_range:bool=False
    fit_xmin:float=0.0
    fit_xmax:float=180.0

class GenxScriptModule(types.ModuleType):
    data: DataList

    def __init__(self, data: DataList):
        types.ModuleType.__init__(self, 'genx_script_module')
        self.data=data
        self._sim=False

    @staticmethod
    def Sim(data: DataList):
        # default implementation of Sim function, will be overwritten by user.
        return [di.y*0 for di in data]

class Model(H5HintedExport):
    ''' A class that holds the model i.e. the script that defines
        the model and the data + various other attributes.
    '''
    h5group_name='current'

    saved=True
    fom=None
    fom_mask_func=None

    # parameters stored to file
    script:str
    fomfunction:str
    solver_pars:Dict[str, bool]
    data:DataList
    parameters:Parameters

    @property
    def solver_pars(self):
        return {'fom_ignore_nan': self.solver_parameters.ignore_fom_nan,
                'fom_ignore_inf': self.solver_parameters.ignore_fom_inf}
    @solver_pars.setter
    def solver_pars(self, value):
        try:
            self.solver_parameters.ignore_fom_nan = value['fom_ignore_nan']
        except KeyError:
            iprint("Could not load parameter fom_ignore_nan from file")
        try:
            self.solver_parameters.ignore_fom_inf = value['fom_ignore_inf']
        except KeyError:
            iprint("Could not load parameter fom_ignore_inf from file")

    @property
    def fomfunction(self):
        return self.fom_func.__name__
    @fomfunction.setter
    def fomfunction(self, value):
        if value in fom_funcs.func_names:
            self.set_fom_func(eval('fom_funcs.'+value))
        else:
            iprint("Can not find fom function name %s"%value)


    def __init__(self):
        '''
        Create a instance and init all the variables.
        '''
        self.opt = ModelParameters()
        self.startup_script = StartupScript()
        self.solver_parameters = SolverParameters()
        self.read_config()

        self.data = DataList()
        self.script = ''
        # noinspection PyBroadException
        try:
            self.script = "\n".join(eval(self.startup_script.script))
        except:
            debug('Issue when loading script from config:', exc_info=True)
        self.parameters = Parameters(model=self)
        self.fom_func = fom_funcs.log

        self._reset_module()

        # Temporary stuff that needs to keep track on
        self.filename = ''
        self.compiled = False

        self.limit_fit_range = False
        self.fit_xmin = 0.01
        self.fit_xmax = 0.1

        self.extra_analysis = {}

    def read_config(self):
        self.opt.load_config()
        self.startup_script.load_config()
        self.solver_parameters.load_config()
        self.create_fom_mask_func()

    def load(self, filename):
        ''' 
        Function to load the necessary parameters from a model file.
        '''
        try:
            loadfile = zipfile.ZipFile(filename, 'r')
        except Exception as e:
            raise GenxIOError(f'Could not open file, python error {e!r}', filename)
        try:
            new_data = pickle.loads(loadfile.read('data'), encoding='latin1', errors='ignore')
            self.data.safe_copy(new_data)
        except Exception as e:
            iprint('Data section loading (gx file) error:\n ', e, '\n')
            raise GenxIOError('Could not locate the data section.', filename)
        try:
            self.script = pickle.loads(loadfile.read('script'), encoding='latin1', errors='ignore')
        except Exception as e:
            iprint('Script section loading (gx file) error:\n ', e, '\n')
            raise GenxIOError('Could not locate the script.', filename)

        try:
            new_parameters = pickle.loads(loadfile.read('parameters'), encoding='latin1', errors='ignore')
            self.parameters.safe_copy(new_parameters)
        except Exception as e:
            iprint('Script section loading (gx file) error:\n ', e, '\n')
            raise GenxIOError('Could not locate the parameters section.', filename)
        try:
            self.fom_func = pickle.loads(loadfile.read('fomfunction'), encoding='latin1', errors='ignore')
        except Exception:
            raise GenxIOError('Could not locate the fomfunction section.', filename)

        loadfile.close()

        self.filename = os.path.abspath(filename)
        self.compiled = False
        self.saved = True
        self.script_module = GenxScriptModule(self.data)
        self.compiled = False

    def save(self, filename):
        '''
        Function to save the model to file filename
        '''
        try:
            savefile = zipfile.ZipFile(filename, 'w')
        except Exception as e:
            raise GenxIOError(str(e), filename)

        # Save the data structures to file
        try:
            savefile.writestr('data', pickle.dumps(self.data))
        except Exception as e:
            raise GenxIOError('Error writing data: '+str(e), filename)
        try:
            savefile.writestr('script', pickle.dumps(self.script))
        except Exception as e:
            raise GenxIOError('Error writing script: '+str(e), filename)
        self.parameters.model = None
        try:
            savefile.writestr('parameters', pickle.dumps(self.parameters))
        except Exception as e:
            raise GenxIOError('Error writing parameters: '+str(e), filename)
        try:
            savefile.writestr('fomfunction', pickle.dumps(self.fom_func))
        except Exception as e:
            raise GenxIOError('Error writing fom_func:  '+str(e), filename)

        savefile.close()

        self.filename = os.path.abspath(filename)
        self.saved = True

    def read_h5group(self, group):
        """
        Read the parameters from a hdf5 group
        """
        H5HintedExport.read_h5group(self, group)
        self.create_fom_mask_func()
        # TODO: Get rid of the interdependence model-optimizer here
        try:
            self.limit_fit_range, self.fit_xmin, self.fit_xmax = (
                bool(group['optimizer']['limit_fit_range'][()]),
                float(group['optimizer']['fit_xmin'][()]), float(group['optimizer']['fit_xmax'][()]))
        except Exception:
            iprint("Could not load limite_fit_range from file")
        self.saved = True
        self.script_module = GenxScriptModule(self.data)
        self.compiled = False

    def save_addition(self, name, text):
        '''
        Save additional text sub-file with name name/text to the current file.
        '''
        if self.filename=='':
            raise GenxIOError('File must be saved before new information is added', '')
        try:
            savefile = zipfile.ZipFile(self.filename, 'a')
        except Exception as e:
            raise GenxIOError(str(e), self.filename)

        # Check so the model data is not overwritten
        if name=='data' or name=='script' or name=='parameters':
            raise GenxIOError('It not alllowed to save a subfile with name: %s'%name)

        try:
            savefile.writestr(name, text)
        except Exception as e:
            raise GenxIOError(str(e), self.filename)
        savefile.close()

    def load_addition(self, name)->str:
        '''
        Load additional text from sub-file
        '''
        if self.filename=='':
            raise GenxIOError('File must be loaded before additional information is read', '')
        try:
            loadfile = zipfile.ZipFile(self.filename, 'r')
        except Exception:
            raise GenxIOError('Could not open the file', self.filename)

        try:
            text = loadfile.read(name).decode('utf-8')
        except Exception:
            raise GenxIOError('Could not read the section named: %s'%name, self.filename)
        loadfile.close()
        return text

    def reset(self):
        self._reset_module()

    def _reset_module(self):
        ''' 
        Internal method for resetting the module before compilation
        '''
        self.create_fom_mask_func()
        self.script_module = GenxScriptModule(self.data)
        self.compiled = False

    def compile_script(self):
        ''' 
        compile the script in a seperate module.
        '''
        self._reset_module()
        # Testing to see if this works under windows
        self.script = '\n'.join(self.script.splitlines())
        try:
            exec(self.script, self.script_module.__dict__)
        except Exception:
            outp = io.StringIO()
            traceback.print_exc(200, outp)
            val = outp.getvalue()
            outp.close()
            raise ModelError(str(val), 0)
        else:
            self.compiled = True

    def eval_in_model(self, codestring):
        '''
        Excecute the code in codestring in the namespace of model module
        '''
        result = eval(codestring, self.script_module.__dict__)
        return result

    def set_fom_ignore_inf(self, flag):
        """
        Sets if the fom calculation should ignore infs
        """
        self.solver_parameters.ignore_fom_inf = bool(flag)
        self.create_fom_mask_func()

    def set_fom_ignore_nan(self, flag):
        """
        Sets if fom calculations should ignore nan's
        """
        self.solver_parameters.ignore_fom_nan = bool(flag)
        self.create_fom_mask_func()

    def create_fom_mask_func(self):
        """
        Create a mask func for fom to take care of unallowed values.
        """
        if self.solver_parameters.ignore_fom_nan and self.solver_parameters.ignore_fom_inf:
            fom_mask = lambda a: np.where(np.isfinite(a), a, np.zeros_like(a))
        elif self.solver_parameters.ignore_fom_nan:
            fom_mask = lambda a: np.where(np.isnan(a), a, np.zeros_like(a))
        elif self.solver_parameters.ignore_fom_inf:
            fom_mask = lambda a: np.where(np.isinf(a), a, np.zeros_like(a))
        else:
            fom_mask = lambda a: a

        self.fom_mask_func = fom_mask

    def calc_fom(self, simulated_data):
        '''
        Sums up the evaluation of the fom values calculated for each
        data point to form the overall fom function for all data sets.
        '''
        fom_raw = self.fom_func(simulated_data, self.data)
        # limit the x-range of fitting
        if self.limit_fit_range:
            for i, di in enumerate(self.data):
                fltr = (di.x<self.fit_xmin) | (di.x>self.fit_xmax)
                fom_raw[i][fltr] = 0.
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
            fom = fom/((N-p)*1.0)

        penalty_funcs = self.get_par_penalty()
        if len(penalty_funcs)>0 and fom is not np.NAN:
            fom += sum([pf() for pf in penalty_funcs])
        return fom_raw, fom_indiv, fom

    def evaluate_fit_func(self):
        '''
        Evalute the Simulation fucntion and returns the fom. Use this one
        for fitting. Use evaluate_sim_func(self) for updating of plots
        and such.
        '''
        self.script_module._sim = False
        simulated_data = self.script_module.Sim(self.data)
        fom_raw, fom_inidv, fom = self.calc_fom(simulated_data)
        return fom

    def evaluate_sim_func(self):
        '''
        Evalute the Simulation function and updates the data simulated data
        as well as the fom of the model. Use this one for calculating data to
        update plots, simulations and such.
        '''
        self.script_module._sim = True
        try:
            simulated_data = self.script_module.Sim(self.data)
        except Exception:
            outp = io.StringIO()
            traceback.print_exc(200, outp)
            val = outp.getvalue()
            outp.close()
            raise ModelError(str(val), 1)

        # check so that the Sim function returns anything
        if not simulated_data:
            text = 'The Sim function does not return anything, it should'+ \
                   ' return a list of the same length as the number of data sets.'
            raise ModelError(text, 1)
        # Check so the number of data sets is correct
        if len(simulated_data)!=len(self.data):
            text = 'The number of simulated data sets returned by the Sim function' \
                   +' has to be same as the number of loaded data sets.\n'+ \
                   'Number of loaded data sets: '+str(len(self.data))+ \
                   '\nNumber of simulated data sets: '+str(len(simulated_data))
            raise ModelError(text, 1)

        self.data.set_simulated_data(simulated_data)

        try:
            fom_raw, fom_inidv, fom = self.calc_fom(simulated_data)
            self.fom = fom
        except Exception:
            outp = io.StringIO()
            traceback.print_exc(200, outp)
            val = outp.getvalue()
            outp.close()
            raise FomError(str(val))
        self.data.set_fom_data(fom_raw)

    def create_fit_func(self, identifier):
        '''
        Creates a function from the string expression in string. 
        If the string is a function in the model this function will be
        returned if string represents anything else a function that sets that 
        object will be returned.
        '''
        obj = self.eval_in_model(identifier)
        # Is it a function or a method!
        if callable(obj):
            return obj
        # Make a function to set the object
        elif isinstance(obj, NumericParameter):
            # We have a NumericParameter that should be set
            exec('def __tempfunc__(val):\n'
                 '    %s.value = val'%identifier, self.script_module.__dict__)
            # noinspection PyUnresolvedReferences
            return self.script_module.__tempfunc__
        else:
            # The function must be created in the module in order to access
            # the different variables
            exec('def __tempfunc__(val):\n\t%s = val'%identifier, self.script_module.__dict__)
            # noinspection PyUnresolvedReferences
            return self.script_module.__tempfunc__

    def get_fit_pars(self):
        '''
        Returns the parameters used with fitting. i.e. the function to 
        set the paraemters, the guess value (values), minimum allowed values
        and the maximum allowed values
        '''
        (row_numbers, sfuncs, vals, minvals, maxvals) = self.parameters.get_fit_pars()
        if len(sfuncs)==0:
            raise ParameterError(sfuncs, 0, None, 4)
        # Check for min and max on all the values
        for i in range(len(vals)):
            # parameter less than min
            if vals[i]<minvals[i]:
                raise ParameterError(sfuncs[i], row_numbers[i], None, 3)
            # parameter larger than max
            if vals[i]>maxvals[i]:
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
            if var.__class__.__name__=='UserVars':
                # noinspection PyProtectedMember
                return var._penalty_funcs
        return []

    def get_fit_values(self):
        '''
        Returns the current parameters values that the user has ticked as
        fittable.
        '''
        (row_numbers, sfuncs, vals, minvals, maxvals) = \
            self.parameters.get_fit_pars()
        return vals

    def get_sim_pars(self):
        '''
        Returns the parameters used with simulations. i.e. the function to 
        set the parameters, the guess value (values). Used for simulation, 
        for fitting see get_fit_pars(self).s
        '''
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
        '''
        Simulates the data sets using the values given in parameters...
        also compiles the script if asked for (default)
        '''
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
        '''
        Reinitilizes the model. Thus, removes all the traces of the
        previous model. 
        '''
        iprint("class Model: new_model")
        self.data = DataList()
        self.script = ''
        self.parameters = Parameters(self)

        self.fom_func = fom_funcs.log
        self._reset_module()

        # Temporary stuff that needs to keep track on
        self.filename = ''
        self.saved = False

        self.extra_analysis = {}

    def pickable_copy(self):
        '''
        Creates a pickable object of the model. Can be used for saving or
        sending to other processes, i.e., parallel processing.
        '''
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
        model_copy.startup_script=self.startup_script.copy()
        model_copy.opt = self.opt.copy()
        model_copy.saved = self.saved
        # Needs to reset the fom_mask_func since this fails under windows.
        model_copy.fom_mask_func = None
        #
        model_copy.limit_fit_range = self.limit_fit_range
        model_copy.fit_xmin = self.fit_xmin
        model_copy.fit_xmax = self.fit_xmax

        return model_copy

    def get_table_as_ascii(self):
        '''
        Just a copy of the parameters class method get_ascii_output()
        '''
        return self.parameters.get_ascii_output()

    def get_data_as_asciitable(self, indices=None):
        '''
        Just a copy of the method defined in data with the same name.
        '''
        return self.data.get_data_as_asciitable(indices)

    def export_table(self, filename):
        '''
        Export the table to filename. ASCII output.
        '''
        self._save_to_file(filename, self.parameters.get_ascii_output())

    def export_orso(self, basename):
        '''
        Export the data to files with basename filename. ORT output.
        The fileending will be .ort
        '''
        self.simulate(True)
        from genx.lib.orso_io import ort, data as odata
        from genx.version import __version__ as version
        para_list = [dict([(nj, tpj(pij)) for nj, pij, tpj in zip(self.parameters.data_labels, pi,
                                                                  [str, float, bool, float, float, str])])
                     for pi in self.parameters.data if pi[0].strip()!='']
        add_header = {
            'analysis': {
                'software': {'name': 'GenX', 'version': version},
                'script': self.script,
                'parameters': para_list,
                }
            }
        # add possible additional analysis data to the header
        add_header['analysis'].update(self.extra_analysis)
        ds = []
        for di in self.data:
            header = dict(di.meta)
            try:
                import getpass
                # noinspection PyTypeChecker
                header['creator']['name'] = getpass.getuser()
            except Exception:
                pass
            import datetime
            # noinspection PyTypeChecker
            header['creator']['time'] = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            header.update(add_header)
            header['data_set'] = di.name
            columns = [di.x, di.y, di.error]
            column_names = ['Qz', 'R', 'sR']
            if 'res' in di.extra_data:
                columns.append(di.extra_data['res'])
                column_names.append('sQz')
            columns.append(di.y_sim)
            column_names.append('Rsim')
            columns.append(di.y_fom)
            column_names.append('FOM')
            for name, col in sorted(di.extra_data.items()):
                if name=='res':
                    continue
                columns.append(col)
                column_names.append(name)
            header['columns'] = [{'name': cn} for cn in column_names]
            ds.append(odata.ORSOData(header, columns))
        try:
            ort.write_file(basename, ds)
        except GenxIOError as e:
            raise GenxIOError(e.error_message, e.file)

    def export_data(self, basename):
        '''
        Export the data to files with basename filename. ASCII output. 
        The fileending will be .dat
        First column is the x-values. 
        Second column is the data y-vales.
        Third column the error on the data y-values.
        Fourth column the calculated y-values.
        '''
        try:
            self.data.export_data_to_files(basename)
        except GenxIOError as e:
            raise GenxIOError(e.error_message, e.file)

    def export_script(self, filename):
        '''
        Export the script to filename. Will be a python script with ASCII 
        output (naturally).
        '''
        self._save_to_file(filename, self.script)

    def import_script(self, filename):
        '''
        Imports the script from file filename
        '''
        read_string = self._read_from_file(filename)
        self.set_script(read_string)
        self.compiled = False

    def import_table(self, filename):
        '''
        import the table from filename. ASCII input. tab delimited
        '''
        read_string = self._read_from_file(filename)
        self.parameters.set_ascii_input(read_string)

    def _save_to_file(self, filename, save_string):
        '''
        Save the string to file with filename.
        '''
        try:
            savefile = open(filename, 'w')
        except Exception as e:
            raise GenxIOError(e.__str__(), filename)

        # Save the string to file
        try:
            savefile.write(save_string)
        except Exception as e:
            raise GenxIOError(e.__str__(), filename)

        savefile.close()

    def _read_from_file(self, filename):
        '''
        Reads the entrie file into string and returns it.
        '''
        try:
            loadfile = open(filename, 'r')
        except Exception as e:
            raise GenxIOError(e.__str__(), filename)

        # Read the text from file
        try:
            read_string = loadfile.read()
        except Exception as e:
            raise GenxIOError(e.__str__(), filename)

        loadfile.close()

        return read_string

    def get_parameters(self)->Parameters:
        return self.parameters

    def get_data(self)->DataList:
        return self.data

    def get_script(self)->str:
        return self.script

    def get_filename(self)->str:
        return self.filename

    def get_possible_parameters(self)->dict:
        """
        Returns all the parameters that can be fitted. Is used by the parameter grid.
        """
        par_dict = get_parameters(self.script_module, numeric_types_only=True)
        if len(par_dict)==0:
            par_dict = self.get_possible_set_functions()
        return par_dict

    def get_possible_set_functions(self)->dict:
        """
        Returns all the parameters that can be fitted given by the old style of defining parameters GenX2.4.X
        """
        # Start by updating the config file
        self.read_config()
        # First we should see if any of the 
        # classes is defined in model.__pars__
        # or in __pars__
        pars = []
        try:
            # Check if the have a pars in module named model
            pars_tmp = self.eval_in_model('model.__pars__')
            pars_tmp = ['model.%s'%p for p in pars_tmp]
            pars += pars_tmp
        except:
            pass

        # Check if we have a __pars__ in the main script
        try:
            pars_tmp = self.eval_in_model('__pars__')
            pars_tmp = ['%s'%p for p in pars_tmp]
            pars += pars_tmp
        except:
            pass

        isstrings = sum([type(p)==type('') for p in pars])==len(pars)

        if not isstrings:
            pars = []

        # First find the classes that exists..
        # and defined in self.registred_classes
        classes = []
        for c in self.opt.registred_classes+pars:
            try:
                ctemp = self.eval_in_model(c)
            except:
                pass
            else:
                if inspect.isclass(ctemp):
                    classes.append(ctemp)
        # Check if there are any classes defined before we proceed
        if len(classes)>0:
            # Get all the objects in the compiled module
            names = list(self.script_module.__dict__.keys())
            # Create a tuple of the classes we defined above
            tuple_of_classes = tuple(classes)
            # Creating a dictionary that holds the name of the classes
            # each item for a classes is a new dictionary that holds the
            # object name and then a list of the methods.
            par_dict = {}
            [par_dict.__setitem__(clas.__name__, {}) for clas in classes]
            # find all the names of the objects that belongs to 
            # one of the classes
            objs = [(name, self.eval_in_model(name)) for name in names]
            valid_objs = [(name, obj) for name, obj in objs
                          if isinstance(obj, tuple_of_classes)]
            # nested for loop for finding for each valid object
            # the right name as given by self.model_parameters.set_func
            # Add this to the right item in par_dict given
            # its class and name.
            [par_dict[obj.__class__.__name__].__setitem__(name,
                                                        [member for member in dir(obj)
                                                        if member.startswith(self.opt.set_func)])
                                                        for name, obj in valid_objs]
            return par_dict
        return {}

    def set_script(self, text: str):
        self.script = text
        self.compiled = False

    def set_fom_func(self, fom_func: callable):
        self.fom_func = fom_func

    def is_compiled(self)->bool:
        return self.compiled

    def bumps_problem(self):
        """
        Return the fit function as a bumps problem to be used with the bumps.fitters.fit function.
        """
        if not self.compiled:
            self.compile_script()
        indep_module = self.script_module
        data = self.data
        # compile again to have a completely independent script module
        self.compile_script()

        from bumps.fitproblem import FitProblem
        from bumps.curve import Curve
        from numpy import hstack

        namespace = {
            'script_module': indep_module, 'data': data,
            'model_data': None, 'bmodel': None,
            'Curve': Curve, 'hstack': hstack
            }

        # build function for full model parameters out of parameter grid
        mstring = 'def model_data(ignore'
        fstring = ''
        pstart = []
        for p in self.parameters:
            if p.name.strip()=='':
                continue
            name = p.name.replace('.set', '_')
            fstring += 'script_module.%s(%s)\n'%(p.name, name)
            mstring += ', '+name
            pstart.append((name, p.value, p.min, p.max, p.fit))
        mstring += '):'
        fstring += 'res=script_module.Sim(data)\nreturn hstack(res)'
        namespace['pstart'] = pstart
        exec(mstring+'\n    '+fstring.replace('\n', '\n    '), namespace)
        exec('''bmodel=Curve(model_data, hstack([di.x for di in data]), 
                       hstack([di.y for di in data]), 
                       hstack([di.error for di in data]), 
                       '''+
             ', \n                   '.join(['%s=%s'%(pi[0], pi[1]) for pi in pstart])+')',
             namespace)
        for pname, pval, pmin, pmax, pfit in pstart:
            exec('bmodel.%s.range(%s, %s)'%(pname, pmin, pmax), namespace)
            if not pfit:
                exec('bmodel.%s.fixed=True'%pname, namespace)
        bproblem = FitProblem(namespace['bmodel'])
        return bproblem

    def bumps_fit(self, method='dream',
                  pop=15, samples=1e5, burn=100, steps=0,
                  thin=1, alpha=0, outliers='none', trim=False,
                  monitors=None, problem=None, **options):
        # create a fitter similar to the bumps.fitters.fit function but with option for GUI monitoring
        if monitors is None:
            monitors = []
        from scipy.optimize import OptimizeResult
        from bumps.fitters import FitDriver, FIT_AVAILABLE_IDS, FITTERS, FIT_ACTIVE_IDS
        options['pop'] = pop
        options['samples'] = samples
        options['burn'] = burn
        options['steps'] = steps
        options['thin'] = thin
        options['alpha'] = alpha
        options['outliers'] = outliers
        options['trim'] = trim

        if problem is None:
            problem = self.bumps_problem()

        # verbose = True
        if method not in FIT_AVAILABLE_IDS:
            raise ValueError("unknown method %r not one of %s"
                             %(method, ", ".join(sorted(FIT_ACTIVE_IDS))))
        for fitclass in FITTERS:
            if fitclass.id==method:
                break
        # noinspection PyUnboundLocalVariable
        driver = FitDriver(fitclass=fitclass, problem=problem, monitors=monitors, **options)
        driver.clip()  # make sure fit starts within domain
        x0 = problem.getp()
        x, fx = driver.fit()
        problem.setp(x)
        dx = driver.stderr()
        result = OptimizeResult(x=x, dx=driver.stderr(), fun=fx, cov=driver.cov(),
                                success=True, status=0, message="successful termination")
        if hasattr(driver.fitter, 'state'):
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
        mstring = 'def model_data(p):\n    '
        fstring = ''
        pstart = []
        i = 0
        for p in self.parameters:
            if not p.fit:
                continue
            name = p.name.replace('.set', '_')
            fstring += 'script_module.%s(p[%i])\n'%(p.name, i)
            pstart.append((name, p.value, p.min, p.max))
            i += 1
        fstring += 'res=script_module.Sim(data)\nreturn hstack(res)'
        namespace = {
            'script_module': indep_module, 'data': data,
            'model_data': None, 'hstack': hstack
            }
        exec(mstring+fstring.replace('\n', '\n    '), namespace)

        n = len(pstart)
        if sigmas is None:
            # if no ranges are supplied, use fit range from model
            bounds = (tuple(p[2] for p in pstart), tuple(p[3] for p in pstart))
        else:
            bounds = (tuple(p[1]+3*s[0] for p, s in zip(pstart, sigmas)),
                      tuple(p[1]+3*s[1] for p, s in zip(pstart, sigmas)))
        bsim = Simulation(f=namespace['model_data'],
                          data=hstack([di.y for di in data]),
                          sigma=hstack([di.error for di in data]),
                          bounds=bounds,
                          labels=[p[0] for p in pstart])
        bdream = Dream(model=bsim, draws=20000,
                       population=random.randn(x_n_gen*n, x_n_chain*n, n))  # n_gen, n_chain, n_var
        return bdream

    def bumps_update_parameters(self, res):
        """
        Update the GenX model from a bumps fit result.
        """
        x = res.x
        bproblem = self.bumps_problem()
        names = list(bproblem.opt().keys())
        if len(names)!=len(x):
            raise ValueError('The number of parameters does not fit the model parameters, was the model changed?')

        for pi in self.parameters:
            if pi.fit:
                pi.value = x[names.index(pi.name.replace('.set', '_'))]

    def plot(self, data_labels=None, sim_labels=None):
        """
        Plot all datasets in the model wtih matplotlib similar to the display in the GUI.
        data_labels and sim_labels are lists of strings for each curve,
        if none is supplied the curve label is created from the dataset names and index.
        """
        self.simulate()
        return self.data.plot(data_labels=data_labels, sim_labels=sim_labels)

    def __repr__(self):
        """
        Display information about the model.
        """
        output = "Genx Model"
        if self.compiled:
            output += ' - compiled'
        else:
            output += ' - not compiled yet'
        output += "\n"
        output += "File: %s\n"%self.filename
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
            output += ' - compiled'
        else:
            output += ' - not compiled yet'
        output += "</h3>\n"
        output += "<p>File: %s</p>\n"%self.filename

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
        plt.xlabel('q/tth')
        plt.ylabel('Intensity')
        fig.canvas.print_png(sio)
        plt.close()
        img_data = binascii.b2a_base64(sio.getvalue()).decode('ascii')
        output += '<div style="margin-left: 40%;"><img src="data:image/png;base64,{}&#10;"></div></div>'.format(
            img_data)

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
            plt.xlabel('q/tth')
            plt.ylabel('Intensity')
            from IPython.display import display
            display(fig)
            plt.close()

        dataw = self.data._repr_ipyw_()
        top = ipw.HBox([dataw, graphw])
        parameters = self.parameters._repr_ipyw_()

        replot = ipw.Button(description='simulate')
        replot.on_click(self._ipyw_replot)
        replot._plot_output = graphw

        script_area = ipw.Textarea(self.script, layout=ipw.Layout(width='100%'))
        script_area.observe(self._ipyw_script, names='value')
        tabs = ipw.Tab(children=[parameters, script_area])
        tabs.set_title(0, 'Parameters')
        tabs.set_title(1, 'Script')

        return ipw.VBox([replot, top, tabs])

    def _ipyw_replot(self, button):
        try:
            self.simulate()
        except Exception as error:
            print(error)
        with button._plot_output:
            from IPython.display import display, clear_output
            from matplotlib import pyplot as plt
            fig = plt.figure(figsize=(10, 8))
            self.data.plot()
            plt.xlabel('q/tth')
            plt.ylabel('Intensity')
            plt.draw()
            clear_output()
            display(fig)
            plt.close()

    def _ipyw_script(self, change):
        self.script = change.new
