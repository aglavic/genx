'''
Library that contains the the class Model. This
is a general class that store and binds togheter all the other
classes that is part of model parameters, such as data and parameters.
Programmer: Matts Bjorck
Last changed: 2016 09 20
'''

# Standard libraries
import shelve, os, types, zipfile
import pickle as pickle
import io, traceback
import inspect
import numpy as np
import h5py

# GenX libraries
from . import data
from . import parameters
from . import fom_funcs
from .gui_logging import iprint
from logging import debug

from .models.lib.parameters import NumericParameter, get_parameters

# assure compatibility with scripts that don't include genx in import
import sys

sys.modules['models']=sys.modules['genx.models']

# ==============================================================================
# BEGIN: Class Model

class Model:
    ''' A class that holds the model i.e. the script that defines
        the model and the data + various other attributes.
    '''

    def __init__(self, config=None):
        '''
        Create a instance and init all the varaibles.
        '''
        self.config=config
        self.data=data.DataList()
        self.script=''
        if config is not None:
            try:
                self.script="\n".join(eval(config.get('startup', 'script', fallback='[""]')))
            except:
                debug('Issue when loading script from config:', exc_info=True)
        self.parameters=parameters.Parameters(model=self)

        # self.fom_func = default_fom_func
        self.fom_func=fom_funcs.log  # The function that evaluates the fom
        self.fom=None  # The value of the fom function

        self.fom_mask_func=None
        self.fom_ignore_nan=False
        self.fom_ignore_inf=False
        self.create_fom_mask_func()

        # Registred classes that is looked for in the model
        self.registred_classes=[]
        # self.registred_classes = ['Layer','Stack','Sample','Instrument',\
        #                            'model.Layer', 'model.Stack',\
        #                             'model.Sample','model.Instrument',\
        #                             'UserVars','Surface','Bulk']
        self.set_func='set'  # 'set'
        self._reset_module()

        # Temporary stuff that needs to keep track on
        self.filename=''
        self.saved=True
        self.compiled=False

        self.limit_fit_range=False
        self.fit_xmin=0.01
        self.fit_xmax=0.1

        self.extra_analysis={}

    def read_config(self):
        '''Read in the config file
        '''
        # Ceck so that config is loaded
        if not self.config:
            return
        try:
            val=self.config.get('parameters', 'registred classes')
        except:
            iprint('Could not find config for parameters, registered classes')
        else:
            self.registred_classes=[s.strip() for s in val.split(';')]
        try:
            val=self.config.get('parameters', 'set func')
        except:
            iprint('Could not find config for parameters, set func')
        else:
            self.set_func=val

        try:
            val=self.config.get_boolean('solver', 'ignore fom nan')
        except:
            iprint('Could not find config for solver, ignore fom nan')
        else:
            self.fom_ignore_nan=val

        try:
            val=self.config.get_boolean('solver', 'ignore fom inf')
        except:
            iprint('Could not find config for solver, ignore fom inf')
        else:
            self.fom_ignore_inf=val

        self.create_fom_mask_func()

    def load(self, filename):
        ''' 
        Function to load the necessary parameters from a model file.
        '''
        try:
            loadfile=zipfile.ZipFile(filename, 'r')
        except Exception as e:
            raise IOError('Could not open file.', filename)
        try:
            new_data=pickle.loads(loadfile.read('data'), encoding='latin1', errors='ignore')
            self.data.safe_copy(new_data)
        except Exception as e:
            iprint('Data section loading (gx file) error:\n ', e, '\n')
            raise IOError('Could not locate the data section.', filename)
        try:
            self.script=pickle.loads(loadfile.read('script'), encoding='latin1', errors='ignore')
        except Exception as e:
            iprint('Script section loading (gx file) error:\n ', e, '\n')
            raise IOError('Could not locate the script.', filename)

        try:
            new_parameters=pickle.loads(loadfile.read('parameters'), encoding='latin1', errors='ignore')
            self.parameters.safe_copy(new_parameters)
        except Exception as e:
            iprint('Script section loading (gx file) error:\n ', e, '\n')
            raise IOError('Could not locate the parameters section.', filename)
        try:
            self.fom_func=pickle.loads(loadfile.read('fomfunction'), encoding='latin1', errors='ignore')
        except Exception:
            raise IOError('Could not locate the fomfunction section.', filename)

        loadfile.close()

        self.filename=os.path.abspath(filename)
        self.compiled=False
        self.saved=True
        self.script_module=types.ModuleType('genx_script_module')
        self.script_module.__dict__['data']=self.data
        self.script_module.__dict__['_sim']=False
        self.compiled=False

    def save(self, filename):
        '''
        Function to save the model to file filename
        '''
        try:
            savefile=zipfile.ZipFile(filename, 'w')
        except Exception as e:
            raise IOError(str(e), filename)

        # Save the data structures to file
        try:
            savefile.writestr('data', pickle.dumps(self.data))
        except Exception as e:
            raise IOError('Error writing data: '+str(e), filename)
        try:
            savefile.writestr('script', pickle.dumps(self.script))
        except Exception as e:
            raise IOError('Error writing script: '+str(e), filename)
        self.parameters.model=None
        try:
            savefile.writestr('parameters', pickle.dumps(self.parameters))
        except Exception as e:
            raise IOError('Error writing parameters: '+str(e), filename)
        try:
            savefile.writestr('fomfunction', pickle.dumps(self.fom_func))
        except Exception as e:
            raise IOError('Error writing fom_func:  '+str(e), filename)

        savefile.close()

        self.filename=os.path.abspath(filename)
        self.saved=True

    def write_h5group(self, group, **kwargs):
        """ Write the current parameters into a hdf5 group

        :param group: h5py Group to write into
        :param kwargs: Additional object parameters to be written (stored_name=object), these objects has to incorporate
            a write_h5group.
        :return:
        """
        self.data.write_h5group(group.create_group('data'))
        group['script']=self.script
        self.parameters.write_h5group(group.create_group('parameters'))
        group['fomfunction']=self.fom_func.__name__.encode('utf-8')
        sgrp=group.create_group('solver_pars')
        sgrp['fom_ignore_nan']=self.fom_ignore_nan
        sgrp['fom_ignore_inf']=self.fom_ignore_inf

        for kw in kwargs:
            kwargs[kw].write_h5group(group.create_group(kw))

    def read_h5group(self, group, **kwargs):
        """ Read the parameters from a hdf5 group

        :param group: hdf5 Group to read the parameters from
        :param kwargs: Additional object parameters to be read (stored_name=object), these objects has to incorporate
            a read_h5group.
        :return:
        """
        self.data.read_h5group(group['data'])
        if int(h5py.__version__[0])>2:
            self.script=group['script'][()].decode('utf-8')
        else:
            self.script=group['script'][()]
        self.parameters.read_h5group(group['parameters'])
        fom_func_name=group['fomfunction'][()].decode('utf-8')
        if fom_func_name in fom_funcs.func_names:
            self.set_fom_func(eval('fom_funcs.'+fom_func_name))
        else:
            iprint("Can not find fom function name %s"%fom_func_name[()])

        try:
            self.fom_ignore_nan=bool(group['solver_pars']['fom_ignore_nan'][()])
        except Exception as e:
            iprint("Could not load parameter fom_ignore_nan from file")
        try:
            self.fom_ignore_inf=bool(group['solver_pars']['fom_ignore_inf'][()])
        except Exception as e:
            iprint("Could not load parameter fom_ignore_inf from file")
        self.create_fom_mask_func()

        try:
            self.limit_fit_range, self.fit_xmin, self.fit_xmax=(
                bool(group['optimizer']['limit_fit_range'][()]),
                float(group['optimizer']['fit_xmin'][()]), float(group['optimizer']['fit_xmax'][()]))
        except Exception as e:
            iprint("Could not load limite_fit_range from file")

        for kw in kwargs:
            kwargs[kw].read_h5group(group[kw])

        self.compiled=False
        self.saved=True
        self.script_module=types.ModuleType('genx_script_module')
        self.script_module.__dict__['data']=self.data
        self.script_module.__dict__['_sim']=False
        self.compiled=False

    def save_addition(self, name, text):
        '''save_addition(self, name, text) --> None
        
        save additional text [string] subfile with name name [string]\
         to the current file.
        '''
        # Check so the filename is ok i.e. has been saved
        if self.filename=='':
            raise IOError('File must be saved before new information is added', '')
        try:
            savefile=zipfile.ZipFile(self.filename, 'a')
        except Exception as e:
            raise IOError(str(e), self.filename)

        # Check so the model data is not overwritten
        if name=='data' or name=='script' or name=='parameters':
            raise IOError('It not alllowed to save a subfile with name: %s'%name)

        try:
            savefile.writestr(name, text)
        except Exception as e:
            raise IOError(str(e), self.filename)
        savefile.close()

    def load_addition(self, name):
        '''load_addition(self, name) --> text
        
        load additional text [string] subfile with name name [string]\
         to the current model file.
        '''
        # Check so the filename is ok i.e. has been saved
        if self.filename=='':
            raise IOError('File must be loaded before additional information is read', '')
        try:
            loadfile=zipfile.ZipFile(self.filename, 'r')
        except Exception as e:
            raise IOError('Could not open the file', self.filename)

        try:
            text=loadfile.read(name)
        except Exception as e:
            raise IOError('Could not read the section named: %s'%name, self.filename)
        loadfile.close()
        return text

    def _reset_module(self):
        ''' 
        Internal method for resetting the module before compilation
        '''
        self.create_fom_mask_func()
        self.script_module=types.ModuleType('genx_script_module')
        # self.script_module = Temp()
        # self.script_module.__dict__ = {}
        # Bind data for preprocessing with the script
        self.script_module.__dict__['data']=self.data
        # Flag to indicate to the Sim funtion if a simulation is conducted (True)
        # or a fit is running (False). 
        self.script_module.__dict__['_sim']=False
        self.compiled=False

    def compile_script(self):
        ''' 
        compile the script in a seperate module.
        '''

        self._reset_module()
        # Testing to see if this works under windows
        self.script='\n'.join(self.script.splitlines())
        try:
            exec(self.script, self.script_module.__dict__)
        except Exception as e:
            outp=io.StringIO()
            traceback.print_exc(200, outp)
            val=outp.getvalue()
            outp.close()
            raise ModelError(str(val), 0)
        else:
            self.compiled=True

    def eval_in_model(self, codestring):
        '''
        Excecute the code in codestring in the namespace of
        model module
        '''
        # exec codestring in self.script_module.__dict__
        result=eval(codestring, self.script_module.__dict__)
        # print 'Sucessfully evaluted: ', codestring
        return result

    def set_fom_ignore_inf(self, flag):
        """ Sets if the fom calculation should ignore infs

        :param flag: boolean
        :return:
        """
        self.fom_ignore_inf=bool(flag)
        self.create_fom_mask_func()

    def set_fom_ignore_nan(self, flag):
        """ Sets if fom calculations should ignore nan's

        :param flag: boolean flag
        :return:
        """
        self.fom_ignore_nan=bool(flag)
        self.create_fom_mask_func()

    def create_fom_mask_func(self):
        """ Create a mask func for fom to take care of unallowed values.

        :param fom: an array
        :return: an masked array
        """
        if self.fom_ignore_nan and self.fom_ignore_inf:
            fom_mask=lambda a: np.where(np.isfinite(a), a, np.zeros_like(a))
        elif self.fom_ignore_nan:
            fom_mask=lambda a: np.where(np.isnan(a), a, np.zeros_like(a))
        elif self.fom_ignore_inf:
            fom_mask=lambda a: np.where(np.isinf(a), a, np.zeros_like(a))
        else:
            fom_mask=lambda a: a*1.0

        self.fom_mask_func=fom_mask

    def calc_fom(self, simulated_data):
        '''calc_fom(self, fomlist) -> fom_raw (list of arrays), 
                                      fom_indiv(list of floats), 
                                      fom(float)
        
        Sums up the evaluation of the fom values calculated for each
         data point to form the overall fom function for all data sets.
        '''
        fom_raw=self.fom_func(simulated_data, self.data)
        # limit the x-range of fitting
        if self.limit_fit_range:
            for i, di in enumerate(self.data):
                fltr=(di.x<self.fit_xmin) | (di.x>self.fit_xmax)
                fom_raw[i][fltr]=0.
        # Sum up a unique fom for each data set in use
        fom_indiv=[np.sum(np.abs(self.fom_mask_func(fom_set))) for fom_set in fom_raw]
        fom=np.sum([f for f, d in zip(fom_indiv, self.data) if d.use])

        # Lets extract the number of datapoints as well:
        N=np.sum([len(self.fom_mask_func(fom_set)) for fom_set, d in zip(fom_raw, self.data) if d.use])
        # And the number of fit parameters
        p=self.parameters.get_len_fit_pars()
        # self.fom_dof = fom/((N-p)*1.0)
        try:
            use_dif=self.fom_func.__div_dof__
        except Exception:
            use_dif=False
        if use_dif:
            fom=fom/((N-p)*1.0)

        penalty_funcs=self.get_par_penalty()
        if len(penalty_funcs)>0 and fom is not np.NAN:
            fom+=sum([pf() for pf in penalty_funcs])
        return fom_raw, fom_indiv, fom

    def evaluate_fit_func(self):
        ''' evaluate_fit_func(self) --> fom (float)
        
        Evalute the Simulation fucntion and returns the fom. Use this one
        for fitting. Use evaluate_sim_func(self) for updating of plots
        and such.
        '''
        self.script_module._sim=False
        simulated_data=self.script_module.Sim(self.data)
        # fom = self.fom_func(simulated_data, self.data)
        fom_raw, fom_inidv, fom=self.calc_fom(simulated_data)
        return fom

    def evaluate_sim_func(self):
        '''evaluate_sim_func(self) --> None
        
        Evalute the Simulation function and updates the data simulated data
        as well as the fom of the model. Use this one for calculating data to
        update plots, simulations and such.
        '''
        self.script_module._sim=True
        try:
            simulated_data=self.script_module.Sim(self.data)
        except Exception as e:
            outp=io.StringIO()
            traceback.print_exc(200, outp)
            val=outp.getvalue()
            outp.close()
            raise ModelError(str(val), 1)

        # check so that the Sim function returns anything
        if not simulated_data:
            text='The Sim function does not return anything, it should'+ \
                 ' return a list of the same length as the number of data sets.'
            raise ModelError(text, 1)
        # Check so the number of data sets is correct
        if len(simulated_data)!=len(self.data):
            text='The number of simulated data sets returned by the Sim function' \
                 +' has to be same as the number of loaded data sets.\n'+ \
                 'Number of loaded data sets: '+str(len(self.data))+ \
                 '\nNumber of simulated data sets: '+str(len(simulated_data))
            raise ModelError(text, 1)

        self.data.set_simulated_data(simulated_data)

        try:
            # self.fom = self.fom_func(simulated_data, self.data)
            fom_raw, fom_inidv, fom=self.calc_fom(simulated_data)
            self.fom=fom
        except Exception as e:
            outp=io.StringIO()
            traceback.print_exc(200, outp)
            val=outp.getvalue()
            outp.close()
            raise FomError(str(val))
        # print len(fom_raw)
        self.data.set_fom_data(fom_raw)

    def create_fit_func(self, str):
        '''create_fit_func(self, str) --> function
        
        Creates a function from the string expression in string. 
        If the string is a function in the model this function will be
        returned if string represents anything else a function that sets that 
        object will be returned.
        '''
        object=self.eval_in_model(str)
        # Is it a function or a method!
        name=type(object).__name__
        if callable(object):
            return object
        # Make a function to set the object
        elif isinstance(object, NumericParameter):
            # We have a NumericParameter that should be set
            exec('def __tempfunc__(val):\n\t%s.value = val'%str, self.script_module.__dict__)
            return self.script_module.__tempfunc__
        else:
            # The function must be created in the module in order to access
            # the different variables
            exec('def __tempfunc__(val):\n\t%s = val'%str, self.script_module.__dict__)
            return self.script_module.__tempfunc__

    def get_fit_pars(self):
        ''' get_fit_pars(self) --> (funcs, values, min_values, max_values)
        
        Returns the parameters used with fitting. i.e. the function to 
        set the paraemters, the guess value (values), minimum allowed values
        and the maximum allowed values
        '''
        (row_numbers, sfuncs, vals, minvals, maxvals)=self.parameters.get_fit_pars()
        if len(sfuncs)==0:
            raise ParameterError(sfuncs, 0, 'None', 4)
        # Check for min and max on all the values
        for i in range(len(vals)):
            # parameter less than min
            if vals[i]<minvals[i]:
                raise ParameterError(sfuncs[i], row_numbers[i], 'None', 3)
            # parameter larger than max
            if vals[i]>maxvals[i]:
                raise ParameterError(sfuncs[i], row_numbers[i], 'None', 2)

        # Compile the strings to create the functions..
        funcs=[]
        for func in sfuncs:
            try:
                funcs.append(self.create_fit_func(func))
            except Exception as e:
                raise ParameterError(func, row_numbers[len(funcs)], str(e), 0)
        return funcs, vals, minvals, maxvals

    def get_par_penalty(self):
        for var in self.script_module.__dict__.values():
            if var.__class__.__name__=='UserVars':
                return var._penalty_funcs
        return []

    def get_fit_values(self):
        '''get_fit_values(self) --> values
        
        Returns the current parameters values that the user has ticked as
        fittable.
        '''
        (row_numbers, sfuncs, vals, minvals, maxvals)= \
            self.parameters.get_fit_pars()
        return vals

    def get_sim_pars(self):
        ''' get_sim_pars(self) --> (funcs, values)
        
        Returns the parameters used with simulations. i.e. the function to 
        set the parameters, the guess value (values). Used for simulation, 
        for fitting see get_fit_pars(self).s
        '''
        (sfuncs, vals)=self.parameters.get_sim_pars()
        # Compile the strings to create the functions..
        funcs=[]
        for func in sfuncs:
            try:
                funcs.append(self.create_fit_func(func))
            except Exception as e:
                raise ParameterError(func, len(funcs), str(e), 0)

        return funcs, vals

    def simulate(self, compile=True):
        '''simulate(self, compile = True) --> None
        
        Simulates the data sets using the values given in parameters...
        also compiles the script if asked for (default)
        '''
        if compile:
            self.compile_script()
        (funcs, vals)=self.get_sim_pars()
        # print 'Functions to evulate: ', funcs
        # Set the parameter values in the model
        # [func(val) for func,val in zip(funcs, vals)]
        i=0
        for func, val in zip(funcs, vals):
            try:
                func(val)
            except Exception as e:
                (sfuncs_tmp, vals_tmp)=self.parameters.get_sim_pars()
                raise ParameterError(sfuncs_tmp[i], i, str(e), 1)
            i+=1

        self.evaluate_sim_func()

    def new_model(self):
        '''
        new_model(self) --> None
        
        Reinitilizes the model. Thus, removes all the traces of the
        previous model. 
        '''
        iprint("class Model: new_model")
        self.data=data.DataList()
        self.script=''
        self.parameters=parameters.Parameters(self)

        # self.fom_func = default_fom_func
        self.fom_func=fom_funcs.log
        self._reset_module()

        # Temporary stuff that needs to keep track on
        self.filename=''
        self.saved=False

        self.extra_analysis={}

    def pickable_copy(self):
        '''pickable_copy(self) --> model
        
        Creates a pickable object of the model. Can be used for saving or
        sending to other processes, i.e., parallel processing.
        '''
        model_copy=Model(self.config)
        model_copy.data=self.data
        model_copy.script=self.script
        model_copy.parameters=self.parameters.copy()
        model_copy.fom_func=self.fom_func
        # The most important stuff - a module is not pickable
        model_copy.script_module=None
        model_copy.filename=self.filename
        model_copy.compiled=self.compiled
        model_copy.fom=self.fom
        model_copy.fom_ignore_nan=self.fom_ignore_nan
        model_copy.fom_ignore_inf=self.fom_ignore_inf
        model_copy.set_func=self.set_func
        model_copy.saved=self.saved
        # Needs to reset the fom_mask_func since this fails under windows.
        model_copy.fom_mask_func=None
        #
        model_copy.limit_fit_range=self.limit_fit_range
        model_copy.fit_xmin=self.fit_xmin
        model_copy.fit_xmax=self.fit_xmax

        return model_copy

    def get_table_as_ascii(self):
        '''get_table_as_ascii(self) --> None
        
        Just a copy of the parameters class method get_ascii_output()
        '''
        return self.parameters.get_ascii_output()

    def get_data_as_asciitable(self, indices=None):
        '''get_data_as_asciitable(self, indices None) --> string
        
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
        para_list=[dict([(nj, tpj(pij)) for nj, pij, tpj in zip(self.parameters.data_labels, pi,
                                                           [str, float, bool, float, float, str])])
                   for pi in self.parameters.data if pi[0].strip()!='']
        add_header={'analysis': {
            'software': {'name': 'GenX', 'version': version},
            'script': self.script,
            'parameters': para_list,
            }}
        # add possible additional analysis data to the header
        add_header['analysis'].update(self.extra_analysis)
        ds=[]
        for di in self.data:
            header=dict(di.meta)
            try:
                import getpass
                header['creator']['name']=getpass.getuser()
            except Exception:
                pass
            import datetime
            header['creator']['time']=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            header.update(add_header)
            header['data_set']=di.name
            columns=[di.x, di.y, di.error]
            column_names=['Qz', 'R', 'sR']
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
            header['columns']=[{'name': cn} for cn in column_names]
            ds.append(odata.ORSOData(header, columns))
        try:
            ort.write_file(basename, ds)
        except data.IOError as e:
            raise IOError(e.error_message, e.file)

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
        except data.IOError as e:
            raise IOError(e.error_message, e.file)

    def export_script(self, filename):
        '''
        Export the script to filename. Will be a python script with ASCII 
        output (naturally).
        '''
        self._save_to_file(filename, self.script)

    def import_script(self, filename):
        '''import_script(self, filename) --> None
        
        Imports the script from file filename
        '''
        read_string=self._read_from_file(filename)
        self.set_script(read_string)
        self.compiled=False

    def import_table(self, filename):
        '''
        import the table from filename. ASCII input. tab delimited
        '''
        read_string=self._read_from_file(filename)
        self.parameters.set_ascii_input(read_string)

    def _save_to_file(self, filename, save_string):
        '''_save_to_file(self, filename, save_string) --> None
        
        Save the string to file with filename.
        '''
        try:
            savefile=open(filename, 'w')
        except Exception as e:
            raise IOError(e.__str__(), filename)

        # Save the string to file
        try:
            savefile.write(save_string)
        except Exception as e:
            raise IOError(e.__str__(), filename)

        savefile.close()

    def _read_from_file(self, filename):
        '''_read_from_file(self, filename) --> string
        
        Reads the entrie file into string and returns it.
        '''
        try:
            loadfile=open(filename, 'r')
        except Exception as e:
            raise IOError(e.__str__(), filename)

        # Read the text from file
        try:
            read_string=loadfile.read()
        except Exception as e:
            raise IOError(e.__str__(), filename)

        loadfile.close()

        return read_string

    # Get functions

    def get_parameters(self):
        '''
        get_parameters(self) --> parameters
        
        returns the parameters of the model. Instace of Parameters class
        '''
        return self.parameters

    def get_data(self):
        '''
        get_data(self) --> self.data
        
        Returns the DataList object.
        '''

        return self.data

    def get_script(self):
        '''
        get_script(self) --> self.script
        
        Returns the model script (string).
        '''
        return self.script

    def get_filename(self):
        '''
        get_filename(self) --> string
        
        returns the filename of the model id the model has not been saved
        it returns an empty string 
        '''
        return self.filename

    def get_possible_parameters(self):
        """ Returns all the parameters that can be fitted. Is used by the parameter grid.

        Returns:
             par_dict (list):  all the current objects that are of the classes defined by self.registred_classes.

        """
        # loop through all objects and locate NumericParameters or HasParameters
        par_dict=get_parameters(self.script_module, numeric_types_only=True)

        if len(par_dict)==0:
            par_dict=self.get_possible_set_functions()

        return par_dict

    def get_possible_set_functions(self):
        """Returns all the parameters that can be fitted given by the old style of defining parameters GenX2.4.X

        Returns:
             par_dict (list):  all the current objects that are of the classes defined by self.registred_classes.

        """
        # Start by updating the config file
        self.read_config()
        # First we should see if any of the 
        # classes is defnined in model.__pars__
        # or in __pars__
        pars=[]
        try:
            # Check if the have a pars in module named model
            pars_tmp=self.eval_in_model('model.__pars__')
            pars_tmp=['model.%s'%p for p in pars_tmp]
            pars+=pars_tmp
        except:
            pass

        # Check if we have a __pars__ in the main script
        try:
            pars_tmp=self.eval_in_model('__pars__')
            pars_tmp=['%s'%p for p in pars_tmp]
            pars+=pars_tmp
        except:
            pass

        isstrings=sum([type(p)==type('') for p in pars])==len(pars)

        if not isstrings:
            pars=[]

        # First find the classes that exists..
        # and defined in self.registred_classes
        classes=[]
        for c in self.registred_classes+pars:
            try:
                ctemp=self.eval_in_model(c)
            except:
                pass
            else:
                if inspect.isclass(ctemp):
                    classes.append(ctemp)
        # Check so there are any classes defined before we proceed
        if len(classes)>0:
            # Get all the objects in the compiled module
            names=list(self.script_module.__dict__.keys())
            # Create a tuple of the classes we defined above
            tuple_of_classes=tuple(classes)
            # Creating a dictionary that holds the name of the classes
            # eaxh item for a classes is a new dictonary that holds the
            # object name and then a list of the methods.
            par_dict={}
            [par_dict.__setitem__(clas.__name__, {}) for clas in classes]
            # find all the names of the objects that belongs to 
            # one of the classes
            objs=[(name, self.eval_in_model(name)) for name in names]
            valid_objs=[(name, obj) for name, obj in objs
                        if isinstance(obj, tuple_of_classes)]
            # nested for loop for finding for each valid object
            # the right name as given by self.set_func
            # Add this to the right item in par_dict given
            # its class and name.
            [par_dict[obj.__class__.__name__].__setitem__(name,
                                                          [member for member in dir(obj)
                                                           if member[:len(self.set_func)]==self.set_func])
             for name, obj in valid_objs]

            return par_dict

        return {}

    # Set functions
    def set_script(self, text):
        '''
        Set the text in the script use this to change the model script. 
        '''
        self.script=text
        self.compiled=False

    def set_fom_func(self, fom_func):
        '''
        Set the fucntion that calculates the figure of merit between the model
        and the data.
        '''
        self.fom_func=fom_func

    def is_compiled(self):
        '''is_compiled(self) --> compiled [boolean]
        
        Returns true if the model script has been sucessfully 
        compiled.
        '''
        return self.compiled

    def bumps_problem(self):
        """
        Return the fit function as a bumps problem to be used with the bumps.fitters.fit function.
        """
        if not self.compiled:
            self.compile_script()
        indep_module=self.script_module
        data=self.data
        # compile again to have a completely independent script module
        self.compile_script()

        from bumps.fitproblem import FitProblem
        from bumps.curve import Curve
        from numpy import hstack

        namespace={'script_module': indep_module, 'data': data,
                   'model_data': None, 'bmodel': None,
                   'Curve': Curve, 'hstack': hstack}

        # build function for full model parameters out of parameter grid
        mstring='def model_data(ignore'
        fstring=''
        pstart=[]
        for p in self.parameters:
            if p.name.strip()=='':
                continue
            name=p.name.replace('.set', '_')
            fstring+='script_module.%s(%s)\n'%(p.name, name)
            mstring+=', '+name
            pstart.append((name, p.value, p.min, p.max, p.fit))
        mstring+='):'
        fstring+='res=script_module.Sim(data)\nreturn hstack(res)'
        namespace['pstart']=pstart
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
        bproblem=FitProblem(namespace['bmodel'])
        return bproblem

    def bumps_fit(self, method='dream',
                  pop=15, samples=1e5, burn=100, steps=0,
                  thin=1, alpha=0, outliers='none', trim=False,
                  monitors=[], problem=None, **options):
        # create a fitter similar to the bumps.fitters.fit function but with option for GUI monitoring
        from scipy.optimize import OptimizeResult
        from bumps.fitters import FitDriver, FIT_AVAILABLE_IDS, FITTERS, FIT_ACTIVE_IDS
        options['pop']=pop
        options['samples']=samples
        options['burn']=burn
        options['steps']=steps
        options['thin']=thin
        options['alpha']=alpha
        options['outliers']=outliers
        options['trim']=trim

        if problem is None:
            problem=self.bumps_problem()

        # verbose = True
        if method not in FIT_AVAILABLE_IDS:
            raise ValueError("unknown method %r not one of %s"
                             %(method, ", ".join(sorted(FIT_ACTIVE_IDS))))
        for fitclass in FITTERS:
            if fitclass.id==method:
                break
        driver=FitDriver(fitclass=fitclass, problem=problem, monitors=monitors, **options)
        driver.clip()  # make sure fit starts within domain
        x0=problem.getp()
        x, fx=driver.fit()
        problem.setp(x)
        dx=driver.stderr()
        result=OptimizeResult(x=x, dx=driver.stderr(), fun=fx, cov=driver.cov(),
                              success=True, status=0, message="successful termination")
        if hasattr(driver.fitter, 'state'):
            result.state=driver.fitter.state
        return result

    def build_dream(self, x_n_gen=25, x_n_chain=5, sigmas=None):
        """
        Compile an independent script model and function to include in the
        Dream MCMC model and return that model.
        """
        if not self.compiled:
            self.compile_script()
        indep_module=self.script_module

        from bumps.dream import Dream
        from bumps.dream.model import Simulation
        from numpy import hstack, random

        data=self.data
        # compile again to have a completely independent script module
        self.compile_script()

        # generate a function that takes a single parameter list and returns the stacked data
        mstring='def model_data(p):\n    '
        fstring=''
        pstart=[]
        i=0
        for p in self.parameters:
            if not p.fit:
                continue
            name=p.name.replace('.set', '_')
            fstring+='script_module.%s(p[%i])\n'%(p.name, i)
            pstart.append((name, p.value, p.min, p.max))
            i+=1
        fstring+='res=script_module.Sim(data)\nreturn hstack(res)'
        namespace={'script_module': indep_module, 'data': data,
                   'model_data': None, 'hstack': hstack}
        exec(mstring+fstring.replace('\n', '\n    '), namespace)

        n=len(pstart)
        if sigmas is None:
            # if no ranges are supplied, use fit range from model
            bounds=(tuple(p[2] for p in pstart), tuple(p[3] for p in pstart))
        else:
            bounds=(tuple(p[1]+3*s[0] for p, s in zip(pstart, sigmas)),
                    tuple(p[1]+3*s[1] for p, s in zip(pstart, sigmas)))
        bsim=Simulation(f=namespace['model_data'],
                        data=hstack([di.y for di in data]),
                        sigma=hstack([di.error for di in data]),
                        bounds=bounds,
                        labels=[p[0] for p in pstart])
        bdream=Dream(model=bsim, draws=20000,
                     population=random.randn(x_n_gen*n, x_n_chain*n, n))  # n_gen, n_chain, n_var
        return bdream

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
        output="Genx Model"
        if self.compiled:
            output+=' - compiled'
        else:
            output+=' - not compiled yet'
        output+="\n"
        output+="File: %s\n"%self.filename
        output+=self.parameters.__repr__()
        output+=self.data.__repr__()
        return output

    def _repr_html_(self):
        """
        Display information about the model.
        """
        try:
            self.simulate()
        except Exception as error:
            print(error)

        output="<h3>Genx Model"
        if self.compiled:
            output+=' - compiled'
        else:
            output+=' - not compiled yet'
        output+="</h3>\n"
        output+="<p>File: %s</p>\n"%self.filename

        output+='<div style="width: 100%;"><div style="width: 40%; float: left;">'
        output+=self.data._repr_html_()
        output+="</div>"

        # generate a plot of the model
        import binascii
        from io import BytesIO
        from matplotlib import pyplot as plt
        sio=BytesIO()
        fig=plt.figure(figsize=(10, 8))
        self.data.plot()
        plt.xlabel('q/tth')
        plt.ylabel('Intensity')
        fig.canvas.print_png(sio)
        plt.close()
        img_data=binascii.b2a_base64(sio.getvalue()).decode('ascii')
        output+='<div style="margin-left: 40%;"><img src="data:image/png;base64,{}&#10;"></div></div>'.format(img_data)

        output+=self.parameters._repr_html_()
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

        graphw=ipw.Output()
        with graphw:
            from matplotlib import pyplot as plt
            fig=plt.figure(figsize=(10, 8))
            self.data.plot()
            plt.xlabel('q/tth')
            plt.ylabel('Intensity')
            from IPython.display import display
            display(fig)
            plt.close()

        dataw=self.data._repr_ipyw_()
        top=ipw.HBox([dataw, graphw])
        parameters=self.parameters._repr_ipyw_()

        replot=ipw.Button(description='simulate')
        replot.on_click(self._ipyw_replot)
        replot._plot_output=graphw

        script_area=ipw.Textarea(self.script, layout=ipw.Layout(width='100%'))
        script_area.observe(self._ipyw_script, names='value')
        tabs=ipw.Tab(children=[parameters, script_area])
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
            fig=plt.figure(figsize=(10, 8))
            self.data.plot()
            plt.xlabel('q/tth')
            plt.ylabel('Intensity')
            plt.draw()
            clear_output()
            display(fig)
            plt.close()

    def _ipyw_script(self, change):
        self.script=change.new

# END: Class Model
# ==============================================================================
# Some Exception definition for errorpassing
class GenericError(Exception):
    ''' Just a empty class used for inheritance. Only useful
    to check if the errors are originating from the model library.
    All these errors are controllable. If they not originate from
    this class something has passed trough and that should be impossible '''
    pass

class ParameterError(GenericError):
    ''' Class for yielding Parameter errors
    '''

    def __init__(self, parameter, parameter_number, error_message, what=-1):
        '''__init__(self, parameter, parameter_number, error_message) --> None
        
        parameter: the name of the parameter [string]
        parameter_number: the position of the parameter in the list [int]
        error_mesage: pythons error message from the original exception
        set: int to show where the error lies. 
            -1 : undefined
             0 : an not find the parameter
             1 : can not evaluate i.e. set the parameter
             2 : value are larger than max
             3 : value are smaller than min
             4 : No parameters to fit
        '''
        self.parameter=parameter
        self.parameter_number=parameter_number
        self.error_message=error_message
        self.what=what

    def __str__(self):
        ''' __str__(self) --> text [string]
        Yields a human readable description of the problem
        '''
        text=''
        text+='Parameter number %i, %s, '%(self.parameter_number,
                                           self.parameter)

        # Take care of the different cases
        if self.what==0:
            text+='could not be found. Check the spelling.\n'
        elif self.what==1:
            text+='could not be evaluated. Check the code of the function.\n'
        elif self.what==2:
            text+='is larger than the value in the max column.\n'
        elif self.what==3:
            text+='is smaller than the value in the min column\n'
        elif self.what==4:
            text='There are no parameter selcted to be fitted.\n'+ \
                 'Select the parameters you want to fit by checking the '+ \
                 'boxes in the fit column, folder grid'
        else:
            text+='yielded an undefined error. Check the Python output\n'

        if self.error_message!='None':
            text+='\nPython error output:\n'+self.error_message

        return text

class ModelError(GenericError):
    ''' Class for yielding compile or evaluation errors in the model text
    '''

    def __init__(self, error_message, where):
        '''__init__(self, error_message, where = -1) --> None
        
        error_mesage: pythons error message from the original exception
        where: integer describing where the error was raised.
                -1: undef
                 0: compile error
                 1: evaulation error
        '''
        self.error_message=error_message
        self.where=where

    def __str__(self):
        ''' __str__(self) --> text [string]
        Yields a human readable description of the problem
        '''
        text=''
        if self.where==0:
            text+='It was not possible to compile the model script.\n'
        elif self.where==1:
            text+='It was not possible to evaluate the model script.\n' \
                  +'Check the Sim function.\n'
        elif self.where==-1:
            text+='Undefined error from the Model. See below.\n'

        text+='\n'+self.error_message

        return text

class FomError(GenericError):
    '''Error class for the fom evaluation'''

    def __init__(self, error_message):
        ''' __init__(self, error_message) --> None'''
        self.error_message=error_message

    def __str__(self):
        text='Could not evaluate the FOM function. See python output.\n' \
             +'\n'+self.error_message
        return text

class IOError(GenericError):
    ''' Error class for input output, mostly concerning files'''

    def __init__(self, error_message, file=''):
        '''__init__(self, error_message)'''
        self.error_message=error_message
        self.file=file

    def __str__(self):
        text='Input/Output error for file:\n'+self.file+ \
             '\n\n Python error:\n '+self.error_message
        return text

# Some small default function that are needed for initilization

def default_fom_func(simulated_data, data):
    '''
    The default fom function. Its just a dummy so far dont use it!
    '''
    return sum([abs(d.y-sim_d).sum() for sim_d, d \
                in zip(simulated_data, data)])

class Temp:
    pass
