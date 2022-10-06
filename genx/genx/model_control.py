"""
GenX model and optimizer control classes. All functional aspects should be covered here (no GUI).
"""
import os
import sys
from typing import List

import h5py
import numpy as np
from logging import warning

from .core.config import config
from .data import DataList
from .exceptions import ErrorBarError, GenxIOError
from .model import Model
from .model_actions import ActionHistory, DeleteParams, ExchangeModel, InsertParam, ModelAction, MoveParam, NoOp, \
    UpdateModelScript, SortAndGroupParams, UpdateColorCycle, UpdateParams, UpdateParamValue, \
    UpdateSolverOptoins, UpdateDataPlotSettings
from .solver_basis import GenxOptimizer, GenxOptimizerCallback
from .plugins.data_loader_framework import Template as DataLoaderTemplate


class ModelController:
    model: Model
    optimizer: GenxOptimizer
    history: ActionHistory
    model_store: List[Model]

    def __init__(self, optimizer: GenxOptimizer):
        self.model = Model()
        self.optimizer = optimizer
        self.history = ActionHistory()
        self.model.saved = True
        self.model_store = []

    def action_callback(self, action: ModelAction):
        pass

    def set_callbacks(self, callbacks: GenxOptimizerCallback):
        self.optimizer.set_callbacks(callbacks)

    def set_action_callback(self, func):
        # sets a function to be called when aver an action is applied.
        self.action_callback = func

    def perform_action(self, action_class: type(ModelAction), *params):
        action = action_class(self.model, *params)
        self.history.execute(action)
        self.action_callback(action)

    def undo_action(self):
        action = self.history.undo()
        self.action_callback(action)
        return action

    def redo_action(self):
        action = self.history.redo()
        self.action_callback(action)
        return action

    def history_stacks(self):
        return self.history.undo_stack, self.history.redo_stack

    def history_clear(self):
        self.history.clear()
        self.action_callback(NoOp(None))

    def history_remove(self, start, length=1):
        actions = self.history.remove_actions(start, length)
        self.action_callback(actions)

    def is_configured(self):
        return self.optimizer.is_configured()

    def get_result_info(self):
        return self.optimizer.get_result_info()

    def get_fitted_model(self):
        return self.optimizer.get_model()

    def get_fom_log(self):
        return self.optimizer.get_fom_log()

    def ReadConfig(self):
        '''
        Reads the parameter that should be read from the config file.
        And set the parameters in both the optimizer and this class.
        '''
        self.model.ReadConfig()
        self.optimizer.ReadConfig()

    def WriteConfig(self):
        '''
        Writes the current configuration of the solver to file.
        '''
        self.model.WriteConfig()
        self.optimizer.WriteConfig()

    def new_model(self):
        self.model.new_model()
        self.history_clear()

    def set_model(self, model: Model):
        self.perform_action(ExchangeModel, model)

    def get_model(self) -> Model:
        return self.model

    def activate_model(self, store_index):
        '''
        Takes a model from the model_store list and sets it as active model_store.
        '''
        self.set_model(self.model_store[store_index])

    def put_in_store(self, store_index=None):
        '''
        Places a copy of the current model in
        '''
        if store_index is not None:
            self.model_store.insert(store_index, self.model.copy())
        else:
            self.model_store.append(self.model.deepcopy())

    def read_sequence(self, data_loader: DataLoaderTemplate, file_lists,
                      name_by_file=False, callback=None):
        data = self.get_data()
        if len(data)!=len(file_lists):
            raise ValueError("You have to provide a list of datafiles for each dataset in the model")
        for i, fi in enumerate(zip(*file_lists)):
            if callback:
                callback(i)
            for j, dj in enumerate(data):
                data_loader.LoadDataset(dj, fi[j])
            if name_by_file:
                fname = os.path.basename(fi[0])
                self.model.h5group_name = fname
            else:
                self.model.h5group_name = f'sequence_{len(self.model_store):05}'
            self.model.sequence_value = float(len(self.model_store))
            self.put_in_store()
        # activate the first dataset
        self.activate_model(0)

    def active_index(self):
        try:
            return self.model_store.index(self.model)
        except ValueError:
            return -1

    def set_model_script(self, text):
        old_script = self.model.get_script().replace('\r\n', '\n').replace('\r', '\n').strip()
        if text.strip().replace('\r\n', '\n').replace('\r', '\n')==old_script:
            # nothing to do, same script
            return
        self.perform_action(UpdateModelScript, text)

    def get_model_script(self):
        return self.model.get_script()

    def set_model_params(self, params):
        self.model.parameters = params

    def get_model_params(self):
        return self.model.parameters

    def set_data(self, data: DataList):
        self.model.data = data

    def get_data(self) -> DataList:
        return self.model.data

    def set_data_plotsettings(self, indices, sim_par, data_par):
        self.perform_action(UpdateDataPlotSettings, indices, sim_par, data_par)

    def update_color_cycle(self, source):
        if source!=self.model.data.color_source:
            self.perform_action(UpdateColorCycle, source)

    def get_color_cycle(self):
        return self.model.data.color_source

    def get_parameters(self):
        return self.model.parameters

    def get_sim_pars(self):
        return self.model.get_sim_pars()

    def get_parameter_data(self, row):
        return self.model.parameters.get_data()[row]

    def get_parameter_name(self, row):
        return self.model.parameters.get_names()[row]

    def get_possible_parameters(self):
        return self.model.get_possible_parameters()

    def set_error_pars(self, error_values):
        self.model.parameters.set_error_pars(error_values)

    def set_value_pars(self, new_values):
        pars = self.model.parameters.get_value_pars()
        if all([pi==pj for pi, pj in zip(new_values, pars)]):
            return
        self.perform_action(UpdateParams, new_values)

    def set_parameter_value(self, row, col, value):
        if value==self.model.parameters.get_value(row, col):
            return
        self.perform_action(UpdateParamValue, row, col, value)

    def move_parameter(self, row, step):
        self.perform_action(MoveParam, row, step)

    def insert_parameter(self, row):
        self.perform_action(InsertParam, row)

    def delete_parameter(self, rows):
        self.perform_action(DeleteParams, rows)

    def sort_and_group_parameters(self, sort_params):
        self.perform_action(SortAndGroupParams, sort_params)

    def get_fom(self):
        return self.model.fom

    def get_fom_name(self):
        return self.model.fom_func.__name__

    def set_filename(self, filename):
        self.model.filename = filename

    def get_filename(self):
        return self.model.filename

    def get_model_name(self):
        module = self.script_module
        name = module.model.__name__.rsplit('.', 1)[1]
        return name

    def force_compile(self):
        self.model.compile_script()

    def compile_if_needed(self):
        if not self.model.compiled:
            self.model.compile_script()

    def simulate(self, recompile=False):
        self.model.simulate(compile=(recompile or not self.model.compiled))

    def export_data(self, basename):
        self.model.export_data(basename)

    def export_table(self, basename):
        self.model.export_table(basename)

    def export_script(self, basename):
        self.model.export_script(basename)

    def export_orso(self, basename):
        self.model.export_orso(basename)

    def import_table(self, filename):
        self.model.import_table(filename)

    def import_script(self, filename):
        self.model.import_script(filename)

    def get_data_as_asciitable(self, indices=None):
        return self.model.get_data_as_asciitable(indices=indices)

    def get_combined_options(self):
        # Return a configuration object with all parameters relevant for fitting
        return self.model.solver_parameters | self.optimizer.opt

    def update_combined_options(self, new_values: dict):
        combined_options = self.get_combined_options()
        difference = {}
        for key, value in new_values.items():
            if getattr(combined_options, key, None)!=value:
                difference[key] = value
        self.perform_action(UpdateSolverOptoins, self.optimizer, difference)

    @property
    def saved(self):
        return self.model.saved

    @saved.setter
    def saved(self, value):
        self.model.saved = value

    @property
    def eval_in_model(self):
        return self.model.eval_in_model

    @property
    def script_module(self):
        self.compile_if_needed()
        return self.model.script_module

    def CalcErrorBars(self):
        '''
        Method that calculates the errorbars for the fit that has been
        done. Note that the fit has to been conducted before this is run.
        '''
        if self.optimizer.n_fom_evals==0:
            raise ErrorBarError('Can not find any stored evaluations of the model in the optimizer.\n'
                                'Run a fit before calculating the errorbars.')
        if self.optimizer.get_start_guess() is not None and not self.optimizer.is_running():
            n_elements = len(self.optimizer.get_start_guess())
            error_values = []
            for index in range(n_elements):
                # calculate the error, this is threshold based and not rigours
                error_values.append(self.optimizer.calc_error_bar(index))
            return np.array(error_values)
        else:
            raise ErrorBarError('Wait for fit to finish or fit to changed model first.')

    def ProjectEvals(self, parameter: int):
        '''
        Projects the parameter number parameter on one axis and returns
        the fom values.
        '''
        row = self.model.parameters.get_pos_from_row(parameter)
        if self.optimizer.get_start_guess() is not None and not self.optimizer.is_running():
            return self.optimizer.project_evals(row)
        else:
            raise ErrorBarError()

    def StartFit(self):
        '''
        Function to start running the fit
        '''
        if not self.model.compiled:
            self.simulate(recompile=True)
        # Make sure that the config of the solver is updated..
        self.optimizer.ReadConfig()
        # Reset all the errorbars
        self.model.parameters.clear_error_pars()
        self.optimizer.start_fit(self.model)

    def StopFit(self):
        '''
        Function to stop a running fit
        '''
        self.optimizer.stop_fit()

    def ResumeFit(self):
        '''
        Function to resume the fitting after it has been stopped
        '''
        self.compile_if_needed()
        self.optimizer.ReadConfig()
        self.optimizer.resume_fit(self.model)

    def IsFitted(self):
        '''
        Returns true if a fit has been started otherwise False
        '''
        return self.optimizer.is_fitted()

    def save(self):
        self.save_file(self.model.get_filename())

    def save_file(self, fname: str, update_callback=None):
        """
        Saves objects model, optimiser and config into file fname
        """
        if fname.lower().endswith('.gx'):
            self.save_gx(fname)
        elif fname.lower().endswith('.hgx'):
            self.save_hgx(fname, update_callback=update_callback)
        else:
            raise GenxIOError('Wrong file ending, should be .gx or .hgx')
        self.model.filename = os.path.abspath(fname)
        self.model.saved = True

    def load_file(self, fname: str, update_callback=None):
        """
        Loads parameters from fname into model, optimizer and config
        """
        if fname.lower().endswith('.gx'):
            self.load_gx(fname)
        elif fname.lower().endswith('.hgx'):
            self.load_hgx(fname, update_callback=update_callback)
        else:
            raise GenxIOError('Wrong file ending, should be .gx or .hgx')
        self.model.filename = os.path.abspath(fname)
        self.model.saved = True
        self.history_clear()

    def save_hgx(self, fname: str, update_callback=None):
        f = h5py.File(fname.encode('utf-8'), 'w')
        g = f.create_group('current')
        self.model.write_h5group(g)
        opt_group = g.create_group(self.optimizer.h5group_name)
        opt_group['solver'] = self.optimizer.__class__.__name__
        opt_group['solver_module'] = self.optimizer.__class__.__module__
        self.optimizer.write_h5group(opt_group)
        self.WriteConfig()
        g['config'] = config.model_dump().encode('utf-8')
        N = len(self.model_store)+1
        lN = len(f'{N}')
        fmt = '%%0%ii-%%s'%lN  # put an index before the name to keep list order on load
        for i, modeli in enumerate(self.model_store):
            if update_callback:
                update_callback(i+1, N)
            g = f.create_group(fmt%(i, modeli.h5group_name))
            modeli.write_h5group(g)
        f.close()

    def load_hgx(self, fname: str, update_callback=None):
        f = h5py.File(fname.encode('utf-8'), 'r')
        g = f['current']
        self.model.read_h5group(g)
        opt_group = g[self.optimizer.h5group_name]
        try:
            solver_class = opt_group.get('solver')[()]
            solver_module = opt_group.get('solver_module')[()]
        except (KeyError, TypeError):
            solver_class = 'DiffEv'
            solver_module = 'genx.diffev'
        else:
            if type(solver_class) is bytes:
                solver_class = solver_class.decode('utf-8')
                solver_module = solver_module.decode('utf-8')
        if solver_class!=self.optimizer.__class__.__name__:
            try:
                exec(f'from {solver_module} import {solver_class}')
            except ImportError:
                warning(f'Could not import solver {solver_class} from moudle {solver_module}')
            else:
                prev_optimizer = self.optimizer
                exec(f'self.optimizer={solver_class}()')
                self.optimizer.set_callbacks(prev_optimizer.get_callbacks())
        self.optimizer.read_h5group(opt_group)
        try:
            config.load_string(g['config'][()].decode('utf-8'))
            self.ReadConfig()
        except KeyError:
            pass
        except AttributeError:
            pass
        self.model_store = []
        N = len(f)
        names = [fi for fi in f.keys() if fi!='current']
        names.sort()  # make sure
        for i, gname in enumerate(names):
            if update_callback:
                update_callback(i, N)
            g = f[gname]
            modeli = Model()
            modeli.h5group_name = gname.split('-', 1)[-1]
            modeli.read_h5group(g)
            self.model_store.append(modeli)
        f.close()

    def save_gx(self, fname: str):
        self.model.save(fname)
        self.model.save_addition('config', config.model_dump())
        self.model.save_addition('optimizer', self.optimizer.pickle_string(clear_evals=
                                                                           not config.getboolean('solver',
                                                                                                 'save all evals')))

    def load_gx(self, fname: str):
        self._patch_modules()  # for compatibility with old files
        self.model.load(fname)
        config.load_string(self.model.load_addition('config').decode('utf-8'))
        self.optimizer.pickle_load(self.model.load_addition('optimizer'))

    def _patch_modules(self):
        # add legacy items to genx for loading of pickled strings from old program
        from genx import diffev
        diffev.defualt_autosave = diffev.DiffEv._callbacks.autosave
