"""
GenX model and optimizer control classes. All functional aspects should be covered here (no GUI).
"""
import os
import sys
import h5py
from logging import warning

from .core.config import config
from .exceptions import ErrorBarError, GenxIOError
from .model import Model
from .solver_basis import GenxOptimizer, GenxOptimizerCallback

class ModelController:
    def __init__(self, optimizer: GenxOptimizer):
        self.model=Model()
        self.optimizer=optimizer

    def set_callbacks(self, callbacks: GenxOptimizerCallback):
        self.optimizer.set_callbacks(callbacks)

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

    def CalcErrorBars(self):
        '''
        Method that calculates the errorbars for the fit that has been
        done. Note that the fit has to been conducted before this is run.
        '''
        if self.optimizer.n_fom_evals==0:
            raise ErrorBarError('Can not find any stored evaluations of the model in the optimizer.\n'
                                'Run a fit before calculating the errorbars.')
        if self.optimizer.get_start_guess() is not None and not self.optimizer.is_running():
            n_elements=len(self.optimizer.get_start_guess())
            error_values=[]
            for index in range(n_elements):
                # calculate the error, this is threshold based and not rigours
                (error_low, error_high)=self.optimizer.calc_error_bar(index)
                error_str='(%.3e, %.3e)'%(error_low, error_high)
                error_values.append(error_str)
            return error_values
        else:
            raise ErrorBarError('Wait for fit to finish or fit to changed model first.')

    def ProjectEvals(self, parameter: int):
        '''
        Projects the parameter number parameter on one axis and returns
        the fom values.
        '''
        row=self.model.parameters.get_pos_from_row(parameter)
        if self.optimizer.get_start_guess() is not None and not self.optimizer.is_running():
            return self.optimizer.project_evals(row)
        else:
            raise ErrorBarError()

    def StartFit(self):
        '''
        Function to start running the fit
        '''
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
        self.optimizer.ReadConfig()
        self.optimizer.resume_fit(self.model)

    def IsFitted(self):
        '''
        Returns true if a fit has been started otherwise False
        '''
        return self.optimizer.is_fitted()

    def save(self):
        self.save_file(self.model.get_filename())

    def save_file(self, fname: str):
        """
        Saves objects model, optimiser and config into file fname
        """
        if fname.endswith('.gx'):
            self.save_gx(fname)
        elif fname.endswith('.hgx'):
            self.save_hgx(fname)
        else:
            raise GenxIOError('Wrong file ending, should be .gx or .hgx')
        self.model.filename=os.path.abspath(fname)
        self.model.saved=True

    def load_file(self, fname: str):
        """
        Loads parameters from fname into model, optimizer and config
        """
        if fname.endswith('.gx'):
            self.load_gx(fname)
        elif fname.endswith('.hgx'):
            self.load_hgx(fname)
        else:
            raise GenxIOError('Wrong file ending, should be .gx or .hgx')
        self.model.filename=os.path.abspath(fname)

    def save_hgx(self, fname: str):
        f=h5py.File(fname.encode('utf-8'), 'w')
        g=f.create_group(self.model.h5group_name)
        self.model.write_h5group(g)
        opt_group=g.create_group(self.optimizer.h5group_name)
        opt_group['solver']=self.optimizer.__class__.__name__
        opt_group['solver_module']=self.optimizer.__class__.__module__
        self.optimizer.write_h5group(opt_group)
        self.WriteConfig()
        g['config']=config.model_dump().encode('utf-8')
        f.close()

    def load_hgx(self, fname: str):
        f=h5py.File(fname.encode('utf-8'), 'r')
        g=f[self.model.h5group_name]
        self.model.read_h5group(g)
        opt_group=g[self.optimizer.h5group_name]
        try:
            solver_class=opt_group.get('solver')[()]
            solver_module=opt_group.get('solver_module')[()]
        except (KeyError, TypeError):
            solver_class='DiffEv'
            solver_module='genx.diffev'
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
                prev_optimizer=self.optimizer
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
        f.close()

    def save_gx(self, fname: str):
        self.model.save(fname)
        self.model.save_addition('config', config.model_dump())
        self.model.save_addition('optimizer', self.optimizer.pickle_string(clear_evals=
                                        not config.getboolean('solver', 'save all evals')))

    def load_gx(self, fname: str):
        self._patch_modules() # for compatibility with old files
        self.model.load(fname)
        config.load_string(self.model.load_addition('config').decode('utf-8'))
        self.optimizer.pickle_load(self.model.load_addition('optimizer'))

    def _patch_modules(self):
        # add legacy items to genx for loading of pickled strings from old program
        from genx import diffev
        diffev.defualt_autosave=diffev.DiffEv._callbacks.autosave