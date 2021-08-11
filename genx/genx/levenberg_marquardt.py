'''
Use Levenberg-Marquardt as minimizer to estimate errors and for fast decent to next local minimum.
'''
import pickle
from dataclasses import dataclass

from numpy import *
from scipy.optimize import leastsq

from .core.config import BaseConfig
from .core.custom_logging import iprint
from .model import Model
from .solver_basis import GenxOptimizer, GenxOptimizerCallback, SolverParameterInfo, SolverResultInfo, SolverUpdateInfo

class LMDefaultCallbacks(GenxOptimizerCallback):

    def text_output(self, text):
        iprint(text)
        sys.stdout.flush()

    def plot_output(self, update_data):
        pass

    def parameter_output(self, param_info):
        pass

    def fitting_ended(self, result_data):
        pass

    def autosave(self):
        pass

@dataclass
class LMConfig(BaseConfig):
    section='solver'
    limit_fit_range:bool=False
    fit_xmin:float=0.0
    fit_xmax:float=180.0

class LMOptimizer(GenxOptimizer):
    '''
    Optimizer based on Levenberg-Marquardt algorithm.
    '''
    opt: LMConfig
    model: Model
    fom_log: ndarray
    start_guess: ndarray
    cover: ndarray

    _callbacks: GenxOptimizerCallback=LMDefaultCallbacks()

    n_fom_evals=0

    def is_running(self):
        return False

    def __init__(self):
        GenxOptimizer.__init__(self)
        self.model=Model()
        self.fom_log=array([[0, 0]])[0:0]
        self.covar=array([[0, 0]])[0:0]

    def pickle_string(self, clear_evals: bool = False):
        return pickle.dumps(self)

    def pickle_load(self, pickled_string: str):
        pass

    def get_start_guess(self):
        return self.start_guess

    def get_model(self) -> Model:
        return self.model

    def get_fom_log(self):
        return self.fom_log

    def connect_model(self, model_obj: Model):
        '''
        Connects the model [model] to this object. Retrives the function
        that sets the variables  and stores a reference to the model.
        '''
        # Retrieve parameters from the model
        (param_funcs, start_guess, par_min, par_max)=model_obj.get_fit_pars()

        # Control parameter setup
        self.par_funcs=param_funcs
        self.model=model_obj
        self.n_dim=len(param_funcs)
        model_obj.opt.limit_fit_range, model_obj.opt.fit_xmin, model_obj.opt.fit_xmax=(
            self.opt.limit_fit_range,
            self.opt.fit_xmin,
            self.opt.fit_xmax)
        self.start_guess=start_guess

    def calc_fom(self, vec):
        '''
        Function to calcuate the figure of merit for parameter vector
        vec.
        '''
        model_obj=self.model
        model_obj.opt.limit_fit_range, model_obj.opt.fit_xmin, model_obj.opt.fit_xmax=(
            self.opt.limit_fit_range,
            self.opt.fit_xmin,
            self.opt.fit_xmax)

        # Set the parameter values
        list(map(lambda func, value: func(value), self.par_funcs, vec))
        fom=self.model.evaluate_fit_func(get_elements=True)
        self.n_fom_evals+=1
        return fom

    def calc_error_bar(self, index: int) -> (float, float):
        err=self.covar[index, index]
        return err, err

    def project_evals(self, index: int):
        # -> (ArrayLike, ArrayLike)
        pass

    def start_fit(self, model: Model):
        self.n_fom_evals=0
        self.connect_model(model)
        res=leastsq(self.calc_fom, self.start_guess, full_output=True)
        self.best_vec=res[0]
        self.covar=res[1]

        result = self.get_result_info()
        self._callbacks.fitting_ended(result)

    def stop_fit(self):
        pass

    def resume_fit(self, model: Model):
        pass

    def is_fitted(self):
        return self.n_fom_evals>0

    def is_configured(self) -> bool:
        pass

    def set_callbacks(self, callbacks: GenxOptimizerCallback):
        self._callbacks=callbacks


    def get_result_info(self):
        result = SolverResultInfo(
            start_guess=self.start_guess.copy(),
            error_message="",
            values=self.best_vec.copy(),
            new_best=True,
            population=[],
            max_val=[],
            min_val=[],
            fitting=False
            )
        return result

