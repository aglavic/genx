"""
Use Levenberg-Marquardt as minimizer to estimate errors and for fast decent to next local minimum.
"""

import _thread
import pickle
import sys

from dataclasses import dataclass

from numpy import array, ndarray, sign, sqrt
from scipy.optimize import leastsq

from .core.config import BaseConfig
from .core.custom_logging import iprint
from .exceptions import ErrorBarError, OptimizerInterrupted
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
    section = "solver"

    groups = {}


class LMOptimizer(GenxOptimizer):
    """
    Optimizer based on Levenberg-Marquardt algorithm.
    """

    opt: LMConfig
    model: Model
    fom_log: ndarray
    start_guess: ndarray
    covar: ndarray

    _callbacks: GenxOptimizerCallback = LMDefaultCallbacks()

    n_fom_evals = 0

    def is_running(self):
        return False

    def __init__(self):
        GenxOptimizer.__init__(self)
        self.model = Model()
        self.fom_log = array([[0, 0]])[0:0]
        self.covar = None

    def pickle_string(self, clear_evals: bool = False):
        return pickle.dumps(self)

    def pickle_load(self, pickled_string: bytes):
        obj = pickle.loads(pickled_string, encoding="latin1", errors="ignore")
        # TODO: set own options from object

    def get_start_guess(self):
        return self.start_guess

    def get_model(self) -> Model:
        return self.model

    def get_fom_log(self):
        return self.fom_log

    def connect_model(self, model_obj: Model):
        """
        Connects the model [model] to this object. Retrives the function
        that sets the variables  and stores a reference to the model.
        """
        # Retrieve parameters from the model
        (param_funcs, start_guess, par_min, par_max) = model_obj.get_fit_pars(use_bounds=False)

        # Control parameter setup
        self.par_funcs = param_funcs
        self.model = model_obj
        self.n_dim = len(param_funcs)
        self.start_guess = start_guess

    def calc_sim(self, vec):
        """calc_sim(self, vec) --> None
        Function that will evaluate the the data points for
        parameters in vec.
        """
        model_obj = self.model
        # Set the parameter values
        list(map(lambda func, value: func(value), self.par_funcs, vec))

        self.model.evaluate_sim_func()
        return self.model.fom

    def calc_fom(self, vec):
        """
        Function to calcuate the figure of merit for parameter vector
        vec.
        """
        if self._stop_fit:
            raise OptimizerInterrupted("interrupted")
        # Set the parameter values
        list(map(lambda func, value: func(value), self.par_funcs, vec))
        fom = self.model.evaluate_fit_func(get_elements=True)  # fom is squared in leastsq
        chi = sign(fom) * sqrt(abs(fom))
        self.n_fom_evals += 1
        return chi

    def calc_error_bar(self, index: int) -> (float, float):
        if self.covar is None:
            raise ErrorBarError(
                "Could not get covariance matrix from fit, maybe the parameters are coupled/have no influence?"
            )
        err = sqrt(self.covar[index, index])
        return -err, err

    def project_evals(self, index: int):
        # -> (ArrayLike, ArrayLike)
        pass

    def start_fit(self, model: Model):
        self.n_fom_evals = 0
        self.connect_model(model)
        self._stop_fit = False
        _thread.start_new_thread(self.optimize, ())

    def optimize(self):
        try:
            res = leastsq(self.calc_fom, self.start_guess, full_output=True)
        except OptimizerInterrupted:
            self._callbacks.fitting_ended(self.get_result_info(interrupted=True))
            return
        self.best_vec = res[0]
        if res[1] is None:
            self.covar = None
        else:
            Chi2Res = self.calc_fom(self.best_vec) ** 2
            s_sq = Chi2Res.sum() / (len(Chi2Res) - len(res[0]))  # variance of the residuals
            self.covar = res[1] * s_sq

        self.plot_output()
        self._callbacks.fitting_ended(self.get_result_info())

    def stop_fit(self):
        self._stop_fit = True

    def resume_fit(self, model: Model):
        pass

    def is_fitted(self):
        return self.n_fom_evals > 0

    def is_configured(self) -> bool:
        pass

    def set_callbacks(self, callbacks: GenxOptimizerCallback):
        self._callbacks = callbacks

    def get_callbacks(self) -> "GenxOptimizerCallback":
        return self._callbacks

    def plot_output(self):
        self.calc_sim(self.best_vec)
        data = SolverUpdateInfo(
            fom_value=self.model.fom,
            fom_name=self.model.fom_func.__name__,
            fom_log=self.get_fom_log(),
            new_best=True,
            data=self.model.data,
        )
        self._callbacks.plot_output(data)

    def get_result_info(self, interrupted=False):
        result = SolverResultInfo(
            start_guess=self.start_guess.copy(),
            error_message="",
            values=self.best_vec.copy(),
            new_best=not interrupted,
            population=[],
            max_val=[],
            min_val=[],
            fitting=False,
        )
        return result
