"""
Use Levenberg-Marquardt as minimizer to estimate errors and for fast decent to next local minimum.
"""

import multiprocessing
import pickle
import sys

from dataclasses import dataclass
from logging import debug
from threading import Thread
from typing import Dict

import numpy as np

from bumps.dream.stats import var_stats
from bumps.fitproblem import BaseFitProblem, FitProblem, nllf_scale
from bumps.fitters import FITTERS, FitDriver
from bumps.formatnum import format_uncertainty
from bumps.monitor import TimedUpdate

from .core import custom_logging
from .core.config import BaseConfig
from .exceptions import ErrorBarError, OptimizerInterrupted
from .model import GenxCurve, Model
from .solver_basis import GenxOptimizer, GenxOptimizerCallback, SolverParameterInfo, SolverResultInfo, SolverUpdateInfo

_cpu_count = multiprocessing.cpu_count()
iprint = custom_logging.iprint


class BumpsDefaultCallbacks(GenxOptimizerCallback):

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


class FitterMonitor(TimedUpdate):

    def __init__(self, problem: FitProblem, parent: "BumpsOptimizer", progress=0.5, improvement=2.0):
        TimedUpdate.__init__(self, progress=progress, improvement=improvement)
        self.problem = problem
        self.parent = parent
        self.chis = []
        self.steps = []
        self.p = None
        self.last_time = 0.0
        self.last_step = 0

    def show_progress(self, history):
        scale, err = nllf_scale(self.problem)
        chisq = scale * history.value[0]
        n_fev = len(history.population_values[0]) * (history.step[0] - self.last_step)
        dt = history.time[0] - self.last_time
        self.last_step = history.step[0]
        self.last_time = history.time[0]
        self.parent.n_fom_evals = len(history.population_values[0]) * history.step[0]
        self.parent.text_output(f"FOM: {chisq:.3f} Iteration: {history.step[0]} Speed: {n_fev/dt:.1f}")
        self.parent.parameter_output(self.chis, history.population_values[0], history.population_points[0])

    def show_improvement(self, history):
        self.parent.new_beest(self.p, np.array([self.steps, self.chis]).T)

    def __call__(self, history):
        t = history.time[0]
        v = history.value[0]

        scale, err = nllf_scale(self.problem)
        self.steps.append(history.step[0])
        self.chis.append(scale * history.value[0])
        if v < self.value:
            self.improved = True
            self.value = v
            self.p = history.point[0]
        if t > self.progress_time + self.progress_delta:
            self.progress_time = t
            self.show_progress(history)
        if self.improved and t > self.improvement_time + self.improvement_delta:
            self.improved = False
            self.improvement_time = t
            self.show_improvement(history)


if "DREAM" in [fi.name for fi in FITTERS]:
    fitter_default_name = "DREAM"
else:
    fitter_default_name = FITTERS[0].name


@dataclass
class BumpsConfig(BaseConfig):
    section = "solver"

    population: int = 12
    samples: int = 10000
    steps: int = 0
    thin: int = 1
    alpha: int = 0
    burn: int = 1000
    outliers: str = BaseConfig.GChoice("none", ["none", "IQR", "Grubbs", "Mahal"], label="Outlier Test")
    trim: bool = False

    ftol: float = 1e-6
    xtol: float = 1e-12

    method: str = BaseConfig.GChoice(fitter_default_name, selection=[fi.name for fi in FITTERS])

    use_parallel_processing: bool = False
    parallel_processes: int = BaseConfig.GParam(_cpu_count, pmin=2, pmax=_cpu_count, label="# processes")
    parallel_chunksize: int = BaseConfig.GParam(10, pmin=1, pmax=1000, label="items/chunk")

    use_boundaries = True

    groups = {  # for building config dialogs
        "Bumps Fitting": ["method", "steps"],
        "Statistic Solvers": [
            "population",
        ],
        "Tolerances": [["ftol", "xtol"]],
        "DREAM": [["burn", "samples"], ["trim", "thin"], "alpha", "outliers"],
        "Parallel processing": ["use_parallel_processing", "parallel_processes", "parallel_chunksize"],
    }


@dataclass
class BumpsResult:
    x: np.ndarray
    dx: np.ndarray
    dxpm: np.ndarray
    cov: np.ndarray
    chisq: float
    bproblem: "Any"
    state: "Any" = None


class BumpsOptimizer(GenxOptimizer):
    """
    Optimizer based on Levenberg-Marquardt algorithm.
    """

    opt: BumpsConfig
    model: Model
    fom_log: np.ndarray
    start_guess: np.ndarray
    covar: np.ndarray
    errors: np.ndarray

    _callbacks: GenxOptimizerCallback = BumpsDefaultCallbacks()
    _map_indices: Dict[int, int]

    n_fom_evals = 0
    _running = False

    def is_running(self):
        return self._running

    def __init__(self):
        GenxOptimizer.__init__(self)
        self.model = Model()
        self.fom_log = np.array([[0, 0]])[0:0]
        self.start_guess = np.array([[0, 0]])[0:0]
        self.covar = np.array([[0, 0]])[0:0]
        self.errors = np.array([[0, 0]])[0:0]
        self.last_result = None
        self.bproblem = None

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

    def text_output(self, text: str):
        self._callbacks.text_output(text)

    def parameter_output(self, fom_history, chis, population):
        if len(population) == 0:
            return
        best = chis.argmin()
        population = np.array(list(map(self.map_bumps2genx, population)))
        new_best = chis[best] <= min(fom_history)
        if new_best:
            best_pop = population[best]
        else:
            best_pop = self.best_vec
        param_info = SolverParameterInfo(
            values=best_pop,
            new_best=new_best,
            population=[pi for pi in population],
            max_val=self.par_max.copy(),
            min_val=self.par_min.copy(),
            fitting=True,
        )
        self._callbacks.parameter_output(param_info)

    def new_beest(self, p, fom_log):
        self.best_vec = self.map_bumps2genx(p)
        self.fom_log = fom_log
        self.plot_output()

    def map_bumps2genx(self, p):
        # convert Bumps parameter array p to vector with GenX order of indices
        out = np.zeros(len(p))
        for i, pi in enumerate(p):
            out[self._map_indices[i]] = pi
        return out

    def covar_bumps2genx(self, cov):
        if cov is None:
            return None
        out = np.zeros((len(cov), len(cov)))
        for i, rowi in enumerate(cov):
            for j, pij in enumerate(rowi):
                out[self._map_indices[i], self._map_indices[j]] = pij
        return out

    def connect_model(self, model_obj: Model):
        """
        Connects the model [model] to this object. Retrives the function
        that sets the variables  and stores a reference to the model.
        """
        # Retrieve parameters from the model
        (param_funcs, start_guess, par_min, par_max) = model_obj.get_fit_pars()

        # Control parameter setup
        self.par_min = np.array(par_min)
        self.par_max = np.array(par_max)
        self.par_funcs = param_funcs
        self.model = model_obj
        self.n_dim = len(param_funcs)
        self.start_guess = start_guess
        self.best_vec = start_guess
        self.bproblem: BaseFitProblem = self.model.bumps_problem()
        self.last_result = None

    def calc_sim(self, vec):
        """calc_sim(self, vec) --> None
        Function that will evaluate the the data points for
        parameters in vec.
        """
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
        fom = self.model.evaluate_fit_func()  # fom is squared in leastsq
        self.n_fom_evals += 1
        return fom

    def calc_error_bar(self, index: int) -> (float, float):
        if self.errors is None:
            raise ErrorBarError(
                "Could not get covariance matrix from fit, maybe the parameters are coupled/have no influence?"
            )
        return self.errors[index][0], self.errors[index][1]

    def project_evals(self, index: int):
        # -> (ArrayLike, ArrayLike)
        pass

    def start_fit(self, model: Model):
        self.n_fom_evals = 0
        self.connect_model(model)
        self._stop_fit = False

        self._thread = Thread(target=self.optimize, daemon=True)
        self._thread.start()
        self._running = True

    def optimize(self):
        options = {}
        options["pop"] = self.opt.population
        options["samples"] = self.opt.samples
        options["steps"] = self.opt.steps
        options["thin"] = self.opt.thin
        options["alpha"] = self.opt.alpha
        options["outliers"] = self.opt.outliers
        options["trim"] = self.opt.trim
        options["burn"] = self.opt.burn
        options["ftol"] = self.opt.ftol
        options["xtol"] = self.opt.xtol

        problem = self.bproblem
        problem.fitness.stop_fit = False
        options["abort_test"] = lambda: problem.fitness.stop_fit
        pnames = list(problem.model_parameters().keys())
        mnames = problem.labels()
        self._map_indices = dict(((i, pnames.index(ni)) for i, ni in enumerate(mnames)))

        fitclass = None
        for fitclass in FITTERS:
            if fitclass.name == self.opt.method:
                break

        if self.opt.use_parallel_processing:
            from .models.lib import USE_NUMBA, paratt

            use_cuda = paratt.Refl.__module__.rsplit(".", 1)[1] == "paratt_cuda"
            # reduce numba thread count for numba functions
            if USE_NUMBA:
                numba_procs = max(1, _cpu_count // self.opt.parallel_processes)
            else:
                numba_procs = None
            self.text_output("Starting a pool with %i workers ..." % (self.opt.parallel_processes,))
            self.pool = multiprocessing.Pool(
                processes=self.opt.parallel_processes,
                initializer=parallel_init,
                initargs=(numba_procs, custom_logging.mp_logger.queue),
            )
            if use_cuda:
                self.pool.apply_async(init_cuda)
            options["mapper"] = lambda p: list(self.pool.map(problem.nllf, p, chunksize=self.opt.parallel_chunksize))
            # TODO: investigate why function connection is lost here
            (param_funcs, start_guess, par_min, par_max) = self.model.get_fit_pars()
            self.par_funcs = param_funcs

        monitors = [FitterMonitor(problem, self)]
        driver = FitDriver(fitclass=fitclass, problem=problem, monitors=monitors, **options)
        driver.clip()  # make sure fit starts within domain
        x0 = problem.getp()
        self.start_guess = self.map_bumps2genx(x0)
        x, fx = driver.fit()
        problem.setp(x)
        dx = driver.stderr()
        if self.opt.method.lower() == "dream":
            dxpm = self.model.asym_stderr(driver.fitter)
        else:
            dxpm = None
        cov = driver.cov()

        if self.opt.use_parallel_processing:
            self.pool.close()
            self.pool.join()
            self.pool = None

        result = BumpsResult(x=x, dx=dx, dxpm=dxpm, cov=cov, chisq=driver.chisq(), bproblem=self.bproblem)
        if hasattr(driver.fitter, "state"):
            result.state = driver.fitter.state
        self.last_result = result

        self.best_vec = self.map_bumps2genx(x)
        if dxpm is not None:
            dxm = self.map_bumps2genx(dxpm[:, 0])
            dxp = self.map_bumps2genx(dxpm[:, 1])
            self.errors = np.array([dxm, dxp]).T
        else:
            dxmap = self.map_bumps2genx(dx)
            self.errors = np.array([-1, 1])[np.newaxis, :] * dxmap[:, np.newaxis]
        self.covar = self.covar_bumps2genx(cov)

        self.plot_output()
        self._callbacks.fitting_ended(self.get_result_info(interrupted=problem.fitness.stop_fit))
        self._running = False

    def stop_fit(self):
        if self.bproblem is None:
            return
        self._stop_fit = True
        self.bproblem.fitness.stop_fit = True
        self._thread.join(1.0)

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
        fom_log = self.get_fom_log()
        data = SolverUpdateInfo(
            fom_value=fom_log[-1, 1], fom_name="chi2bars", fom_log=fom_log, new_best=True, data=self.model.data
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


def parallel_init(numba_procs=None, log_queue=None):
    """
    parallel initialization of a pool of processes. The function takes a
    pickle safe copy of the model and resets the script module and the compiles
    the script and creates function to set the variables.
    """
    if log_queue is not None:
        custom_logging.setup_mp(log_queue)
    if numba_procs is not None:
        try:
            import numba
        except ImportError:
            pass
        else:
            if hasattr(numba, "set_num_threads") and numba.get_num_threads() > numba_procs:
                debug(f"Setting numba threads to {numba_procs}")
                numba.set_num_threads(numba_procs)
    debug(f"Initialize multiprocessing for bumps")


def init_cuda():
    iprint("Init CUDA in one worker")
    # activate cuda in subprocesses
    from .models.lib import neutron_cuda, neutron_refl, paratt, paratt_cuda

    paratt.Refl = paratt_cuda.Refl
    paratt.ReflQ = paratt_cuda.ReflQ
    paratt.Refl_nvary2 = paratt_cuda.Refl_nvary2
    neutron_refl.Refl = neutron_cuda.Refl
    from .models.lib import neutron_refl, paratt

    paratt.Refl = paratt_cuda.Refl
    paratt.ReflQ = paratt_cuda.ReflQ
    paratt.Refl_nvary2 = paratt_cuda.Refl_nvary2
    neutron_refl.Refl = neutron_cuda.Refl
    iprint("CUDA init done, go to work")
