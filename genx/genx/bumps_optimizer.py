'''
Use Levenberg-Marquardt as minimizer to estimate errors and for fast decent to next local minimum.
'''
import pickle
from threading import Thread
from dataclasses import dataclass
from typing import Dict

from numpy import *
from bumps.fitters import FitDriver, FIT_AVAILABLE_IDS, FITTERS, FIT_ACTIVE_IDS
from bumps.monitor import TimedUpdate
from bumps.fitproblem import FitProblem, nllf_scale, BaseFitProblem
from bumps.formatnum import format_uncertainty

from .exceptions import ErrorBarError, OptimizerInterrupted
from .core.config import BaseConfig
from .core.custom_logging import iprint
from .model import Model, GenxCurve
from .solver_basis import GenxOptimizer, GenxOptimizerCallback, SolverParameterInfo, SolverResultInfo, SolverUpdateInfo

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
    def __init__(self, problem: FitProblem, parent: 'BumpsOptimizer', progress=0.25, improvement=5.0,
                 max_steps=1000):
        TimedUpdate.__init__(self, progress=progress, improvement=improvement)
        self.problem=problem
        self.parent=parent
        self.max_stepst=max_steps
        self.chis=[]
        self.steps=[]

    def show_progress(self, history):
        scale, err=nllf_scale(self.problem)
        chisq=format_uncertainty(scale*history.value[0], err)
        self.parent.text_output('step: %s/%s  cost: %s'%(history.step[0], self.max_stepst, chisq))
        self.steps.append(history.step[0])
        self.chis.append(scale*history.value[0])

    def show_improvement(self, history):
        self.show_progress(history)
        self.value=history.value[0]
        p=self.problem.getp()
        self.parent.new_beest(p, array([self.steps, self.chis]).T)

if 'DREAM' in [fi.name for fi in FITTERS]:
    fitter_default_name='DREAM'
else:
    fitter_default_name = FITTERS[0].name
@dataclass
class BumpsConfig(BaseConfig):
    section='solver'

    population: int = 12
    samples: int = 10000
    steps:int = 0
    thin: int = 1
    alpha: int = 0
    burn: int = 1000
    outliers: str='none'
    trim: bool = False

    method:str=BaseConfig.GChoice(fitter_default_name, selection=[fi.name for fi in FITTERS])


class BumpsOptimizer(GenxOptimizer):
    '''
    Optimizer based on Levenberg-Marquardt algorithm.
    '''
    opt: BumpsConfig
    model: Model
    fom_log: ndarray
    start_guess: ndarray
    covar: ndarray

    _callbacks: GenxOptimizerCallback=BumpsDefaultCallbacks()
    _map_indices: Dict[int, int]

    n_fom_evals=0

    def is_running(self):
        return False

    def __init__(self):
        GenxOptimizer.__init__(self)
        self.model=Model()
        self.fom_log=array([[0, 0]])[0:0]
        self.covar=None

    def pickle_string(self, clear_evals: bool = False):
        return pickle.dumps(self)

    def pickle_load(self, pickled_string: str):
        obj=pickle.loads(pickled_string)
        # TODO: set own options from object

    def get_start_guess(self):
        return self.start_guess

    def get_model(self) -> Model:
        return self.model

    def get_fom_log(self):
        return self.fom_log

    def text_output(self, text: str):
        self._callbacks.text_output(text)

    def new_beest(self, p, fom_log):
        self.best_vec=self.p_to_vec(p)
        self.fom_log=fom_log
        self.plot_output()

    def p_to_vec(self, p):
        # convert Bumps parameter array p to vector expected by GenX (reorder indices)
        out=list(range(len(p)))
        for i, pi in enumerate(p):
            out[self._map_indices[i]]=pi
        return out

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
        self.start_guess=start_guess
        self.bproblem:BaseFitProblem=self.model.bumps_problem()

    def calc_sim(self, vec):
        ''' calc_sim(self, vec) --> None
        Function that will evaluate the the data points for
        parameters in vec.
        '''
        # Set the parameter values
        list(map(lambda func, value: func(value), self.par_funcs, vec))

        self.model.evaluate_sim_func()
        return self.model.fom

    def calc_fom(self, vec):
        '''
        Function to calcuate the figure of merit for parameter vector
        vec.
        '''
        if self._stop_fit:
            raise OptimizerInterrupted("interrupted")
        # Set the parameter values
        list(map(lambda func, value: func(value), self.par_funcs, vec))
        fom=self.model.evaluate_fit_func() # fom is squared in leastsq
        self.n_fom_evals+=1
        return fom

    def calc_error_bar(self, index: int) -> (float, float):
        if self.covar is None:
            raise ErrorBarError("Could not get covariance matrix from fit, maybe the parameters are coupled/have no influence?")
        err=sqrt(self.covar[index, index])
        return err, err

    def project_evals(self, index: int):
        # -> (ArrayLike, ArrayLike)
        pass

    def start_fit(self, model: Model):
        self.n_fom_evals=0
        self.connect_model(model)
        self._stop_fit=False
        self._thread=Thread(target=self.optimize, daemon=True)
        self._thread.start()

    def optimize(self):
        options={}
        options['pop'] = self.opt.population
        options['samples'] = self.opt.samples
        options['steps'] = self.opt.steps
        options['thin'] = self.opt.thin
        options['alpha'] = self.opt.alpha
        options['outliers'] = self.opt.outliers
        options['trim'] = self.opt.trim
        options['burn'] = self.opt.burn

        problem = self.bproblem
        problem.fitness.stop_fit=False
        options['abort_test']=lambda: problem.fitness.stop_fit
        pnames=list(problem.model_parameters().keys())
        mnames=problem.labels()
        self._map_indices=dict(((i, pnames.index(ni)) for i,ni in enumerate(mnames)))

        fitclass=None
        for fitclass in FITTERS:
            if fitclass.name==self.opt.method:
                break
        # noinspection PyUnboundLocalVariable
        monitors=[FitterMonitor(problem, self)]
        driver = FitDriver(fitclass=fitclass, problem=problem, monitors=monitors, **options)
        driver.clip()  # make sure fit starts within domain
        x0 = problem.getp()
        x, fx = driver.fit()
        problem.setp(x)
        dx = driver.stderr()

        self.best_vec=self.p_to_vec(x)
        #self.covar=driver.cov()

        self.plot_output()
        self._callbacks.fitting_ended(self.get_result_info())

    def stop_fit(self):
        self._stop_fit=True
        self.bproblem.fitness.stop_fit=True
        self._thread.join(1.0)

    def resume_fit(self, model: Model):
        pass

    def is_fitted(self):
        return self.n_fom_evals>0

    def is_configured(self) -> bool:
        pass

    def set_callbacks(self, callbacks: GenxOptimizerCallback):
        self._callbacks=callbacks

    def plot_output(self):
        self.calc_sim(self.best_vec)
        data=SolverUpdateInfo(
            fom_value=self.model.fom,
            fom_name=self.model.fom_func.__name__,
            fom_log=self.get_fom_log(),
            new_best=True,
            data=self.model.data
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
            fitting=False
            )
        return result

