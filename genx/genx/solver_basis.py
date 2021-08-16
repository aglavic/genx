"""
Define abstract base classes for solvers to be used in DiffEv and any future solvers.
"""
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import List, Union

from numpy.typing import ArrayLike

from .core.config import Configurable
from .core.h5_support import H5HintedExport
from .data import DataList
from .model import Model


class GenxOptimizer(Configurable, H5HintedExport, metaclass=ABCMeta):
    """
    Defines an abstract base class for a optimizer for GenX models.
    DiffEv is implementing this abstraction.
    """
    h5group_name='optimizer'
    _export_ignore = ['opt', 'model']

    @property
    @abstractmethod
    def n_fom_evals(self)->int:
        """Returns teh number of FOM evaluations"""

    @abstractmethod
    def pickle_string(self, clear_evals: bool=False):
        """ Return a pickle string for the object """

    @abstractmethod
    def pickle_load(self, pickled_string: str):
        """ Configure object from pickled copy """

    @abstractmethod
    def is_running(self)->bool:
        """Indicate if a refinement is currently running"""

    @abstractmethod
    def get_start_guess(self)->ArrayLike:
        """Return the used start_guess for parameters"""

    @abstractmethod
    def get_model(self)->Model:
        """Return the used model object"""

    @abstractmethod
    def get_fom_log(self)->ArrayLike:
        """Return array of FOM evaluations from fit"""

    @abstractmethod
    def calc_error_bar(self, index:int)->(float, float):
        """Use simple threshold based calculation for errorbar estimation"""

    @abstractmethod
    def project_evals(self, index: int) -> (ArrayLike, ArrayLike):
        """Generate parameter value vs. FOM for previous run of solver"""

    @abstractmethod
    def start_fit(self, model: Model):
        """Start refining the model parameters"""

    @abstractmethod
    def stop_fit(self):
        """Stop refining the model parameters"""

    @abstractmethod
    def resume_fit(self, model: Model):
        """Continue a previously stopped fit"""

    @abstractmethod
    def is_fitted(self)->bool:
        """Indicate if a refinement was done earlier"""

    @abstractmethod
    def is_configured(self)->bool:
        """Has the refinement been setup for a fit"""

    @abstractmethod
    def set_callbacks(self, callbacks: 'GenxOptimizerCallback'):
        """Setup the callback mechanism"""

    @abstractmethod
    def get_callbacks(self)-> 'GenxOptimizerCallback':
        """return the callback mechanism object"""

    @abstractmethod
    def get_result_info(self)->'SolverResultInfo':
        """Return the result info of a previous run"""

@dataclass(frozen=True)
class SolverParameterInfo:
    # Used to report model parameter updates
    values: ArrayLike
    new_best: bool
    population: List[ArrayLike]
    max_val: ArrayLike
    min_val: ArrayLike
    fitting: bool

@dataclass(frozen=True)
class SolverUpdateInfo:
    # Used to report model update and result data
    new_best: bool
    fom_value: float
    fom_name: str
    fom_log: ArrayLike
    data: DataList

@dataclass(frozen=True)
class SolverResultInfo(SolverParameterInfo):
    # Used to report the model parameters of the fit result
    start_guess: ArrayLike
    error_message: Union[str, None]

class GenxOptimizerCallback(ABC):
    """
    Defines the callbacks used by the optimizer to give feedback on the state of refinement.
    It defines a set of methods that are used by the solver.
    """

    @abstractmethod
    def text_output(self, text)->None:
        """Send small update string to give status feedback"""

    @abstractmethod
    def plot_output(self, update_data: SolverUpdateInfo)->None:
        """Send full function results back for plotting"""

    @abstractmethod
    def parameter_output(self, param_info: SolverParameterInfo)->None:
        """Update the state of the model parameters"""

    @abstractmethod
    def fitting_ended(self, result_data: SolverResultInfo)->None:
        """Indicate a fit has ended"""

    @abstractmethod
    def autosave(self)->None:
        """Send small update string to give status feedback"""