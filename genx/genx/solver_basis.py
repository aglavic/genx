"""
Define abstract base classes for solvers to be used in DiffEv and any future solvers.
"""
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import List, Union

from .core.config import Configurable
from .core.h5_support import H5HintedExport
from .data import DataList
from .model import Model

try:
    from numpy.typing import ArrayLike
except ImportError:
    from numpy import ndarray as ArrayLike

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
    def pickle_load(self, pickled_string: bytes):
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

    def __repr__(self):
        output=f"{self.__class__.__name__} Optimizer:\n"
        for gname, group in self.opt.groups.items():
            output+='    %s:\n'%gname
            for attr in group:
                if type(attr) is list:
                    output+='        '
                    for ati in attr:
                        try:
                            output += '%s=%s\t'%(ati, getattr(self.opt, ati))
                        except AttributeError:
                            continue
                    output=output[:-1]+'\n'
                else:
                    output+='        %-30s %s\n'%(attr, getattr(self.opt, attr))
        return output

    @property
    def widget(self):
        return self._repr_ipyw_()

    def _ipyw_entry_for_item(self, attr):
        import ipywidgets as ipw
        if type(attr) is list:
            sublist = []
            for subattr in attr:
                try:
                    subentry = self._ipyw_entry_for_item(subattr)
                except AttributeError:
                    continue
                sublist.append(subentry)
            item = ipw.VBox(sublist)
            return item

        val = getattr(self.opt, attr)
        if type(val) is bool:
            item = ipw.Checkbox(value=val, indent=False, description=attr, layout=ipw.Layout(width='24ex'))
            entry = item
        elif type(val) is int:
            entry = ipw.IntText(value=val, layout=ipw.Layout(width='18ex'))
            item = ipw.VBox([ipw.Label(attr), entry])
        elif type(val) is float:
            entry = ipw.FloatText(value=val, layout=ipw.Layout(width='18ex'))
            item = ipw.VBox([ipw.Label(attr), entry])
        elif attr=='method':
            entry = ipw.Dropdown(value=val, options=self.methods, layout=ipw.Layout(width='18ex'))
            item = ipw.VBox([ipw.Label(attr), entry])
        else:
            entry = ipw.Text(value=val, layout=ipw.Layout(width='14ex'))
            item = ipw.VBox([ipw.Label(attr), entry])
        entry.change_item = attr
        entry.observe(self._ipyw_change, names='value')
        return item

    def _repr_ipyw_(self):
        import ipywidgets as ipw
        entries=[]
        for gname, group in self.opt.groups.items():
            gentries=[ipw.HTML("<b>%s:</b>"%gname)]
            for attr in group:
                item=self._ipyw_entry_for_item(attr)
                gentries.append(item)
            entries.append(ipw.VBox(gentries, layout=ipw.Layout(width='26ex')))
        return ipw.VBox([ipw.HTML("<h3>Optimizer Settings:</h3>"), ipw.HBox(entries)])

    def _ipyw_change(self, change):
        setattr(self.opt, change.owner.change_item, change.new)


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