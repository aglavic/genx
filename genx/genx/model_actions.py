"""
Command classes that perform actions on a model. Allows
to implement undo/redo functionality and tracking of
actions in logs.
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Flag, auto
from typing import List

from .model import Model
from .solver_basis import GenxOptimizer

class ModelInfluence(Flag):
    # The part of the model that gets altered by n action
    NONE = auto()
    SCRIPT = auto()
    PARAM = auto()
    DATA = auto()
    OPTIONS = auto()

class ModelAction(ABC):
    """
    Represents an action performed on the model.
    """
    model: Model

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def influences(self) -> ModelInfluence:
        ...

    @abstractmethod
    def __init__(self, model: Model, *params):
        ...

    @abstractmethod
    def execute(self):
        ...

    @abstractmethod
    def undo(self):
        ...

    def redo(self):
        self.execute()

    def __str__(self):
        return self.name

class NoOp(ModelAction):
    influences = ModelInfluence.NONE
    name = 'no action'
    def __init__(self, model): self.model = model
    def execute(self): pass
    def undo(self): pass
    def redo(self): pass

@dataclass
class ActionHistory:
    undo_stack: List[ModelAction] = field(default_factory=list)
    redo_stack: List[ModelAction] = field(default_factory=list)
    max_stack=100

    def execute(self, action: ModelAction):
        action.execute()
        self.undo_stack.append(action)
        if len(self.undo_stack)>self.max_stack:
            self.undo_stack.pop(0)
        self.redo_stack=[]

    def undo(self) -> ModelAction:
        if len(self.undo_stack)>0:
            action=self.undo_stack.pop()
            action.undo()
            self.redo_stack.append(action)
            return action
        else:
            return NoOp(None)

    def redo(self) -> ModelAction:
        if len(self.redo_stack)>0:
            action=self.redo_stack.pop()
            action.redo()
            self.undo_stack.append(action)
            return action
        else:
            return NoOp(None)

    def clear(self):
        self.undo_stack=[]
        self.redo_stack=[]

class SetModelScript(ModelAction):
    influences = ModelInfluence.SCRIPT
    name = 'edit script'

    def __init__(self, model, text):
        self.model=model
        self.new_text=text
        self.old_text=None

    def execute(self):
        # Replace model script with new text, store previous script as new text (toggles)
        self.old_text=self.model.get_script()
        self.model.set_script(self.new_text)

    def undo(self):
        self.model.set_script(self.old_text)

    def redo(self):
        self.model.set_script(self.new_text)

    def __str__(self):
        import difflib
        old=self.old_text or self.model.get_script()
        new=self.new_text
        diff=''.join(difflib.unified_diff(old.splitlines(keepends=True),
                                          new.splitlines(keepends=True),
                                          fromfile='old script', tofile='new script',n=1))
        return diff

class UpdateSolverOptoins(ModelAction):
    influences = ModelInfluence.OPTIONS
    name = 'optimizer options'

    def __init__(self, model, optimizer: GenxOptimizer, new_values: dict):
        self.model=model
        self.optimizer=optimizer
        self.new_values=new_values.copy()
        self.old_values={}
        self.combined_options=self.model.solver_parameters|self.optimizer.opt
        for key in new_values.keys():
            self.old_values[key]=getattr(self.combined_options, key, None)

    def execute(self):
        # Update parameters and write to config
        for key, value in self.new_values.items():
            setattr(self.combined_options, key, value)
        self.model.WriteConfig()
        self.optimizer.WriteConfig()

    def undo(self):
        # Reset previous settings
        for key, value in self.old_values.items():
            setattr(self.combined_options, key, value)
        self.model.WriteConfig()
        self.optimizer.WriteConfig()

class UpdateDataPlotSettings(ModelAction):
    influences = ModelInfluence.DATA
    name = 'plot settings'

    def __init__(self, model, indices: List[int], sim_par: dict, data_par: dict):
        self.model=model
        self.indices=indices
        self.new_sim_par=sim_par.copy()
        self.new_data_par=data_par.copy()
        self.old_sim_pars={}
        self.old_data_pars={}

    def execute(self):
        # Update parameters for each dataset index
        for index in self.indices:
            di=self.model.data[index]
            self.old_sim_pars[index]=di.get_sim_plot_items()
            self.old_data_pars[index]=di.get_data_plot_items()
            di.set_sim_plot_items(self.new_sim_par)
            di.set_data_plot_items(self.new_data_par)

    def undo(self):
        # Reset previous settings
        for index in self.indices:
            di=self.model.data[index]
            di.set_sim_plot_items(self.old_sim_pars[index])
            di.set_data_plot_items(self.old_data_pars[index])

class UpdateColorCycle(ModelAction):
    influences = ModelInfluence.DATA
    name = 'color cycle'

    def __init__(self, model, source):
        self.model=model
        self.new_source=source
        self.old_source=None
        self.old_sim_colors={}
        self.old_data_colors={}

    def execute(self):
        # Update colors from each dataset index
        for i, di in enumerate(self.model.data):
            self.old_sim_colors[i]=di.sim_color
            self.old_data_colors[i]=di.data_color
        self.old_source=self.model.data.color_source
        self.model.data.update_color_cycle(self.new_source)

    def undo(self):
        # Reset previous settings and colors
        self.model.data.update_color_cycle(self.old_source)
        for i, di in enumerate(self.model.data):
            di.sim_color=self.old_sim_colors[i]
            di.data_color=self.old_data_colors[i]
