"""
Command classes that perform actions on a model. Allows
to implement undo/redo functionality and tracking of
actions in logs.
"""
import difflib

from logging import debug
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Flag, auto
from typing import List

from .model import Model
from .parameters import SortSplitItem
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
        # generate string representation replacing format string items interactively
        obj_items=self.__dict__.copy()
        obj_items.update(self.__class__.__dict__)
        return self.description.format(**obj_items)

    @property
    def description(self):
        return self.name

    @property
    def action_name(self):
        return self.name.format(**self.__dict__)

    def should_update(self, other: 'ModelAction'):
        # some actions prefer an update over keeping all changes in the todo
        return False

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
        action_string=str(action)
        action_string=action_string.replace('\n', '\n    ')
        debug(f'Action Executed - {action.__class__.__name__}: {action_string}')
        if len(self.undo_stack)>0 and action.should_update(self.undo_stack[-1]):
            self.undo_stack.pop()
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

    def remove_actions(self, start, length=1):
        """
        Undo the last 'start' actions, skipe 'length'
        redo actions and then redo the residual actions.
        """
        current_undo=len(self.redo_stack)
        changed_actions=[]
        if len(self.undo_stack)<start:
            raise ValueError(f"Only {len(self.undo_stack)} items on the stack to undo.")
        for i in range(start):
            changed_actions.append(self.undo())
        for i in range(length):
            self.redo_stack.pop()
        while len(self.redo_stack)>current_undo:
            # don't redo action the where on the redo stack before the start
            changed_actions.append(self.redo())
        return ActionBlock(changed_actions[-1].model, changed_actions)

    def clear(self):
        self.undo_stack=[]
        self.redo_stack=[]

class ActionBlock(ModelAction):
    influences = ModelInfluence.NONE
    name = 'action block'

    def __init__(self, model, actions: List[ModelAction]):
        self.model=model
        for action in actions:
            self.influences=self.influences|action.influences
        self.actions=actions

    def execute(self):
        for i, action in enumerate(self.actions):
            try:
                action.execute()
            except Exception as e:
                for rev in reversed(self.actions[:i]):
                    rev.undo()
                raise e

    def undo(self):
        for action in reversed(self.actions):
            action.undo()

    @property
    def description(self):
        return '|'.join(map(str, self.actions))


class SetModelScript(ModelAction):
    influences = ModelInfluence.SCRIPT
    name = 'edit script'

    def __init__(self, model, text):
        self.model=model
        self.new_text=text
        self.old_text=self.model.get_script()

    def execute(self):
        # Replace model script with new text, script as new text (toggles)
        self.old_text=self.model.get_script()
        self.model.set_script(self.new_text)

    def undo(self):
        self.model.set_script(self.old_text)

    @property
    def description(self):
        old=self.old_text or self.model.get_script()
        new=self.new_text
        diff=''.join(difflib.unified_diff(old.splitlines(keepends=True),
                                          new.splitlines(keepends=True),
                                          fromfile='old script', tofile='new script',n=1))
        return self.name+': '+diff

class UpdateSolverOptoins(ModelAction):
    influences = ModelInfluence.OPTIONS
    name = 'optimizer options'
    description = 'set optimizer parameters: {new_values}'

    def __init__(self, model, optimizer: GenxOptimizer, new_values: dict):
        self.model=model
        self.optimizer=optimizer
        self.new_values=new_values.copy()
        self.old_values={}
        self.combined_options=self.model.solver_parameters|self.optimizer.opt
        for key in new_values.keys():
            self.old_values[key]=getattr(self.combined_options, key, None)

    def execute(self):
        self.combined_options=self.model.solver_parameters|self.optimizer.opt
        for key in self.new_values.keys():
            self.old_values[key]=getattr(self.combined_options, key, None)
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

class UpdateParams(ModelAction):
    influences = ModelInfluence.PARAM
    name = 'parameter values'
    description = 'set parameter values\n{param_names}\nfrom\n{old_values}\nto\n{new_values}'

    def __init__(self, model, new_values):
        self.model=model
        self.new_values=list(new_values)
        self.param_names=self.model.parameters.get_fit_pars()[1]
        self.old_values=self.model.parameters.get_value_pars()

    def execute(self):
        param_names=self.model.parameters.get_fit_pars()[1]
        if self.param_names!=param_names:
            raise ValueError("Current fit parameters do not correspond to setting during action init")
        self.old_values=self.model.parameters.get_value_pars()
        self.model.parameters.set_value_pars(self.new_values)

    def undo(self):
        self.model.parameters.set_value_pars(self.old_values)

class UpdateParamValue(ModelAction):
    influences = ModelInfluence.PARAM
    name = 'update parameter [{param_name}]'
    description = 'parameter[{param_name},{col}]: {old_value!r} -> {new_value!r}'

    def __init__(self, model, row, col, new_value):
        self.model=model
        self.col=col
        self.new_value=new_value
        if col!=0:
            self.param_name=self.model.parameters.get_names()[row]
        else:
            self.param_name=row
        self.old_value=self.model.parameters.get_value(row, col)

    def execute(self):
        if self.col!=0:
            row=self.model.parameters.get_names().index(self.param_name)
        else:
            row=self.param_name
        self.old_value=self.model.parameters.get_value(row, self.col)
        self.model.parameters.set_value(row, self.col, self.new_value)

    def undo(self):
        if self.col!=0:
            row=self.model.parameters.get_names().index(self.param_name)
        else:
            row=self.param_name
        self.model.parameters.set_value(row, self.col, self.old_value)

    def should_update(self, other):
        # if the same parameter is changed multiple time, just keep one change
        if type(other)!=type(self) or self.param_name!=getattr(other, 'param_name', None):
            return False
        else:
            self.old_value=other.old_value
            return True

class MoveParam(ModelAction):
    influences = ModelInfluence.PARAM
    name = 'move parameter'

    def __init__(self, model, row, step):
        self.model=model
        self.step=step
        self.param_name=self.model.parameters.get_names()[row]

    def execute(self):
        row=self.model.parameters.get_names().index(self.param_name)
        self.model.parameters.move_row(row, self.step)

    def undo(self):
        row=self.model.parameters.get_names().index(self.param_name)
        self.model.parameters.move_row(row, -self.step)

class DeleteParams(ModelAction):
    influences = ModelInfluence.PARAM
    name = 'delete parameters'
    description = 'deleted {parameter_names}'

    def __init__(self, model, rows):
        self.model=model
        names=self.model.parameters.get_names()
        self.param_names=[names[ri] for ri in rows]
        self.old_data=[]

    def execute(self):
        names=self.model.parameters.get_names()
        rows=[names.index(ni) for ni in self.param_names]
        old_rows=self.model.parameters.data
        self.old_data=[(ri, old_rows[ri]) for ri in rows]
        self.model.parameters.delete_rows(rows)

    def undo(self):
        for ri, di in self.old_data:
            self.model.parameters.data.insert(ri, di)

class InsertParam(ModelAction):
    influences = ModelInfluence.PARAM
    name = 'insert parameter'

    def __init__(self, model, row):
        self.model=model
        self.param_name=self.model.parameters.get_names()[row]

    def execute(self):
        row=self.model.parameters.get_names().index(self.param_name)
        self.insert_index=row
        self.model.parameters.insert_row(row)

    def undo(self):
        self.model.parameters.delete_rows([self.insert_index])

class SortAndGroupParams(ModelAction):
    influences = ModelInfluence.PARAM
    name = 'sort parameters'
    description = 'sort by {sort_params}'

    def __init__(self, model, sort_params: SortSplitItem):
        self.model=model
        self.sort_params=sort_params
        self.old_parameters=[]

    def execute(self):
        self.old_parameters=self.model.parameters.data.copy()
        if not self.model.compiled: self.model.compile_script()
        self.model.parameters.sort_rows(self.model, self.sort_params)
        self.model.parameters.group_rows(self.model, self.sort_params)
        self.model.parameters.strip()

    def undo(self):
        self.model.parameters.data=self.old_parameters
        self.old_parameters=[]
