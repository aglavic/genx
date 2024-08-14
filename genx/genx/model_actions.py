"""
Command classes that perform actions on a model. Allows
to implement undo/redo functionality and tracking of
actions in logs.
"""

import difflib

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Flag, auto
from logging import debug
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
    def name(self) -> str: ...

    @property
    @abstractmethod
    def influences(self) -> ModelInfluence: ...

    @abstractmethod
    def __init__(self, model: Model, *params): ...

    @abstractmethod
    def execute(self): ...

    @abstractmethod
    def undo(self): ...

    def redo(self):
        self.execute()

    def __str__(self):
        # generate string representation replacing format string items interactively
        obj_items = self.__dict__.copy()
        obj_items.update(self.__class__.__dict__)
        try:
            return self.description.format(**obj_items)
        except KeyError:
            return self.description

    @property
    def description(self):
        return self.name

    @property
    def action_name(self):
        return self.name.format(**self.__dict__)

    def should_update(self, other: "ModelAction"):
        # some actions prefer an update over keeping all changes in the todo
        return False


class NoOp(ModelAction):
    influences = ModelInfluence.NONE
    name = "no action"

    def __init__(self, model):
        self.model = model

    def execute(self):
        pass

    def undo(self):
        pass

    def redo(self):
        pass


@dataclass
class ActionHistory:
    undo_stack: List[ModelAction] = field(default_factory=list)
    redo_stack: List[ModelAction] = field(default_factory=list)
    max_stack = 100

    def execute(self, action: ModelAction):
        action.execute()
        action_string = str(action)
        action_string = action_string.replace("\n", "\n    ")
        debug(f"Action Executed - {action.__class__.__name__}: {action_string}")
        if len(self.undo_stack) > 0 and action.should_update(self.undo_stack[-1]):
            self.undo_stack.pop()
        self.undo_stack.append(action)
        if len(self.undo_stack) > self.max_stack:
            self.undo_stack.pop(0)
        self.redo_stack = []

    def undo(self) -> ModelAction:
        if len(self.undo_stack) > 0:
            action = self.undo_stack[-1]
            action.undo()
            self.undo_stack.pop()
            self.redo_stack.append(action)
            return action
        else:
            return NoOp(None)

    def redo(self) -> ModelAction:
        if len(self.redo_stack) > 0:
            action = self.redo_stack[-1]
            action.redo()
            self.redo_stack.pop()
            self.undo_stack.append(action)
            return action
        else:
            return NoOp(None)

    def remove_actions(self, start, length=1):
        """
        Undo the last 'start' actions, skipe 'length'
        redo actions and then redo the residual actions.
        """
        current_undo = len(self.redo_stack)
        changed_actions = []
        skipped_action = []
        redone_action = []
        if len(self.undo_stack) < start:
            raise ValueError(f"Only {len(self.undo_stack)} items on the stack to undo.")
        for i in range(start):
            changed_actions.append(self.undo())
        for i in range(length):
            skipped_action.append(self.redo_stack.pop())
        while len(self.redo_stack) > current_undo:
            # redo actions that where not on the redo stack before the start
            try:
                redone_action.append(self.redo())
            except Exception as e:
                # there was an error during a redo, hald the process and
                # revert to previous state
                for _ in reversed(redone_action):
                    self.undo()
                skipped_action.reverse()
                self.redo_stack += skipped_action
                while len(self.redo_stack) > current_undo:
                    self.redo()
                raise e
        return ActionBlock(changed_actions[-1].model, changed_actions)

    def clear(self):
        self.undo_stack = []
        self.redo_stack = []


class ActionBlock(ModelAction):
    influences = ModelInfluence.NONE
    name = "action block"

    def __init__(self, model, actions: List[ModelAction]):
        self.model = model
        for action in actions:
            self.influences = self.influences | action.influences
        self.actions = actions

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
        return "|".join(map(str, self.actions))


class UpdateModelScript(ModelAction):
    influences = ModelInfluence.SCRIPT

    def __init__(self, model, text):
        """
        Generate a diff that can be applied to the current script
        to get the new one. This allows the action to be performed
        even if a previous change is undone.
        """
        self.model = model
        self.old_text = self.model.get_script().replace("\r\n", "\n").replace("\r", "\n")
        debug(f"UpdateModelScript Init\nold text:\n    {self.old_text!r}\nnew text:\n    {text!r}")
        self.diff = self.generate_diff(self.old_text, text.replace("\r\n", "\n").replace("\r", "\n"))

    def execute(self):
        # Replace model script with new text, script as new text (toggles)
        self.old_text = self.model.get_script().replace("\r\n", "\n").replace("\r", "\n")
        new_text = self.apply_diff(self.old_text)
        self.model.set_script(new_text)

    def generate_diff(self, old, new):
        """
        Generate a list of changes with their line numbers
        as it is not clear how to change a script if more
        """
        d = difflib.Differ()
        diff = list(d.compare(old.splitlines(keepends=True), new.splitlines(keepends=True)))
        debug(f"generate_diff:\n{diff}")
        cdiff = []
        old_indx = 0
        while len(diff) > 0:
            ndiff = diff.pop(0)
            if ndiff.startswith(" "):
                old_indx += 1
                if len(diff) > 0 and diff[0][0] == "+":
                    nextcdiff = [old_indx, [], []]
                    while len(diff) > 0 and diff[0][0] == "+":
                        nextcdiff[2].append(diff.pop(0)[2:])
                    cdiff.append(nextcdiff)
            elif ndiff.startswith("-"):
                nextcdiff = [old_indx, [ndiff[2:]], []]
                old_indx += 1
                # collect old lines to remove
                while len(diff) > 0 and diff[0][0] == "-":
                    ndiff = diff.pop(0)
                    nextcdiff[1].append(ndiff[2:])
                    old_indx += 1
                while len(diff) > 0 and diff[0][0] in ["?", "+"]:
                    if diff[0][0] == "?":
                        ndiff = diff.pop(0)
                    else:
                        # collect new lines to insert
                        nextcdiff[2].append(diff.pop(0)[2:])
                cdiff.append(nextcdiff)
            elif old_indx == 0 and ndiff.startswith("+"):
                nextcdiff = [old_indx, [], [ndiff[2:]]]
                while len(diff) > 0 and diff[0][0] == "+":
                    nextcdiff[2].append(diff.pop(0)[2:])
                cdiff.append(nextcdiff)
            else:
                debug(f"ignored diff line {old_indx}: {ndiff}")
        return cdiff

    def apply_diff(self, old):
        old_list = old.splitlines(keepends=True)
        new_list = []
        current_index = 0
        for idx, diff_from, diff_to in self.diff:
            if old_list[idx : idx + len(diff_from)] == diff_from:
                new_list += old_list[current_index:idx]
                new_list += diff_to
                current_index = idx + len(diff_from)
            else:
                raise ValueError(
                    f"The script hase been modefined in a way "
                    f"that this change can no longer be applied: "
                    f"line {idx} was {diff_from} but now is {old_list[idx:idx+len(diff_from)]}."
                )
        new_list += old_list[current_index:]
        return "".join(new_list)

    def format_diff(self):
        output = "\n"
        for idx, diff_from, diff_to in self.diff:
            output += f"line {idx}:\n    "
            output += "    ".join(["-" + ln for ln in diff_from])
            if len(diff_from) > 0:
                output += "    "
            output += "    ".join(["+" + ln for ln in diff_to])
        return output[:-1].replace("\n", "\n    ")

    def undo(self):
        self.model.set_script(self.old_text)

    @property
    def name(self):
        name = "edit script ("
        name += "|".join([str(d[0]) for d in self.diff])
        name += ")"
        return name

    @property
    def description(self):
        return self.name + ": " + self.format_diff()


class UpdateSolverOptoins(ModelAction):
    influences = ModelInfluence.OPTIONS
    name = "optimizer options"
    description = "set optimizer parameters: {new_values}"

    def __init__(self, model, optimizer: GenxOptimizer, new_values: dict):
        self.model = model
        self.optimizer = optimizer
        self.new_values = new_values.copy()
        self.old_values = {}
        self.combined_options = self.model.solver_parameters | self.optimizer.opt
        for key in new_values.keys():
            self.old_values[key] = getattr(self.combined_options, key, None)

    def execute(self):
        self.combined_options = self.model.solver_parameters | self.optimizer.opt
        for key in self.new_values.keys():
            self.old_values[key] = getattr(self.combined_options, key, None)
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
    name = "plot settings"

    def __init__(self, model, indices: List[int], sim_par: dict, data_par: dict):
        self.model = model
        self.indices = indices
        self.new_sim_par = sim_par.copy()
        self.new_data_par = data_par.copy()
        self.old_sim_pars = {}
        self.old_data_pars = {}

    def execute(self):
        # Update parameters for each dataset index
        for index in self.indices:
            di = self.model.data[index]
            self.old_sim_pars[index] = di.get_sim_plot_items()
            self.old_data_pars[index] = di.get_data_plot_items()
            di.set_sim_plot_items(self.new_sim_par)
            di.set_data_plot_items(self.new_data_par)

    def undo(self):
        # Reset previous settings
        for index in self.indices:
            di = self.model.data[index]
            di.set_sim_plot_items(self.old_sim_pars[index])
            di.set_data_plot_items(self.old_data_pars[index])


class UpdateColorCycle(ModelAction):
    influences = ModelInfluence.DATA
    name = "color cycle"

    def __init__(self, model, source):
        self.model = model
        self.new_source = source
        self.old_source = None
        self.old_sim_colors = {}
        self.old_data_colors = {}

    def execute(self):
        # Update colors from each dataset index
        for i, di in enumerate(self.model.data):
            self.old_sim_colors[i] = di.sim_color
            self.old_data_colors[i] = di.data_color
        self.old_source = self.model.data.color_source
        self.model.data.update_color_cycle(self.new_source)

    def undo(self):
        # Reset previous settings and colors
        self.model.data.update_color_cycle(self.old_source)
        for i, di in enumerate(self.model.data):
            di.sim_color = self.old_sim_colors[i]
            di.data_color = self.old_data_colors[i]


class UpdateParams(ModelAction):
    influences = ModelInfluence.PARAM
    name = "parameter values"
    description = "set parameter values\n{param_names}\nfrom\n{old_values}\nto\n{new_values}"

    def __init__(self, model, new_values):
        self.model = model
        self.new_values = list(new_values)
        self.param_names = self.model.parameters.get_fit_pars()[1]
        self.old_values = self.model.parameters.get_value_pars()

    def execute(self):
        param_names = self.model.parameters.get_fit_pars()[1]
        if self.param_names != param_names:
            raise ValueError("Current fit parameters do not correspond to setting during action init")
        self.old_values = self.model.parameters.get_value_pars()
        self.model.parameters.set_value_pars(self.new_values)

    def undo(self):
        self.model.parameters.set_value_pars(self.old_values)


class UpdateParamValue(ModelAction):
    influences = ModelInfluence.PARAM
    name = "update parameter [{param_name}]"
    description = "parameter[{param_name},{col}]: {old_value!r} -> {new_value!r}"

    def __init__(self, model, row, col, new_value):
        self.model = model
        self.col = col
        self.new_value = new_value
        if col != 0:
            self.param_name = self.model.parameters.get_names()[row]
        else:
            self.param_name = row
        self.old_value = self.model.parameters.get_value(row, col)

    def execute(self):
        if self.col != 0:
            row = self.model.parameters.get_names().index(self.param_name)
        else:
            row = self.param_name
        self.old_value = self.model.parameters.get_value(row, self.col)
        self.model.parameters.set_value(row, self.col, self.new_value)

    def undo(self):
        if self.col != 0:
            row = self.model.parameters.get_names().index(self.param_name)
        else:
            row = self.param_name
        self.model.parameters.set_value(row, self.col, self.old_value)

    def should_update(self, other):
        # if the same parameter is changed multiple time, just keep one change
        if type(other) != type(self) or self.param_name != getattr(other, "param_name", None):
            return False
        else:
            self.old_value = other.old_value
            return True


class MoveParam(ModelAction):
    influences = ModelInfluence.PARAM
    name = "move parameter"

    def __init__(self, model, row: int, step: int):
        self.model = model
        self.step = step
        name = self.model.parameters.get_names()[row]
        if name.strip() == "":
            self.param_name = row
        else:
            self.param_name = name

    def execute(self):
        if type(self.param_name) is int:
            row = self.param_name
        else:
            row = self.model.parameters.get_names().index(self.param_name)
        self.model.parameters.move_row(row, self.step)

    def undo(self):
        if type(self.param_name) is int:
            row = self.param_name + self.step
        else:
            row = self.model.parameters.get_names().index(self.param_name)
        self.model.parameters.move_row(row, -self.step)


class DeleteParams(ModelAction):
    influences = ModelInfluence.PARAM
    name = "delete parameters"
    description = "deleted {param_names}"

    def __init__(self, model, rows):
        self.model = model
        names = self.model.parameters.get_names()
        self.param_names = [names[ri] for ri in rows if ri < len(names) and names[ri].strip() != ""]
        self.empty_rows = [ri for ri in rows if ri < len(names) and names[ri].strip() == ""]
        self.old_data = []

    def execute(self):
        names = self.model.parameters.get_names()
        rows = [names.index(ni) for ni in self.param_names]
        # make sure we only delete empty rows if they are still empty
        rows += [row for row in self.empty_rows if names[row].strip() == ""]
        old_rows = self.model.parameters.data
        self.old_data = [(ri, old_rows[ri]) for ri in rows]
        self.model.parameters.delete_rows(rows)

    def undo(self):
        for ri, di in self.old_data:
            self.model.parameters.data.insert(ri, di)


class InsertParam(ModelAction):
    influences = ModelInfluence.PARAM
    name = "insert parameter"

    def __init__(self, model, row):
        self.model = model
        if row >= self.model.parameters.get_len_rows():
            self.param_name = self.model.parameters.get_len_rows()
        else:
            name = self.model.parameters.get_names()[row]
            if name.strip() == "":
                self.param_name = row
            else:
                self.param_name = name

    def execute(self):
        if type(self.param_name) is int:
            row = self.param_name
        else:
            row = self.model.parameters.get_names().index(self.param_name)
        self.insert_index = row
        self.model.parameters.insert_row(row)

    def undo(self):
        self.model.parameters.delete_rows([self.insert_index])


class SortAndGroupParams(ModelAction):
    influences = ModelInfluence.PARAM
    name = "sort parameters"
    description = "sort by {sort_params}"

    def __init__(self, model, sort_params: SortSplitItem):
        self.model = model
        self.sort_params = sort_params
        self.old_parameters = []

    def execute(self):
        self.old_parameters = self.model.parameters.data.copy()
        if not self.model.compiled:
            self.model.compile_script()
        self.model.parameters.sort_rows(self.model, self.sort_params)
        self.model.parameters.group_rows(self.model, self.sort_params)
        self.model.parameters.strip()

    def undo(self):
        self.model.parameters.data = self.old_parameters
        self.old_parameters = []


class ExchangeModel(ModelAction):
    influences = ModelInfluence.PARAM | ModelInfluence.DATA | ModelInfluence.SCRIPT
    name = "exchange model"
    description = "replace model, data and parameter grid from internal storage memory"

    def __init__(self, model, new_model: Model):
        self.model = model
        self.new_state = new_model.__getstate__().copy()
        self.new_state["filename"] = self.model.filename
        self.old_state = model.__getstate__().copy()

    def execute(self):
        self.model.__setstate__(self.new_state)

    def undo(self):
        self.model.__setstate__(self.old_state)
