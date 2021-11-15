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

    @abstractmethod
    def redo(self):
        ...

class NoOp(ModelAction):
    influence = ModelInfluence.NONE
    name = 'no action'
    def __init__(self, model): self.model = model
    def execute(self): pass
    def undo(self): pass
    def redo(self): pass

@dataclass
class ActionHistory:
    undo_stack: List[ModelAction] = field(default_factory=list)
    redo_stack: List[ModelAction] = field(default_factory=list)

    def execute(self, action: ModelAction):
        action.execute()
        self.undo_stack.append(action)
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
        self.text=text

    def execute(self):
        # Replace model script with new text, store previous script as new text (toggles)
        old_text=self.model.get_script()
        self.model.set_script(self.text)
        self.text=old_text

    undo=execute
    redo=execute

