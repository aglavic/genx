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