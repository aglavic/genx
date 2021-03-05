"""
Scripting interface for GenX for use in python scripting or Jupyter Notebooks.
"""

__all__=[]

import sys
from genx import models as _models
if not "models" in sys.modules:
    sys.modules["models"]=_models

from genx.model import Model
from genx.diffev import DiffEv
from genx import filehandling as io
_config=io.Config()

def load(fname, compile=True):
    model=Model()
    optimizer=DiffEv()
    io.load_file(fname, model, optimizer, _config)
    if compile:
        model.compile_script()
    optimizer.model=model
    return model, optimizer

def save(fname, model, optimizer):
    io.save_file(fname, model, optimizer, _config)
