"""
Dataclasses used as Qt signal payloads across gui_qt modules.
"""

from dataclasses import dataclass

from .. import data


@dataclass(frozen=True)
class UpdatePlotEvent:
    data: object | None = None
    fom_value: float | None = None
    fom_name: str | None = None
    fom_log: object | None = None
    update_fit: bool = False
    desc: str = ""
    model: object | None = None


@dataclass(frozen=True)
class UpdateParametersEvent:
    values: object
    new_best: bool
    population: object
    max_val: object
    min_val: object
    fitting: bool
    desc: str
    update_errors: bool = False
    permanent_change: bool = False


@dataclass(frozen=True)
class FittingEndedEvent:
    start_guess: object
    error_message: str | None
    values: object
    new_best: bool
    population: object
    max_val: object
    min_val: object
    fitting: bool
    desc: str


@dataclass(frozen=True)
class BatchNextEvent:
    last_index: int
    finished: bool


@dataclass(frozen=True)
class UpdatePlotSettingsEvent:
    indices: list[int]
    sim_par: dict
    data_par: dict


@dataclass(frozen=True)
class SetParameterValueEvent:
    row: int
    col: int
    value: object


@dataclass(frozen=True)
class MoveParameterEvent:
    row: int
    step: int


@dataclass
class DataListEvent:
    data: data.DataList
    data_changed: bool = True
    new_data: bool = False
    new_model: bool = False
    description: str = ""
    data_moved: bool = False
    position: int | list[int] | None = None
    up: bool = False
    deleted: bool = False
    name_change: bool = False
