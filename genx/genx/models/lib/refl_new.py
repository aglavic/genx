'''
Module to provide the Layer - Stack - Sample classes to build a sample for reflectivity modelling.
The new implementation is based on python dataclasses.

Classes:
ReflFunction - A function class that can be used as a parameter in the other classes.
is_reflfunction - Funtion that checks if an object belongs to the class ReflFunction
ReflBase - Base class for all the physical classes.
LayerBase - Base Layer class.
StackBase - Base Stack class.
SampleBase - Base Sample class.
InstrumentBase - Base Instrument class.

A model should derive its own data classes as base classe of the above like:
::
    @dataclass
    class Instrument(ReflBase):
        wavelength: float = 0.0

Choices should use appropriate Enum to simplify comparison. An example would be:
::
    class Polarization(str, enum.Enum):
        up_up = 'uu'
        down_down = 'dd'
        up_down = 'ud'
        down_up = 'du'

    @dataclass
    class Instrument(ReflBase):
        pol: Polarization = Polarization.up_up
'''
import sys
import inspect
import ast
from typing import List, Optional
# noinspection PyUnresolvedReferences
from dataclasses import (dataclass, _FIELD, _FIELD_INITVAR, _FIELDS, _HAS_DEFAULT_FACTORY, _POST_INIT_NAME, MISSING,
                         _create_fn, _field_init, _init_param, _set_new_attribute, field, fields, )


# change of signature introduced in python 3.10.1
if sys.version_info>=(3, 10, 1):
    _field_init_real = _field_init


    def _field_init(f, frozen, locals, self_name):
        return _field_init_real(f, frozen, locals, self_name, False)


def _custom_init_fn(fieldsarg, frozen, has_post_init, self_name, globals):
    """
    _init_fn from dataclasses adapted to accept additional keywords.
    See dataclasses source for comments on code.
    """
    seen_default = False
    for f in fieldsarg:
        if f.init:
            if not (f.default is MISSING and f.default_factory is MISSING):
                seen_default = True
            elif seen_default:
                raise TypeError(f"non-default argument {f.name!r} " "follows default argument")

    locals = {f"_type_{f.name}": f.type for f in fieldsarg}
    locals.update({"MISSING": MISSING, "_HAS_DEFAULT_FACTORY": _HAS_DEFAULT_FACTORY})

    body_lines = []
    for f in fieldsarg:
        line = _field_init(f, frozen, locals, self_name)
        if line:
            body_lines.append(line)

    if has_post_init:
        params_str = ",".join(f.name for f in fieldsarg if f._field_type is _FIELD_INITVAR)
        body_lines.append(f"{self_name}.{_POST_INIT_NAME}({params_str})")

    # processing of additional user keyword arguments
    body_lines += ["setattr(self, '_unused_kwds', user_kwds)"]

    return _create_fn(
        "__init__",
        [self_name, '*']+[_init_param(f) for f in fieldsarg if f.init]+["**user_kwds"],
        body_lines,
        locals=locals,
        globals=globals,
        return_type=None,
        )


class ReflMeta(type):
    """
    Metaclass that overwrites a classes __init__ method to make sure dataclass is generated without
    own __init__ method and thus does ignore additional keyword arguments.
    """

    def __new__(meta, name, bases, attrs):
        cls = super().__new__(meta, name, bases, attrs)

        if "__annotations__" in attrs and len([k for k in attrs["__annotations__"].keys() if not k.startswith("_")])>0:
            as_dataclass = dataclass(cls)
            fieldsarg = getattr(as_dataclass, _FIELDS)

            # Generate custom __init__ method that allows arbitrary extra keyword arguments
            has_post_init = hasattr(as_dataclass, _POST_INIT_NAME)
            # Include InitVars and regular fields (so, not ClassVars).
            flds = [f for f in fieldsarg.values() if f._field_type in (_FIELD, _FIELD_INITVAR)]
            init_fun = _custom_init_fn(flds, False, has_post_init, "self", globals())
            setattr(cls, '__init__', init_fun)
        return cls


class ReflBase(metaclass=ReflMeta):
    """
    Base class for all reflectometry dataclass objects. It uses the ReflMeta class to
    allow arbitrary ununsed keyword arguments (for switching models) and stores
    the call strings used in when creating a class instance to allow re-creation
    within the GUI.
    """

    List = List
    Optional = Optional
    field = field

    def __post_init__(self):
        self._generate_setters()
        self._generate_getters()
        call_lines = ''.join(inspect.stack()[2].code_context)
        # try to extract call arguments for the keywords
        ca = {}
        try:
            p = ast.parse(call_lines)
            kws = p.body[0].value.keywords
            for kw in kws:
                ca[kw.arg] = call_lines[kw.value.col_offset:kw.end_col_offset]
        except Exception:
            pass
        for fi in fields(self):
            if fi.type in [str, int, float, complex] and not isinstance(getattr(self, fi.name), fi.type):
                # convert parameter to correct type
                setattr(self, fi.name, fi.type(getattr(self, fi.name)))
            if fi.name in ca:
                setattr(self, '_source_'+fi.name, ca[fi.name])
        self._ca = ca

    @staticmethod
    def _get_setter(object, par):
        def set_func(value):
            setattr(object, par, value)

        set_func.__name__ = 'set'+par.capitalize()
        return set_func

    @staticmethod
    def _get_real_setter(object, par):
        def set_func(value):
            setattr(object, par, value+getattr(object, par).imag*1J)

        set_func.__name__ = 'set'+par.capitalize()+'real'
        return set_func

    @staticmethod
    def _get_imag_setter(object, par):
        def set_func(value):
            setattr(object, par, value*1J+getattr(object, par).real)

        set_func.__name__ = 'set'+par.capitalize()+'imag'
        return set_func

    def _generate_setters(self):
        for fi in fields(self):
            par = str(fi.name)

            set_func = ReflBase._get_setter(self, par)
            setattr(self, set_func.__name__, set_func)

            if fi.type is complex:
                set_real_func = ReflBase._get_real_setter(self, par)
                setattr(self, set_real_func.__name__, set_real_func)

                set_imag_func = ReflBase._get_imag_setter(self, par)
                setattr(self, set_imag_func.__name__, set_imag_func)

    @staticmethod
    def _get_getter(object, par):
        def get_func():
            return getattr(object, par)

        get_func.__name__ = 'get'+par.capitalize()
        return get_func

    @staticmethod
    def _get_real_getter(object, par):
        def get_func():
            return getattr(object, par).real

        get_func.__name__ = 'get'+par.capitalize()+'real'
        return get_func

    @staticmethod
    def _get_imag_getter(object, par):
        def get_func():
            return getattr(object, par).imag

        get_func.__name__ = 'get'+par.capitalize()+'imag'
        return get_func

    def _generate_getters(self):
        for fi in fields(self):
            par = str(fi.name)
            get_func = ReflBase._get_getter(self, par)
            setattr(self, get_func.__name__, get_func)

            if fi.type is complex:
                get_real_func = ReflBase._get_real_getter(self, par)
                setattr(self, get_real_func.__name__, get_real_func)

                get_imag_func = ReflBase._get_imag_getter(self, par)
                setattr(self, get_imag_func.__name__, get_imag_func)

    def _repr_call(self):
        output = self.__class__.__name__+'('
        for fi in fields(self):
            call = getattr(self, '_source_'+fi.name, None)
            if call is not None:
                output += f'{fi.name}={call}, '
            else:
                output += f'{fi.name}={getattr(self, fi.name)!r}, '
        return output[:-2]+')'

@dataclass
class StackBase(ReflBase):
    Layers: List[ReflBase] = field(default_factory=list)
    Repetitions: int = 1

    def resolveLayerParameter(self, name):
        return [getattr(li, name) for li in self.Layers]*self.Repetitions

@dataclass
class SampleBase(ReflBase):
    Stacks: List[StackBase] = field(default_factory=list)
    Ambient: ReflBase = None
    Substrate: ReflBase = None

    def _resolve_parameter(self, obj, key):
        return getattr(obj, key)

    def resolveLayerParameters(self):
        par = {}
        for fi in fields(self.Substrate):
            par[fi.name] = []
            par[fi.name]=[self._resolve_parameter(self.Substrate, fi.name)]
            for stack in self.Stacks:
                par[fi.name]+=stack.resolveLayerParameter(fi.name)
            par[fi.name].append(self._resolve_parameter(self.Ambient, fi.name))
        return par
