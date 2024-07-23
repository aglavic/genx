"""
Define base classes that can be used in models for prameterization. This classes are model independent.
"""

import sys
import inspect
import ast
from enum import Enum
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


class ModelParamMeta(type):
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
        # adding documentation for class into module doc-string
        # this leads to much cleaner code in model module as beginning of file only contains
        # a general introduction and each dataclass gets documented where it's defined.
        if cls.__doc__ is not None:
            doc = inspect.cleandoc(cls.__doc__)
            module = inspect.getmodule(cls)
            docout = f'{cls.__name__}\n'
            docout += '~'*len(cls.__name__)+'\n'
            sig = eval('inspect.signature(cls)', module.__dict__, {'cls': cls, 'inspect':inspect})
            units = getattr(cls, 'Units', {})
            call_params = []
            for ni, si in sig.parameters.items():
                if ni=='user_kwds':
                    continue
                if ni in units and units[ni].strip()!='':
                    call_params.append(f'{ni}={si.default} {units[ni]}')
                else:
                    call_params.append(f'{ni}={si.default}')
            call_string = f'{cls.__name__}({", ".join(call_params)})'
            docout += f'``{call_string}``\n\n'
            docout += doc
            module.__doc__+=docout+'\n\n'

        return cls


class ModelParamBase(metaclass=ModelParamMeta):
    """
    Base class for all reflectometry dataclass objects. It uses the ReflMeta class to
    allow arbitrary ununsed keyword arguments (for switching models) and stores
    the call strings used when creating a class instance to allow re-creation
    within the GUI.

    In addition to the dataclass attributes a subclass can define a Units
    dictionary that can be used in GUI and documentation to annotate the
    physical unit used for the parameter.
    """
    Units = {}

    def __post_init__(self):
        self._generate_setters()
        self._generate_getters()
        # record the line number in the script when executed by exec, see model.Model.compile_script
        self._lno_context = inspect.stack()[2].lineno-1
        self._orig_params = {}
        self._ca = {}
        for fi in fields(self):
            self._orig_params[fi.name] = getattr(self, fi.name)
            if inspect.isclass(fi.type) and not isinstance(getattr(self, fi.name), fi.type):
                # convert parameter to correct type
                setattr(self, fi.name, fi.type(getattr(self, fi.name)))

    def _extract_callpars(self, source):
        # try to extract call arguments for the keywords
        context = [source.splitlines()[self._lno_context]]
        call_lines = ''.join(context)
        ca = {}
        try:
            p = ast.parse(call_lines)
            kws = p.body[0].value.keywords
            for kw in kws:
                if hasattr(ast, 'get_source_segment'):
                    ca[kw.arg] = ast.get_source_segment(call_lines, kw.value)
                else:
                    ca[kw.arg] = call_lines[kw.value.col_offset:kw.value.end_col_offset]
        except Exception:
            pass
        for fi in fields(self):
            if not fi.name in ca:
                ca[fi.name] = str(self._orig_params[fi.name])
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

            set_func = ModelParamBase._get_setter(self, par)
            setattr(self, set_func.__name__, set_func)

            if fi.type is complex:
                set_real_func = ModelParamBase._get_real_setter(self, par)
                setattr(self, set_real_func.__name__, set_real_func)

                set_imag_func = ModelParamBase._get_imag_setter(self, par)
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
            get_func = ModelParamBase._get_getter(self, par)
            setattr(self, get_func.__name__, get_func)

            if fi.type is complex:
                get_real_func = ModelParamBase._get_real_getter(self, par)
                setattr(self, get_real_func.__name__, get_real_func)

                get_imag_func = ModelParamBase._get_imag_getter(self, par)
                setattr(self, get_imag_func.__name__, get_imag_func)

    def _repr_call(self):
        output = self.__class__.__name__+'('
        for fi in fields(self):
            call = self._ca.get(fi.name, None)
            if call is not None:
                output += f'{fi.name}={call}, '
            else:
                output += f'{fi.name}={getattr(self, fi.name)!r}, '
        return output[:-2]+')'

class AltStrEnum(str, Enum):
    """
    A string based Enum that allows to create equivalent values from different strings.
    Therefore, the user can select various values but the code compares just to one item.

    Any alternative attributes have to start with "alternate??_" and end with `name`, where
    `name` is the original attribute to be doublicated.
    """

    def __init__(self, string):
        if self.name.startswith('alternate'):
            self.alt_for = self.name.split('_', 1)[1]
        else:
            self.alt_for = None

    def __eq__(self, other):
        if getattr(other, 'alt_for', None) is not None:
            return self.name == other.alt_for or str.__eq__(self, other)
        elif getattr(self, 'alt_for', None) is not None:
            return self.alt_for==other.name or str.__eq__(self, other)
        else:
            return str.__eq__(self, other)

    def __neq__(self, other):
        return not self.__eq__(other)