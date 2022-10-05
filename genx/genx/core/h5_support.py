"""
Items used to support saving classes to HDF 5 files.
"""
import h5py
import pickle
from abc import ABC, abstractmethod
from typing import get_type_hints, List, Union
from inspect import isclass
from numpy import ndarray, void
from logging import debug, warning
from datetime import datetime

try:
    from typing import get_args, get_origin
except ImportError:
    def get_args(tp): return getattr(tp, '__args__', ())


    def get_origin(tp): return getattr(tp, '__extra__', None) or getattr(tp, '__origin__', None)


class H5Savable(ABC):

    # Defines minimum required methods for a class to be used upon saving
    @property
    @abstractmethod
    def h5group_name(self):
        """
        Attribute that stores the name of the group it will be supplied in the write_/read_h5group methods.
        The main reason to fix this in the H5Savable class is to make it more transparent, where this
        classes attributes will be saved in the hdf5 file.
        """

    @abstractmethod
    def write_h5group(self, group: h5py.Group):
        """ Save configuration to hdf5 group """

    @abstractmethod
    def read_h5group(self, group: h5py.Group):
        """ Configure object from hdf5 group """

    def h5_write_free_dict(self, group: h5py.Group, obj: dict):
        """ Help method to write an arbitrary dictionary of variable depth to hdf5 group """
        for key, value in obj.items():
            vtyp = type(value)
            if vtyp is dict:
                sub_group = group.create_group(key, track_order=True)
                self.h5_write_free_dict(sub_group, value)
                group[key].attrs['genx_type']='free_dict'.encode('ascii')
            elif any([issubclass(vtyp, typ) for typ in [float, int, ndarray, complex]]):
                group[key] = value
                group[key].attrs['genx_type']=vtyp.__name__.encode('ascii')
            elif isinstance(value, datetime):
                group[key] = value.isoformat().encode('ascii')
                group[key].attrs['genx_type']=vtyp.__name__.encode('ascii')
            elif issubclass(vtyp, str):
                group[key] = value.encode('utf-8')
                group[key].attrs['genx_type']=vtyp.__name__.encode('ascii')
            else:
                # if the type can't be handled like this, create a pickle string
                group[key] = void(pickle.dumps(value))
                group[key].attrs['genx_type']='dump'.encode('ascii')

    def h5_read_free_dict(self, output: dict, group: Union[h5py.Group, h5py.Dataset], item_path: List[str]):
        """
        Recursive read of meta data from hdf5 group
        """
        node = output
        for pathi in item_path[:-1]:
            # decent into sub-node, create if it does not exist
            if pathi in node:
                node = node[pathi]
            else:
                new_node = {}
                node[pathi] = new_node
                node = new_node
        if type(group) is h5py.Dataset:
            prev_typ=group.attrs.get('genx_type', None)
            value = group[()]
            vtyp = type(value)
            if issubclass(vtyp, float) or prev_typ=='float':
                value = float(value)
            elif issubclass(vtyp, int) or prev_typ=='int':
                value = int(value)
            elif issubclass(vtyp, complex) or prev_typ=='complex':
                value = complex(value)
            elif issubclass(vtyp, void) or prev_typ=='dump':
                value = pickle.loads(value.tobytes())
            elif vtyp is bytes:
                if prev_typ=='datetime':
                    try:
                        value = datetime.fromisoformat(value.decode('ascii'))
                    except AttributeError:
                        value = datetime.strptime(lue.decode('ascii'), "%Y-%m-%dT%H:%M:%S")
                else:
                    value = value.decode('utf-8')
            node[item_path[-1]] = value
            return
        for key in group:
            self.h5_read_free_dict(output, group[key], item_path+[key])


# noinspection PyAbstractClass
class H5HintedExport(H5Savable):
    """
    Base class that exports all its attributes that have a type hint to the hdf5 group using their name as key.

    This allows clean classes where parameters are automatically written correctly and type checked.
    """
    _export_ignore = []  # allows type hinted parameters that should not be exported
    _group_attr = {} # hdf attributes to be added to the group on export

    def init_defaults(self):
        """Allows the class to automatically initialize the exported parameters from their defaults"""
        for attr, typ in get_type_hints(self).items():
            if attr in self._export_ignore or attr.startswith('_'):
                continue
            if not hasattr(self.__class__, attr):
                continue
            default = getattr(self.__class__, attr)
            if hasattr(default, 'copy'):
                default = default.copy()  # for array, dict and lists, make sure a copy is used
            setattr(self, attr, default)

    def write_h5group(self, group: h5py.Group):
        """
        Uses this objects type hints to write attributes to hdf5 group.

        Two object types are treated specifically:
            -dict will be written in sub-group items
            -H5Savable derived classes will be written in sub-group
             using their write_h5group method and h5group_name attribute
        """
        for attr, typ in get_type_hints(self).items():
            if attr in self._export_ignore or attr.startswith('_'):
                continue
            value = getattr(self, attr)
            if typ is dict:
                # free dictionary, save every str, int, float type values and ignore rest
                self.h5_write_free_dict(group.create_group(attr, track_order=True), value)
            elif get_origin(typ) is dict:
                sub_group = group.create_group(attr)
                styp = get_args(typ)[1]
                for key, subval in value.items():
                    if styp is str:
                        sub_group[key] = subval.encode('utf-8')
                    else:
                        sub_group[key] = subval
            elif isclass(typ) and issubclass(typ, H5Savable):
                # the attribute is savable, create a sub-group
                sub_group = group.create_group(value.h5group_name)
                value.write_h5group(sub_group)
            else:
                if typ is str:
                    value = value.encode('utf-8')
                if getattr(value, 'nbytes', 0)>1024*128:
                    # this is an array with significant size, use compression
                    group.create_dataset(attr, data=value, dtype=value.dtype, chunks=True,
                                         compression="gzip", compression_opts=9, shuffle=True)
                else:
                    try:
                        group[attr] = value
                    except Exception:
                        warning(f'Error in writing value={value}', exc_info=True)
        for key, value in self._group_attr.items():
            group.attrs[key] = value

    def read_h5group(self, group: h5py.Group):
        """
        Uses this objects type hints to read attributes from hdf5 group.

        Two object types are treated specifically:
            -dict will be read from sub-group items
            -H5Savable derived classes will be read from sub-group
             using their read_h5group method and h5group_name attribute

        Values missing in file are handled according to two cases:
            1. If the class itself defines values for the parameters, these are taken as
               fallback and only debug info is provided
            2. Otherwise a warning message for missing parameter is given and attribute is not set at all
        """
        for attr, typ in get_type_hints(self).items():
            if attr in self._export_ignore or attr.startswith('_'):
                continue
            if hasattr(self.__class__, attr):
                fallback = getattr(self.__class__, attr)
            else:
                fallback = None
            write_attr = None
            err_mesg = ''

            if typ is dict:
                opt = {}
                try:
                    subgroup = group[attr]
                except KeyError:
                    warning(f'Did not find hdf5 group for {attr} in {group.name}')
                    continue
                self.h5_read_free_dict(opt, subgroup, [])
                write_attr = opt
            elif get_origin(typ) is dict:
                try:
                    subgroup = group[attr]
                except KeyError:
                    warning(f'Did not find hdf5 group for {attr} in {group.name}')
                    continue
                opt = {}
                vtyp = get_args(typ)[1]
                for key in subgroup:
                    if vtyp is ndarray:
                        opt[key] = subgroup[key][()]
                    elif vtyp is str:
                        value = subgroup[key][()]
                        if type(value) is str:
                            opt[key] = value
                        else:
                            try:
                                opt[key] = value.decode('utf-8')
                            except ValueError:
                                warning(f'Could not convert {attr}/{key} in {group.name} to string')
                                continue
                    else:
                        try:
                            opt[key] = vtyp(subgroup[key][()])
                        except ValueError:
                            warning(f'Could not convert {attr}/{key} in {group.name} to {vtyp.__name__}')
                            continue
                write_attr = opt
            elif isclass(typ) and issubclass(typ, H5Savable):
                # the attribute is loadable from sub-group
                obj = getattr(self, attr)
                try:
                    subgroup = group[obj.h5group_name]
                except KeyError:
                    warning(f'Did not find hdf5 group for {attr} in {group.name}')
                    continue
                obj.read_h5group(subgroup)
                continue
            else:
                try:
                    value = group[attr][()]
                except KeyError:  # resulting item did not exist
                    err_mesg = f'\n    Reason: The entry with name {attr} did not exist.'
                except AttributeError:  # result was not a dataset
                    err_mesg = f'\n    Reason: The item with name {attr} was not a Dataset but {type(group[attr])}.'
                else:
                    if get_origin(typ) is tuple:
                        valtyp = get_args(typ)
                        try:
                            write_attr = tuple(ti(vi) for ti, vi in zip(valtyp, value))
                        except ValueError:
                            err_mesg = f'\n    Reason: Could not convert value {value} to types {valtyp}.'
                    elif typ is ndarray:
                        write_attr = value
                    elif typ is str and type(value) is bytes:
                        try:
                            write_attr = value.decode('utf-8')
                        except UnicodeDecodeError:
                            err_mesg = f'\n    Reason: Could not convert bytes to string using "utf-8" encoding.'
                    else:
                        try:
                            write_attr = typ(value)
                        except ValueError:
                            err_mesg = f'\n    Reason: Could not convert value to type {typ.__name__}.'

            if write_attr is not None:
                setattr(self, attr, write_attr)
            elif fallback is not None:
                debug(f'Could not load {attr} from {group.name}, setting fallback value.'+err_mesg)
                if hasattr(fallback, 'copy'):
                    fallback = fallback.copy()  # for array, dict and lists, make sure a copy is used
                setattr(self, attr, fallback)
            else:
                warning(f'Could not load {attr} from {group.name} and no fallback was available, ignoring item.\n'
                        f'(This may result in strange behavior as the model will have values from before the load.)'+
                        err_mesg)
