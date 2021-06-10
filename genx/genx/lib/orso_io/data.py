"""
Define a class to hold all data for ORSO data files.
"""

import os
import json
import warnings
import numpy as np
import datetime
from collections import OrderedDict
from . import ORSOStandardError, ORSOStandardWarning

try:
    # read header schema for validation
    import jsonschema

    SCHEMA=json.load(
        open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reduced_data_header_schema.json'), 'r'))
    validator=jsonschema.Draft7Validator(SCHEMA)
except Exception as err:
    warnings.warn("Can't validate headers: %s"%str(err), UserWarning)
    validator=None

def validate_header(header, strict=True):
    if validator is None:
        return True
    valid=validator.is_valid(header)
    if valid:
        return True
    text='Header does not comply with ORSO standard:\n'
    for err in validator.iter_errors(header):
        text+='  path /%s: %s\n'%('/'.join(err.absolute_path), err.message)
    if strict:
        raise ORSOStandardError(text)
    else:
        warnings.warn(text, ORSOStandardWarning)
        return False

def get_schema_from_path(node_path):
    node=SCHEMA
    for pi in node_path:
        node=node[pi]
    return node

def get_schema_child_nodes(node_path):
    node=get_schema_from_path(node_path)
    items=[(key, value, node_path+[key]) for key, value in node['properties'].items() if key in node['required']]
    output={}
    for key, value, npath in items:
        # make sure the item is not a reference or has multiple options
        while not 'type' in value:
            if 'anyOf' in value:
                # if there are several choices, use the first
                value=value['anyOf'][0]
                npath.append(0)
            elif '$ref' in value:
                npath=value['$ref'][2:].split('/')
                value=get_schema_from_path(npath)
            else:
                raise ValueError("Could not find correct item to referenc, path="+'/'.join(node_path))
        if value['type']=='object':
            output[key]=get_schema_child_nodes(npath)
        elif value['type']=='string':
            if 'enum' in value:
                output[key]=value['enum'][0]
            elif 'format' in value and value['format']=='date-time':
                output[key]=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            else:
                output[key]='unspecified'
        elif value['type']=='number':
            output[key]=0.
    return output

class DataColumn(np.ndarray):
    name=None
    unit=None
    dimension=None

    def __new__(cls, data, name='unset', unit=None, dimension=None, min_size=None, **opts):
        adata=np.array(data, dtype=np.float64, **opts)
        if adata.shape[0]<min_size:
            appends=np.zeros(min_size-adata.shape[0], dtype=np.float64)
            adata=np.hstack([adata, appends])
        out=adata.view(cls)
        out.name=name
        out.unit=unit
        out.dimension=dimension
        return out

    def __repr__(self):
        output=np.ndarray.__repr__(self)[:-1]
        if self.name is not None:
            output+=', name="%s"'%self.name
        if self.unit is not None:
            output+=', unit="%s"'%self.unit
        if self.dimension is not None:
            output+=', dimension="%s"'%self.dimension
        return output+')'

class ORSOData():
    """
    Representation of a reflectometry dataset with metadata.
    According to ORSO standard definition.
    """

    def __init__(self, header, data, strict=True):
        if len(data)<3:
            raise ORSOStandardError("Need at least 3 data columns, Qz, R, dR")
        # make sure the data consists of DataColumn arrays with same length
        maxlen=max([len(di) for di in data])
        self._data=[DataColumn(di, min_size=maxlen) for di in data]
        if 'columns' in header and len(header['columns'])==len(data):
            for i, ci in enumerate(header['columns']):
                try:
                    self._data[i].name=ci['name']
                except KeyError:
                    raise ORSOStandardError("Each column needs a name")
                if 'unit' in ci:
                    self._data[i].unit=ci['unit']
                if 'dimension' in ci:
                    self._data[i].dimension=ci['dimension']
        else:
            raise ORSOStandardError("Header requires 'columns' field entry for every data column")
        validate_header(header, strict)
        self.header=OrderedDict(header)
        self.header.move_to_end('columns')

    @classmethod
    def minimal_header(cls):
        # return a minimal OrderedDict that fulfills the SCHEMA
        output=OrderedDict(get_schema_child_nodes([]))
        output['columns']=[{'name': 'Qz'}, {'name': 'R'}, {'name': 'sR'}]
        return output

    @property
    def x(self):
        return self._data[0]

    @property
    def y(self):
        return self._data[1]

    @property
    def dy(self):
        return self._data[2]

    @property
    def dx(self):
        if len(self._data)>3 and self._data[3].name=='sQz':
            return self._data[3]
        else:
            return None

    @property
    def columns(self):
        return [di.name for di in self._data]

    @property
    def data(self):
        return np.vstack(self._data)

    @property
    def units(self):
        return [di.unit if di.unit is not None else '1' for di in self._data]

    def __getitem__(self, item):
        return self._data[item]

    def __repr__(self):
        output='ORSOData('
        output+='columns=%s,\n'%repr(self.columns)
        output+='         units=%s'%repr(self.units)
        return output+') '+self.name

    def __eq__(self, other):
        output=(self.header==other.header)
        for i, col in enumerate(self):
            output&=(col.shape==other[i].shape)
            if not output:
                return False
            output&=bool((col==other[i]).all())
        return output

    @property
    def name(self):
        if 'data_set' in self.header:
            return self.header['data_set']
        else:
            return ''

    def plot(self, axes=None):
        if axes is None:
            from matplotlib import pylab
            axes=pylab.gca()
        if self.dx is None:
            axes.errorbar(self.x, self.y, yerr=self.dy, label=self.name)
        else:
            axes.errorbar(self.x, self.y, yerr=self.dy, xerr=self.dx, label=self.name)
        axes.yscale('log')
        axes.xlabel('%s (%s)'%(self.x.name, self.x.unit))
        axes.ylabel('%s (%s)'%(self.x.name, self.x.unit))
