"""
File reader and writer for ORSO text file standard .ort
"""
import re
import numpy as np
import json
from datetime import datetime
from collections import OrderedDict
from . import ORSOIOError
from .data import ORSOData

# for keeping datetime as a string (schema validation)
def yaml_timestamp_constructor(loader, node):
    return str(node.value)

def parse_yaml(header):
    import yaml
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    class StringDatetimeLoader(Loader):
        pass

    StringDatetimeLoader.add_constructor(u'tag:yaml.org,2002:timestamp', yaml_timestamp_constructor)

    return yaml.load(header, Loader=StringDatetimeLoader)

# in case we want to represent datetime in python
# def date_hook(json_dict):
#     for (key, value) in json_dict.items():
#         try:
#             json_dict[key] = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
#         except:
#             pass
#     return json_dict

def parse_json(header):
    return json.loads(header)  # , object_hook=date_hook)

def read_data(text_data):
    # read the data from the text, faster then numpy loadtxt with StringIO
    data=[li.split() for li in text_data.strip().splitlines() if not li.startswith('#')]
    return np.array(data, dtype=float).T

def recursive_update(main, sub):
    # create a dictionary that replaces every item from main with sub, if it exists
    # includes sub-key updates
    output=main.copy()
    for key, value in sub.items():
        if not key in output:
            output[key]=value
        elif type(value) is dict:
            output[key]=recursive_update(main[key], value)
        else:
            output[key]=value
    return output

def read_file(fname):
    with open(fname, 'r') as fh:
        # check if this is the right file type
        l1=fh.readline()
        if not l1.lower().startswith('# orso'):
            raise ORSOIOError('Not an ORSO reflectivity text file, wrong header')
        # find end of comment block (first line not starting with #
        text=fh.read()
    ftype_info=list(map(str.strip, l1.split('|')))
    data_start=re.search(r'\n[^#]', text).start()
    header=text[:data_start-1].rsplit('\n', 1)[0]  # remove last header line that only contains column info
    header=header[2:].replace('\n# ', '\n')  # remove header comment to make the text valid yaml/json
    header_encoding=ftype_info[2].lower().split()[0]
    if header_encoding=='yaml':
        main_header=parse_yaml(header)
        ds_string='\n# data_set:'
    elif header_encoding=='json':
        main_header=parse_json(header)
        ds_string='\n# {\n#     "data_set":'
    else:
        raise ORSOIOError('Unknown header encoding "%s"'%header_encoding)

    # implement possibility of more then one data block:
    if ds_string in text:
        split_indices=[match.start()+data_start for match in re.finditer(ds_string, text[data_start:])]+[-1]
        output=[ORSOData(main_header, read_data(text[data_start:split_indices[0]]), strict=False)]
        for i, si in enumerate(split_indices[:-1]):
            sub_header_length=re.search(r'\n[^#]', text[si:]).start()
            data=read_data(text[si+sub_header_length:split_indices[i+1]])
            sub_header_text=text[si+2:si+sub_header_length].rsplit('\n', 1)[0].replace('\n# ', '\n').strip()
            if header_encoding=='yaml':
                sub_header_data=parse_yaml(sub_header_text)
            elif header_encoding=='json':
                sub_header_data=parse_json(sub_header_text)
            # create a merged dictionary
            sub_headers_dict=recursive_update(main_header, sub_header_data)
            output.append(ORSOData(sub_headers_dict, data, strict=False))
        return output
    else:
        data=read_data(text[data_start:])
        return ORSOData(main_header, data, strict=False)

def encode_yaml(header):
    import yaml
    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())

    try:
        from yaml import CDumper as Dumper
    except ImportError:
        from yaml import Dumper

    class DumperInd(yaml.Dumper):
        # Dumper the fixes indentation of list items
        def increase_indent(self, flow=False, *args, **kwargs):
            return super().increase_indent(flow=flow, indentless=False)

    DumperInd.add_representer(OrderedDict, _dict_representer)
    if 'columns' in header:
        header=header.copy()
        columns=header['columns']
        del (header['columns'])
        output=yaml.dump(header, Dumper=DumperInd,
                         indent=4, width=78, sort_keys=False)
        output+='columns:\n'
        for col in columns:
            output+='    - %s'%yaml.dump(col, Dumper=DumperInd,
                                         indent=4, width=72, sort_keys=False, default_flow_style=True)
    else:
        output=yaml.dump(header, Dumper=DumperInd, indent=4, width=78, sort_keys=False)
    return output

def encode_json(header):
    # in case we want to represent date as datetime in python
    # class DateEncoder(json.JSONEncoder):
    #     def default(self, o):
    #         if type(o)==datetime:
    #             return o.strftime('%Y-%m-%dT%H:%M:%S')
    #         else:
    #             return json.JSONEncoder.default(self, o)
    # return json.dumps(header, indent=4, cls=DateEncoder)+'\n'
    return json.dumps(header, indent=4)+'\n'

def dict_change(main, new):
    # get a dictionary with any changed or new items including sub-dictionaries
    output=OrderedDict({})
    for mkey in main.keys():
        if mkey in new and main[mkey]!=new[mkey]:
            item=new[mkey]
            if type(item) is dict:
                output[mkey]=dict_change(main[mkey], item)
            else:
                output[mkey]=item
    for nkey in new.keys():
        if not nkey in main:
            output[nkey]=new[nkey]
    return output

def data_header(columns):
    output="%-15s"%("1 %s"%columns[0])+' '.join(["%-16s"%("%i %s"%(i+2, ci)) for i, ci in enumerate(columns[1:])])
    return output

def write_file(fname, data, header_encoding='yaml', force_ending=True):
    if force_ending and not fname.endswith('.ort'):
        fname=fname+'.ort'
    if type(data) is list:
        main_dict=data[0].header
        if header_encoding=='yaml':
            header='ORSO reflectivity data file | 0.1 standard | YAML encoding | https://www.reflectometry.org/\n'
            header+=encode_yaml(main_dict)
        elif header_encoding=='json':
            header='ORSO reflectivity data file | 0.1 standard | JSON encoding | https://www.reflectometry.org/\n'
            header+=encode_json(main_dict)
        else:
            raise ORSOIOError('Unknown header encoding "%s"'%header_encoding)
        with open(fname, 'w') as fh:
            header+=data_header(data[0].columns)
            np.savetxt(fh, data[0].data.T, header=header, fmt='%-16.9e')
            for i, di in enumerate(data[1:]):
                fh.write('\n')
                update_dict=dict_change(main_dict, di.header)
                if not 'data_set' in update_dict:
                    update_dict['data_set']='data_set'
                update_dict.move_to_end('data_set', False)
                if header_encoding=='yaml':
                    header=encode_yaml(update_dict)
                elif header_encoding=='json':
                    header=encode_json(update_dict)
                header+=data_header(di.columns)
                np.savetxt(fh, di.data.T, header=header, fmt='%-16.9e')
    else:
        main_dict=data.header
        if header_encoding=='yaml':
            header='ORSO reflectivity data file | 0.1 standard | YAML encoding | https://www.reflectometry.org/\n'
            header+=encode_yaml(main_dict)
        elif header_encoding=='json':
            header='ORSO reflectivity data file | 0.1 standard | JSON encoding | https://www.reflectometry.org/\n'
            header+=encode_json(main_dict)
        else:
            raise ORSOIOError('Unknown header encoding "%s"'%header_encoding)
        header+=data_header(data.columns)
        np.savetxt(fname, data.data.T, header=header, fmt='%-16.9e')
