'''
Library for the classes to store the data. The class DataSet stores 
on set and the class DataList stores multiple DataSets.
'''

import time
import os
from sys import platform
from numpy import *
from typing import List, Dict, Tuple, Union

from .exceptions import GenxIOError
from .core.custom_logging import iprint
from .core.colors import CyclicList
from .core.h5_support import H5Savable, H5HintedExport


try:
    from orsopy.fileio import Orso


    META_DEFAULT = Orso.empty().to_dict()
except ImportError:
    META_DEFAULT = {'data_source': {'owner': {'name': None, 'affiliation': None},
                                    'experiment': {'title': None, 'instrument': None, 'start_date': None, 'probe': None},
                                    'sample': {'name': None},
                                    'measurement': {'instrument_settings': {
                                        'incident_angle': {'magnitude': None},
                                        'wavelength': {'magnitude': None}},
                                        'data_files': []}},
                    'reduction': {'software': {'name': None}},
                    'columns': [{'name': 'Qz', 'unit': '1/angstrom'}, {'name': 'R'}]}


class DataSet(H5HintedExport):
    '''
    Class to store each dataset to fit. To fit several items instead the.
    Contains x,y,error values and xraw,yraw,errorraw for the data.
    '''
    h5group_name = 'datasets'
    # Parameters used for saving the object state
    x: ndarray = array([])
    x_raw: ndarray = array([])
    y: ndarray = array([])
    y_raw: ndarray = array([])
    error: ndarray = array([])
    error_raw: ndarray = array([])
    x_command: str = 'x'
    y_command: str = 'y'
    error_command: str = 'e'

    extra_commands: Dict[str, str] = {}
    extra_data: Dict[str, ndarray] = {}
    extra_data_raw: Dict[str, ndarray] = {}

    y_sim: ndarray = array([])
    y_fom: ndarray = array([])
    show: bool = True  # Should we display the dataset, ie plot it
    name: str = 'New Data'
    use: bool = True  # Should the dataset be used for fitting?
    use_error: bool = False  # Should the error be used
    cols: Tuple[int, int, int] = (0, 1, 1)  # The columns to load

    data_color: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    data_symbol: str = 'o'
    data_symbolsize: int = 4
    data_linetype: str = '-'
    data_linethickness: int = 2

    sim_color: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    sim_symbol: str = ''
    sim_symbolsize: int = 1
    sim_linetype: str = '-'
    sim_linethickness: int = 2

    meta: dict = META_DEFAULT

    simulation_params = [0.01, 6.01, 600]

    def __init__(self, name='', copy_from=None):
        self.init_defaults()

        # Special list for settings when setting the plotting properties
        self.plot_setting_names = ['color', 'symbol', 'symbolsize', 'linetype',
                                   'linethickness']
        if name!='':
            self.name = name

        if copy_from:
            # Should the dataset be used for fitting?
            self.use = copy_from.use
            # Should the error be used
            self.use_error = copy_from.use_error
            # The columns to load
            self.cols = copy_from.cols  # Columns to load (xcol,ycol,ecol)
            # The different colors for the data and simulation
            self.data_color = copy_from.data_color
            self.sim_color = copy_from.sim_color
            # The different linetypes and symbols incl. sizes
            self.data_symbol = copy_from.data_symbol
            self.data_symbolsize = copy_from.data_symbolsize
            self.data_linetype = copy_from.data_linetype
            self.data_linethickness = copy_from.data_linethickness
            self.sim_symbol = copy_from.sim_symbol
            self.sim_symbolsize = copy_from.sim_symbolsize
            self.sim_linetype = copy_from.sim_linetype
            self.sim_linethickness = copy_from.sim_linethickness
        self.run_command()

    def copy(self):
        ''' Make a copy of the current Data Set'''
        cpy = DataSet()
        cpy.safe_copy(self)
        return cpy

    def safe_copy(self, new_set):
        '''
        A safe copy from one dataset to another. 
        Note, not totally safe since references are not broken
        '''
        self.name = new_set.name
        self.x = new_set.x
        self.y = new_set.y
        self.y_sim = new_set.y_sim
        self.error = new_set.error
        # The raw data
        self.x_raw = new_set.x_raw
        self.y_raw = new_set.y_raw
        self.error_raw = new_set.error_raw

        # The dictonaries for the extra data
        try:
            self.extra_data = new_set.extra_data
        except AttributeError:
            self.extra_data = {}
        try:
            self.extra_data_raw = new_set.extra_raw
        except AttributeError:
            self.extra_data_raw = self.extra_data.copy()
        try:
            self.extra_commands = new_set.extra_commands
        except AttributeError:
            self.extra_commands = {}

        # The different commands to transform raw data to normal data
        self.x_command = new_set.x_command
        self.y_command = new_set.y_command
        self.error_command = new_set.error_command
        try:
            self.show = new_set.show
        except AttributeError:
            self.show = True

        self.use = new_set.use
        # Should the error be used
        self.use_error = new_set.use_error
        # The columns to load
        # The different colors for the data and simulation
        self.data_color = new_set.data_color
        self.sim_color = new_set.sim_color
        # The different linetypes and symbols incl. sizes
        self.data_symbol = new_set.data_symbol
        self.data_symbolsize = new_set.data_symbolsize
        self.data_linetype = new_set.data_linetype
        self.data_linethickness = new_set.data_linethickness
        self.sim_symbol = new_set.sim_symbol
        self.sim_symbolsize = new_set.sim_symbolsize
        self.sim_linetype = new_set.sim_linetype
        self.sim_linethickness = new_set.sim_linethickness

    def __getattr__(self, attr):
        """Overloading __getattr__ for using direct access to extra data"""
        if attr in ["extra_data", "extra_data_raw"]:
            raise AttributeError()
        if attr in self.extra_data:
            return self.extra_data[attr]
        elif attr.rstrip('_raw') in self.extra_data_raw:
            return self.extra_data_raw[attr.rstrip('_raw')]
        raise AttributeError("%r object has no attribute %r"%(self.__class__, attr))

    def has_data(self):
        return len(self.x_raw)>0

    def get_extra_data_names(self):
        return list(self.extra_data.keys())

    def set_extra_data(self, name, value, command=None):
        '''
        Sets extra data name, if it does not exist a new entry is created.
        name should be a string and value can be any object.
        If command is set, this means that the data set can be operated upon
        with commands just as the x,y and e data members.
        '''
        if name in ['x', 'y', 'e']:
            raise KeyError('The extra data can not support the key'
                           'names x, y or e.')
        self.extra_data[name] = value
        self.extra_data_raw[name] = value
        if command:
            self.extra_commands[name] = command
        else:
            self.extra_commands[name] = str(name)

    def get_extra_data(self, name):
        '''
        returns the extra_data object with name name [string] if does not
        exist an LookupError is raised.
        '''
        if name not in self.extra_data:
            raise LookupError('Can not find extra data with name %s'%name)
        return self.extra_data[name]

    # TODO: Remove load and save methods from dataset, should be handled soley by data loaders
    def loadfile(self, filename, sep='\t', pos=0):
        '''
        Function to load data from a file.
        Note that the data should be in ASCII format and it can
        be gzipped (handled automagically if the filname ends with .gz
        Possible extras:
        comments - string of chars that shows that a line is a comment
        delimeter - chars that are spacers between values default None
            all whitespaces 
        skiprows - number of rows to skip before starting to read the data
        '''
        if not os.path.exists(filename):
            iprint("Can't open file: %s does not exist"%filename)
            return False
        try:
            A = loadtxt(filename)
            # , comments = '#', delimeter = None, skiprows = 0
        except:
            iprint("Can't read the file %s, check the format"%filename)
        else:
            xcol = self.cols[0]
            ycol = self.cols[1]
            ecol = self.cols[2]
            if xcol<A.shape[1] and ycol<A.shape[1] and ecol<A.shape[1]:
                self.x_raw = A[:, xcol].copy()
                self.y_raw = A[:, ycol].copy()
                self.error_raw = A[:, ecol].copy()
                self.x = A[:, xcol]
                self.y = A[:, ycol]
                self.error = A[:, ecol]
                self.y_sim = array([])
                iprint("Sucessfully loaded %i datapoints"%(A.shape[0]))
                return True
            else:
                iprint("There are not enough columns in your data\n\
                       There are %i columns"%A.shape[1])
            return False

    def save_file(self, filename):
        '''
        saves the dataset to a file with filename.
        '''
        if self.x.shape==self.y_sim.shape and \
                self.y.shape==self.error.shape and \
                self.x.shape==self.y.shape:
            header = f'Dataset "{self.name}" exported from GenX3 on {time.ctime()}\n' \
                     f'Column lables:\n' \
                     f'x                I_simulated        I                  error(I)'
            savetxt(filename, c_[self.x, self.y_sim, self.y, self.error], header=header, fmt='%.12e')
        else:
            debug = 'y_sim.shape: '+str(self.y_sim.shape)+'\ny.shape: '+ \
                    str(self.y.shape)+'\nx.shape: '+str(self.x.shape)+ \
                    '\nerror.shape: '+str(self.error.shape)
            raise GenxIOError('The data is not in the correct format all the'+ \
                              'arrays have to have the same shape:\n'+debug, filename)

    @staticmethod
    def rms(*items):
        # combine errors to RMS using sqrt(s1**2+s2**2+...+si**2)
        sitems = items[0]**2
        for itm in items[1:]:
            sitems += itm**2
        return sqrt(sitems)

    def run_x_command(self):
        x = self.x_raw
        y = self.y_raw
        e = self.error_raw
        rms = self.rms

        for key in self.extra_data:
            exec('%s = self.extra_data_raw["%s"]'%(key, key))

        self.x = eval(self.x_command)

    def run_y_command(self):
        x = self.x_raw
        y = self.y_raw
        e = self.error_raw
        rms = self.rms

        for key in self.extra_data:
            exec('%s = self.extra_data_raw["%s"]'%(key, key))

        self.y = eval(self.y_command)

    def run_error_command(self):
        x = self.x_raw
        y = self.y_raw
        e = self.error_raw
        rms = self.rms

        xt = self.x
        yt = self.y

        for key in self.extra_data:
            exec('%s = self.extra_data_raw["%s"]'%(key, key))

        def fpe(xmax=0.05, relerr=0.01, x=xt, y=yt):
            '''
            Estimate intensity error due to beam crossection deviating from model foot print
            xmax: the full beam hits the sample at locations larger than this x-value
            relerr: relative intensity error expected wrt. footprint
            '''
            return where(x<xmax, y*relerr, 0.0)

        def dydx(x=xt, y=yt):
            # numerical calculation of local derivative from data
            return hstack([
                (y[1]-y[0])/(x[1]-x[0]),
                (y[2:]-y[:-2])/(x[2:]-x[:-2]),
                (y[-1]-y[-2])/(x[-1]-x[-2])
                ])

        self.error = eval(self.error_command)

    def run_extra_commands(self):
        x = self.x_raw
        y = self.y_raw
        e = self.error_raw
        rms = self.rms

        for key in self.extra_data_raw:
            exec('%s = self.extra_data_raw["%s"]'%(key, key))

        for key in self.extra_data_raw:
            if not key in self.extra_commands:
                self.extra_commands[key] = "%s"%key
            self.extra_data[key] = eval(self.extra_commands["%s"%key])

        if 'res' in self.extra_data:
            self.res = self.extra_data['res']

    def set_simulation(self):
        self.x = linspace(*self.simulation_params)
        self.y = nan*self.x
        self.error = nan*self.x

        for key in self.extra_data_raw:
            self.extra_data[key] = self.x*0.0

    def run_command(self):
        if len(self.x_raw)==0 and self.x_command=='x':
            # if no data is loaded and no user setting for simulation, use default
            self.set_simulation()
        else:
            self.run_x_command()
            self.run_y_command()
            self.run_error_command()
            self.run_extra_commands()

    def try_commands(self, command_dict):
        '''
        Evals the commands to locate any errors. Used to 
        test the commands before doing the actual setting of x,y and z
        '''
        result = ''

        x = self.x_raw
        y = self.y_raw
        e = self.error_raw
        rms = self.rms
        xt, yt, et = x, y, e  # make sure values are always defined

        # Know we have to do this with the extra data
        for key in self.extra_data_raw:
            exec('%s = self.extra_data_raw["%s"]'%(key, key))

        # Try to evaluate all the expressions
        if command_dict['x']!='':
            try:
                xt = eval(command_dict['x'])
            except Exception as e:
                result += 'Error in evaluating x expression.\n\nPython output:\n' \
                          +e.__str__()+'\n'

        if command_dict['y']!='':
            try:
                yt = eval(command_dict['y'])
            except Exception as e:
                result += 'Error in evaluating y expression.\n\nPython output:\n' \
                          +e.__str__()+'\n'

        if command_dict['e']!='':
            def fpe(xmax=0.05, relerr=0.01, x=xt, y=yt):
                '''
                Estimate intensity error due to beam crossection deviating from model foot print
                xmax: the full beam hits the sample at locations larger than this x-value
                relerr: relative intensity error expected wrt. footprint
                '''
                return where(x<xmax, y*relerr, 0.0)

            def dydx(x=xt, y=yt):
                # numerical calculation of local derivative from data
                return hstack([
                    (y[1]-y[0])/(x[1]-x[0]),
                    (y[2:]-y[:-2])/(x[2:]-x[:-2]),
                    (y[-1]-y[-2])/(x[-1]-x[-2])
                    ])

            try:
                et = eval(command_dict['e'])
            except Exception as e:
                result += 'Error in evaluating e expression.\n\nPython output:\n' \
                          +e.__str__()+'\n'

        extra_results = {}
        for key in self.extra_commands.keys():
            value = command_dict[key]
            if command_dict[key]!='':
                try:
                    extra_results[key] = eval(value)
                except Exception as e:
                    result += 'Error in evaluating %s expression.\n\nPython output:\n'%key \
                              +e.__str__()+'\n'

        # If we got an error - report it
        if result!='':
            return result
        # Finally check so that all the arrays have the same size
        extra_shape = any([resi.shape!=xt.shape for resi in extra_results.values()])
        if (xt.shape!=yt.shape or xt.shape!=et.shape or extra_shape) \
                and result=='':
            result += 'The resulting arrays are not of the same size:\n'+ \
                      'len(x) = %d, len(y) = %d, len(e) = %d' \
                      %(xt.shape[0], yt.shape[0], et.shape[0])
            for key, value in extra_results.items():
                result += ', len(%s) = %d'%(key, extra_results[key].shape[0])
        return result

    def get_commands(self):
        '''
        Returns the commnds as a dictonary with items x, y, z
        '''
        cmds = {'x': self.x_command, 'y': self.y_command, 'e': self.error_command}
        for key in self.extra_commands:
            cmds[key] = self.extra_commands[key]
        return cmds

    def set_commands(self, command_dict):
        '''
        Sets the commands in the data accroding to values in command dict
        See get_commands for more details
        '''
        if command_dict['x']!='':
            self.x_command = command_dict['x']
        if command_dict['y']!='':
            self.y_command = command_dict['y']
        if command_dict['e']!='':
            self.error_command = command_dict['e']
        # Lets do it for the extra commands as well
        for key in command_dict:
            if key in self.extra_commands:
                if command_dict[key]!='':
                    self.extra_commands[key] = command_dict[key]

    def set_simulated_data(self, simulated_data):
        self.y_sim = simulated_data

    def set_fom_data(self, fom_data):
        self.y_fom = fom_data

    def get_sim_plot_items(self):
        '''
        Returns a dictonary of color [tuple], symbol [string], 
        sybolsize [float], linetype [string], linethickness [float].
        Used for plotting the simulation.
        '''
        return {
            'color': (self.sim_color[0]*255, self.sim_color[1]*255,
                      self.sim_color[2]*255),
            'symbol': self.sim_symbol,
            'symbolsize': self.sim_symbolsize,
            'linetype': self.sim_linetype,
            'linethickness': self.sim_linethickness
            }

    def get_data_plot_items(self):
        '''
        Returns a dictonary of color [tuple], symbol [string], 
        sybolsize [float], linetype [string], linethickness [float].
        Used for plotting the data.
        '''
        return {
            'color': (self.data_color[0]*255, self.data_color[1]*255,
                      self.data_color[2]*255),
            'symbol': self.data_symbol,
            'symbolsize': self.data_symbolsize,
            'linetype': self.data_linetype,
            'linethickness': self.data_linethickness
            }

    def set_data_plot_items(self, pars):
        '''
        Sets the plotting parameters for the data by a dictonary of the
        same structure as in get_data_plot_items(). If one of items in the 
        pars [dictonary] is None that item will be skipped, i.e. keep its old
        value.
        '''
        for name in self.plot_setting_names:
            if pars[name] is not None:
                if type(pars[name])==type(''):
                    exec('self.data_'+name+' = "' \
                         +pars[name].__str__()+'"')
                elif name=='color':
                    c = pars['color']
                    self.data_color = (c[0]/255.0, c[1]/255.0, c[2]/255.0)
                else:
                    exec('self.data_'+name+' = '+pars[name].__str__())

    def set_sim_plot_items(self, pars):
        '''
        Sets the plotting parameters for the data by a dictonary of the
        same structure as in get_data_plot_items(). If one of items in the 
        pars [dictonary] is None that item will be skipped, i.e. keep its old
        value.
        '''
        for name in self.plot_setting_names:
            if pars[name] is not None:
                if type(pars[name])==type(''):
                    exec('self.sim_'+name+' = "' \
                         +pars[name].__str__()+'"')
                elif name=='color':
                    c = pars['color']
                    self.sim_color = (c[0]/255.0, c[1]/255.0, c[2]/255.0)
                else:
                    exec('self.sim_'+name+' = '+pars[name].__str__())

    def set_show(self, val):
        self.show = bool(val)

    @property
    def data_kwds(self):
        # return all keywords to supply to matplotlib plot functions for this dataset graph
        return dict(color=self.data_color,
                    lw=self.data_linethickness, ls=self.data_linetype,
                    marker=self.data_symbol, ms=self.data_symbolsize)

    @property
    def sim_kwds(self):
        # return all keywords to supply to matplotlib plot functions for this simulation graph
        return dict(color=self.sim_color,
                    lw=self.sim_linethickness, ls=self.sim_linetype,
                    marker=self.sim_symbol, ms=self.sim_symbolsize)

    def __eq__(self, other: "DataSet"):
        if not isinstance(other, DataSet):
            return False

        result = (self.use==other.use)
        result &= (array(self.x_raw)==array(other.x_raw)).all()
        result &= (array(self.y_raw)==array(other.y_raw)).all()
        result &= (array(self.error_raw)==array(other.error_raw)).all()
        for key, value in self.extra_data_raw.items():
            result &= (array(value)==array(other.extra_data_raw[key])).all()
            result &= (self.extra_commands[key]==other.extra_commands[key])
        result &= (self.x_command==other.x_command)
        result &= (self.y_command==other.y_command)
        result &= (self.error_command==other.error_command)
        result &= (self.name==other.name)

        return result

    def __repr__(self):
        output = "DataSet(name=%-15s, show=%s, use=%s, error=%s)"%(self.name, self.show, self.use, self.use_error)
        return output

    def _repr_html_(self):
        items = ['name', 'show', 'use', 'use_error']
        output = '<table><tr><th colspan="%i"><center>DataSet</center></th></tr>\n'%len(items)
        output += "           <tr><th>"+"</th><th>".join(items)+"</th></tr>\n"
        output += "<tr><td>"+"</td><td>".join([str(getattr(self, ii)) for ii in items])
        output += '<tr><th colspan="%i"><center>Commands</center></th></tr>\n'%len(items)
        for col in ['x', 'y', 'error']:
            output += '<tr><td>%s</td><td colspan="%i"><center>%s</center></td></tr>\n'%(
                col, len(items)-1,
                eval('self.%s_command'%col, globals(), locals()))
        for key, value in self.extra_commands.items():
            output += '<tr><td>%s</td><td colspan="%i"><center>%s</center></td></tr>\n'%(
                key, len(items)-1, value)
        output += "</tr></table>\n"
        return output

    @property
    def widget(self):
        return self._repr_ipyw_()

    def _repr_ipyw_(self, add_header=True):
        import ipywidgets as ipw
        vlist = []
        if add_header:
            vlist.append(ipw.HTML("<h3>Dataset</h3>"))
            header = ipw.HBox([ipw.HTML('<b>%s</b>'%txt[0], layout=ipw.Layout(width=txt[1])) for txt in
                               [('Name', '30ex'), ('show', '6ex'), ('use', '6ex'),
                                ('error', '6ex')]])
            vlist.append(header)

        entries = []
        entries.append(ipw.Text(self.name, layout=ipw.Layout(width='30ex')))
        entries.append(ipw.Checkbox(self.show, indent=False, layout=ipw.Layout(width='6ex')))
        entries.append(ipw.Checkbox(self.use, indent=False, layout=ipw.Layout(width='6ex')))
        entries.append(ipw.Checkbox(self.use_error, indent=False, layout=ipw.Layout(width='6ex')))
        items = ['name', 'show', 'use', 'use_error']
        for j, entr in enumerate(entries):
            entr._child_val = items[j]
            entr.observe(self._ipyw_change, names='value')
        vlist.append(ipw.HBox(entries))
        cbox = ipw.VBox()
        plotbox = ipw.VBox()
        clist = []
        commands = ipw.Accordion(children=[cbox, plotbox], selected_index=None, layout=ipw.Layout(width='46x'))
        commands.set_title(0, 'Commands')
        commands.set_title(1, 'Plotting')
        vlist.append(commands)
        for col in ['x', 'y', 'error']:
            entr = ipw.Text(eval('self.%s_command'%col, globals(), locals()),
                            description=col)
            entr._command = col
            entr.observe(self._ipyw_command, names='value')
            clist.append(entr)
        for key, value in self.extra_commands.items():
            entr = ipw.Text(value, description=key)
            entr._command = key
            clist.append(entr)
            entr.observe(self._ipyw_command, names='value')
        cbox.children = tuple(clist)
        clist = []
        clist.append(ipw.ColorPicker(description="Sim Color", value=c2html(self.sim_color)))
        clist[-1]._child_val = 'sim_color'
        clist[-1].observe(self._ipyw_change_color)
        clist.append(ipw.ColorPicker(description="Data Color", value=c2html(self.data_color)))
        clist[-1]._child_val = 'data_color'
        clist[-1].observe(self._ipyw_change_color)

        plotbox.children = tuple(clist)
        return ipw.VBox(vlist)

    def _ipyw_change(self, change):
        exec('self.%s=change.new'%(change.owner._child_val))

    def _ipyw_change_color(self, change):
        if type(change.new) is str:
            exec('self.%s=html2c(change.new)'%(change.owner._child_val))

    def _ipyw_command(self, change):
        if change.owner._command in ['x', 'y', 'error']:
            exec('self.%s_command=change.new'%change.owner._command)
        else:
            self.extra_commands[change.owner._command] = change.new
        try:
            self.run_command()
        except Exception as error:
            change.owner.description = 'ERR!'
        else:
            change.owner.description = change.owner._command


def c2html(colors):
    out = '#'
    for ci in colors:
        out += "%02x"%int(ci*255.)
    return out


def html2c(colors):
    out = (
        int("0x%s"%colors[1:3], 16)/255.,
        int("0x%s"%colors[3:5], 16)/255.,
        int("0x%s"%colors[5:7], 16)/255.,
        )
    return tuple(out)


class DataList(H5Savable):
    ''' Class to store a list of DataSets'''
    h5group_name = 'data'
    items: List[DataSet]
    color_source: Union[CyclicList, None]

    def __init__(self, items=None):
        ''' init function - creates a list with one DataSet'''
        self.color_source = None
        if items is None:
            self.items = [DataSet(name='Data 0')]
            self._counter = 1
        else:
            self.items = list(items)
            self._counter = len(items)

    def write_h5group(self, group):
        """
        Write parameters to a hdf group
        """
        data_group = group.create_group(DataSet.h5group_name)
        for index, data in enumerate(self.items):
            data.write_h5group(data_group.create_group('%d'%index))
        group['_counter'] = self._counter

    def read_h5group(self, group):
        """
        Read parameters from a hdf group
        """
        data_group = group[DataSet.h5group_name]
        self.items = []
        for index in range(len(data_group)):
            self.items.append(DataSet())
            self.items[-1].read_h5group(data_group['%d'%index])
        self._counter = int(group['_counter'][()])

    def __getitem__(self, key):
        return self.items[key]

    def __setitem__(self, key, value):
        if not type(key) is int or key>=len(self):
            raise IndexError("Can only replace existing datasets")
        self.items[key] = value

    def __iter__(self):
        ''' __iter__(self) --> iterator
        
        Opertor definition. Good to have in case one needs to loop over
        all datasets
        '''
        return self.items.__iter__()

    def __len__(self):
        '''
        Returns the nmber of datasers in the list.
        '''
        return self.items.__len__()

    def safe_copy(self, new_data):
        '''
        Conduct a safe copy of a data set into this data set.
        This is intended to produce version safe import of data sets.
        '''
        self.items = []
        for new_set in new_data:
            self.items.append(DataSet())
            self.items[-1].safe_copy(new_set)

    def add_new(self, name=''):
        '''
        Adds a new DataSet with the optional name. If name not sets it 
        will be given an automatic name
        '''
        if name=='':
            self.items.append(DataSet('Data %d'%self._counter,
                                      copy_from=self.items[-1]))
            self._counter += 1
        else:
            self.items.append(DataSet(name, copy_from=self.items[-1]))
        if self.color_source:
            self.items[-1].data_color = self.color_source[len(self)-1][0]
            self.items[-1].sim_color = self.color_source[len(self)-1][1]

    def delete_item(self, pos):
        '''
        Deletes the item at position pos. Only deletes if the pos is an 
        element and the number of datasets are more than one.
        '''
        if pos<len(self.items) and len(self.items)>1:
            self.items.pop(pos)
            return True
        else:
            return False

    def move_up(self, pos):
        '''
        Move the data set at position pos up one step. If it is at the top
        it will not be moved.
        '''
        if pos!=0:
            tmp = self.items.pop(pos)
            self.items.insert(pos-1, tmp)

    def move_down(self, pos):
        '''
        Move the dataset at postion pos down one step. If it is at the bottom
        it will not be moved.
        '''
        if pos!=len(self.items):
            tmp = self.items.pop(pos)
            self.items.insert(pos+1, tmp)

    def update_data(self):
        '''
        Calcultes all the values for the current items. 
        '''
        [item.run_command() for item in self.items]

    def update_color_cycle(self, source):
        self.color_source = source
        if source is None:
            return
        for i, item in enumerate(self.items):
            item.data_color = self.color_source[i][0]
            item.sim_color = self.color_source[i][1]

    def set_simulated_data(self, sim_data):
        '''
        set_simulated_data(self, sim_data) --> None
        
        Sets the simualted data in the data. Note this will depend on the
        flag use in the data.
        '''
        [self.items[i].set_simulated_data(sim_data[i]) for i in \
         range(self.get_len())]

    def set_fom_data(self, fom_data):
        '''
        Sets the point by point fom data in the data. Note this will depend on the
        flag use in the data.
        '''
        [self.items[i].set_fom_data(fom_data[i]) for i in \
         range(self.get_len())]

    def get_len(self):
        return len(self.items)

    def get_name(self, pos):
        '''
        Yields the name(string) of the dataset at position pos(int). 
        '''
        return self.items[pos].name

    def get_cols(self, pos):
        return self.items[pos].cols

    def get_use(self, pos):
        '''
        returns the flag use for dataset at pos [int].
        '''
        return self.items[pos].use

    def get_use_error(self, pos):
        '''
        returns the flag use_error for dataset at pos [int].
        '''
        return self.items[pos].use_error

    def toggle_use_error(self, pos):
        '''
        Toggles the use_error flag for dataset at position pos.
        '''
        self.items[pos].use_error = not self.items[pos].use_error

    def toggle_use(self, pos):
        '''
        Toggles the use flag for dataset at position pos.
        '''
        self.items[pos].use = not self.items[pos].use

    def toggle_show(self, pos):
        '''
        Toggles the show flag for dataset at position pos.
        '''
        self.items[pos].show = not self.items[pos].show

    def show_items(self, positions):
        '''
        Will put the datasets at positions [list] to show all 
        other of no show, hide.
        '''
        [item.set_show(i in positions) for i, item in enumerate(self.items)]

    def set_name(self, pos, name):
        '''
        Sets the name of the data set at position pos (int) to name (string)
        '''
        self.items[pos].name = name

    def export_data_to_files(self, basename, indices=None):
        '''
        saves the data to files with base name basename and extentions .dat
        If indices are used only the data given in the list indices are 
        exported.
        '''
        # Check if we shoudlstart picking data sets to export

        if indices:
            if not sum([i<len(self.items) for i in indices])==len(indices):
                raise GenxIOError('Error in export_data_to_files')
        else:
            indices = list(range(len(self.items)))
        for index in indices:
            base, ext = os.path.splitext(basename)
            if ext=='':
                ext = '.dat'
            self.items[index].save_file(base+'%03d'%index+ext)

    def get_data_as_asciitable(self, indices=None):
        '''
        Yields the data sets as a ascii table with tab seperated values.
        This makes it possible to export the data to for example spreadsheets.
        Each data set will be four columns with x, Meas, Meas error and Calc.
        If none is given all the data sets are transformed otherwise incdices
        shouldbe a list.
        '''

        if indices:
            if not sum([i<len(self.items) for i in indices])==len(indices):
                raise IndexError('Error in get_data_as_asciitable')
        else:
            indices = list(range(len(self.items)))

        # making some nice looking header so the user know what is what
        header1 = ''.join(['%s\t\t\t\t'%self.items[index].name \
                           for index in indices])
        header2 = ''.join(['x\ty\ty error\ty sim\t' for index in indices])

        # Find the maximum extent of the data sets
        maxlen = max([len(item.y_sim) for item in self.items])

        # Create the funtion that actually do the exporting
        def exportFunc(index, row):
            item = self.items[index]
            if row<len(item.x):
                return '%e\t%e\t%e\t%e\t'%(item.x[row], item.y[row],
                                           item.error[row], item.y_sim[row])
            else:
                return ' \t \t \t \t'

        # Now create the data
        text_data = ''.join(['\n'+''.join([exportFunc(index, row) \
                                           for index in indices]) \
                             for row in range(maxlen)])
        return header1+'\n'+header2+text_data

    def __eq__(self, other):
        """
        Check if two DataList objects have the same names and data is equal.
        """
        if not isinstance(other, DataList):
            return False
        else:
            return self.items==other.items

    def __repr__(self):
        output = "DataList([\n"
        for item in self.items:
            output += "           %s,\n"%repr(item)
        output += "           ])\n"
        return output

    def _repr_html_(self):
        items = ['name', 'show', 'use', 'use_error']
        output = '<table><tr><th colspan="%i"><center>DataList</center></th></tr>\n'%(len(items)+1)
        output += "           <tr><th>No.</th><th>"+"</th><th>".join(items)+"</th></tr>\n"
        for i, item in enumerate(self.items):
            output += "           <tr><td>#%i</td><td>"%i
            output += "</td><td>".join([str(getattr(item, ii)) for ii in items])+"</td></tr>\n"
        output += "</table>"
        return output

    @property
    def widget(self):
        return self._repr_ipyw_()

    def _repr_ipyw_(self):
        import ipywidgets as ipw
        vlist = []
        header = ipw.HBox([ipw.HTML('<b>%s</b>'%txt[0], layout=ipw.Layout(width=txt[1])) for txt in
                           [('No.', '6ex'), ('Name', '30ex'), ('show', '6ex'), ('use', '6ex'),
                            ('error', '6ex')]])
        vlist.append(header)
        items = ['name', 'show', 'use', 'use_error']
        for i, item in enumerate(self.items):
            entries = []
            entries.append(ipw.Label('#%i'%i, layout=ipw.Layout(width='6ex')))
            # entries.append(ipw.Text(item.name, layout=ipw.Layout(width='30ex')))
            # entries.append(ipw.Checkbox(item.show, indent=False, layout=ipw.Layout(width='6ex')))
            # entries.append(ipw.Checkbox(item.use, indent=False, layout=ipw.Layout(width='6ex')))
            # entries.append(ipw.Checkbox(item.use_error, indent=False, layout=ipw.Layout(width='6ex')))
            # for j, entr in enumerate(entries[1:]):
            #     entr._child_id=i
            #     entr._child_val=items[j]
            #     entr.observe(self._ipyw_change, names='value')
            entries.append(item._repr_ipyw_(add_header=False))

            vlist.append(ipw.HBox(entries))

        add_button = ipw.Button(description='Add Dataset')
        vlist.append(add_button)
        add_button.on_click(self._ipyw_add)
        vbox = ipw.VBox(vlist)
        add_button.vbox = vbox
        return vbox

    def _ipyw_change(self, change):
        exec('self.items[%i].%s=change.new'%(change.owner._child_id, change.owner._child_val))

    def _ipyw_add(self, button):
        import ipywidgets as ipw
        self.add_new()

        items = ['name', 'show', 'use', 'use_error']
        item = self.items[-1]
        i = (len(self.items)-1)
        entries = []
        entries.append(ipw.Label('#%i'%i, layout=ipw.Layout(width='6ex')))
        # entries.append(ipw.Text(item.name, layout=ipw.Layout(width='30ex')))
        # entries.append(ipw.Checkbox(item.show, indent=False, layout=ipw.Layout(width='6ex')))
        # entries.append(ipw.Checkbox(item.use, indent=False, layout=ipw.Layout(width='6ex')))
        # entries.append(ipw.Checkbox(item.use_error, indent=False, layout=ipw.Layout(width='6ex')))
        # for j, entr in enumerate(entries[1:]):
        #     entr._child_id=i
        #     entr._child_val=items[j]
        #     entr.observe(self._ipyw_change, names='value')

        entries.append(item._repr_ipyw_(add_header=False))
        prev_box = button.vbox.children
        button.vbox.children = prev_box[:-1]+(ipw.HBox(entries), prev_box[-1])

    def plot(self, data_labels=None, sim_labels=None):
        # convenience function to plot all datasets with matplotlib
        from matplotlib import pyplot as plt
        for i, ds in enumerate(self.items):
            if data_labels is None:
                dl = 'data-%i: %s'%(i, ds.name)
            else:
                dl = data_labels[i]
            if sim_labels is None:
                sl = 'model-%i: %s'%(i, ds.name)
            else:
                sl = sim_labels[i]
            if not ds.show:
                continue
            plt.semilogy(ds.x, ds.y, label=dl, **ds.data_kwds)
            if ds.y_sim.shape==ds.y.shape:
                plt.semilogy(ds.x, ds.y_sim, label=sl, **ds.sim_kwds)
        plt.legend()
