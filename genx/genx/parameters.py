'''
Definition for the class Parameters. Used for storing the fitting parameters
in GenX.
'''
from typing import Union, List, Tuple
from enum import Enum, auto

from .core.custom_logging import iprint
from .core.h5_support import H5Savable


# ==============================================================================
class SortSplitItem(Enum):
    ATTRIBUTE = auto()
    OBJ_NAME = auto()
    CLASS = auto()


class Parameters(H5Savable):
    """
    Class for storing the fitting parameters in GenX
    """
    h5group_name = 'parameters'
    data_labels: Tuple[str] = ('Parameter', 'Value', 'Fit', 'Min', 'Max', 'Error')
    init_data: Tuple[str, float, bool, float, float, str] = ('', 0.0, False, 0.0, 0.0, 'None')

    def __init__(self):
        self._data = [list(self.init_data)]

    @property
    def data(self) -> list:
        return self._data

    @data.setter
    def data(self, value):
        plen = len(self.init_data)
        if not all([len(item)==plen for item in value]):
            raise ValueError(f'All parameter rows must have {plen} items')
        self._data = list(value)

    def write_h5group(self, group):
        """
        Export the members in the object to a h5py group.
        """
        group['data_labels'] = [label.encode('utf-8') for label in self.data_labels]
        group['data col 0'] = [r[0].encode('utf-8') for r in self.data]
        group['data col 1'] = [r[1] for r in self.data]
        group['data col 2'] = [r[2] for r in self.data]
        group['data col 3'] = [r[3] for r in self.data]
        group['data col 4'] = [r[4] for r in self.data]
        group['data col 5'] = [r[5].encode('utf-8') for r in self.data]

    def read_h5group(self, group):
        """ Import data to the object from a h5py group

        :param group: h5py Group to import from
        :return:
        """
        self.data_labels = tuple(str(item.decode('utf-8')) for item in list(group['data_labels'][()]))
        self.data = [[c0.decode('utf-8'), float(c1), bool(c2), float(c3), float(c4), c5.decode('utf-8')]
                     for (c0, c1, c2, c3, c4, c5) in
                     zip(group['data col 0'][()], group['data col 1'][()],
                         group['data col 2'][()], group['data col 3'][()],
                         group['data col 4'][()], group['data col 5'][()])]

    def to_dict(self):
        """Creates a dict from the names in the columns, returns a Dict"""
        d = {}
        for i in range(len(self.data_labels)):
            d[self.data_labels[i]] = [r[i] for r in self.data]
        return d

    def set_value(self, row, col, value):
        """ Set a value in the parameter grid """
        self.data[row][col] = value

    def get_value(self, row, col):
        """ Get the value in the grid """
        return self.data[row][col]

    def get_names(self):
        """ Returns the parameter names """
        par_names = [row[0] for row in self.data]
        return par_names

    def get_value_by_name(self, name):
        """Get the value for parameter name. Returns None if name can not be found."""
        par_names = [row[0] for row in self.data]
        if name in par_names:
            value = self.data[par_names.index(name)][1]
        else:
            value = None
        return value

    def get_fit_state_by_name(self, name):
        """
        Get the fitting state for parameter name. Returns None if name can not be found.

        :return: int 0 - not found, 1 - fitted, 2 - defined but constant
        """
        par_names = [row[0] for row in self.data]
        state = 0
        if name in par_names:
            if self.data[par_names.index(name)][2]:
                state = 1
            else:
                state = 2
        return state

    def set_fit_state_by_name(self, name, value, state, min_val, max_val):
        """ Set the fit state by name accoring to the following states
        0 -- remove from grid.
        1 -- add if not exist, set parameter to be fitted (True)
        2 -- add if not exist, set parameter to not be fitted (False)

        :param name: parameter name
        :param state: state, see above.
        :return:
        """
        par_names = [row[0] for row in self.data]
        if state==0 and name in par_names:
            self.delete_rows([par_names.index(name), ])
        elif state==1 or state==2:
            if state==1:
                fit = True
            else:
                fit = False
            if name in par_names:
                self.data[par_names.index(name)][1] = value
                self.data[par_names.index(name)][2] = fit
            else:
                self.append()
                self.data[-1][0] = name
                self.data[-1][1] = value
                self.data[-1][2] = fit
                self.data[-1][3] = min_val
                self.data[-1][4] = max_val
                self.data[-1][5] = '-'

    def get_len_rows(self):
        return len(self.data)

    def get_len_cols(self):
        return len(self.data[0])

    def get_len_fit_pars(self):
        return sum([row[2] for row in self.data])

    def get_col_headers(self):
        return self.data_labels[:]

    def delete_rows(self, rows):
        ''' Delete the rows in the list rows ...'''
        delete_count = 0
        rows = rows[:]
        rows.sort()

        for i in rows:
            # Note that index changes as we delete values, that's why rows has to be sorted
            try:
                self.data.pop(i-delete_count)
            except IndexError:
                pass
            else:
                delete_count += 1

        return delete_count

    def insert_row(self, row):
        ''' Insert a new row at row(int). '''
        self.data.insert(row, list(self.init_data))

    def move_row_up(self, row):
        self.move_row(row, -1)

    def move_row_down(self, row):
        self.move_row(row, 1)

    def move_row(self, row, step):
        if self.can_move_row(row, step):
            self.data.insert(row+step, self.data.pop(row))

    def can_move_row(self, row, step=1):
        if 0<=row+step<self.get_len_rows():
            return True
        else:
            return False

    def sort_rows(self, model=None, sort_params: SortSplitItem = SortSplitItem.ATTRIBUTE):
        def _sort_key_func(item):
            class_name = ''
            pname = item[0]
            obj_name = item[0]
            if model is not None:
                if item[0].count('.')>0 and model.compiled:
                    pieces = item[0].split('.')
                    obj_name = pieces[0]
                    pname = pieces[1]
                    obj = model.eval_in_model(obj_name)
                    class_name = obj.__class__.__name__
                    if sort_params is SortSplitItem.OBJ_NAME:
                        return class_name.lower(), obj_name.lower(), pname.lower()
            return class_name.lower(), pname.lower(), obj_name.lower()

        self.data.sort(key=_sort_key_func)
        return True

    def group_rows(self, model, split_params: SortSplitItem = SortSplitItem.ATTRIBUTE):
        # generate empty lines between blocks of same class objects
        def get_cls(item):
            if item[0].count('.')>0:
                pieces = item[0].split('.')
                obj_name = pieces[0]
                pname = pieces[1]
                obj = model.eval_in_model(obj_name)
                if split_params is SortSplitItem.ATTRIBUTE:
                    return obj.__class__.__name__+'->'+pname
                elif split_params is SortSplitItem.OBJ_NAME:
                    return obj.__class__.__name__+'->'+obj_name
                elif split_params is SortSplitItem.CLASS:
                    return obj.__class__.__name__
            return None

        inserts = []
        cls = None
        for i, row in enumerate(self.data):
            next_obj = get_cls(row)
            if cls is not None and next_obj!=cls:
                inserts.append(i)
            cls = next_obj
        for i in reversed(inserts):
            self.data.insert(i, list(self.init_data))

    def strip(self):
        # remove empty parameters at beginning and end
        while self.data[0][0].strip()=='':
            self.data.pop(0)
        while self.data[-1][0].strip()=='':
            self.data.pop()

    def append(self, parameter=None, model=None):
        data = list(self.init_data)
        out = ConnectedParameter(self, data)
        out._model = model
        if parameter is not None:
            out.name = parameter
        self.data.append(data)
        return out

    def get_fit_pars(self):
        ''' Returns the variables needed for fitting '''
        # print 'Data in the parameters class: ', self.data
        rows = list(range(len(self.data)))
        row_nmb = [nmb for nmb in rows if self.data[nmb][2] and \
                   not self.data[nmb][0]=='']
        funcs = [row[0] for row in self.data if row[2] and not row[0]=='']
        values = [row[1] for row in self.data if row[2] and not row[0]=='']
        min_ = [row[3] for row in self.data if row[2] and not row[0]=='']
        max_ = [row[4] for row in self.data if row[2] and not row[0]=='']
        return row_nmb, funcs, values, min_, max_

    def get_pos_from_row(self, row):
        '''get_pos_from_row(self) --> pos [int]

        Transform the row row to the position in the fit_pars list
        '''
        rows = list(range(row+1))
        row_nmb = [nmb for nmb in rows if self.data[nmb][2] and \
                   not self.data[nmb][0]=='']
        return len(row_nmb)-1

    def get_sim_pars(self):
        ''' Returns the variables needed for simulation '''
        funcs = [row[0] for row in self.data if not row[0]=='']
        value = [row[1] for row in self.data if not row[0]=='']
        return funcs, value

    def get_sim_pos_from_row(self, row):
        '''Transform a row to a psoitions in the sim list
        that is returned by get_sim_pars
        '''
        rows = list(range(row+1))
        row_nmb = [nmb for nmb in rows if not self.data[nmb][0]=='']
        return len(row_nmb)-1

    def set_value_pars(self, value):
        ''' Set the values of the parameters '''
        valueindex = 0
        for row in self.data:
            if row[2] and not row[0]=='':
                row[1] = value[valueindex]
                valueindex = valueindex+1

    def get_value_pars(self):
        output = []
        for row in self.data:
            if row[2] and not row[0]=='':
                output.append(row[1])
        return output

    def set_error_pars(self, value):
        ''' Set the errors on the parameters '''
        valueindex = 0
        for row in self.data:
            if row[2] and not row[0]=='':
                row[5] = value[valueindex]
                valueindex = valueindex+1
            else:
                row[5] = '-'

    def get_error_pars(self):
        ''' Get the errors on the parameters '''
        output = []
        for row in self.data:
            if row[2] and not row[0]=='':
                value = row[5]
                try:
                    slow, shigh = map(float, value.lstrip('(').rstrip(')').split(','))
                except Exception:
                    slow, shigh = 0., 0.
                output.append((slow, shigh))
        return output

    def clear_error_pars(self):
        ''' clears the errors in the parameters'''
        for row in self.data:
            row[5] = '-'

    def set_data(self, data):
        rowi = 0
        coli = 0
        for row in data:
            for col in row:
                self.set_value(rowi, coli, col)
                coli = coli+1
            rowi = rowi+1
            coli = 0

    def get_data(self):
        return self.data[:]

    def get_ascii_output(self):
        '''
        Returns the parameters grid as an ascii string.
        '''
        text = '#'
        # Show the data labels but with a preceding # to denote a comment
        for label in self.data_labels:
            text += label+'\t'
        text += '\n'
        for row in self.data:
            for item in row:
                # special handling of floats to reduce the
                # col size use 5 significant digits
                if type(item)==type(10.0):
                    text += '%.4e\t'%item
                else:
                    text += item.__str__()+'\t'
            text += '\n'
        return text

    @staticmethod
    def _parse_ascii_input(text: str) -> Union[list, None]:
        '''
        Parses an ascii string to a parameter table. returns a list table if
        sucessful otherwise it returns None
        '''
        table = []
        lines = text.split('\n')
        for line in lines[:-1]:
            # If line not commented
            if line[0]!='#' and line[0]!='\n':
                line_strs = line.split('\t')
                # Check the format is it valid?
                if len(line_strs)>7 or len(line_strs)<6:
                    if len(line_strs)==1:
                        break
                    else:
                        return None
                # noinspection PyBroadException
                try:
                    par = line_strs[0]
                    val = float(line_strs[1])
                    fitted = line_strs[2].strip()=='True' \
                             or line_strs[2].strip()=='1'
                    min_ = float(line_strs[3])
                    max_ = float(line_strs[4])
                    error = line_strs[5]
                except:
                    return None
                else:
                    table.append([par, val, fitted, min_, max_, error])
        return table

    def set_ascii_input(self, text):
        '''
        If possible parse the text source and set the current parameters table
        to the one given in text.
        '''
        table = self._parse_ascii_input(text)
        if table is not None:
            self.data = table
            return True
        else:
            return False

    def safe_copy(self, obj):
        '''
        Does a safe copy from object into this object.
        '''
        self.data = [di.copy() for di in obj.data]

    def copy(self):
        '''
        Does a copy of the current object.
        '''
        new_pars = Parameters()
        new_pars.data = [di.copy() for di in self.data]
        return new_pars

    def __len__(self):
        return len([di for di in self.data if di[0].strip()!=''])

    def __getitem__(self, item):
        return ConnectedParameter(self, self.data[item])

    def __eq__(self, other):
        if not isinstance(other, Parameters):
            return False
        return self.data==other.data

    def __repr__(self):
        """
        Display information about the fit parameters.
        """
        output = "Parameters:\n"
        output += "           "+" ".join(["%-16s"%label for label in self.data_labels])+"\n"
        for line in self.data:
            output += "           "+" ".join(["%-16s"%col for col in line])+"\n"
        return output

    def _repr_html_(self):
        output = '<table><tr><th colspan="%i"><center>Parameters</center></th></tr>\n'%(len(self.data_labels)+1)
        output += "           <tr><th>No.</th><th>"+"</th><th>".join(
            ["%s"%label for label in self.data_labels])+"</th></tr>\n"
        for i, line in enumerate(self.data):
            output += "           <tr><td>%i</td><td>"%i
            output += "</td><td>".join(["%s"%col for col in line])+"\n"
        output += "</table>"
        return output

    @property
    def widget(self):
        return self._repr_ipyw_()

    def _repr_ipyw_(self):
        import ipywidgets as ipw
        vlist = []
        header = ipw.HBox([ipw.HTML('<b>%s</b>'%txt[0], layout=ipw.Layout(width=txt[1])) for txt in
                           [('Parameter', '35%'), ('Value', '20%'), ('fit', '5%'),
                            ('min', '20%'), ('max', '20%')]])
        vlist.append(header)
        for par in self:
            # noinspection PyProtectedMember
            vlist.append(par._repr_ipyw_())

        add_button = ipw.Button(description='Add Parameter')
        vlist.append(add_button)
        add_button.on_click(self._ipyw_add)
        vbox = ipw.VBox(vlist)
        add_button.vbox = vbox
        return vbox

    def _ipyw_add(self, button):
        self.append()

        prev_box = button.vbox.children
        # noinspection PyProtectedMember
        button.vbox.children = prev_box[:-1]+(self[-1]._repr_ipyw_(), prev_box[-1])


class ConnectedParameter:
    """
    A representation of a single fittable parameter for use in api access to GenX.
    """

    def __init__(self, parent, data):
        self.data = data
        self._parent = parent
        self._model = None

    @property
    def name(self):
        return self.data[0]

    @name.setter
    def name(self, value):
        if self._model is None:
            raise ValueError("Name can't be set directly, use set_name(value, model)")
        else:
            self.set_name(value, self._model)

    def set_name(self, value, model):
        if not model.is_compiled():
            model.compile_script()
        par_value = eval('self._model.script_module.'+value.replace('.set', '.get')+'()',
                         globals(), locals())
        self.data[0] = value
        self.value = par_value
        self.min = 0.25*par_value
        self.max = 4.*par_value

    @property
    def value(self):
        return self.data[1]

    @value.setter
    def value(self, value):
        self.data[1] = float(value)

    @property
    def fit(self):
        return self.data[2]

    @fit.setter
    def fit(self, value):
        self.data[2] = bool(value)

    @property
    def min(self):
        return self.data[3]

    @min.setter
    def min(self, value):
        self.data[3] = float(value)

    @property
    def max(self):
        return self.data[4]

    @max.setter
    def max(self, value):
        self.data[4] = float(value)

    @property
    def error(self):
        return self.data[5]

    def __repr__(self):
        """
        Display information about the parameter.
        """
        output = "Parameter:\n"
        output += "           "+" ".join(["%-16s"%label for label in self._parent.data_labels])+"\n"
        output += "           "+" ".join(["%-16s"%col for col in self.data])+"\n"
        return output

    def _repr_html_(self):
        output = '<table><tr><th colspan="%i"><center>Parameter</center></th></tr>\n'%(len(self._parent.data_labels)+1)
        output += "           <tr><th>No.</th><th>"+"</th><th>".join(
            ["%s"%label for label in self._parent.data_labels])+"</th></tr>\n"
        output += "           <tr><td>%i</td><td>"%self._parent.data.index(self.data)
        output += "</td><td>".join(["%s"%col for col in self.data])+"\n"
        output += "</table>"
        return output

    def _repr_ipyw_(self):
        import ipywidgets as ipw
        wname = ipw.Combobox(value=self.name,
                             options=[ni for ni, oi in self._parent.model.script_module.__dict__.items()
                                      if type(oi).__name__ in ['Layer', 'Stack', 'Instrument']])
        wname.change_item = 'name'
        wval = ipw.FloatText(value=self.value, layout=ipw.Layout(width='20%'))
        wval.change_item = 'value'
        wfit = ipw.Checkbox(value=self.fit, indent=False, layout=ipw.Layout(width='5%'))
        wfit.change_item = 'fit'
        wmin = ipw.FloatText(value=self.min, layout=ipw.Layout(width='20%'))
        wmin.change_item = 'min'
        wmax = ipw.FloatText(value=self.max, layout=ipw.Layout(width='20%'))
        wmax.change_item = 'max'
        entries = [wname, wval, wfit, wmin, wmax]
        wname.others = entries[1:]
        for entr in entries:
            entr.observe(self._ipyw_change, names='value')
        return ipw.HBox(entries)

    def _ipyw_change(self, change):
        if change.owner.change_item=='name':
            if '.' in change.new:
                # noinspection PyBroadException
                try:
                    name = change.new.split('.')[0]
                    change.owner.options = tuple("%s.%s"%(name, si)
                                                 for si in dir(self._parent.model.script_module.__dict__[name])
                                                 if si.startswith('set'))
                except:
                    pass
            else:
                change.owner.options = tuple(ni for ni, oi in self._parent.model.script_module.__dict__.items()
                                             if type(oi).__name__ in ['Layer', 'Stack', 'Instrument'])

            if change.new=='':
                self.data[0] = ''
                change.owner.description = ''
                return
            prev = self.name
            # noinspection PyBroadException
            try:
                self.name = change.new
            except Exception:
                self.data[0] = prev
                change.owner.description = 'ERR'
            else:
                change.owner.description = ''
                change.owner.others[0].value = self.value
                change.owner.others[1].value = self.fit
                change.owner.others[2].value = self.min
                change.owner.others[3].value = self.max
        elif change.owner.change_item=='fit':
            self.fit = change.new
        elif change.owner.change_item=='value':
            self.value = change.new
        elif change.owner.change_item=='min':
            self.min = change.new
        elif change.owner.change_item=='min':
            self.min = change.new


if __name__=='__main__':
    p = Parameters()
    p.append()
    import h5py


    f = h5py.File('test.hdf', 'w')
    g = f.create_group('parameters')
    p.write_h5group(g)
    f.close()
    p2 = Parameters()
    f = h5py.File('test.hdf', 'r')
    p2.read_h5group(f['parameters'])
    iprint(p2.data)
