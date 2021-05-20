'''
Definition for the class Parameters. Used for storing the fitting parameters
in GenX.
Programmer: Matts Bjorck
'''

import numpy as np
import string
from genx.gui_logging import iprint

# ==============================================================================


class Parameters:
    """
    Class for storing the fitting parameters in GenX
    """
    # Parameters used for saving the object state
    export_parameters={'data_labels': list, 'init_data': list, 'data': list}

    def __init__(self, model=None):
        self.data_labels=['Parameter', 'Value', 'Fit', 'Min', 'Max', 'Error']
        self.init_data=['', 0.0, False, 0.0, 0.0, 'None']
        self.data=[self.init_data[:]]
        self.model=model
        self.string_dtype="S100"

    def write_h5group(self, group):
        """ Export the members in the object to a h5py group.
        :param group: h5py Group to export to
        :return:
        """
        group['data_labels']=[label.encode('utf-8') for label in self.data_labels]
        # print np.array([r[0] for r in self.data], dtype='S50')
        group['data col 0']=[r[0].encode('utf-8') for r in self.data]
        group['data col 1']=[r[1] for r in self.data]
        group['data col 2']=[r[2] for r in self.data]
        group['data col 3']=[r[3] for r in self.data]
        group['data col 4']=[r[4] for r in self.data]
        group['data col 5']=[r[5].encode('utf-8') for r in self.data]

    def read_h5group(self, group):
        """ Import data to the object from a h5py group

        :param group: h5py Group to import from
        :return:
        """
        self.data_labels=[item.decode('utf-8') for item in list(group['data_labels'][()])]
        self.data=[[c0.decode('utf-8'), float(c1), bool(c2), float(c3), float(c4), c5.decode('utf-8')]
                   for (c0, c1, c2, c3, c4, c5) in
                   zip(group['data col 0'][()], group['data col 1'][()],
                       group['data col 2'][()], group['data col 3'][()],
                       group['data col 4'][()], group['data col 5'][()])]

    def to_dict(self):
        """Creates a dict from the names in the columns, returns a Dict"""
        d={}
        for i in range(len(self.data_labels)):
            d[self.data_labels[i]]=[r[i] for r in self.data]
        return d

    def set_value(self, row, col, value):
        """ Set a value in the parameter grid """
        self.data[row][col]=value

    def get_value(self, row, col):
        """ Get the value in the grid """
        return self.data[row][col]

    def name_in_grid(self, name):
        """ Checks if name is a parameter in the grid

        :param name: string representation of a parameter
        :return:
        """
        par_names=[row[0] for row in self.data]
        return name in par_names

    def get_names(self):
        """ Returns the parameter names

        :return:
        """
        par_names=[row[0] for row in self.data]

        return par_names

    def get_value_by_name(self, name):
        """Get the value for parameter name. Returns None if name can not be found.

        :param name:
        :return: Value or None
        """
        par_names=[row[0] for row in self.data]
        if name in par_names:
            value=self.data[par_names.index(name)][1]
        else:
            value=None
        return value

    def get_fit_state_by_name(self, name):
        """Get the fitting state for parameter name. Returns None if name can not be found.

        :param name:
        :return: int 0 - not found, 1 - fitted, 2 - defined but constant
        """
        par_names=[row[0] for row in self.data]
        state=0
        if name in par_names:
            if self.data[par_names.index(name)][2]:
                state=1
            else:
                state=2
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
        par_names=[row[0] for row in self.data]
        if state==0 and name in par_names:
            self.delete_rows([par_names.index(name), ])
        elif state==1 or state==2:
            if state==1:
                fit=True
            else:
                fit=False
            if name in par_names:
                self.data[par_names.index(name)][1]=value
                self.data[par_names.index(name)][2]=fit
            else:
                self.append()
                self.data[-1][0]=name
                self.data[-1][1]=value
                self.data[-1][2]=fit
                self.data[-1][3]=min_val
                self.data[-1][4]=max_val
                self.data[-1][5]='-'

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
        delete_count=0
        rows=rows[:]
        rows.sort()

        for i in rows:
            # Note index changes as we delete values. thats why rows has to be sorted
            try:
                self.data.pop(i-delete_count)
            except:
                pass
            else:
                delete_count+=1

        return delete_count

    def insert_row(self, row):
        ''' Insert a new row at row(int). '''
        self.data.insert(row, self.init_data[:])

    def move_row_up(self, row):
        """ Move row up.

        :param row: Move row up one row.
        :return: Boolean True if the row was moved, otherwise False.
        """

        if row!=0 and row<self.get_len_rows():
            self.data.insert(row-1, self.data.pop(row))
            return True
        else:
            return False

    def move_row_down(self, row):
        """ Move row up.

        :param row: Move row down one row.
        :return: Boolean True if the row was moved, otherwise False.
        """

        if row<self.get_len_rows()-1:
            self.data.insert(row+1, self.data.pop(row))
            return True
        else:
            return False

    def _sort_key_func(self, item):
        class_name=''
        pname=item[0]
        obj_name=item[0]
        if self.model is not None:
            if item[0].count('.')>0 and self.model.compiled:
                pieces=item[0].split('.')
                obj_name=pieces[0]
                pname=pieces[1]
                obj=self.model.eval_in_model(obj_name)
                class_name=obj.__class__.__name__

        return string.lower(class_name), string.lower(pname), string.lower(obj_name)

    def sort_rows(self):
        """ Sort the rows in the table

        :return: Boolean to indicate success (True)
        """
        self.data.sort(key=self._sort_key_func)
        return True

    def append(self, parameter=None):
        data=list(self.init_data)
        out=ConnectedParameter(self, data)
        if parameter is not None:
            out.name=parameter
        self.data.append(data)
        return out

    def get_fit_pars(self):
        ''' Returns the variables needed for fitting '''
        # print 'Data in the parameters class: ', self.data
        rows=list(range(len(self.data)))
        row_nmb=[nmb for nmb in rows if self.data[nmb][2] and \
                 not self.data[nmb][0]=='']
        funcs=[row[0] for row in self.data if row[2] and not row[0]=='']
        mytest=[row[1] for row in self.data if row[2] and not row[0]=='']
        min=[row[3] for row in self.data if row[2] and not row[0]=='']
        max=[row[4] for row in self.data if row[2] and not row[0]=='']
        return row_nmb, funcs, mytest, min, max

    def get_pos_from_row(self, row):
        '''get_pos_from_row(self) --> pos [int]

        Transform the row row to the position in the fit_pars list
        '''
        rows=list(range(row+1))
        row_nmb=[nmb for nmb in rows if self.data[nmb][2] and \
                 not self.data[nmb][0]=='']
        return len(row_nmb)-1

    def get_sim_pars(self):
        ''' Returns the variables needed for simulation '''
        funcs=[row[0] for row in self.data if not row[0]=='']
        mytest=[row[1] for row in self.data if not row[0]=='']
        return funcs, mytest

    def get_sim_pos_from_row(self, row):
        '''Transform a row to a psoitions in the sim list
        that is returned by get_sim_pars
        '''
        rows=list(range(row+1))
        row_nmb=[nmb for nmb in rows if not self.data[nmb][0]=='']
        return len(row_nmb)-1

    def set_value_pars(self, value):
        ''' Set the values of the parameters '''
        valueindex=0
        for row in self.data:
            if row[2] and not row[0]=='':
                row[1]=value[valueindex]
                valueindex=valueindex+1

    def set_error_pars(self, value):
        ''' Set the errors on the parameters '''
        valueindex=0
        for row in self.data:
            if row[2] and not row[0]=='':
                row[5]=value[valueindex]
                valueindex=valueindex+1

    def clear_error_pars(self):
        ''' clears the errors in the parameters'''
        for row in self.data:
            row[5]='-'

    def set_data(self, data):
        rowi=0
        coli=0
        for row in data:
            for col in row:
                self.set_value(rowi, coli, col)
                coli=coli+1
            rowi=rowi+1
            coli=0

    def get_data(self):
        return self.data[:]

    def get_ascii_output(self):
        '''get_ascii_output(self) --> text [string]

        Returns the parameters grid as an ascii string.
        '''
        text='#'
        # Show the data labels but with a preceeding # to denote a comment
        for label in self.data_labels:
            text+=label+'\t'
        text+='\n'
        for row in self.data:
            for item in row:
                # special handling of floats to reduce the
                # col size use 5 significant digits
                if type(item)==type(10.0):
                    text+='%.4e\t'%item
                else:
                    text+=item.__str__()+'\t'
            text+='\n'
        return text

    def _parse_ascii_input(self, text):
        '''parse_ascii_input(self, text) --> list table

        Parses an ascii string to a parameter table. returns a list table if
        sucessful otherwise it returns None
        '''
        table=[]
        sucess=True
        lines=text.split('\n')
        for line in lines[:-1]:
            # If line not commented
            if line[0]!='#' and line[0]!='\n':
                line_strs=line.split('\t')
                # Check the format is it valid?
                if len(line_strs)>7 or len(line_strs)<6:
                    if len(line_strs)==1:
                        break
                    else:
                        sucess=False
                        break
                try:
                    par=line_strs[0]
                    val=float(line_strs[1])
                    fitted=line_strs[2].strip()=='True' \
                           or line_strs[2].strip()=='1'
                    min=float(line_strs[3])
                    max=float(line_strs[4])
                    error=line_strs[5]
                except Exception as e:
                    sucess=False
                    break
                else:
                    table.append([par, val, fitted, min, max, error])
        if sucess:
            return table
        else:
            return None

    def set_ascii_input(self, text):
        '''set_ascii_input(self, text) --> None

        If possible parse the text source and set the current parameters table
        to the one given in text.
        '''
        table=self._parse_ascii_input(text)
        if table:
            self.data=table
            return True
        else:
            return False

    def safe_copy(self, object):
        '''safe_copy(self, object) --> None

        Does a safe copy from object into this object.
        '''
        self.data=object.data[:]

    def copy(self):
        '''get_copy(self) --> copy of Parameters

        Does a copy of the current object.
        '''
        new_pars=Parameters()
        new_pars.data=self.data[:]

        return new_pars

    def __len__(self):
        return len([di for di in self.data if di[0].strip()!=''])

    def __getitem__(self, item):
        return ConnectedParameter(self, self.data[item])

    def __repr__(self):
        """
        Display information about the model.
        """
        output="Parameters:\n"
        output+="           "+" ".join(["%-16s"%label for label in self.data_labels])+"\n"
        for line in self.data:
            output+="           "+" ".join(["%-16s"%col for col in line])+"\n"
        return output

    def _repr_html_(self):
        output='<table><tr><th colspan="%i"><center>Parameters</center></th></tr>\n'%(len(self.data_labels)+1)
        output+="           <tr><th>No.</th><th>"+"</th><th>".join(
            ["%s"%label for label in self.data_labels])+"</th></tr>\n"
        for i, line in enumerate(self.data):
            output+="           <tr><td>%i</td><td>"%i
            output+="</td><td>".join(["%s"%col for col in line])+"\n"
        output+="</table>"
        return output

    @property
    def widget(self):
        return self._repr_ipyw_()

    def _repr_ipyw_(self):
        import ipywidgets as ipw
        vlist=[]
        header=ipw.HBox([ipw.HTML('<b>%s</b>'%txt[0], layout=ipw.Layout(width=txt[1])) for txt in
                         [('Parameter', '35%'), ('Value', '20%'), ('fit', '5%'),
                          ('min', '20%'), ('max', '20%')]])
        vlist.append(header)
        for par in self:
            vlist.append(par._repr_ipyw_())

        add_button=ipw.Button(description='Add Parameter')
        vlist.append(add_button)
        add_button.on_click(self._ipyw_add)
        vbox=ipw.VBox(vlist)
        add_button.vbox=vbox
        return vbox

    def _ipyw_add(self, button):
        self.append()

        prev_box=button.vbox.children
        button.vbox.children=prev_box[:-1]+(self[-1]._repr_ipyw_(), prev_box[-1])

class ConnectedParameter():
    """
    A representation of a single fittable parameter for use in api access to GenX.
    """

    def __init__(self, parent, data):
        self.data=data
        self._parent=parent

    @property
    def name(self):
        return self.data[0]

    @name.setter
    def name(self, value):
        model=self._parent.model
        if not model.is_compiled():
            model.compile_script()
        par_value=eval('self._parent.model.script_module.'+value.replace('.set', '.get')+'()',
                       globals(), locals())
        self.data[0]=value
        self.value=par_value
        self.min=0.25*par_value
        self.max=4.*par_value

    @property
    def value(self):
        return self.data[1]

    @value.setter
    def value(self, value):
        self.data[1]=float(value)

    @property
    def fit(self):
        return self.data[2]

    @fit.setter
    def fit(self, value):
        self.data[2]=bool(value)

    @property
    def min(self):
        return self.data[3]

    @min.setter
    def min(self, value):
        self.data[3]=float(value)

    @property
    def max(self):
        return self.data[4]

    @max.setter
    def max(self, value):
        self.data[4]=float(value)

    @property
    def error(self):
        return self.data[5]

    def __repr__(self):
        """
        Display information about the parameter.
        """
        output="Parameter:\n"
        output+="           "+" ".join(["%-16s"%label for label in self._parent.data_labels])+"\n"
        output+="           "+" ".join(["%-16s"%col for col in self.data])+"\n"
        return output

    def _repr_html_(self):
        output='<table><tr><th colspan="%i"><center>Parameter</center></th></tr>\n'%(len(self._parent.data_labels)+1)
        output+="           <tr><th>No.</th><th>"+"</th><th>".join(
            ["%s"%label for label in self._parent.data_labels])+"</th></tr>\n"
        output+="           <tr><td>%i</td><td>"%self._parent.data.index(self.data)
        output+="</td><td>".join(["%s"%col for col in self.data])+"\n"
        output+="</table>"
        return output

    def _repr_ipyw_(self):
        import ipywidgets as ipw
        wname=ipw.Text(value=self.name, layout=ipw.Layout(width='35%'))
        wname=ipw.Combobox(value=self.name,
                           options=[ni for ni, oi in self._parent.model.script_module.__dict__.items()
                                    if type(oi).__name__ in ['Layer', 'Stack', 'Instrument']])
        wname.change_item='name'
        wval=ipw.FloatText(value=self.value, layout=ipw.Layout(width='20%'))
        wval.change_item='value'
        wfit=ipw.Checkbox(value=self.fit, indent=False, layout=ipw.Layout(width='5%'))
        wfit.change_item='fit'
        wmin=ipw.FloatText(value=self.min, layout=ipw.Layout(width='20%'))
        wmin.change_item='min'
        wmax=ipw.FloatText(value=self.max, layout=ipw.Layout(width='20%'))
        wmax.change_item='max'
        entries=[wname, wval, wfit, wmin, wmax]
        wname.others=entries[1:]
        for entr in entries:
            entr.observe(self._ipyw_change, names='value')
        return ipw.HBox(entries)

    def _ipyw_change(self, change):
        if change.owner.change_item=='name':
            if '.' in change.new:
                try:
                    name=change.new.split('.')[0]
                    change.owner.options=tuple("%s.%s"%(name, si)
                                               for si in dir(self._parent.model.script_module.__dict__[name])
                                               if si.startswith('set'))
                except:
                    pass
            else:
                change.owner.options=tuple(ni for ni, oi in self._parent.model.script_module.__dict__.items()
                                           if type(oi).__name__ in ['Layer', 'Stack', 'Instrument'])

            if change.new=='':
                self.data[0]=''
                change.owner.description=''
                return
            prev=self.name
            try:
                self.name=change.new
            except Exception as err:
                self.data[0]=prev
                change.owner.description='ERR'
            else:
                change.owner.description=''
                change.owner.others[0].value=self.value
                change.owner.others[1].value=self.fit
                change.owner.others[2].value=self.min
                change.owner.others[3].value=self.max
        elif change.owner.change_item=='fit':
            self.fit=change.new
        elif change.owner.change_item=='value':
            self.value=change.new
        elif change.owner.change_item=='min':
            self.min=change.new
        elif change.owner.change_item=='min':
            self.min=change.new

if __name__=='__main__':
    p=Parameters()
    p.append()
    import h5py

    f=h5py.File('test.hdf', 'w')
    g=f.create_group('parameters')
    p.write_h5group(g)
    f.close()
    p2=Parameters()
    f=h5py.File('test.hdf', 'r')
    p2.read_h5group(f['parameters'])
    iprint(p2.data)
