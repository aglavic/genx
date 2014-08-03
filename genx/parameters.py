'''
Definition for the class Parameters. Used for storing the fitting parameters
in GenX.
Programmer: Matts Bjorck
'''

import numpy as np
import string

#==============================================================================
class Parameters:
    """
    Class for storing the fitting parameters in GenX
    """
    def __init__(self, model=None):
        self.data_labels = ['Parameter', 'Value', 'Fit', 'Min', 'Max', 'Error']
        self.init_data = ['', 0.0, False, 0.0, 0.0, 'None']
        self.data = [self.init_data[:]]
        self.model = model
        
    def set_value(self, row, col, value):
        """ Set a value in the parameter grid """
        self.data[row][col] = value
    
    def get_value(self, row, col):
        """ Get the value in the grid """
        return self.data[row][col]

    def get_value_by_name(self, name):
        """Get the value for parameter name. Returns None if name can not be found.

        :param name:
        :return: Value or None
        """
        par_names = [row[0] for row in self.data]
        if name in par_names:
            value = self.data[par_names.index(name)][1]
        else:
            value = None
        return value
        
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
        rows=rows[:]
        rows.sort()
        
        for i in rows:
            #Note index changes as we delete values. thats why rows has to be sorted
            try:
                self.data.pop(i - delete_count)
            except:
                pass
            else:
                delete_count += 1
                
        return delete_count
    
    def insert_row(self, row):
        ''' Insert a new row at row(int). '''
        self.data.insert(row, self.init_data[:])

    def move_row_up(self, row):
        """ Move row up.

        :param row: Move row up one row.
        :return: Boolean True if the row was moved, otherwise False.
        """

        if row != 0 and row < self.get_len_rows():
            self.data.insert(row - 1, self.data.pop(row))
            return True
        else:
            return False

    def move_row_down(self, row):
        """ Move row up.

        :param row: Move row down one row.
        :return: Boolean True if the row was moved, otherwise False.
        """

        if row < self.get_len_rows() - 1:
            self.data.insert(row + 1, self.data.pop(row))
            return True
        else:
            return False

    def _sort_key_func(self, item):
        class_name = ''
        pname = item[0]
        obj_name = item[0]
        if self.model is not None:
            if item[0].count('.') > 0 and self.model.compiled:
                pieces = item[0].split('.')
                obj_name = pieces[0]
                pname = pieces[1]
                obj = self.model.eval_in_model(obj_name)
                class_name = obj.__class__.__name__
        print string.lower(class_name), string.lower(pname), string.lower(obj_name)

        return string.lower(class_name), string.lower(pname), string.lower(obj_name)

    def sort_rows(self):
        """ Sort the rows in the table

        :return: Boolean to indicate success (True)
        """
        self.data.sort(key=self._sort_key_func)
        return True



    def append(self):
        self.data.append(self.init_data[:])
        
    def get_fit_pars(self):
        ''' Returns the variables needed for fitting '''
        #print 'Data in the parameters class: ', self.data
        rows = range(len(self.data))
        row_nmb=[nmb for nmb in rows if self.data[nmb][2] and\
                not self.data[nmb][0]=='']
        funcs=[row[0] for row in self.data if row[2] and not row[0] == '']
        mytest=[row[1] for row in self.data if row[2] and not row[0] == '']
        min=[row[3] for row in self.data if row[2] and not row[0] == '']
        max=[row[4] for row in self.data if row[2] and not row[0] == '']
        return (row_nmb, funcs, mytest, min, max)
    
    def get_pos_from_row(self, row):
        '''get_pos_from_row(self) --> pos [int]
        
        Transform the row row to the position in the fit_pars list
        '''
        rows = range(row + 1)
        row_nmb=[nmb for nmb in rows if self.data[nmb][2] and\
                not self.data[nmb][0] == '']
        return len(row_nmb) - 1
    
    def get_sim_pars(self):
        ''' Returns the variables needed for simulation '''
        funcs = [row[0] for row in self.data if not row[0] == '']
        mytest = [row[1] for row in self.data if not row[0] == '']
        return (funcs, mytest)

    def get_sim_pos_from_row(self, row):
        '''Transform a row to a psoitions in the sim list 
        that is returned by get_sim_pars
        '''
        rows = range(row + 1)
        row_nmb=[nmb for nmb in rows if not self.data[nmb][0] == '']
        return len(row_nmb) - 1
       
    def set_value_pars(self, value):
        ''' Set the values of the parameters '''
        valueindex = 0
        for row in  self.data:
            if row[2] and not row[0] == '':
                row[1] = value[valueindex]
                valueindex = valueindex + 1
                
    def set_error_pars(self, value):
        ''' Set the errors on the parameters '''
        valueindex = 0
        for row in  self.data:
            if row[2] and not row[0] == '':
                row[5] = value[valueindex]
                valueindex = valueindex + 1

    def clear_error_pars(self):
        ''' clears the errors in the parameters'''
        for row in  self.data:
            row[5] = '-' 
                
    def set_data(self, data):
        rowi = 0
        coli = 0
        for row in data:
            for col in row:
                self.set_value(rowi, coli, col)
                coli = coli + 1
            rowi = rowi + 1
            coli = 0
            
    def get_data(self):
        return self.data[:]
    
    def get_ascii_output(self):
        '''get_ascii_output(self) --> text [string]
        
        Returns the parameters grid as an ascii string.
        '''
        text = '#'
        # Show the data labels but with a preceeding # to denote a comment
        for label in self.data_labels:
            text += label + '\t'
        text += '\n'
        for row in self.data:
            for item in row:
                # special handling of floats to reduce the 
                # col size use 5 significant digits
                if type(item) == type(10.0):
                    text += '%.4e\t'%item
                else:
                    text += item.__str__() + '\t'
            text += '\n'
        return text
    
    def _parse_ascii_input(self, text):
        '''parse_ascii_input(self, text) --> list table
        
        Parses an ascii string to a parameter table. returns a list table if 
        sucessful otherwise it returns None
        '''
        table = []
        sucess = True
        lines = text.split('\n')
        for line in lines[:-1]:
            # If line not commented
            if line[0] != '#' and line[0] != '\n':
                line_strs = line.split('\t')
                # Check the format is it valid?
                if len(line_strs) > 7 or len(line_strs) < 6:
                    if len(line_strs) == 1:
                        break
                    else:
                        sucess = False
                        break
                try:
                    par = line_strs[0]
                    val = float(line_strs[1])
                    fitted = line_strs[2].strip() == 'True'\
                                or line_strs[2].strip() == '1'
                    min = float(line_strs[3])
                    max = float(line_strs[4])
                    error = line_strs[5]
                except Exception, e:
                    sucess = False
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
        table = self._parse_ascii_input(text)
        if table:
            self.data = table
            return True
        else:
            return False
    
    def safe_copy(self, object):
        '''safe_copy(self, object) --> None
        
        Does a safe copy from object into this object.
        '''
        self.data = object.data[:]
    
    def copy(self):
        '''get_copy(self) --> copy of Parameters
        
        Does a copy of the current object.
        '''
        new_pars = Parameters()
        new_pars.data = self.data[:]
        
        return new_pars
