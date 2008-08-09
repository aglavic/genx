'''
Definition for the class Parameters. Used for storing the fitting parameters
in GenX.
Programmer: Matts Bjorck
Last changed: 2008 02 21
'''

import numpy as np

#==============================================================================
class Parameters:
    '''
    Class for storing the fitting parameters in GenX
    '''
    def __init__(self):
        self.data_labels = ['Parameter', 'Value', 'Fit', 'Min', 'Max', 'Error']
        self.init_data = ['', 0.0, False, 0.0, 0.0, 'None']
        self.data = [self.init_data[:]]
        
        
    def set_value(self, row, col, value):
        ''' Set a value in the parameter grid '''
        self.data[row][col] = value
    
    def get_value(self, row, col):
        ''' Get the value in the grid '''
        return self.data[row][col]
        
    def get_len_rows(self):
        return len(self.data)
    
    def get_len_cols(self):
        return len(self.data[0])
        
    def get_col_headers(self):
        return self.data_labels[:]
        
    def delete_rows(self, rows):
        ''' Delete the rows in the list rows ...'''
        delete_count=0
        rows=rows[:]
        rows.sort()
        
        for i in rows:
            #Note index changes as we delete values. thats why rows has to be sorted
            try:
                self.data.pop(i-delete_count)
            except:
                pass
            else:
                delete_count+=1
                
        return delete_count
    
    def insert_row(self, row):
        ''' Insert a new row at row(int). '''
        self.data.insert(row,self.init_data[:])
        
    def append(self):
        self.data.append(self.init_data[:])
        
    def get_fit_pars(self):
        ''' Returns the variables needed for fitting '''
        print 'Data in the parameters class: ', self.data
        rows = range(len(self.data))
        row_nmb=[nmb for nmb in rows if self.data[nmb][2] and\
                not self.data[nmb][0]=='']
        funcs=[row[0] for row in self.data if row[2] and not row[0]=='']
        mytest=[row[1] for row in self.data if row[2] and not row[0]=='']
        min=[row[3] for row in self.data if row[2] and not row[0]=='']
        max=[row[4] for row in self.data if row[2] and not row[0]=='']
        return (row_nmb, funcs, mytest, min, max)
    
    def get_pos_from_row(self, row):
        '''get_pos_from_row(self) --> pos [int]
        
        Transform the row row to the position in the fit_pars list
        '''
        rows = range(row+1)
        row_nmb=[nmb for nmb in rows if self.data[nmb][2] and\
                not self.data[nmb][0]=='']
        return len(row_nmb) - 1
        
    def get_sim_pars(self):
        ''' Returns the variables needed for simulation '''
        funcs=[row[0] for row in self.data if not row[0]=='']
        mytest=[row[1] for row in self.data if not row[0]=='']
        return (funcs, mytest)
       
    def set_value_pars(self, value):
        ''' Set the values of the parameters '''
        valueindex=0
        for row in  self.data:
            if row[2] and not row[0]=='':
                row[1]=value[valueindex]
                valueindex=valueindex+1
                
    def set_error_pars(self, value):
        ''' Set the errors on the parameters '''
        valueindex=0
        for row in  self.data:
            if row[2] and not row[0]=='':
                row[5]=value[valueindex]
                valueindex=valueindex+1
                
    def set_data(self, data):
        rowi=0
        coli=0
        for row in data:
            for col in row:
                self.set_value(rowi,coli,col)
                coli=coli+1
            rowi=rowi+1
            coli=0
            
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
                    text += item.__str__()+'\t'
        return text
    
    def _parse_ascii_input(self, text):
        '''parse_ascii_input(self, text) --> list table
        
        Parses an ascii string to a parameter table. returns a list table if 
        sucessful otherwise it returns None
        '''
        table = []
        sucess = True
        lines = text.split('\n')
        for line in lines:
            # If line not commented
            if line[0] != '#':
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
                    fitted = bool(line_strs[2])
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
        print table
        if table:
            self.data = table
            return True
        else:
            return False
        
    def copy(self):
        '''get_copy(self) --> copy of Parameters
        
        Does a copy of the current object.
        '''
        new_pars = Parameters()
        new_pars.data = self.data[:]
        
        return new_pars