'''
Library for the classes to store the data. The class DataSet stores 
on set and the class DataList stores multiple DataSets.
Programmer Matts Bjorck
Last changed: 2008 08 22
'''

from numpy import *
import os
import time

#==============================================================================
#BEGIN: Class DataSet

class DataSet:
    ''' Class to store each dataset to fit. To fit several items instead the.
        Contains x,y,error values and xraw,yraw,errorraw for the data.
    '''
    # Parameters used for saving the object state
    export_parameters = {'x':array, 'y':array, 'y_sim':array, 'y_fom':array, 'error':array, 'x_raw':array,
                         'y_raw':array, 'error_raw':array, 'extra_data':array,
                         'extra_data_raw':array, 'extra_commands':str, 'x_command':str, 'y_command':str,
                         'error_command':str, 'show':bool, 'name':str, 'use':bool, 'use_error':bool, 'cols':tuple,
                         'data_color':tuple, 'sim_color':tuple, 'data_symbol':str,
                         'data_symbolsize':int, 'data_linetype':str, 'data_linethickness':int, 'sim_symbol':str,
                         'sim_linetype':str, 'sim_linethickness':int,
                         }

    def __init__(self, name='', copy_from=None):
        #Processed data
        self.x = array([])
        self.y = array([])
        self.y_sim = array([])
        self.y_fom = array([])
        self.error = array([])
        # The raw data
        self.x_raw = array([])
        self.y_raw = array([])
        self.error_raw = array([])
        
        self.extra_data = {}
        self.extra_data_raw = {}
        # This is to add datasets that can be oprated upon as x,y and z
        self.extra_commands = {}
        
        # The different commands to transform raw data to normal data
        self.x_command = 'x'
        self.y_command = 'y'
        self.error_command = 'e'
        
        # Should we display the dataset, ie plot it
        # This should be default for ALL datasets..
        self.show = True
        
        # Special list for settings when setting the plotting properties
        self.plot_setting_names = ['color', 'symbol', 'symbolsize', 'linetype',\
                              'linethickness']
        # Name of the data set
        if name == '':
            self.name = 'New Data'
        else:
            self.name = name
            
        if copy_from:
            #Should the dataset be used for fitting?
            self.use = copy_from.use
            #Should the error be used
            self.use_error = copy_from.use_error
            #The columns to load
            self.cols = copy_from.cols # Columns to load (xcol,ycol,ecol)
            #The different colors for the data and simulation
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
        else:
            #Should the dataset be used for fitting?
            self.use = True
            #Should the error be used
            self.use_error = False
            #The columns to load
            self.cols = (0,1,1) # Columns to load (xcol,ycol)
            #The different colors for the data and simulation
            self.data_color = (0.0, 0.0, 1.0)
            self.sim_color = (1.0, 0.0, 0.0)
            # The different linetypes and symbols incl. sizes
            self.data_symbol = 'o'
            self.data_symbolsize = 4
            self.data_linetype = '-'
            self.data_linethickness = 2
            self.sim_symbol = ''
            self.sim_symbolsize = 1
            self.sim_linetype = '-'
            self.sim_linethickness = 2

    def write_h5group(self, group):
        """ Write the parameters to a hdf group

        :param group: h5py Group to write to
        :return:
        """
        for par in self.export_parameters:
            obj = getattr(self, par)
            if type(obj) is dict:
                sub_group = group.create_group(par)
                for key in obj:
                    item = obj[key]
                    if type(item) in [float, int, str, ndarray, array]:
                        sub_group[key] = item
            else:
                group[par] = obj

    def read_h5group(self, group):
        """ Read parameters from a hdf group

        :param group: h5py Group to read from
        :return:
        """
        for par in self.export_parameters:
            obj = getattr(self, par)
            if type(obj) is dict:
                sub_group = group[par]
                for key in sub_group:
                    if self.export_parameters[par] is array:
                        obj[key] = sub_group[key].value
                    else:
                        obj[key] = self.export_parameters[par](sub_group[key].value)
            else:
                if self.export_parameters[par] is array:
                    setattr(self, par, group[par].value)
                else:
                    setattr(self, par, self.export_parameters[par](group[par].value))

            
    def copy(self):
        ''' Make a copy of the current Data Set'''
        cpy = DataSet()
        cpy.safe_copy(self)
        return cpy
            
    def safe_copy(self, new_set):
        '''safe_copy(self, new_set) --> None
        
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
        #Should the error be used
        self.use_error = new_set.use_error
        #The columns to load
        #The different colors for the data and simulation
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
        # This is protection for an infinite recursion bug in python 2.X!
        # See: http://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
        if attr in ["extra_data", "extra_data_raw"]:
            raise AttributeError()
        if attr in self.extra_data:
            return self.extra_data[attr]
        elif attr.rstrip('_raw') in self.extra_data_raw:
            return self.extra_data_raw[attr.rstrip('_raw')]

        raise AttributeError("%r object has no attribute %r" %
                            (self.__class__, attr))
            
    def get_extra_data_names(self):
        '''get_extra_data_names(self) --> names [list]
        
        returns the names of the extra data
        '''
        return self.extra_data.keys()
    
    def set_extra_data(self, name, value, command = None):
        '''set_extra_data_names(self, name, value, command = None)
        
        sets extra data name, if it does not exist a new entry is created.
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
    
    def get_extra_data(self, name):
        '''get_extra_data(self, name) --> object
        
        returns the extra_data object with name name [string] if does not
        exist an LookupError is yielded.
        '''
        if not self.extra_data.has_key(name):
            raise LookupError('Can not find extra data with name %s'%name)
        return self.extra_data[name]
    
    def loadfile(self,filename, sep='\t', pos=0):
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
        try:
            f=open(filename)
            #f.close()
        except:
            print "Can't open file: %s"%filename
        else:
            try:
                A = loadtxt(f)
                #, comments = '#', delimeter = None, skiprows = 0
            except:
                print "Can't read the file %s, check the format"%filename
            else:
                #print A
                xcol=self.cols[0]
                ycol=self.cols[1]
                ecol=self.cols[2]
                #print xcol,ycol
                if xcol<A.shape[1] and ycol<A.shape[1] and ecol<A.shape[1]:
                    self.x_raw=A[:,xcol].copy()
                    self.y_raw=A[:,ycol].copy()
                    self.error_raw=A[:,ecol].copy()
                    self.x=A[:,xcol]
                    self.y=A[:,ycol]
                    self.error=A[:,ecol]
                    self.y_sim = array([])
                    print "Sucessfully loaded %i datapoints"%(A.shape[0])
                    return True
                else:
                    print "There are not enough columns in your data\n\
                     As I see it there are %i columns"%A.shape[1]
            return False
        
    def save_file(self, filename):
        '''save_file(self, filename) --> None
        
        saves the dataset to a file with filename.
        '''
        if  self.x.shape == self.y_sim.shape and \
            self.y.shape == self.error.shape and\
            self.x.shape == self.y.shape:
            # save the file
            #print self.y.shape, self.y_sim.shape
            #print c_[self.x, self.y_sim, self.y, self.error]
            f = open(filename, 'w')
            f.write('# Dataset "%s" exported from GenX on %s\n'%\
                            (self.name, time.ctime()))
            f.write('# Column lables:\n')
            f.write('# x\tI_simulated\tI\terror(I)\n')
            savetxt(f, c_[self.x, self.y_sim, self.y, self.error])
        else:
            debug = 'y_sim.shape: ' + str(self.y_sim.shape) + '\ny.shape: ' +\
            str(self.y.shape) + '\nx.shape: ' + str(self.x.shape) +\
            '\nerror.shape: ' + str(self.error.shape)
            #print debug
            raise IOError('The data is not in the correct format all the' +\
                    'arrays have to have the same shape:\n' + debug, filename)

    
    def run_x_command(self):
        x = self.x_raw
        y = self.y_raw
        e = self.error_raw
        
        for key in self.extra_data:
            exec('%s = self.%s_raw'%(key, key))
        
        self.x = eval(self.x_command)
        #print self.x

    def run_y_command(self):
        x = self.x_raw
        y = self.y_raw
        e = self.error_raw
        
        for key in self.extra_data:
            exec('%s = self.%s_raw'%(key, key))
        
        self.y = eval(self.y_command)
        #print self.y
        #print self.y_command
        
    def run_error_command(self):
        x = self.x_raw
        y = self.y_raw
        e = self.error_raw
        
        for key in self.extra_data:
            exec('%s = self.%s_raw'%(key, key))
        
        self.error = eval(self.error_command)
        
    def run_extra_commands(self):
        x = self.x_raw
        y = self.y_raw
        e = self.error_raw
        
        for key in self.extra_data_raw:
            exec('%s = self.%s_raw' % (key, key))

        for key in self.extra_commands:
            self.extra_data[key] = eval('self.extra_commands["%s"]' % key)
        
        
    def run_command(self):
        self.run_x_command()
        self.run_y_command()
        self.run_error_command()
        self.run_extra_commands()
        
    def try_commands(self, command_dict):
        ''' try_commands(self, command_dict) --> tuple of bool
        Evals the commands to locate any errors. Used to 
        test the commands before doing the actual setting of x,y and z
        '''
        result = ''
        
        x = self.x_raw
        y = self.y_raw
        e = self.error_raw
        
        #Know we have to do this with the extra data
        for key in self.extra_data_raw:
            exec('%s = self.%s_raw'%(key, key))
            
        xt = self.x
        yt = self.y
        et = self.error
        
        #Know we have to do this with the extra data
        for key in self.extra_data_raw:
            exec('%st = self.%s'%(key, key))
        
        # Try to evaluate all the expressions
        if command_dict['x'] != '':
            try:
                xt = eval(command_dict['x'])
            except Exception, e:
                result += 'Error in evaluating x expression.\n\nPython output:\n'\
                            + e.__str__() + '\n'
        
        if command_dict['y'] != '':
            try:
                yt = eval(command_dict['y'])
            except Exception, e:
                result += 'Error in evaluating y expression.\n\nPython output:\n'\
                        + e.__str__() + '\n'
        
        if command_dict['e'] != '':
            try:
                et = eval(command_dict['e'])
            except Exception, e:
                result += 'Error in evaluating e expression.\n\nPython output:\n'\
                        + e.__str__() + '\n'
                        
        for key in self.extra_commands:
            if command_dict[key] != '':
                try:
                    exec('%st = eval(command_dict["%s"])'%(key, key))
                except Exception, e:
                    result += 'Error in evaluating %s expression.\n\nPython output:\n'%key\
                            + e.__str__() + '\n'
            
        # If we got an error - report it
        if result != '':
            return result
        #print 'Debug, datatry: ', xt, yt, et
        # Finally check so that all the arrays have the same size
        extra_shape = not all([eval('%st.shape'%key) == xt.shape for \
                            key in self.extra_commands])
        if (xt.shape != yt.shape or xt.shape != et.shape or extra_shape)\
            and result == '':
            result += 'The resulting arrays are not of the same size:\n' + \
                       'len(x) = %d, len(y) = %d, len(e) = %d'\
                    %(xt.shape[0], yt.shape[0], et.shape[0])
            for key in self.extra_commands:
                result += ', len(%s) = %d'%(key, eval('%st.shape[0]'%key))
        return result
            
    def get_commands(self):
        ''' get_commands(self) --> list of dicts
        Returns the commnds as a dictonary with items x, y, z
        '''
        cmds = {'x':self.x_command, 'y':self.y_command, 'e':self.error_command}
        for key in self.extra_commands:
            cmds[key] = self.extra_commands[key]
        return cmds
    
    def set_commands(self, command_dict):
        ''' set_commands(self, command_dict) --> None
        Sets the commands in the data accroding to values in command dict
        See get_commands for more details
        '''
        if command_dict['x'] != '':
            self.x_command = command_dict['x']
        if command_dict['y'] != '':
            self.y_command = command_dict['y']
        if command_dict['e'] != '':
            self.error_command = command_dict['e']
        # Lets do it for the extra commands as well
        for key in command_dict:
            if self.extra_commands.has_key(key):
                if command_dict[key] != '':
                    self.extra_commands[key] = command_dict[key]
    
    def set_simulated_data(self, simulated_data):
        self.y_sim = simulated_data
        
    def set_fom_data(self, fom_data):
        self.y_fom = fom_data
        
    def get_sim_plot_items(self):
        '''get_sim_plot_items(self) --> dict
        Returns a dictonary of color [tuple], symbol [string], 
        sybolsize [float], linetype [string], linethickness [float].
        Used for plotting the simulation.
        '''
        return {'color': (self.sim_color[0]*255, self.sim_color[1]*255,\
     self.sim_color[2]*255),\
                'symbol': self.sim_symbol,\
                'symbolsize': self.sim_symbolsize,\
                'linetype': self.sim_linetype,\
                'linethickness': self.sim_linethickness\
               }
    def get_data_plot_items(self):
        '''get_data_plot_items(self) --> dict
        Returns a dictonary of color [tuple], symbol [string], 
        sybolsize [float], linetype [string], linethickness [float].
        Used for plotting the data.
        '''
        return {'color': (self.data_color[0]*255, self.data_color[1]*255,\
     self.data_color[2]*255),\
                'symbol': self.data_symbol,\
                'symbolsize': self.data_symbolsize,\
                'linetype': self.data_linetype,\
                'linethickness': self.data_linethickness\
               }
    
    def set_data_plot_items(self, pars):
        ''' set_data_plot_items(self, pars) --> None
        Sets the plotting parameters for the data by a dictonary of the
        same structure as in get_data_plot_items(). If one of items in the 
        pars [dictonary] is None that item will be skipped, i.e. keep its old
        value.
        '''
        #print 'data set_data_plot_items: '
        #print pars
        for name in self.plot_setting_names:
            if pars[name] != None:
                if type(pars[name]) == type(''):
                    exec('self.data_' + name + ' = "' \
                            + pars[name].__str__() + '"')
                elif name == 'color':
                    c = pars['color']
                    self.data_color = (c[0]/255.0, c[1]/255.0, c[2]/255.0)
                else:
                    exec('self.data_' + name + ' = ' + pars[name].__str__())
                
    def set_sim_plot_items(self, pars):
        ''' set_data_plot_items(self, pars) --> None
        Sets the plotting parameters for the data by a dictonary of the
        same structure as in get_data_plot_items(). If one of items in the 
        pars [dictonary] is None that item will be skipped, i.e. keep its old
        value.
        '''
        #print 'data set_sim_plot_items: '
        #print pars
        for name in self.plot_setting_names:
            if pars[name] != None:
                if type(pars[name]) == type(''):
                    exec('self.sim_' + name + ' = "' \
                            + pars[name].__str__() + '"')
                elif name == 'color':
                    c = pars['color']
                    self.sim_color = (c[0]/255.0, c[1]/255.0, c[2]/255.0)
                else:
                    exec('self.sim_' + name + ' = ' + pars[name].__str__())

    def set_show(self, val):
        '''Set show true - show data set in plots
        '''
        self.show = bool(val)
        
        
#END: Class DataSet
#==============================================================================
#BEGIN: Class DataList
class DataList:
    ''' Class to store a list of DataSets'''
    
    def __init__(self):
        ''' init function - creates a list with one DataSet'''
        self.items = [DataSet(name='Data 0')]
        self._counter = 1

    def write_h5group(self, group):
        """ Write parameters to a hdf group

        :param group: h5py Group to write to
        :return:
        """
        data_group = group.create_group('datasets')
        for index, data in enumerate(self.items):
            data.write_h5group(data_group.create_group('%d'%index))
        group['_counter'] = self._counter

    def read_h5group(self, group):
        """ Read parameters from a hdf group

        :param group: h5py Group to read from
        :return:
        """
        data_group = group['datasets']
        self.items = []
        for index in range(len(data_group)):
            self.items.append(DataSet())
            self.items[-1].read_h5group(data_group['%d'%index])
        self._counter = int(group['_counter'].value)

        
    def __getitem__(self,key):
        '''__getitem__(self,key) --> DataSet
        
        returns item at position key
        '''
        return self.items[key]
        
    def __iter__(self):
        ''' __iter__(self) --> iterator
        
        Opertor definition. Good to have in case one needs to loop over
        all datasets
        '''
        return self.items.__iter__()
        
    def __len__(self):
        '''__len__(self) --> length (integer)
        
        Returns the nmber of datasers in the list.
        '''
        return self.items.__len__()
    
    def safe_copy(self, new_data):
        '''safe_copy(self, new_data) --> None
        
        Conduct a safe copy of a data set into this data set.
        This is intended to produce version safe import of data sets.
        '''
        self.items = []
        for new_set in new_data:
            self.items.append(DataSet())
            self.items[-1].safe_copy(new_set)
        
    def add_new(self,name=''):
        ''' add_new(self,name='') --> None
        
        Adds a new DataSet with the optional name. If name not sets it 
        will be given an automatic name
        '''
        if name=='':
            self.items.append(DataSet('Data %d'%self._counter,\
                        copy_from=self.items[-1]))
            self._counter += 1
        else:
            self.items.append(DataSet(name,copy_from=self.items[-1]))
        #print "An empty dataset is appended at postition %i."%(len(self.items)-1)

    def delete_item(self, pos):
        '''delete_item(self,pos) --> None        
        
        Deletes the item at position pos. Only deletes if the pos is an 
        element and the number of datasets are more than one.
        ''' 
        if pos<len(self.items) and len(self.items)>1:
            self.items.pop(pos)
            #print "Data set number %i have been removed."%pos
            return True
        else:
            #print 'Can not remove dataset number %i.'%pos
            return False
            
    def move_up(self, pos):
        '''move_up(self, pos) --> None
        
        Move the data set at position pos up one step. If it is at the top
        it will not be moved.
        '''
        if pos != 0:
            tmp = self.items.pop(pos)
            self.items.insert(pos-1, tmp)
            
    def move_down(self,pos):
        '''
        move_down(self,pos) --> None
        
        Move the dataset at postion pos down one step. If it is at the bottom
        it will not be moved.
        '''
        if pos != len(self.items):
            tmp = self.items.pop(pos)
            self.items.insert(pos + 1, tmp)

    def update_data(self):
        ''' update_data(self) --> None
        
        Calcultes all the values for the current items. 
        '''
        [item.run_command() for item in self.items]
        
    def set_simulated_data(self, sim_data):
        '''
        set_simulated_data(self, sim_data) --> None
        
        Sets the simualted data in the data. Note this will depend on the
        flag use in the data.
        '''
        [self.items[i].set_simulated_data(sim_data[i]) for i in\
            range(self.get_len())]
        
    def set_fom_data(self, fom_data):
        '''
        set_fom_data(self, fom_data) --> None
        
        Sets the point by point fom data in the data. Note this will depend on the
        flag use in the data.
        '''
        [self.items[i].set_fom_data(fom_data[i]) for i in\
            range(self.get_len())]
        
    def get_len(self):
        return len(self.items)
        
    def get_name(self,pos):
        '''
        get_name(self,pos) --> name (string)
        
        Yields the name(string) of the dataset at position pos(int). 
        '''
        return self.items[pos].name
        
    def get_cols(self,pos):
        return self.items[pos].cols
    
    def get_use(self, pos):
        '''get_use_error(self, pos) --> bool
        returns the flag use for dataset at pos [int].
        '''
        return self.items[pos].use
    
    def get_use_error(self, pos):
        '''get_use_error(self, pos) --> bool
        returns the flag use_error for dataset at pos [int].
        '''
        return self.items[pos].use_error
    
    def toggle_use_error(self, pos):
        '''toggle_use_error(self, pos) --> None
        Toggles the use_error flag for dataset at position pos.
        '''
        self.items[pos].use_error = not self.items[pos].use_error
        
    def toggle_use(self, pos):
        '''toggle_use(self, pos) --> None
        Toggles the use flag for dataset at position pos.
        '''
        self.items[pos].use = not self.items[pos].use
        
    def toggle_show(self, pos):
        '''toggle_show(self, pos) --> None
        Toggles the show flag for dataset at position pos.
        '''
        self.items[pos].show = not self.items[pos].show

    def show_items(self, positions):
        '''show_items(self, positions) --> None
        Will put the datasets at positions [list] to show all 
        other of no show, hide.
        '''
        [item.set_show(i in positions) for i, item in enumerate(self.items)]

    def set_name(self,pos,name):
        '''
        set_name(self,pos,name) --> None
        
        Sets the name of the data set at position pos (int) to name (string)
        '''
        self.items[pos].name=name
    
    def export_data_to_files(self, basename, indices = None):
        '''export_data_to_files(self, basename, indices = None) --> None
        
        saves the data to files with base name basename and extentions .dat
        If indices are used only the data given in the list indices are 
        exported.
        '''
        # Check if we shoudlstart picking data sets to export
        
        if indices:
            if not sum([i < len(self.items) for i in indices]) == len(indices):
                raise 'Error in export_data_to_files'
        else:
            indices = range(len(self.items))
        #print 'Output: ', indices, len(self.items)
        for index in indices:
            base, ext = os.path.splitext(basename)
            if ext == '':
                ext = '.dat'
            self.items[index].save_file(base + '%03d'%index + ext)
        
    
    def get_data_as_asciitable(self, indices = None):
        ''' get_data_as_table(self, indices = None) --> string
        
        Yields the data sets as a ascii table with tab seperated values.
        This makes it possible to export the data to for example spreadsheets.
        Each data set will be four columns with x, Meas, Meas error and Calc.
        If none is given all the data sets are transformed otherwise incdices
        shouldbe a list.
        '''
        
        if indices:
            if not sum([i < len(self.items) for i in indices]) == len(indices):
                raise 'Error in get_data_as_asciitable'
        else:
            indices = range(len(self.items))
        
        #making some nice looking header so the user know what is what
        header1=''.join(['%s\t\t\t\t'%self.items[index].name\
                            for index in indices])
        header2=''.join(['x\ty\ty error\ty sim\t' for index in indices])
        
        # Find the maximum extent of the data sets
        maxlen=max([len(item.y_sim) for item in self.items])
        
        # Create the funtion that actually do the exporting
        def exportFunc(index,row):
            item = self.items[index]
            if row < len(item.x):
                return '%e\t%e\t%e\t%e\t'%(item.x[row], item.y[row],\
                                            item.error[row], item.y_sim[row])
            else:
                return ' \t \t \t \t'
        # Now create the data
        text_data = ''.join(['\n' + ''.join([exportFunc(index,row)\
                        for index in indices])\
                        for row in range(maxlen)])
        return header1 + '\n' + header2 + text_data
        
#==============================================================================
#Some Exception definition for errorpassing
class GenericError(Exception):
    ''' Just a empty class used for inheritance. Only useful
    to check if the errors are originating from the model library.
    All these errors are controllable. If they not originate from
    this class something has passed trough and that should be impossible '''
    pass
    
class IOError(GenericError):
    ''' Error class for input output, mostly concerning files'''
    
    def __init__(self, error_message, file = ''):
        '''__init__(self, error_message)'''
        self.error_message = error_message
        self.file = file
        
    def __str__(self):
        text = 'Input/Output error for file:\n' + self.file +\
                '\n\n Python error:\n ' + self.error_message
        return text

if __name__ == '__main__':
    import h5py
    d = DataList()
    d[0].x = arange(0, 10)
    f = h5py.File('myfile.hdf5', 'w')
    dic = f.create_group('data')
    d.write_h5group(dic)
    f.close()

    d = DataList()
    print 'x ', d[0].x
    f = h5py.File('myfile.hdf5', 'r')
    dic = f['data']
    d.read_h5group(dic)
    print 'x ', d[0].x
    f.close()
    print 'x ', d[0].x
    print type(d[0].data_color), d[0].data_color
    print d[0].extra_data
    print type(d[0].data_symbolsize)
