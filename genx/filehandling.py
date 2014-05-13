'''I/O functions for GenX. 
These include loading of initilazation files.
Also included is the config object.
File started by: Matts Bjorck

$Rev::                                  $:  Revision of last commit
$Author::                               $:  Author of last commit
$Date::                                 $:  Date of last commit
'''

import ConfigParser as CP
import StringIO

# Functions to save the gx files
#==============================================================================
def save_gx(fname, model, optimizer, config):
    model.save(fname)
    model.save_addition('config', config.model_dump())
    model.save_addition('optimizer', 
            optimizer.pickle_string(clear_evals = 
                                    not config.get_boolean('solver', 
                                                           'save all evals')))
# Not yet used ...
def load_gx(fname, model, optimizer, config):
    model.load(fname)
    config.load_model(model.load_addition('config'))   
    optimizer.pickle_load(model.load_addition('optimizer'))




#==============================================================================
class Config:
    def __init__(self):
        self.default_config = CP.ConfigParser()
        self.model_config = CP.ConfigParser()
    
    def load_default(self, filename):
        '''load_default(self, filename) --> None
        
        Loads the default config from file filename. Raises a IOError if the
        can not be found.
        '''
        try:
            self.default_config.read(filename)
        except Exception, e:
            print e
            raise IOError('Could not load default config file', filename)
    
    def write_default(self, filename):
        '''write_default(self, filename) --> None
        
        Writes the current defualt configuration to filename
        '''
        try:
            cfile = open(filename, 'wb')
            self.default_config.write(cfile)
        except Exception, e:
            print e
            raise IOError('Could not write default config file', filename)
    
    def load_model(self, str):
        '''load_model(self, str) --> None
        
        Loads a config from a string str.  Raises an IOError if the string can not be
        read.
        '''
        buffer = StringIO.StringIO(str)
        try:
            self.model_config.readfp(buffer)
        except Exception, e:
            raise IOError('Could not load model config file')
        
    def _getf(self, default_function, model_function, section, option):
        '''_getf(default_function, model_function, section, option) --> object
        
        For the function function try to locate the section and option first in
        model_config if that fails try to locate it in default_config. If both
        fails raise an OptionError.
        '''
        value = 0
        try:
            value = model_function(section, option)
        except Exception, e:
            try:
                value = default_function(section, option)
            except Exception, e:
                raise OptionError(section, option)
        
        return value
    
    def get_float(self, section, option):
        '''get_float(self, section, option) --> float
        
        returns a float value if possible for option in section
        '''
        return self._getf(self.default_config.getfloat,\
                    self.model_config.getfloat, section, option)
                    
    def get_boolean(self, section, option):
        '''get_boolean(self, section, option) --> boolean
        
        returns a boolean value if possible for option in section
        '''
        return self._getf(self.default_config.getboolean,\
                    self.model_config.getboolean, section, option)
                    
    def get_int(self, section, option):
        '''get_int(self, section, option) --> int
        
        returns a int value if possible for option in section
        '''
        return self._getf(self.default_config.getint,\
                    self.model_config.getint, section, option)
                    
    def get(self, section, option):
        '''get(self, section, option) --> string
        
        returns a string value if possible for option in section
        '''
        return self._getf(self.default_config.get,\
                    self.model_config.get, section, option)
                    
    def model_set(self, section, option, value):
        '''model_set(self, section, option, value) --> None
        
        Set a value in section, option for the model configuration.
        '''
        if not self.model_config.has_section(section):
            self.model_config.add_section(section)
        self.model_config.set(section, option, str(value))
        #print 'Model set: ', section, ' ', option, ' ', value
        
    def default_set(self, section, option, value):
        '''model_set(self, section, option, value) --> None
        
        Set a value in section, option for the model configuration.
        '''
        if not self.default_config.has_section(section):
            self.default_config.add_section(section)
        self.default_config.set(section, option, str(value))
        #print 'Default set: ', section, ' ', option, ' ', value
        
    
    def set(self, section, option, value):
        '''set(self, section, option, value) --> None
        
        Set a value in section, option for the model configuration. 
        Just a duplication of model_set.
        '''
        self.model_set(section, option, value)
        
    def model_dump(self):
        '''model_save(self, file_pointer) --> string
        
        dumps the model configuration to a string.
        '''
        # Create a buffer - file like object to trick config parser
        buffer = StringIO.StringIO()
        # write
        self.model_config.write(buffer)
        # get the string values
        str = buffer.getvalue()
        # Close the buffer - destroy it
        buffer.close()
        #print 'model content: ', str
        return str
        
    
# END: class Config
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
        text = 'Input/Output error for file:\n' + file +\
                '\n\n Python error:\n ' + error_message
  
class OptionError(GenericError):
    ''' Error class for not finding an option section pair in the 
    configuration '''
    
    def __init__(self, section, option):
        '''__init__(self, error_message)'''
        #self.error_message = error_message
        self.section = section
        self.option = option
        
    def __str__(self):
        text = 'Error in trying to loacate values in GenX configuration.' +\
                '\nCould not locate the section: ' + self.section +\
                ' or option: ' + self.option + '.'
        return text
