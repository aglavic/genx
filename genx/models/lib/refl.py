import shelve

# Helper functions for meta class

def _addMethod(fldName, clsName, verb, methodMaker, dict):
    """Make a get or set method and add it to dict."""
    compiledName = _getCompiledName(fldName, clsName)
    methodName = _getMethodName(fldName, verb)
    dict[methodName] = methodMaker(compiledName)
    
def _getCompiledName(fldName, clsName):
    """Return mangled fldName if necessary, else no change."""
    # If fldName starts with 2 underscores and does *not* end with 2 underscores...
    if fldName[:2] == '__' and fldName[-2:] != '__':
        return "_%s%s" % (clsName, fldName)
    else:
        return fldName

def _getMethodName(fldName, verb):
    """'_salary', 'get'  => 'getSalary'"""
    s = fldName.lstrip('_') # Remove leading underscores
    return verb + s.capitalize()

def _makeGetter(compiledName):
    """Return a method that gets compiledName's value."""
    return lambda self: self.__getattribute__(compiledName)

def _makeSetter(compiledName):
    """Return a method that sets compiledName's value."""    
    return lambda self, value: setattr(self, compiledName, value)

# Start Metaclass
class Meta(type):
    """Adds accessor methods to a class."""
    def __new__(cls, clsName, bases, dict):
        for fldName in dict['__dict__'].keys():
            _addMethod(fldName, clsName, 'get', _makeGetter, dict)
            _addMethod(fldName, clsName, 'set', _makeSetter, dict)

        def __init__(self, **kw):
            """ Simplistic __init__: first set all attributes to default
                values, then override those explicitly passed in kw.
            """
            for k in self.__dict__: setattr(self, k, self.__dict__[k])
            for k in kw: setattr(self, k, kw[k])
        
        dict['__init__']=__init__
        return type.__new__(cls, clsName, bases, dict)
# End Metaclass


# Class intrument-a class to accomodate the instrumental parameters
_Reciprocal=0
_Angles_=1


# Function! to create classes with the model given by the input parameters
def MakeClasses(InstrumentParameters = {'Wavelength':1.54,'Coordinates':1},\
        LayerParameters = {'d':0.0, 'sigma':0.0, 'n':0.0+0.0j},\
        StackParameters = {'Layers':[], 'Repetitions':1}, \
        SampleParameters = {'Stacks':[], 'Ambient':None, 'Substrate':None}, \
        SimulationFunctions = {'Specular':0,'OffSpecular':0},\
        ModelID = 'Standard'):
    
    class Instrument:
        __dict__=InstrumentParameters
        __metaclass__=Meta

        def __repr__(self):
            s='Instrument('
        
            for k in self.__dict__.keys():
                # if the type is a string...
                if type(self.__getattribute__(k)) == type(''):
                    stemp = "%s = '%s'," % (k, str(self.__getattribute__(k)))
                else:
                    stemp='%s = %s,' % (k, str(self.__getattribute__(k)))
                s=s+stemp
            return s[:-1]+')'

        def _todict(self):
            dic={}
            for key in self.__dict__.keys():
                dic[key] = self.__getattribute__(key)
            return dic
        def _fromdict(self,dic):
            error=0
            for key in dic.keys():
                if self.__dict__.has_key(key):
                    self.__setattr__(key,dic[key])
                else:
                    error=1
            if error:
                print 'Wrong format (Model parameters) for Instrument'
        def save(self,file):
            dbase=shelve.open(file)
            dbase['ModelID'] = ModelID
            dbase['Sample'] = self._todict()
            dbase.close()

        def load(self,file):
            dbase=shelve.open(file)
            if ModelID==dbase['ModelID']:
                self._fromdict(dbase['Sample'])
            else:
                print 'ERROR: File ModelID doesnt match the one for the Model used'
                print 'File ModelID: ', dbase['ModelID'],', Loaded ModelID: ', ModelID
            dbase.close()
        

    # Class Layer
    
    class Layer:
        __dict__= LayerParameters
        __metaclass__ = Meta

        def __repr__(self):
            s='Layer('
        
            for k in self.__dict__.keys():
                stemp='%s = %s, ' % (k, str(self.__getattribute__(k)))
                s=s+stemp
            return s[:-2]+')'
        def _todict(self):
            dic={}
            for key in self.__dict__.keys():
                dic[key] = self.__getattribute__(key) + 0.0
            return dic
        
        def _fromdict(self,dic):
            error=0
            for key in dic.keys():
                if self.__dict__.has_key(key):
                    self.__setattr__(key,dic[key])
                else:
                    error=1
            if error:
                print 'Wrong format (Model parameters) for Layer'

        
    # Class Repeat
    class Stack:
        __dict__= StackParameters
        __metaclass__ = Meta

        def __repr__(self):
            s='Stack: '
            for k in self.__dict__.keys():
                if k != 'Layers':
                    stemp='%s = %s, ' %(k,str(self.__getattribute__(k)))
                    s=s+stemp
            s=s[:-2]+'\n'
            it=len(self.Layers)
            for lay in range(it-1,-1,-1):
                s=s+'\t'+repr(self.Layers[lay])+'\n'
            return s
        def resolveLayerParameter(self,parameter):
            par=[lay.__getattribute__(parameter)+0.0 for lay in self.Layers]*self.Repetitions
            return par

        def _todict(self):
            dic={}
            for key in self.__dict__.keys():
                if key != 'Layers':
                    dic[key]=self.__getattribute__(key)
            dic['Layers']=[]
            for lay in self.Layers:
                dic['Layers'].append(lay._todict())
            return dic

        def _fromdict(self,dic):
            error=False
            for key in dic.keys():
                if self.__dict__.has_key(key):
                    if key != 'Layers':
                        self.__setattr__(key,dic[key])
                else:
                    error=True
            
            self.Layers=[]
            for lay in dic['Layers']:
                layer=Layer()
                layer._fromdict(lay)
                self.Layers.append(layer)
            
            if error:
                print 'Wrong format (Model parameters) for Stack'

    # Class Sample
            
    class Sample:
        __dict__ = SampleParameters
        __metaclass__ = Meta

        def __repr__(self):
            Add='Sample: '
            for k in self.__dict__.keys():
                if k != 'Stacks' and k!='Ambient' and k!='Substrate':
                    temp='%s = %d, ' %(k,self.__getattribute__(k))
                    Add=Add+temp
            Add=Add[:-2]+'\n'
            temp=[repr(x) for x in self.Stacks]
            temp.reverse()
            return Add+'Ambient:\n\t'+repr(self.Ambient)+'\n'+''.join(temp)+'Substrate:\n\t'+repr(self.Substrate)

        def resolveLayerParameters(self):
            par=self.Substrate.__dict__.copy()
            for k in par.keys():
                par[k]=[self.Substrate.__getattribute__(k)+0.0]
            for k in Layer().__dict__.keys():
                for stack in self.Stacks:
                    par[k] = par[k] + stack.resolveLayerParameter(k)
                par[k ]= par[k] + [self.Ambient.__getattribute__(k)+0.0]
            return par

        def _todict(self):
            dic={}
            for key in self.__dict__.keys():
                if key != 'Stacks' and key!='Ambient' and key!='Substrate':
                    dic[key]=self.__getattribute__(key)

            dic['Stacks']=[stack._todict() for stack in self.Stacks]
            dic['Ambient']=self.Ambient._todict()
            dic['Substrate']=self.Substrate._todict()
            return dic

        def _fromdict(self,dic):
            error=False
            for key in dic.keys():
                if self.__dict__.has_key(key):
                    if key != 'Stacks' and key!='Ambient' and key!='Substrate':
                        self.__setattr__(key,dic[key])
                else:
                    error=True
            if error:
                print 'Wrong format (Model parameters) for Stack'                

            self.Stacks=[]
            for sta in dic['Stacks']:
                stack=Stack()
                stack._fromdict(sta)
                self.Stacks.append(stack)
            #self.__setattr__('Stacks',[Stack()._fromdict(stack) for stack in dic['Stacks']])
            amb=Layer()
            amb._fromdict(dic['Ambient'])
            self.__setattr__('Ambient',amb)
            sub=Layer()
            sub._fromdict(dic['Substrate'])
            self.__setattr__('Substrate',sub)

        def save(self,file):
            dbase=shelve.open(file)
            dbase['ModelID'] = ModelID
            dbase['Sample'] = self._todict()
            dbase.close()

        def load(self,file):
            dbase=shelve.open(file)
            if ModelID==dbase['ModelID']:
                self._fromdict(dbase['Sample'])
            else:
                print 'ERROR: File ModelID doesnt match the one for the Model used'
                print 'File ModelID: ', dbase['ModelID'],', Loaded ModelID: ', ModelID
            dbase.close()
        
        def SimSpecular(self,TwoThetaQz,instrument):
            return SimulationFunctions['Specular'](TwoThetaQz,self,instrument)

        def SimOffSpecular(self,TwoThetaQz,ThetaQx,instrument):
            return SimulationFunctions['OffSpecular'](TwoThetaQz,ThetaQx,self,instrument)

        def SimSLD(self, z, instrument):
            if SimulationFunctions.has_key('SLD'):
                return SimulationFunctions['SLD'](z, self, instrument)
            else:
                return {}

    return (Instrument,Layer,Stack,Sample)
