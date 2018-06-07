#!/usr/bin/env python
"""Provide an easy and transparent handling of fit parameters"""


__author__ = "Yannic Utz"
__copyright__ = ""
__credits__ = ["Yannic Utz and Martin Zwiebler"]
__license__ = ""
__version__ = ""
__maintainer__ = "Yannic Utz"
__email__ = "yannic.utz@tu-dresden.de"
__status__ = "Prototype"



import os.path



class Parameter(object):
    """Base class for parameter. Contains read-only property \'value\'."""
    
    def __init__(self, value):
        """Initialize and check if numeric type."""
        if not isinstance(value,(int,float,complex)):
            raise Exception("Property \'value\' has to be either of type int, float or complex.")
        self._value=value
        
    def _getValue(self):
        return self._value
    
    #exposed properties (feel like instance variables but are protected via getter and setter methods)
    value=property(_getValue)
    
    
    
        
class Fitparameter(Parameter):
    """Contains a fit parameter, its name and limits and manages them as protected properties."""
    

    def __init__(self, name, fixed=0,value=None, lower_lim=None, upper_lim=None):
        """Initialize a fitparamter at least with a name.
        
            If \'fixed=1\',the parameter will not be varied during a fit routine and the limits are not necessary."""
        
        #check types
        if not (isinstance(value,(int,float,complex)) or (value is None)):
            raise TypeError("Property \'value\' has to be either of type int, float or complex.")
        if not isinstance(name,str):
            raise TypeError("Property \'name\' has to be a string.")
        if not ((fixed==1 or fixed==0) or (fixed is None)):
            raise ValueError("Property \'fixed\' has to be either 0 or 1.")
        if not (isinstance(lower_lim,(int,float,complex)) or (lower_lim is None)):
            raise TypeError("Property \'lower_lim\' has to be either of type int, float or complex.")
        if not (isinstance(upper_lim,(int,float,complex)) or (upper_lim is None)):
            raise TypeError("Property \'upper_lim\' has to be either of type int, float or complex.")
        

        
        #assign members
        self._name=name
        self._fixed=fixed
        self._value=value
        self._lower_lim=lower_lim
        self._upper_lim=upper_lim
    
    def __str__(self):
        if self._fixed==1:
            return "parameter \'"+self._name+"\': fixed, value="+str(self._value)
        elif self._fixed==0:
            return "parameter \'"+self._name+"\': not fixed, value = "+str(self._value)+", lower limit = " +str(self._lower_lim)+", upper limit = " +str(self._upper_lim)
    
    def _getName(self):
        return self._name
    
    def _getFixed(self):
        return self._fixed
    
    def _setFixed(self, fixed):
        if not (fixed==1 or fixed==0):
            raise ValueError("Property \'fixed\' has to be either 0 or 1.")
        self._fixed=fixed
    
    def _setValue(self,value):
        if not isinstance(value,(int,float,complex)):
            raise TypeError("Property \'value\' has to be either of type int, float or complex.")
        self._value=value
    
    def _getValue(self):
        return Parameter._getValue(self)

    def _getLowerLim(self):
        return self._lower_lim
    
    def _setLowerLim(self,lower_lim):
        if not isinstance(lower_lim,(int,float,complex)) :
            raise TypeError("Property \'lower_lim\' has to be either of type int, float or complex.")
        self._lower_lim=lower_lim
        
    def _getUpperLim(self):
        return self._upper_lim
    
    def _setUpperLim(self,upper_lim):
        if not isinstance(upper_lim,(int,float,complex)) :
            raise TypeError("Property \'upper_lim\' has to be either of type int, float or complex.")
        self._upper_lim=lower_lim
    
    
    #public methods
    
    def fix(self):
        """Fix parameter during fitting."""
        self._fixed=1
    
    def unfix(self):
        """Set parameter as variable during fitting."""
        self._fixed=0

    
    
    #exposed properties (feel like instance variables but are protected via getter and setter methods)
    name=property(_getName)             #read-only
    fixed=property(_getFixed,_setFixed)
    value=property(_getValue,_setValue)
    lower_lim=property(_getLowerLim,_setLowerLim)
    upper_lim=property(_getUpperLim,_setUpperLim)
    


class ParameterPool(object):
    """Collects a pool of Parameter objects and connects them with a parameter file."""
    
    def __init__(self,parfilename=None):
        """Initialize a new ParameterPool. Read parameter initialisation from file \'parfilename\'.
        
           As soon as you connect a parameter file to the pool, its initialisation values have priority 
           over local initialisations with self.NewParameter("name",fixed,start_value, lower_lim, upper_lim).
           Moreover the arrays for fitting are only deliverd when there are the same parameter in the file and
           in the pool and if every parameter is also used somewhere in the code (meaning:created with \'NewParameter()\').
        """           
        self._parPool=[]
        if parfilename is not None:
            self.ReadFromFile(parfilename)
            
    @classmethod                                                                                       #classmethod: Method is bound to the class. We dont need a reference to the instance here, but only to the class variables defining the file format
    def _convertStrToNumber(cls, string):
        """Convert a string to a corresponding number type or None if the corresponding symbol is found.
        
            This method is used to read parameters from the paramter file.
        """
        if string==cls.nonesymbol:
             return None
        elif 'j' in string or 'J' in string:
            return complex(string)
        elif '.' in string:
            return float(string)
        else:
            return int(string)
        
    def _checkParameters(self):
        """Check parameters for consitency (complex<->real), set _realUnfixedArray with a list of indices of parameters which are not fixed and real and set _complexUnfixedArray with a list of indices of parameters which are not fixed and complex."""
        complexArray=set()
        for i in range(len(self._parPool)):
            item=self._parPool[i]
            if (isinstance(item.value, complex) or isinstance(item.value, complex) or isinstance(item.value, complex)): 
                if (isinstance(item.value, complex) and isinstance(item.value, complex) and isinstance(item.value, complex)):
                    complexArray.add(i)
                else:
                    raise Exception("Parameter \'"+item.name+"\' is inconsistent. Use complex values for every number property or for none.")
        unfixedArray=set([i for i,x in enumerate([item.fixed for item in self._parPool]) if x==0])
        self._realUnfixedArray=list(unfixedArray.difference(complexArray))             #removes all complex parameter indices from unfixedArray
        self._complexUnfixedArray=list(unfixedArray.intersection(complexArray))               #keeps only complex parameter indices form unfixedArray
    

        
    #file format
    nonesymbol='-'                          #if this symbol is found in the parameter file the corresponding property is set to "None"
    commentsymbol='#'                       #Lines starting with this symbol in the parameter file are ignored
    lineorder=[0,1,2,3,4]                 #gives the order of parameter properties in a line of the parameter file. [pos of value, pos of fixed, pos of lower_lim, pos of upper_lim, name]
    
    
    #public methods
    
    def NewParameter(self,name,fixed=None,value=None,lower_lim=None,upper_lim=None):
        """Create New Parameter and return a reference to it or return the reference to an existing one."""
        par=self.GetParameter(name)
        if par is None:
            par=Fitparameter(name,fixed,value,lower_lim,upper_lim)
            self._parPool.append(par)
        return par
        
        
    def GetParameter(self,name):
        """Return existing parameter with name \'name\' or return \'None\' if not existing."""
        if not isinstance(name,str):
            raise TypeError("\'name\' has to be a string.")
        try:
            index=[item.name for item in self._parPool].index(name)
            return self._parPool[index]
        except ValueError:
            return None
    
    def ReadFromFile(self,parfilename):        
        """Read parameters from file, append them to the pool or overwrite existing once.
            
           Lines in the paramter file should be in the format: <value>  <fixed> <lower_limit> <upper_limit> <name>
           Lines starting with \'#\' are ignored.
        """
        with open(parfilename,'r') as f:
            for line in f:
                if not line[0]==self.commentsymbol:                       #ignore lines starting with '#', for comments
                    l=line.split()
      
                    #the order of the parameter properties is coded here, with the help of the lineorder array
                    try:
                        value = self._convertStrToNumber(l[self.lineorder[0]])
                        fixed = self._convertStrToNumber(l[self.lineorder[1]])
                        lower_limit =self. _convertStrToNumber(l[self.lineorder[2]])
                        upper_limit = self._convertStrToNumber(l[self.lineorder[3]])
                    except ValueError:
                        print "Invalid number format in line: "+line
                        raise
                        
                    name=l[self.lineorder[4]]

                    if not ((isinstance(value,complex) or value is None) and (isinstance(upper_limit,complex) or upper_limit is None) and (isinstance(lower_limit,complex) or lower_limit is None))  and not ((not isinstance(value,complex) or value is None) and (not isinstance(upper_limit,complex) or upper_limit is None) and (not isinstance(lower_limit,complex) or lower_limit is None)):
                        print line
                        raise ValueError("Value and Limits in the parameter file should be either all complex numbers or none of them.\n Bad line: "+line)                                
                    
                    par=self.GetParameter(name)
                    if par is None:
                        par=Fitparameter(name,fixed,value,lower_limit,upper_limit)
                        self._parPool.append(par)
                    else:
                        par.value=value
                        par.fixed=fixed
                        par.lower_lim=lower_limit
                        par.upper_lim=upper_limit
    
    def WriteToFile(self,parfilename):
        """Write all paramters to file. 
        
           Can be used to create a template for a parameter initialisation file or to store paramters after fitting.
        """   
        columnwidth=25
        if os.path.isfile(parfilename) :
            #ask if overwrite
            answer=raw_input("Do you realy want to overwrite \'"+parfilename+"\'? [y/N]")
            answer=answer.lower()
            if not(answer=='y' or answer=='yes' or answer=='j' or answer=='ja'):
                return
        headerarray=["<value>","<fixed>","<lower_limit>","<upper_limit>","<name>"]
        with open(parfilename,'w') as f:
            f.write((self.commentsymbol+headerarray[self.lineorder[0]]).ljust(columnwidth)+headerarray[self.lineorder[1]].ljust(columnwidth)+headerarray[self.lineorder[2]].ljust(columnwidth)+headerarray[self.lineorder[3]].ljust(columnwidth)+headerarray[self.lineorder[4]]+"\n")
            for p in self._parPool:
                parray=[str(item) for item in [p.value,p.fixed,p.lower_lim,p.upper_lim,p.name]]
                f.write(parray[self.lineorder[0]].ljust(columnwidth)+parray[self.lineorder[1]].ljust(columnwidth)+parray[self.lineorder[2]].ljust(columnwidth)+parray[self.lineorder[3]].ljust(columnwidth)+parray[self.lineorder[4]].ljust(columnwidth)+"\n")
    
    def GetFitValues(self):
        """Return an array of parameter values in the order of occurence in the pool (order of parameter creation) of parameters which are not fixed."""
        self._checkParameters()
        return [self._parPool[index].value for index in self._realUnfixedArray]+[self._parPool[index].value.real for index in self._complexUnfixedArray]+[self._parPool[index].value.imag for index in self._complexUnfixedArray]
    
    def GetUpperLims(self):
        """Return an array of parameter upper limits in the order of occurence in the pool (order of parameter creation) of parameters which are not fixed."""
        self._checkParameters()
        return [self._parPool[index].upper_lim for index in self._realUnfixedArray]+[self._parPool[index].upper_lim.real for index in self._complexUnfixedArray]+[self._parPool[index].upper_lim.imag for index in self._complexUnfixedArray]
    
    def GetLowerLims(self):
        """Return an array of parameter lower limits in the order of occurence in the pool (order of parameter creation) of parameters which are not fixed."""
        self._checkParameters()
        return [self._parPool[index].lower_lim  for index in self._realUnfixedArray]+[self._parPool[index].lower_lim.real for index in self._complexUnfixedArray]+[self._parPool[index].lower_lim.imag for index in self._complexUnfixedArray]
    
    def SetFitValues(self,fitvalues):
        """Set the values of all parameters which are not fixed in the order of occurence in the pool (order of parameter creation).
        
           Be carefull: Don't change the parameter configuration between calls of GetFitValueArray and SetFitValueArray. Otherwise, the two arrays don't mean
           the same.
           GetFitValueArray has to be called once so that SetFitValueArray nows which parameters are not fixed.
        """
        lenreal=len(self._realUnfixedArray)
        lencomplex=len(self._complexUnfixedArray)
        if len(fitvalues)<>(lenreal+2*lencomplex):
            raise Exception("Given fit value array is "+str(len(fitvalues))+" entries long but should have a length of "+str(lenreal+2*lencomplex)+".")
        for i in range(lenreal):
            self._parPool[self._realUnfixedArray[i]].value=fitvalues[i]
        for i in range(lencomplex):
            self._parPool[self._complexUnfixedArray[i]].value=complex(fitvalues[lenreal+i],fitvalues[lenreal+lencomplex+i])
