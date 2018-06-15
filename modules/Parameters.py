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
    """Base class for parameter. Contains read-only property \'value\'. Supports creation of dependent paramters by arithmetic operations."""
    
    def __init__(self, value=None):
        """Initialize and check if numeric type."""
        if not isinstance(value,(int,float,complex)) and not value is None:
            raise Exception("Property \'value\' has to be either of type int, float or complex.")
        self._value=value
        
    def getValue(self,fitpararray=None):
        """Returns the value. fitpararray is not used and only there to make this function forward compatible with the Fitparamter class and with dependent parameters."""
        if self._value is None:
            raise Exception("Value is not initialized.")
        return self._value
    
    
    # overloading operators to allow to create dependent parameters from arithmetic operations
    def __add__(self,other):
        new = Parameter()
        if isinstance(other,Parameter):            
            new.getValue = lambda fitpararray: self.getValue(fitpararray) + other.getValue(fitpararray)
        else: 
            new.getValue = lambda fitpararray: self.getValue(fitpararray) + other
        return new
    def __radd__(self,other):
        new = Parameter()
        if isinstance(other,Parameter):            
            new.getValue = lambda fitpararray: other.getValue(fitpararray) + self.getValue(fitpararray)  
        else: 
            new.getValue = lambda fitpararray:  other + self.getValue(fitpararray)
        return new
    def __sub__(self,other):
        new = Parameter()
        if isinstance(other,Parameter):            
            new.getValue = lambda fitpararray: self.getValue(fitpararray) - other.getValue(fitpararray)
        else: 
            new.getValue = lambda fitpararray: self.getValue(fitpararray) - other
        return new
    def __rsub__(self,other):
        new = Parameter()
        if isinstance(other,Parameter):            
            new.getValue = lambda fitpararray: other.getValue(fitpararray) - self.getValue(fitpararray)  
        else: 
            new.getValue = lambda fitpararray:  other -self.getValue(fitpararray)
        return new
    def __mul__(self,other):
        new = Parameter()
        if isinstance(other,Parameter):            
            new.getValue = lambda fitpararray: self.getValue(fitpararray) * other.getValue(fitpararray)
        else: 
            new.getValue = lambda fitpararray: self.getValue(fitpararray) * other
        return new
    def __rmul__(self,other):
        new = Parameter()
        if isinstance(other,Parameter):            
            new.getValue = lambda fitpararray: other.getValue(fitpararray) * self.getValue(fitpararray)  
        else: 
            new.getValue = lambda fitpararray:  other * self.getValue(fitpararray)
        return new
    def __div__(self,other):
        new = Parameter()
        if isinstance(other,Parameter):            
            new.getValue = lambda fitpararray: self.getValue(fitpararray) / other.getValue(fitpararray)
        else: 
            new.getValue = lambda fitpararray: self.getValue(fitpararray) / other
        return new
    def __rdiv__(self,other):
        new = Parameter()
        if isinstance(other,Parameter):            
            new.getValue = lambda fitpararray: other.getValue(fitpararray) / self.getValue(fitpararray)  
        else: 
            new.getValue = lambda fitpararray:  other / self.getValue(fitpararray)
        return new
    def __pow__(self,other):
        new = Parameter()
        if isinstance(other,Parameter):            
            new.getValue = lambda fitpararray: self.getValue(fitpararray)**other.getValue(fitpararray)
        else: 
            new.getValue = lambda fitpararray: self.getValue(fitpararray)**other
        return new
    def __rpow__(self,other):
        new = Parameter()
        if isinstance(other,Parameter):            
            new.getValue = lambda fitpararray: other.getValue(fitpararray)**self.getValue(fitpararray)  
        else: 
            new.getValue = lambda fitpararray:  other**self.getValue(fitpararray)
        return new
    def __neg__(self):
        new = Parameter()
        new.getValue = lambda fitpararray: - self.getValue(fitpararray)
        return new
    def __pos__(self):
        new = Parameter()
        new.getValue = lambda fitpararray: + self.getValue(fitpararray)
        return new
    def __abs__(self):
        new = Parameter()
        new.getValue = lambda fitpararray: abs(self.getValue(fitpararray))
        return new
    
    
    
        
class Fitparameter(Parameter):
    """Contains name, starting value and limits of a paramter and knows how to make the parameter value out of an array of fitparameters if it is attached to a ParameterPool."""
    
    

    def __init__(self, name, fixed=0,start_val=None, lower_lim=None, upper_lim=None):
        """Initialize a fitparamter at least with a name.
        
            If \'fixed=1\',the parameter will not be varied during a fit routine and the limits are not necessary."""
        
        #check types
        if not (isinstance(start_val,(int,float,complex)) or (start_val is None)):
            raise TypeError("Property \'start_val\' has to be either of type int, float or complex.")
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
        if not (fixed==1 or fixed==0):
            raise ValueError("Property \'fixed\' has to be either 0 or 1.")
        self._fixed=fixed
        if isinstance(start_val,complex) or isinstance(lower_lim,complex) or isinstance(upper_lim,complex):   #if one is complex, treat all as complex
            self._complex=1
            self._start_val=complex(start_val)
            self._lower_lim=complex(lower_lim)
            self._upper_lim=complex(upper_lim)
        else:
            self._complex=0
            self._start_val=start_val
            self._lower_lim=lower_lim
            self._upper_lim=upper_lim
        self._index=None
        self.pool=None
        
    
    def __str__(self):
        if self._fixed==1:
            return "parameter \'"+self._name+"\': fixed, start_val="+str(self._start_val)
        elif self._fixed==0:
            return "parameter \'"+self._name+"\': not fixed, start_val = "+str(self._start_val)+", lower limit = " +str(self._lower_lim)+", upper limit = " +str(self._upper_lim)
    
    def _getName(self):
        return self._name
    
    def _getFixed(self):
        return self._fixed
    
    def _setFixed(self, fixed):
        if not (fixed==1 or fixed==0):
            raise ValueError("Property \'fixed\' has to be either 0 or 1.")
        if not self._fixed==fixed:
            self._fixed=fixed
            if not self.pool is None:
                self.pool.update()
    
    def _setStartVal(self,start_val):
        if not isinstance(start_val,(int,float,complex)):
            raise TypeError("Property \'start_val\' has to be either of type int, float or complex.")
        if isinstance(start_val,complex) or self._complex==1:   #if one is complex, treat all as complex
            self._start_val=complex(start_val)
            if self._complex==0:
                self._complex=1
                if not self.pool is None:
                    self.pool.update()
        else:
            self._start_val=start_val

    
    def _getStartVal(self):
        return self._start_val

    def _getLowerLim(self):
        return self._lower_lim
    
    def _setLowerLim(self,lower_lim):
        if not isinstance(lower_lim,(int,float,complex)) :
            raise TypeError("Property \'lower_lim\' has to be either of type int, float or complex.")
        if isinstance(lower_lim,complex) or self._complex==1:   #if one is complex, treat all as complex
            self._lower_lim=complex(lower_lim)
            if self._complex==0:
                self._complex=1
                if not self.pool is None:
                    self.pool.update()
        else:
            self._lower_lim=lower_lim
        
    def _getUpperLim(self):
        return self._upper_lim
    
    def _setUpperLim(self,upper_lim):
        if not isinstance(upper_lim,(int,float,complex)) :
            raise TypeError("Property \'upper_lim\' has to be either of type int, float or complex.")
        if isinstance(upper_lim,complex) or self._complex==1:   #if one is complex, treat all as complex
            self._upper_lim=complex(upper_lim)
            if self._complex==0:
                self._complex=1
                if not self.pool is None:
                    self.pool.update()
        else:
            self._upper_lim=upper_lim
    
    def _setIndex(self,index=None):
        if not (isinstance(index,int) or index is None):
            raise TypeError("\'index\' has to be of type int or None.")
        if self._fixed==1 and not index is None:
            raise Exception("Parameter \'"+self._name+"\' is fixed and cannot carry an index.")
        self._index=index
        
    
    def _getIndex(self):
        return self._index
    
    def _getComplex(self):
        return self._complex
    
    #public methods
    
    def fix(self):
        """Fix parameter during fitting."""
        self._fixed=1
        if not self.pool is None:
            self.pool.update()
    
    def unfix(self):
        """Set parameter as variable during fitting."""
        self._fixed=0
        if not self.pool is None:
            self.pool.update()
          
    def getValue(self,fitpararray):
        """Return the value of the parameter corresponding to the given array of values."""        
        if self._fixed==1:
            self._start_val
        elif self._index is None:
            raise Exception("Parameter has to be attached to a ParameterPool.")
        elif self.pool.GetFitArrayLen()<>len(fitpararray):
            raise Exception("Given fit value array is "+str(len(fitpararray))+" entries long but should have a length of "+str(self.pool.GetFitArrayLen())+".")
        elif self._complex==1:
            return complex(fitpararray[self._index],fitpararray[self._index+1])
        else:
            return fitpararray[self._index]
  
        
    
    
    #exposed properties (feel like instance variables but are protected via getter and setter methods)
    name=property(_getName)             #read-only
    fixed=property(_getFixed,_setFixed)
    start_val=property(_getStartVal,_setStartVal)
    lower_lim=property(_getLowerLim,_setLowerLim)
    upper_lim=property(_getUpperLim,_setUpperLim)
    index=property(_getIndex,_setIndex)
    complex=property(_getComplex)
    


class ParameterPool(object):
    """Collects a pool of Parameter objects and connects them with a parameter file."""
    
    def __init__(self,parfilename=None):
        """Initialize a new ParameterPool. Read parameter initialisation from file \'parfilename\'.
        
           As soon as you connect a parameter file to the pool, its initialisation values have priority 
           over local initialisations with self.NewParameter("name",fixed,start_value, lower_lim, upper_lim).
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
    
    @classmethod
    def _convertNumberToStr(cls,number):
        """Convert a number to a corresponding string or to \'nonesymbol\' if number is None..
        
            This method is used to write parameters from the paramter file.
        """
        if number is None:
            return cls.nonesymbol
        else: 
            return str(number)

        
    #file format
    nonesymbol='-'                          #if this symbol is found in the parameter file the corresponding property is set to "None"
    commentsymbol='#'                       #Lines starting with this symbol in the parameter file are ignored
    lineorder=[0,1,2,3,4]                 #gives the order of parameter properties in a line of the parameter file. [pos of start_value, pos of fixed, pos of lower_lim, pos of upper_lim, name]
    
    
    #public methods
    
    def NewParameter(self,name,fixed=0,start_val=None,lower_lim=None,upper_lim=None):
        """Create a New Parameter and return a reference to it or return the reference to an existing one."""
        par=self.GetParameter(name)
        if par is None:
            par=Fitparameter(name,fixed,start_val,lower_lim,upper_lim)
            self._parPool.append(par)
            par.pool=self
        self.update()
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
            
           Lines in the paramter file should be in the format: <start_value>  <fixed> <lower_limit> <upper_limit> <name>
           Lines starting with \'#\' are ignored.
        """
        with open(parfilename,'r') as f:
            for line in f:
                if not line[0]==self.commentsymbol:                       #ignore lines starting with '#', for comments
                    l=line.split()
      
                    #the order of the parameter properties is coded here, with the help of the lineorder array
                    try:
                        start_value = self._convertStrToNumber(l[self.lineorder[0]])
                        fixed = self._convertStrToNumber(l[self.lineorder[1]])
                        lower_limit =self. _convertStrToNumber(l[self.lineorder[2]])
                        upper_limit = self._convertStrToNumber(l[self.lineorder[3]])
                    except ValueError:
                        print "Invalid number format in line: "+line
                        raise
                        
                    name=l[self.lineorder[4]]

                    if not ((isinstance(start_value,complex) or start_value is None) and (isinstance(upper_limit,complex) or upper_limit is None) and (isinstance(lower_limit,complex) or lower_limit is None))  and not ((not isinstance(start_value,complex) or start_value is None) and (not isinstance(upper_limit,complex) or upper_limit is None) and (not isinstance(lower_limit,complex) or lower_limit is None)):
                        print line
                        raise ValueError("Start value and limits in the parameter file should be either all complex numbers or none of them.\n Bad line: "+line)                                
                    
                    par=self.GetParameter(name)
                    if par is None:
                        par=Fitparameter(name,fixed,start_value,lower_limit,upper_limit)
                        self._parPool.append(par)
                        par.pool=self
                    else:
                        par.start_val=start_value
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
        headerarray=["<start_value>","<fixed>","<lower_limit>","<upper_limit>","<name>"]
        with open(parfilename,'w') as f:
            f.write((self.commentsymbol+headerarray[self.lineorder[0]]).ljust(columnwidth)+headerarray[self.lineorder[1]].ljust(columnwidth)+headerarray[self.lineorder[2]].ljust(columnwidth)+headerarray[self.lineorder[3]].ljust(columnwidth)+headerarray[self.lineorder[4]]+"\n")
            for p in self._parPool:
                parray=[self._convertNumberToStr(item) for item in [p.start_val,p.fixed,p.lower_lim,p.upper_lim,p.name]]
                f.write(parray[self.lineorder[0]].ljust(columnwidth)+parray[self.lineorder[1]].ljust(columnwidth)+parray[self.lineorder[2]].ljust(columnwidth)+parray[self.lineorder[3]].ljust(columnwidth)+parray[self.lineorder[4]].ljust(columnwidth)+"\n")
    
    def GetStartLowerUpper(self):
        """Return a tupel of arrays of parameter start values and limits in the order of occurence in the pool (order of parameter creation) of parameters which are not fixed. Real parameters first, then the complex ones."""
        self.update()
        start_vals=[]
        lower_lims=[]
        upper_lims=[]       
        for poolindex in self._realUnfixedArray:
            if self._parPool[poolindex].start_val is None or self._parPool[poolindex].lower_lim is None or self._parPool[poolindex].upper_lim is None:
                raise Exception("Parameter \'"+self._parPool[poolindex].name+"\' is not properly initialized.")
            if not (self._parPool[poolindex].lower_lim<=self._parPool[poolindex].start_val and self._parPool[poolindex].start_val<=self._parPool[poolindex].upper_lim):
                raise Exception("Limits of parameter \'"+self._parPool[poolindex].name+"\' are not properly initialized.")
            start_vals.append(self._parPool[poolindex].start_val)
            lower_lims.append(self._parPool[poolindex].lower_lim)
            upper_lims.append(self._parPool[poolindex].upper_lim)
        for poolindex in self._complexUnfixedArray:
            if self._parPool[poolindex].start_val is None or self._parPool[poolindex].lower_lim is None or self._parPool[poolindex].upper_lim is None:
                raise Exception("Parameter \'"+self._parPool[poolindex].name+"\' is not properly initialized.")
            if not (self._parPool[poolindex].lower_lim.real<=self._parPool[poolindex].start_val.real and self._parPool[poolindex].start_val.real<=self._parPool[poolindex].upper_lim.real and self._parPool[poolindex].lower_lim.imag<=self._parPool[poolindex].start_val.imag and self._parPool[poolindex].start_val.imag<=self._parPool[poolindex].upper_lim.imag):
                raise Exception("Limits of parameter \'"+self._parPool[poolindex].name+"\' are not properly initialized.")
            start_vals.append(self._parPool[poolindex].start_val.real)
            start_vals.append(self._parPool[poolindex].start_val.imag)
            lower_lims.append(self._parPool[poolindex].lower_lim.real)
            lower_lims.append(self._parPool[poolindex].lower_lim.imag)
            upper_lims.append(self._parPool[poolindex].upper_lim.real)
            upper_lims.append(self._parPool[poolindex].upper_lim.imag)
        return (start_vals,lower_lims,upper_lims)
    
    def SetStartValues(self,fitvalues):
        """Set the start values of all parameters which are not fixed in the order of occurence in the pool (order of parameter creation). First real and then complex ones.
        
           Can be used befor WriteToFile() if you want to write out results of a fit to a file.
        """
        lenreal=len(self._realUnfixedArray)
        lencomplex=len(self._complexUnfixedArray)
        if len(fitvalues)<>(lenreal+2*lencomplex):
            raise Exception("Given fit value array is "+str(len(fitvalues))+" entries long but should have a length of "+str(lenreal+2*lencomplex)+".")
        for i in range(lenreal):
            self._parPool[self._realUnfixedArray[i]].start_val=fitvalues[i]
        for i in range(lencomplex):
            self._parPool[self._complexUnfixedArray[i]].start_val=complex(fitvalues[lenreal+2*i],fitvalues[lenreal+2*i+1])
            
    
    def GetFitArrayLen(self):
        return len(self._realUnfixedArray)+2*len(self._complexUnfixedArray)
    
    def GetNames(self):
        """Return the names of all registered Fitparamters as a list in the same order they should be in the fitpararrays."""
        self.update()
        namearray=[]
        for poolindex in self._realUnfixedArray:
            namearray.append(self._parPool[poolindex].name)
        for poolindex in self._complexUnfixedArray:
            namearray.append(self._parPool[poolindex].name+".real")
            namearray.append(self._parPool[poolindex].name+".imag")
        return namearray
            
        
    def update(self):
        """Look for changes within the parameter properties and inform them about changes of their indices."""
        self._realUnfixedArray=[]
        self._complexUnfixedArray=[]
        i=0
        for item in self._parPool:
            item.index=None
            if item.fixed==0 and item.complex==0:
                self._realUnfixedArray.append(i)
            elif item.fixed==0 and item.complex==1:
                self._complexUnfixedArray.append(i)
            i+=1
        i=0
        for poolindex in self._realUnfixedArray:
            self._parPool[poolindex].index=i
            i+=1
        for poolindex in self._complexUnfixedArray:
            self._parPool[poolindex].index=i
            i+=2

