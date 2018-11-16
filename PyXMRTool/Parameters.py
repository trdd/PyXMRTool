#<PyXMRTool: A Python Package for the analysis of X-Ray Magnetic Reflectivity data measured on heterostructures>
#    Copyright (C) <2018>  <Yannic Utz>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""Provides an easy and transparent handling of fit parameters.

Fitparameters can be handled as instances of :class:`.Fitparameter` which are collected and managed by an instance of :class:`.ParameterPool`.
To allow for transparent modelling, :class:`.Fitparameter` is derived from :class:`.Parameter` which can hold a constant value.
I.e. the code using a parameter does not have to know if it is a constant :class:`.Parameter` or a variable :class:`.Fitparameter`.
The values of both are obtained in the same way with **getValue**.

Derived parameters which are arbitrary functions of other parameters can be defined with instance of :class:`.DerivedParameter`.

All parameter classes support also the creation of simple derived paramters by arithmetic operations.

.. doctest::

    >>> import Parameters
    >>> import math
    >>> const=Parameters.Parameter(30.7)
    >>> const.getValue()
    30.7
    >>> pp=Parameters.ParameterPool()
    >>> par_real=pp.newParameter('par_real',start_val=0,lower_lim=-10,upper_lim=10)
    >>> par_complex=pp.newParameter('par_complex',start_val=0+0j,lower_lim=-10-1j,upper_lim=100.3+20j)
    >>> fitpararray = [1,2,3]
    >>> par_real.getValue(fitpararray)
    1
    >>> par_complex.getValue(fitpararray)
    (2+3j)
    >>> simple_derived_par = ((par_complex + 5*par_real)*0.68)**const
    >>> simple_derived_par.getValue(fitpararray)
    (8.367641368810413e+21-1.1467109965804664e+21j)
    >>> def sinus(A,w): return A*math.sin(w)
    >>> derived_par = Parameters.DerivedParameter(sinus, const, par_real)
    >>> derived_par.getValue(fitpararray)
    25.833159233602423
    >>> print (derived_par**2).getValue(fitpararray)
    667.352115989
    
"""


#Python Version 2.7
__author__ = "Yannic Utz"
__copyright__ = ""
__credits__ = ["Yannic Utz and Martin Zwiebler"]
__license__ = "GNU General Public License v3.0"
__version__ = "0.9"
__maintainer__ = "Yannic Utz"
__email__ = "yannic.utz@tu-dresden.de"
__status__ = "beta"



import os.path



class Parameter(object):
    """
    Base class for parameters. It contains a (complex) value which is set as instantiation and cannot be changed.
    
    This class (and all derived classes) supports creation of dependent parameters by arithmetic operations.
    """
    
    def __init__(self, value=None):
        """Initialize with a number as (complex) **value**."""
        if not isinstance(value,(int,float,complex)) and not value is None:
            raise Exception("Property \'value\' has to be either of type int, float or complex.")
        self._value=value
        
    def __str__(self):
        """Enables nice representations by str()."""
        return "<" + type(self).__module__ + "." + type(self).__name__ + " object, value="+str(self._value)+">"
        
    def getValue(self,fitpararray=None):
        """
        Returns the **value**.
        
        **fitpararray** is not used and only there to make this function forward compatible with the :class:`.Fitparameter` class and with dependent parameters.
        """
        if self._value is None:
            raise Exception("Value is not initialized.")
        return self._value
    
    
    # overloading operators to allow to create dependent parameters from arithmetic operations
    def __add__(self,other):
        new = Parameter()
        if isinstance(other,Parameter):            
            new.getValue = lambda fitpararray=None: self.getValue(fitpararray) + other.getValue(fitpararray)
        else: 
            new.getValue = lambda fitpararray=None: self.getValue(fitpararray) + other
        return new
    def __radd__(self,other):
        new = Parameter()
        if isinstance(other,Parameter):            
            new.getValue = lambda fitpararray=None: other.getValue(fitpararray) + self.getValue(fitpararray)  
        else: 
            new.getValue = lambda fitpararray=None:  other + self.getValue(fitpararray)
        return new
    def __sub__(self,other):
        new = Parameter()
        if isinstance(other,Parameter):            
            new.getValue = lambda fitpararray=None: self.getValue(fitpararray) - other.getValue(fitpararray)
        else: 
            new.getValue = lambda fitpararray=None: self.getValue(fitpararray) - other
        return new
    def __rsub__(self,other):
        new = Parameter()
        if isinstance(other,Parameter):            
            new.getValue = lambda fitpararray=None: other.getValue(fitpararray) - self.getValue(fitpararray)  
        else: 
            new.getValue = lambda fitpararray=None:  other -self.getValue(fitpararray)
        return new
    def __mul__(self,other):
        new = Parameter()
        if isinstance(other,Parameter):            
            new.getValue = lambda fitpararray=None: self.getValue(fitpararray) * other.getValue(fitpararray)
        else: 
            new.getValue = lambda fitpararray=None: self.getValue(fitpararray) * other
        return new
    def __rmul__(self,other):
        new = Parameter()
        if isinstance(other,Parameter):            
            new.getValue = lambda fitpararray=None: other.getValue(fitpararray) * self.getValue(fitpararray)  
        else: 
            new.getValue = lambda fitpararray=None:  other * self.getValue(fitpararray)
        return new
    def __div__(self,other):
        new = Parameter()
        if isinstance(other,Parameter):            
            new.getValue = lambda fitpararray=None: self.getValue(fitpararray) / other.getValue(fitpararray)
        else: 
            new.getValue = lambda fitpararray=None: self.getValue(fitpararray) / other
        return new
    def __rdiv__(self,other):
        new = Parameter()
        if isinstance(other,Parameter):            
            new.getValue = lambda fitpararray=None: other.getValue(fitpararray) / self.getValue(fitpararray)  
        else: 
            new.getValue = lambda fitpararray=None:  other / self.getValue(fitpararray)
        return new
    def __pow__(self,other):
        new = Parameter()
        if isinstance(other,Parameter):            
            new.getValue = lambda fitpararray=None: self.getValue(fitpararray)**other.getValue(fitpararray)
        else: 
            new.getValue = lambda fitpararray=None: self.getValue(fitpararray)**other
        return new
    def __rpow__(self,other):
        new = Parameter()
        if isinstance(other,Parameter):            
            new.getValue = lambda fitpararray=None: other.getValue(fitpararray)**self.getValue(fitpararray)  
        else: 
            new.getValue = lambda fitpararray=None:  other**self.getValue(fitpararray)
        return new
    def __neg__(self):
        new = Parameter()
        new.getValue = lambda fitpararray=None: - self.getValue(fitpararray)
        return new
    def __pos__(self):
        new = Parameter()
        new.getValue = lambda fitpararray=None: + self.getValue(fitpararray)
        return new
    def __abs__(self):
        new = Parameter()
        new.getValue = lambda fitpararray=None: abs(self.getValue(fitpararray))
        return new
    
    
    
        
class Fitparameter(Parameter):
    """Contains name, starting value and limits of a fitting parameter and knows how to get the parameter value out of an array of values of fitparameters if it is attached to an instance of :class:`.ParameterPool`."""
    
    

    def __init__(self, name, fixed=False,start_val=None, lower_lim=None, upper_lim=None):
        """Initialize a fitparameter at least with a **name**.
        
        Usually, you (the user) should not instantiate a fitparameter directly. Instead use :meth:`.ParameterPool.newParameter`.
        
        Parameters
        ----------
        name : str
            Name of the fit parameter. Use names without whitespaces. Otherwise there can be problems when using a parameter file.
        fixed : bool
            If `True`,the parameter will not be varied during a fit routine and the limits are not necessary.
        start_val : number
            Start value of the fit parameter.
        lower_lim, upper_lim : number
            Lower limit of the fit parameter.
        upper_lim : number 
            Upper limit of the fit parameter.
        """
        
        #check types
        if not (isinstance(start_val,(int,float,complex)) or (start_val is None)):
            raise TypeError("Property \'start_val\' has to be either of type int, float or complex.")
        if not isinstance(name,str):
            raise TypeError("Property \'name\' has to be a string.")
        if not (isinstance(fixed,bool) or (fixed is None)):
            raise TypeError("Property \'fixed\' has to be of type bool.")
        if not (isinstance(lower_lim,(int,float,complex)) or (lower_lim is None)):
            raise TypeError("Property \'lower_lim\' has to be either of type int, float or complex.")
        if not (isinstance(upper_lim,(int,float,complex)) or (upper_lim is None)):
            raise TypeError("Property \'upper_lim\' has to be either of type int, float or complex.")
        

        
        #assign members
        self._name=name
        self._fixed=fixed
        if isinstance(start_val,complex) or isinstance(lower_lim,complex) or isinstance(upper_lim,complex):   #if one is complex, treat all as complex
            self._complex=True
            self._start_val=complex(start_val)
            self._lower_lim=complex(lower_lim)
            self._upper_lim=complex(upper_lim)
        else:
            self._complex=False
            self._start_val=start_val
            self._lower_lim=lower_lim
            self._upper_lim=upper_lim
        self._index=None
        self.pool=None
        
    
    #special methods
    def __str__(self):
        """Determines how an *object* of this class react on ``str(object)``."""
        if self._fixed==True:
            return "<" + type(self).__module__ + "." + type(self).__name__ + " object \'"+self._name+"\': fixed, start value = "+str(self._start_val)+">"
        elif self._fixed==False:
            return "<" + type(self).__module__ + "." + type(self).__name__ + " object \'"+self._name+"\': not fixed, start value = "+str(self._start_val)+", lower limit = " +str(self._lower_lim)+", upper limit = " +str(self._upper_lim)+">"
    
    #private methods
    def _getName(self):
        return self._name
    
    def _getFixed(self):
        return self._fixed
    
    def _setFixed(self, fixed):
        if not isinstance(fixed,bool):
            raise TypeError("Property \'fixed\' has to be of type bool")
        if not self._fixed==fixed:
            self._fixed=fixed
            if not self.pool is None:
                self.pool._update()
    
    def _setStartVal(self,start_val):
        if not isinstance(start_val,(int,float,complex)):
            raise TypeError("Property \'start_val\' has to be either of type int, float or complex.")
        if isinstance(start_val,complex) or self._complex==True:   #if one is complex, treat all as complex
            self._start_val=complex(start_val)
            if self._complex==False:
                self._complex=True
                if not self.pool is None:
                    self.pool._update()
        else:
            self._start_val=start_val

    
    def _getStartVal(self):
        return self._start_val

    def _getLowerLim(self):
        return self._lower_lim
    
    def _setLowerLim(self,lower_lim):
        if not isinstance(lower_lim,(int,float,complex)) :
            raise TypeError("Property \'lower_lim\' has to be either of type int, float or complex.")
        if isinstance(lower_lim,complex) or self._complex==True:   #if one is complex, treat all as complex
            self._lower_lim=complex(lower_lim)
            if self._complex==False:
                self._complex=True
                if not self.pool is None:
                    self.pool._update()
        else:
            self._lower_lim=lower_lim
        
    def _getUpperLim(self):
        return self._upper_lim
    
    def _setUpperLim(self,upper_lim):
        if not isinstance(upper_lim,(int,float,complex)) :
            raise TypeError("Property \'upper_lim\' has to be either of type int, float or complex.")
        if isinstance(upper_lim,complex) or self._complex==True:   #if one is complex, treat all as complex
            self._upper_lim=complex(upper_lim)
            if self._complex==False:
                self._complex=True
                if not self.pool is None:
                    self.pool._update()
        else:
            self._upper_lim=upper_lim
    
    def _setIndex(self,index=None):
        if not (isinstance(index,int) or index is None):
            raise TypeError("\'index\' has to be of type int or None.")
        if self._fixed==True and not index is None:
            raise Exception("Parameter \'"+self._name+"\' is fixed and cannot carry an index.")
        self._index=index
        
    
    def _getIndex(self):
        return self._index
    
    def _getComplex(self):
        return self._complex
    
    #public methods
    
    def fix(self):
        """Fix parameter during fitting."""
        self._fixed=True
        if not self.pool is None:
            self.pool._update()
    
    def unfix(self):
        """Set parameter as variable during fitting."""
        self._fixed=False
        if not self.pool is None:
            self.pool._update()
          
    def getValue(self,fitpararray):
        """Return the value of the parameter corresponding to the given array of values.
        
           This method only works if the fitparameter is connected to an instance of :class:`.ParameterPool`.
           Therefore, create instances of :class:`.Fitparameter` always using :meth:`.ParameterPool.newParameter`.           
        """        
        if self._fixed==True:
            return self._start_val
        elif self._index is None:
            raise Exception("Parameter has to be attached to a ParameterPool.")
        elif self.pool.getFitArrayLen()<>len(fitpararray):
            raise Exception("Given fit value array is "+str(len(fitpararray))+" entries long but should have a length of "+str(self.pool.getFitArrayLen())+".")
        elif self._complex==True:
            return complex(fitpararray[self._index],fitpararray[self._index+1])
        else:
            return fitpararray[self._index]
  
        
    
    
    #exposed properties (feel like instance variables but are protected via getter and setter methods)
    name=property(_getName)             #read-only
    """(*str*) Name of the parameter. Read-only."""
    fixed=property(_getFixed,_setFixed)
    """(*bool*) Determines if fitparameter is fixed (*True*) or unfixed (*False*) during fitting procedures."""
    start_val=property(_getStartVal,_setStartVal)
    """(*number*) Start value of the fitparameter for fitting procedures."""
    lower_lim=property(_getLowerLim,_setLowerLim)
    """(*number*) Lower limit of the fitparameter for fitting procedures."""
    upper_lim=property(_getUpperLim,_setUpperLim)
    """(*number*) Upper limit of the fitparameter for fitting procedures."""
    index=property(_getIndex,_setIndex)
    """(*int*) Determines which entry in the fitparameter array is interpreted as the value of this fitparameter.
    Usually this property should not be changed by you (the user) but by the instance of :class:`.ParameterPool` where the fitparameter is connected to.
    """
    complex=property(_getComplex)
    """(*bool*) Signals if fitparameter is a complex number (*True*) or not (*False*). Read-only."""
    
class DerivedParameter(Parameter):
    """Objects of this class represent derived parameters which can be arbitrary functions of other parameters."""
    
    
    def __init__(self,f, *params):
        """Initialize a derived parameter as function **f** of the parameters ***params** which are instances of the class :class:`.Parameter`.
        
        BEWARE: The user is responsible for a matching number of arguments.
        
        Parameters
        ----------
        f : function
            Function of a arbitrary number of real or complex parameters which returns a real or complex number
        *params : :class:`Parameters.Parameter`
            The parameter objects from which the  :class:`.DerivedParameter` object is derived. Should be the same number of parameters as expected by **f**. (The star means that this is a variable number of parameters.
        """
        if not callable(f):
            raise Exception("\'f\' has to be a function.")
        for par in params:
            if not isinstance(par, Parameter):
                raise TypeError("Each entry of *params has to be an instance of class \'Parameters.Parameter\'.")       
        
        self._f=f
        self._params=params
        
    #special methods
    def __str__(self):
        """Determines how an *object* of this class react on ``str(object)``."""
        return "<" + type(self).__module__ + "." + type(self).__name__ + " object: function of the parameters " + ", ".join([str(param) for param in self._params])+">"
            
    #public methods
    def getValue(self,fitpararray=None):
        """
        Return the value of the derived parameter corresponding to the given array of fit values.
       
        **fitpararray** is only allowed to be *None* if the DerivedParameter is not derived from instances of 
        :class:`.Fitparameter`.
        """
        values=[param.getValue(fitpararray) for param in self._params]
        return self._f(*values)
        

class ParameterPool(object):
    """Collects a pool of Parameter objects and connects them with a parameter file."""
    
    #file format
    _nonesymbol='-'                          #if this symbol is found in the parameter file the corresponding property is set to "None"
    _commentsymbol='#'                       #Lines starting with this symbol in the parameter file are ignored
    _lineorder=[0,1,2,3,4]                 #gives the order of parameter properties in a line of the parameter file. [pos of start_value, pos of fixed, pos of lower_lim, pos of upper_lim, name]
    
    
    def __init__(self,parfilename=None):
        """Initialize a new ParameterPool.
        
        Read parameter initialisation from file **parfilename** if given. As soon as you connect a parameter file to the pool, its initialisation values have priority 
        over local initialisations with :meth:`.newParameter`.
        """           
        self._parPool=[]
        if parfilename is not None:
            self.readFromFile(parfilename)
    
        
    #private methods
    
    def _update(self):
        """Look for changes within the parameter properties and inform them about changes of their indices.
        
           This function should be used by the :class:`.Fitparameter` objects if they are connected to a ParameterPool.
           (So in a sense it is not private to the class, but nevertheless should not be exposed to users.)
        """
        self._realUnfixedArray=[]
        self._complexUnfixedArray=[]
        i=0
        for item in self._parPool:
            item.index=None
            if item.fixed==False and item.complex==False:
                self._realUnfixedArray.append(i)
            elif item.fixed==False and item.complex==True:
                self._complexUnfixedArray.append(i)
            i+=1
        i=0
        for poolindex in self._realUnfixedArray:
            self._parPool[poolindex].index=i
            i+=1
        for poolindex in self._complexUnfixedArray:
            self._parPool[poolindex].index=i
            i+=2
            
    @classmethod                                                                                       #classmethod: Method is bound to the class. We dont need a reference to the instance here, but only to the class variables defining the file format
    def _convertStrToNumber(cls, string):
        """Convert a string to a corresponding number type or None if the corresponding symbol is found.
        
            This method is used to read parameters from the parameter file.
        """
        if string==cls._nonesymbol:
             return None
        elif 'j' in string or 'J' in string:
            return complex(string)
        elif '.' in string:
            return float(string)
        else:
            return int(string)
    
    @classmethod
    def _convertNumberToStr(cls,number):
        """Convert a **number** to a corresponding string or to *_nonesymbol* if number is None.
        
            This method is used to write parameters from the parameter file.
        """
        if number is None:
            return cls._nonesymbol
        else: 
            return str(number)

        

    
    #public methods
    
    def newParameter(self,name,fixed=False,start_val=None,lower_lim=None,upper_lim=None):
        """Create a new :class:`.Fitparameter` inside the parameter pool and return a reference to it or
        return the reference to an already existing one with the same **name**.
        This is the usual way to create instances of :class:`.Fitparameter`.
        
        Parameters
        ----------
        name : str
            Name of the fit parameter. Use names without whitespaces. Otherwise there can be problems when using a parameter file.
        fixed : bool
            If `True`,the parameter will not be varied during a fit routine and the limits are not necessary.
        start_val : number
            Start value of the fit parameter.
        lower_lim, upper_lim : number
            Lower limit of the fit parameter.
        upper_lim : number 
            Upper limit of the fit parameter.        
        """
        par=self.getParameter(name)
        if par is None:
            par=Fitparameter(name,fixed,start_val,lower_lim,upper_lim)
            self._parPool.append(par)
            par.pool=self
        self._update()
        return par
        
        
    def getParameter(self,name):
        """Return existing parameter with name **name** or return *None* if not existing."""
        if not isinstance(name,str):
            raise TypeError("\'name\' has to be a string.")
        try:
            index=[item.name for item in self._parPool].index(name)
            return self._parPool[index]
        except ValueError:
            return None
    
    def readFromFile(self,parfilename):        
        """Read parameters and there initialisation values from file *parfilename**, append them to the pool or overwrite existing once.
            
           | Lines in the parameter file should be in the format:
           | ``<start_value>  <fixed>  <lower_limit>  <upper_limit>  <name>  <comments/other stuff>``
           | Lines starting with *#* are ignored.
           | Values of <fixed> have to be either `0` or `1` (not `False` or `True`).
        """
        with open(parfilename,'r') as f:
            for line in f:
                line=(line.split(self._commentsymbol))[0]   #ignore everythin behind a _commentsymbol
                if not line.isspace() and not line=="":                       #ignore empty lines
                    l=line.split()
      
                    #the order of the parameter properties is coded here, with the help of the lineorder array
                    try:
                        start_value = self._convertStrToNumber(l[self._lineorder[0]])
                        fixed = self._convertStrToNumber(l[self._lineorder[1]])
                        lower_limit =self. _convertStrToNumber(l[self._lineorder[2]])
                        upper_limit = self._convertStrToNumber(l[self._lineorder[3]])
                    except ValueError:
                        print "Invalid number format in line: "+line
                        raise
                        
                    name=l[self._lineorder[4]]
                    

                    if not ((isinstance(start_value,complex) or start_value is None) and (isinstance(upper_limit,complex) or upper_limit is None) and (isinstance(lower_limit,complex) or lower_limit is None))  and not ((not isinstance(start_value,complex) or start_value is None) and (not isinstance(upper_limit,complex) or upper_limit is None) and (not isinstance(lower_limit,complex) or lower_limit is None)):
                        print line
                        raise ValueError("Start value and limits in the parameter file should be either all complex numbers or none of them.\n Bad line: "+line)                                
                    
                    par=self.getParameter(name)
                    if par is None:
                        par=Fitparameter(name,bool(fixed),start_value,lower_limit,upper_limit)
                        self._parPool.append(par)
                        par.pool=self
                    else:
                        par.start_val=start_value
                        par.fixed=bool(fixed)
                        par.lower_lim=lower_limit
                        par.upper_lim=upper_limit
    
    def writeToFile(self,parfilename,fitpararray=None):
        """Write parameters and there initialisation value to file **parfilename**. 
           
           Can be used to create a template for a parameter initialisation file or to store parameters after fitting.
           If **fitpararray** is given, these values are written as start values. Else the stored start values are used.
        """   
        columnwidth=25
        
        if fitpararray is not None and len(fitpararray)<>self.getFitArrayLen():
            raise ValueError("\'fitpararray\' has wrong length.")
        
        if os.path.isfile(parfilename) :
            #ask if overwrite
            answer=raw_input("Do you realy want to overwrite \'"+parfilename+"\'? [y/N]")
            answer=answer.lower()
            if not(answer=='y' or answer=='yes' or answer=='j' or answer=='ja'):
                return
        headerarray=["<start_value>","<fixed>","<lower_limit>","<upper_limit>","<name>"]
        with open(parfilename,'w') as f:
            f.write((self._commentsymbol+headerarray[self._lineorder[0]]).ljust(columnwidth)+headerarray[self._lineorder[1]].ljust(columnwidth)+headerarray[self._lineorder[2]].ljust(columnwidth)+headerarray[self._lineorder[3]].ljust(columnwidth)+headerarray[self._lineorder[4]]+"\n")
            for p in self._parPool:
                entryarray=[p.start_val,int(p.fixed),p.lower_lim,p.upper_lim,p.name]
                if fitpararray is not None:
                    entryarray[0]=p.getValue(fitpararray)
                parray=[self._convertNumberToStr(item) for item in entryarray]
                f.write(parray[self._lineorder[0]].ljust(columnwidth)+parray[self._lineorder[1]].ljust(columnwidth)+parray[self._lineorder[2]].ljust(columnwidth)+parray[self._lineorder[3]].ljust(columnwidth)+parray[self._lineorder[4]].ljust(columnwidth)+"\n")
    
    def getStartLowerUpper(self):
        """Return a tupel of `fitpararrays` of parameter start values, lower and upper limits for usage with :meth:`.Fitparameter.getValue`.
        
        
        Each of the arrays contains the values in the order of occurence of the corresponding parameter in the pool (order of parameter creation), but only of those parameters which are not fixed. Real parameters first, then the complex ones.
        """
        self._update()
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
    
    def setStartValues(self,fitpararray):
        """Set the start values of all parameters which are not fixed using **fitpararray** in the order of occurence in the pool (order of parameter creation). First real and then complex ones.
        
           Can be used e.g. after finding goog start values with an Evolutionary fitter.
        """
        lenreal=len(self._realUnfixedArray)
        lencomplex=len(self._complexUnfixedArray)
        if len(fitpararray)<>(lenreal+2*lencomplex):
            raise Exception("Given fit value array is "+str(len(fitpararray))+" entries long but should have a length of "+str(lenreal+2*lencomplex)+".")
        for i in range(lenreal):
            self._parPool[self._realUnfixedArray[i]].start_val=fitpararray[i]
        for i in range(lencomplex):
            self._parPool[self._complexUnfixedArray[i]].start_val=complex(fitpararray[lenreal+2*i],fitpararray[lenreal+2*i+1])
            
    
    def getFitArrayLen(self):
        """Return length of the *fitpararray* which is needed for :meth:`.setStartValues`, :meth:`.writeToFile` and :meth:`.Fitparameter.getValue` and which is given by :meth:`.getStartLowerUpper`."""
        return len(self._realUnfixedArray)+2*len(self._complexUnfixedArray)
    
    def getNames(self):
        """Return the names of all registered Fitparameters as a list in the same order they should be in the `fitpararray`."""
        self._update()
        namearray=[]
        for poolindex in self._realUnfixedArray:
            namearray.append(self._parPool[poolindex].name)
        for poolindex in self._complexUnfixedArray:
            namearray.append(self._parPool[poolindex].name+".real")
            namearray.append(self._parPool[poolindex].name+".imag")
        return namearray
            
        
    

