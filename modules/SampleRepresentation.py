#!/usr/bin/env python
"""Deals with the sample representation for simulation of the reflectivity."""


#Python Version 2.7


__author__ = "Yannic Utz"
__copyright__ = ""
__credits__ = ["Yannic Utz and Martin Zwiebler"]
__license__ = ""
__version__ = ""
__maintainer__ = "Yannic Utz"
__email__ = "yannic.utz@tu-dresden.de"
__status__ = "Prototype"


import os.path
import numbers
import numpy
import ast
from  scipy import interpolate
import matplotlib.pyplot
import Pythonreflectivity


from . import Parameters


class Heterostructure(object):
    """Represents a heterostructructure as a stack of layers.
    
       In contrast to Martin's list of Layer-type objects, this class contains all information also for different energies.
    """
    
    def __init__(self, number_of_layers=0, multilayer_structure=None):
        """Create heterostructructure object.
        
           \'number_of_layers\' gives the number of different layers.
           With \'multilayer_structure\' multilayers which contain identical layers several times can be defined.
           This can be defined by as a list containing the indices of layer from the lowest (e.g. substrate) to the highest (top layer, hit first by the beam).
           Default is \'[0,1,2,3, ...,number_of_layers-1]\'. Multilayer syntax is e.g. \'[0,1,2,[100,[3,4,5,6]],7,.,1,..]\' which repeats 100 times the sequence of
           layers 3,4,5,6 in between 2 and 7 and later on layer 1 is repeated once.
        """
        if not isinstance(number_of_layers,int):
            raise TypeError("\'number_of_layers\' must be of type int.")
        if number_of_layers<0:
            raise ValueError("\'number_of_layers\' must be positive.")
        if multilayer_structure is None:
            multilayer_structure=range(number_of_layers)
        elif not isinstance(multilayer_structure,list):
            raise TypeError("\'multilayer_structure\' has to be a list.")
        else:
            self._consistency_check_(number_of_layers, multilayer_structure)
        self._number_of_layers=number_of_layers
        self._multilayer_structure=multilayer_structure
        self._listoflayers=[None for i in range(number_of_layers)]
        self._updatePyReflMLString_()
    
    def _consistency_check_(self,number_of_layers,multilayer_structure):
        index_list=[]
        for item in multilayer_structure:
            if isinstance(item, int):
                if item not in index_list:
                    index_list.append(item)
            elif isinstance(item, list):
                if not (len(item)==2 and isinstance(item[0],int) and item[0]>0 and isinstance(item[1],list)):
                    raise ValueError("\'multilayer_structure\' has wrong format.")
                for subitem in item[1]:
                    if not isinstance(subitem, int):
                        raise ValueError("\'multilayer_structure\' has wrong format.")
                    if subitem not in index_list:
                        index_list.append(subitem)
            else:
                raise ValueError("\'multilayer_structure\' has wrong format.")
        index_list.sort()
        if index_list[0]<>0:
            raise ValueError("Indices in \'multilayer_structure\' have to start from 0.")
        if index_list[-1]<>len(index_list)-1:
            raise ValueError("Highest index in \'multilayer_structure\' does not agree with the number of different indices-1.")
        if len(index_list)<>number_of_layers:
            raise Exception("Number of different indices in \'multilayer_structure\' does not agree with \'number_of_layers\'.")
        return len(index_list)
    
    
    def _updatePyReflMLString_(self):
        self._PyReflMLstring=""
        for item in self._multilayer_structure:
            if isinstance(item, int):
                self._PyReflMLstring+=str(item)+","
            elif isinstance(item, list):
                self._PyReflMLstring+=str(item[0])
                self._PyReflMLstring+="*"
                self._PyReflMLstring+="("
                for subitem in item[1]:
                    self._PyReflMLstring+=str(subitem)+","
                self._PyReflMLstring=self._PyReflMLstring[:-1]  #remove last comma
                self._PyReflMLstring+="),"        
        self._PyReflMLstring=self._PyReflMLstring[:-1]  #remove last comma

        
    
    def _getNumberOfLayers_(self):
        """Return number of different layers (i.e. number of different indices)."""
        return self._number_of_layers
    
    
    def _getTotalNumberOfLayers_(self):
        """Return total number of layers (counting also multiple use of the same layer according to \'multilayer_structure\')."""
        n=0
        for item in self._multilayer_structure:
            if isinstance(item, int):
                n+=1
            elif isinstance(item, list):
                n+=item[0]*len(item[1])
        return n
        
    
    def _getMultilayerStructure_(self):
        return self._multilayer_structure
        
    def _mapTotalIndexToInternal(self,tot_ind):
        """Return the index used within \'multilayer_structure\' which corresponds to the total index of the layer counting from the bottom."""
        if tot_ind>=self.N_total:
            raise ValueError("Index out of range.")        
        i=0
        for item in self._multilayer_structure:
            if isinstance(item, int):
                if i==tot_ind:
                    return item
                i+=1
            elif isinstance(item, list):
                for loop in range(item[0]):
                    for subitem in item[1]:
                        if i==tot_ind:
                            return subitem
                        i+=1

    
    #public methods
                                
    def setLayout(self, number_of_layers, multilayer_structure=None):
        """Change the layout of the heterostructure.
        
           See constructor for details. Only difference is: you cannot make changes which would remove layers.
        """
        if not isinstance(number_of_layers,int):
            raise TypeError("\'number_of_layers\' must be of type int.")
        if number_of_layers<0:
            raise ValueError("\'number_of_layers\' must be positive.")
        if multilayer_structure is None:
            multilayer_structure=range(number_of_layers)
            numberofindices=number_of_layers
        elif not isinstance(multilayer_structure,list):
            raise TypeError("\'multilayer_structure\' has to be a list.")
        else:
            numberofindices=self._consistency_check_(number_of_layers, multilayer_structure)
        if numberofindices<len(self._listoflayers):
            for item in self._listoflayers[numberofindices:]:
                if item is not None:
                    raise Exception("Cannot change layout. Remove layers with index > "+str(numberofindices-1)+" first.")
            self._listoflayers=self._listoflayers[:numberofindices]
        elif numberofindices>len(self._listoflayers):
            self._listoflayers+=[None for i in range(numberofindices-len(self._listoflayers))]
        self._number_of_layers=number_of_layers
        self._multilayer_structure=multilayer_structure
        self._updatePyReflMLString_()
        
    def setLayer(self,index, layer):
        """
        Place \'layer\' (instance of LayerObject) at position \'index\' (counting from 0, starting from the bottom or according to \'multilayer_structure\').
        """
        if not isinstance(index,int):
            raise TypeError("\'index\' must be of type int.")
        if index<0:
            raise ValueError("\'index\' must be positive.")
        if index>=self._number_of_layers:
            raise ValueError("\'index\' exceeds defined number of layers.")
        if not isinstance(layer,LayerObject):
            raise TypeError("\'layer\' has to be an instance of \'LayerObject\'.")
        self._listoflayers[index]=layer
    
    def getLayer(self,index):
        """
        Return the instance of LayerObject which is placed at position \'index\' (counting from 0, starting from the bottom or according to \'multilayer_structure\').
        """
        if not isinstance(index,int):
            raise TypeError("\'index\' must be of type int.")
        if index<0:
            raise ValueError("\'index\' must be positive.")
        if index>=self._number_of_layers:
            raise ValueError("\'index\' exceeds defined number of layers.")
        return self._listoflayers[index]
         
    def getTotalLayer(self,index):
        """
        Return the instance of LayerObject which is placed at position \'index\' (counting from 0, starting from the bottom, even if a latter sequence is repeated with help of \'multilayer_structure\').
        """
        return self.getLayer(self._mapTotalIndexToInternal(index))
    
    def removeLayer(self,index):
        """
        Remove the instance of LayerObject which is placed at position \'index\' (counting from 0, starting from the bottom from the  heterostructructure or according to \'multilayer_structure\').
        
        
        \'index\' can also be a list of indices.
        BEWARE: The instance of LayerObject itself and the corresdonding Parameters are not deleted! 
        """
        if isinstance(index,int):
            if index<0:
                raise ValueError("\'index\' must be positive.")
            if index>=self._number_of_layers:
                raise ValueError("\'index\' exceeds defined number of layers.")
            self._listoflayers[index]=None
        elif isinstance(index,list):
            for item in index:
                if not isinstance(item,int):
                    raise "Entries of \'index\' list must be of type int."
                if item<0:
                    raise ValueError("Entries of \'index\' list must be positive.")
                if item>=self._number_of_layers:
                    raise ValueError("Entry of \'index\' list exceeds defined number of layers.")
                self._listoflayers[item]=None 
        else:
            raise TypeError("\'index\' must be of type int or list of int")

    
    
    def GetSingleEnergyStructure(self,fitpararray,energy=None):
        """
        Return list of layers (layer type from Pythonreflectivity) which can be directly used as input for \'Pythonreflectivity.Reflectivity( )\'
        """
        PyReflStructure=Pythonreflectivity.Generate_structure(self._number_of_layers,self._PyReflMLstring)
        index=0
        for layer in self._listoflayers:
            if layer is None:
                print "WARNING: Layer "+str(index)+" is undefined!"
            else:
                PyReflStructure[index].setd(layer.getD(fitpararray))
                PyReflStructure[index].setsigma(layer.getSigma(fitpararray))
                PyReflStructure[index].setmag(layer.getMagDir(fitpararray))
                PyReflStructure[index].setchi(layer.getChi(fitpararray,energy))
            index+=1
        return PyReflStructure
            

    #properties
    N=property(_getNumberOfLayers_)                                             #number of different layers
    N_total=property(_getTotalNumberOfLayers_)                                  #total number of layers (counting also multiple use of the same layer according to \'multilayer_structure\')
    MLstructure=property(_getMultilayerStructure_)
    

#-----------------------------------------------------------------------------------------------------------------------------


class LayerObject(object):
    """Base class for all layer objects as the common interface. Speciallized implementation should inherit from this class.
    
       It handels the basic properties of 
    """
    
    def __init__(self, chitensor=None, d=None,  sigma=None, magdir="0"):
        """Creates a new Layer.
        
        \'d\' is its thickness, \'sigma\' is the roughness of its upper surface, \'chitensor\' is its electric susceptibility tensor, and \'magdir\' gives the magnetization directrion for MOKE
        \'d\', \'sigma\' ,and the entries of \'chitensor\' are expected to be an instances of a \"Parameter\" class (also \"Fitparamter\").
        \'chitensor\' is a list of either  1,3, 4 or 9 elements (see also documentation for Pythonreflectivity).
        [chi] sets chi_xx = chi_yy = chi_zz = chi
        [chi_xx,chi_yy,chi_z] sets  chi_xx,chi_yy,chi_zz, others are zero
        [chi_xx,chi_yy,chi_z,chi_g] sets  chi_xx,chi_yy,chi_zz and depending on 'magdir' chi_yz=-chi_zy=chi_g (if 'x'), chi_xz=-chi_zx=chi_g (if 'y') or chi_xz=-chi_zx=chi_g (if 'z')
        [chi_xx,chi_xy,chi_xz,chi_yx,chi_yy,chi_yz,chi_zx,chi_zy,chi_zz] sets all the corresdonding elements
        """
        
        #check parameters
        if not isinstance(d,Parameters.Parameter) and not d is None:
            raise TypeError("\'d\' must be an instance of class \'Parameters.Parameter\' or of an derived class (e.g. \'Parameters.Fitparameter\'")
        if chitensor is not None:
            for item in chitensor:
                if not isinstance(item,Parameters.Parameter):
                    raise TypeError("Elements of \'d\' must be instances of class \'Parameters.Parameter\' or of an derived class (e.g. \'Parameters.Fitparameter\'")
        if not isinstance(sigma,Parameters.Parameter) and not sigma is None:
            raise TypeError("\'sigma\' must be an instance of class \'Parameters.Parameter\' or of an derived class (e.g. \'Parameters.Fitparameter\'")
        if not isinstance(magdir,str) and not magdir is None:
            raise TypeError("\'magdir\' must be of type \'str\'.")
        elif not (magdir=='x' or magdir=='y' or magdir=='z' or magdir=='0'):
            raise ValueError("Invalid input for \'magdir\'. Valid inputs are \'x\',\'y\',\'z\', and \'0\'")
        
        #asign members
        self._d=d
        self._chitensor=chitensor
        self._sigma=sigma
        self._magdir=magdir
     
    def _setd_(self,d):
        if not isinstance(d,Parameters.Parameter):
            raise TypeError("\'d\' must be an instance of class \'Parameters.Parameter\' or of an derived class (e.g. \'Parameters.Fitparameter\'")
        self._d=d
    
    def _getd_(self):
        return self._d
    
    def _getChitensor_(self):
        return self._chitensor
    
    def _setChitensor_(self,chitensor):
        for item in chitensor:
            if not isinstance(item,Parameters.Parameter):
                raise TypeError("Elements of \'d\' must be instances of class \'Parameters.Parameter\' or of an derived class (e.g. \'Parameters.Fitparameter\'")
        self._chitensor=chitensor
    
    def _getMagdir_(self):
        return self._magdir
    
    def _setMagdir_(self, magdir):
        if not isinstance(magdir,str):
            raise TypeError("\'magdir\' must be of type \'str\'.")
        elif not (magdir=='x' or magdir=='y' or magdir=='z' or magdir=='0'):
            raise ValueError("Invalid input for \'magdir\'. Valid inputs are \'x\',\'y\',\'z\', and \'0\'")
        self._magdir=magdir
    
    def _getSigma_(self):
        return self._sigma
    
    def _setSigma_(self, sigma):
        if not isinstance(sigma,Parameters.Parameter):
            raise TypeError("\'sigma\' must be an instance of class \'Parameters.Parameter\' or of an derived class (e.g. \'Parameters.Fitparameter\'")
        self._sigma=sigma
     
    
    #public methods
    def getChi(self,fitpararray,energy=None):
        """
        Return the chitensor as a list numbers of  for a certain energy.
    
        For the base implementation of class \'LayerObject\' the parameter \'energy\' is not used. But it may be used by derived classes like \'AtomLayerObject\'.
        """
        #check tensor bevor giving it to the outside
        if not (len(self._chitensor)==1 or len(self._chitensor)==3 or len(self._chitensor)==4 or len(self._chitensor)==9):
            raise ValueError("Chitensor must be either of length 1, 3, 4, or 9.")
        if len(self._chitensor)==4 and self._magdir=="0":
            raise ValueError("You have to define the direction of magnetization \'magdir\'.")
        #return the checked list filled with actual values
        return [item.getValue(fitpararray) for item in self._chitensor]
    
    def getD(self,fitpararray):
        """Return the thickness d as a an actual number."""
        if self._d is None:
            return 0
        else:
            return self._d.getValue(fitpararray)
    
    def getSigma(self,fitpararray):
        """Return sigma as a an actual number."""
        if self._sigma is None:
            return 0
        else:
            return self._sigma.getValue(fitpararray)
    
    def getMagDir(self,fitpararray=None):
        """
        Return magdir.
        
        fitpararray is not used and just there to give a common interface. mMybe derived classes will have a benefit from it.
        """
        return self._magdir
    
    
    
    
    #exposed properties (feel like instance variables but are protected via getter and setter methods)
    #BEWARE: theese properties contain the "Parameter" objects. E.g. to get a certain value for the thicknes do this: "layer.d.getValue(parameterarray)"
    d=property(_getd_,_setd_)
    chitensor=property(_getChitensor_,_setChitensor_)
    sigma=property(_getSigma_,_setSigma_)
    magdir=property(_getMagdir_,_setMagdir_)
    
    

        
        
        
        
        
        
class AtomLayerObject(LayerObject):
    """
    Speciallized Layer to deal with compositions of Atoms and their absorption spectra.
    
    Especially usefull to deal with atomic layers, but can also be used for bulk.
    The atoms and their formfactors have to be registered a the class (with registerAtom) before they can be used to instantiate a new AtomLayerObject.
    The atom density can be plotted with plotAtomDensity() (see convenience functions).
    Density in mol/cm$^3$
    """
    
    def __init__(self, densitydict={}, d=None,  sigma=None, magdir="0"):
        """
        Create an AtomLayerObject.
        
        \'densitydict\' ist a dictionary which contains atom names (strings, must agree with before registered atoms) and densities (must be instances of the Parameter class or derived classes).
        """
        if not isinstance(densitydict,dict):
            raise TypeError("\'densitydict\' has to be a dictionary.")
        for atomname in densitydict:
            if not isinstance(atomname,str):
                raise TypeError("The keys of the \'densitydict\' dictionary have to be stings.")
            if not isinstance(densitydict[atomname], Parameters.Parameter):
                raise TypeError("The values of the \'densitydict\' dictionary have to be instances of the \'Parameter\' class or of derived classes.")
            if atomname not in type(self)._atomdict:
                raise ValueError("Atom \'"+atomname+"\' has not been registered yet.")
        
        self._densitydict=densitydict.copy()                                #The usage of "copy" creates a copy of the dictionary. By this, we ensure, that changes of the original dictionary outside the object will not affect the AtomLayerObject
        
         #call constructor of the base class
        super(AtomLayerObject,self).__init__(None, d,  sigma, magdir)     
        
           
    def getDensitydict(self,fitpararray=None):
        """Return the density dictionary either with evaluated paramters (needs fitpararray) or with the raw '\Parameter\' objects (use fitparraray=None)."""
        if fitpararray==None:
            return self._densitydict.copy()
        elif not isinstance(fitpararray,list):
            raise TypeError("\fitparray\' has to be a list.")
        else:
            return dict(zip(self._densitydict.keys(),[item.getValue(fitpararray) for item in self._densitydict.values()]))              #pack new dictionary from atomnames and "unpacked" parameters (actual values instead of abstract parameter)
       
    def getChi(self,fitpararray,energy):
        
        #debug
        print "WARNING: Calculation of susceptibility from formfactor is not verified!"
        
        ffsum=sum([item[1].getValue(fitpararray)*(type(self)._atomdict[item[0]]).getFF(energy,fitpararray) for item in self._densitydict.items()])  #gehe alle Items in self._densitydict durch, item[1].getValue(fitpararray) liefert Dichte der Atomsorte, (type(self)._atomdict[item[0]]).getFF(fitpararray) liefert Formfaktor der Atomsorte, beides wird multipliziert und alles zusammen aufsummiert
               
        return list(ffsum*830.3584763651544/energy**2)                #no idea why there is this strange numerical factor 
        
        
        
        
    
    #classvariables
    _atomdict={}
    
    #classmethods                                               #are not related to an instance of a class and are here used to deal with the collection of all registered atoms
    @classmethod
    def registerAtom(cls, name,formfactor):
        """
        Register an atom for later use to instantiate an AtomLayerObject.
        
        \'formfactor\' as to be an instance of \'Formfactor\' or of a derived class.
        """        
        if not isinstance(name,str):
            raise TypeError("The atom \'name\' has to be a string.")
        if not isinstance(formfactor,Formfactor):
            raise TypeError("\'formfactor\' has to be an instance of \'Formfactor\' or of a derived class.")
        if name in cls._atomdict:
            print "WARNING: Atom \'"+str(name)+"\' is replaced."
        cls._atomdict.update({name: formfactor})
    
    @classmethod
    def getAtom(cls, name):
        """Return the Formfactor object registered for Atom \'name\'."""
        if not isinstance(name,str):
            raise TypeError("The atom \'name\' has to be a string.")
        if name not in cls._atomdict:
            raise ValueError("The atom \'name\' is not registered.")
        return cls._atomdict[name]
    
    @classmethod
    def getAtomNames(cls):
        """Return a list of names of registered atoms."""
        return cls._atomdict.keys()
    



#-----------------------------------------------------------------------------------------------------------------------------

class Formfactor(object):
    """
    Base class to deal with energy-dependent atomic form-factors.
    
    This base class is an abstract class an cannot be used directly.
    The user should derive from this class if he wants to build his own models.
    """
    
    def __init__(self):
        raise NotImplementedError
    
    def getFF(self,energy,fitpararray=None):
        """
        Return the formfactor for \'energy\' corresponding to fitpararray (if it depends on it).
        """
        raise NotImplementedError


class FFfromFile(Formfactor):
    """
    Class to deal with energy-dependent atomic form-factors which are tabulated in files.
    """
  
    def __init__(self, filename, linereaderfctn=None):
        """Initializes the FFfromFile object with the data from filename.
        
           The \'linereaderfctn\' is used to convert one line from the text file to data.
           It should be a function which takes a string and returns a list of 10 values: [energy,f_xx,f_xy,f_xz,f_yx,f_yy,f_yz,f_zx,f_zy,f_zz] 
           It can also return \'None\' if it detects a comment line.
           You can use FFfromFile.getLinereader to get a standard function, which just reads this array as whitespace seperated from the line.
        """
        if not isinstance(filename,str):
            raise TypeError("\'filename\' needs to be a string.")
        if linereaderfctn is None:
            linereaderfctn=self.getLinereader()
        if not callable(linereaderfctn):
            raise TypeError("\'linereaderfctn\' needs to be a callable object.")
        if not os.path.isfile(filename):
            raise Exception("File \'"+filename+"\' does not exist.")
        energies=[]
        formfactors=[]
        with open(filename,'r') as f:
            for line in f:
                linereaderoutput=linereaderfctn(line)
                if linereaderoutput is None:
                    break
                if not isinstance(linereaderoutput,list):
                    raise TypeError("Linereader function has to return a list.")
                if not  len(linereaderoutput)==10:
                    raise ValueError("Linereader function hast to return a list with 10 elements.")
                for item in linereaderoutput:
                    if not isinstance(item,numbers.Number):
                        raise ValueError("Linereader function hast to return a list of numbers.")
                if isinstance(linereaderoutput[0],complex):
                    raise ValueError("Linereader function hast to return a real value for the energy.")
                energies.append(linereaderoutput[0])                                                        #store energies in one list
                formfactors.append(linereaderoutput[1:])                                                    #store corresponding formfactors in another list
        formfactors=numpy.array(formfactors)                                                                #convert list formfactors to a numpy array for convinience
        self._minE=min(energies)
        self._maxE=max(energies)
        #Create an interpolation function based on the given energie-formfactor-points. The formfactors are thererfore transformed to arrays of length 18 but with real values. 
        #After that the array of N arrays of 18 element is transformed to an array of 18 arrays of N elements as needed by the interp1d function.
        #Therefore, this function will return an array of length 18 wich has to be transformed back to 9 complex valued elements.
        #Energies and formfactors don't have to be stored explicitly , because they are contained in the "self._interpolator" function.
        self._interpolator=interpolate.interp1d(energies,numpy.transpose(numpy.concatenate((formfactors.real,formfactors.imag),1))) 
    
    def _getMinE_(self):
        return self._minE
    
    def _getMaxE_(self):
        return self._maxE
    
    
    #public methods
    
    @staticmethod
    def getLinereader(complex_numbers=True):
        """
        Return the standard linereader function.
        
        This standard linereader function reads energy and complex elements of the formfactor tensor as a whitespace-seperated list (i.e. 10 numbers) and interpretes \'#\' as comment sign.
        If complex_numbers==0 then the reader reads real and imaginary part of every element sepeately, i.e. every line has to consist of 19 numbers seperated by whitespaces.
        """
        commentsymbol='#'
        if complex_numbers==True:
            def linereader(line):
                if not isinstance(line,str):
                    raise TypeError("\'line\' needs to be a string.")
                line=(line.split(commentsymbol))[0]                            #ignore everything behind the commentsymbol  #
                if not line.isspace():                               #ignore empty lines        
                    linearray=line.split()
                    if not len(linearray)==10:
                        raise Exception("Formfactor file has wrong format.")
                    linearray=[ast.literal_eval(item) for item in linearray]
                    return linearray
                else:
                    return None
        elif complex_numbers==False:
            def linereader(line):
                if not isinstance(line,str):
                    raise TypeError("\'line\' needs to be a string.")
                line=(line.split(commentsymbol))[0]                            #ignore everything behind the commentsymbol  #
                if not line.isspace():                               #ignore empty lines        
                    linearray=line.split()
                    if not len(linearray)==19:
                        raise Exception("Formfactor file has wrong format.")
                    linearray=[ast.literal_eval(item) for item in linearray]
                    return [linearray[0], linearray[1]+1j*linearray[2], linearray[3]+1j*linearray[4], linearray[5]+1j*linearray[6], linearray[7]+1j*linearray[8], linearray[9]+1j*linearray[10], linearray[11]+1j*linearray[12], linearray[13]+1j*linearray[14], linearray[15]+1j*linearray[16], linearray[17]+1j*linearray[18]]
                else:
                    return None
        else:
            raise TypeError("\'complex_numbers\' has to be boolean.")
        return linereader                                                               #here the FUNKTION linereader is returned
            
    def getFF(self,energy,fitpararray=None):
        """
        Return the formfactor for \'energy\' as an interpolation between the stored values from file as 1-D numpy array.
        
        \'fitpararray\' is not used.
        """
        if energy<self.minE or energy>self.maxE:
            raise ValueError("\'energy="+str(energy)+"\' is out of range ("+str(self.minE)+","+str(self.maxE)+").")
        FFallReal=self._interpolator(energy)
        #return list(FFallReal[:9]+FFallReal[9:]*1j)
        return FFallReal[:9]+FFallReal[9:]*1j
        
        
        
    #properties
    maxE=property(_getMaxE_)
    minE=property(_getMinE_)
    
    
#--------------------------------------------------------------------------------------------------------------------------
# convenience functions

def plotAtomDensity(hs,fitpararray,atomnames=None,colormap=[]):
    """Make a plot of the atom densities of all AtomLayerObjects contained in the \'Heterostructure\' \'hs\' corresdonding to the \'fitpararray\' and return the plotted information.
    
        You can define which atoms you want to plot or in which order. Give \'atomnames\' as a list of strings.
        You can also define the colors of the bars. Just give a list of matplotlib color names. They will be used in the given order.
    """
    if not isinstance(hs,Heterostructure):
        raise TypeError("\'hs\' has to be of type \'SampleRepresentation.Heterostructure\'.")
    elif not isinstance(fitpararray,list):
            raise TypeError("\fitparray\' has to be a list.")
    elif not isinstance(colormap,list):
            raise TypeError("\colormap\' has to be a list.")
    elif not isinstance(atomnames,list):
            raise TypeError("\atomnames\' has to be a list.")
    for item in atomnames:
        if not isinstance(item,str):
            raise TypeError("\atomnames\' has to be a list of strings.")
    
    number_of_layers=hs.N_total
    if atomnames is None:
        atomnames=AtomLayerObject.getAtomNames()
        widthstep=0.9/len(atomnames)                #if no order is given plot each set of bar with smaller width to not cover the underlying bar
    else:
        widthstep=0.0
    if atomnames==[]:
        print "No atoms registered."
        return
    
    densitylistdict={}
    for name in atomnames:
        densitylistdict[name]=numpy.zeros(number_of_layers)                                 #create dictionary, which has an entry for every atom, with its name as key and as value a list. These lists are as long as there are numbers of layers and filled with zeros.
    
    for i in range(hs.N):                                                   
        layer=hs.getLayer(i)                                                    #go through all layers in the heterostructure
        if isinstance(layer,AtomLayerObject):                                   #if it is an instance of AtomLayerObject (i.e. contains information about atom densities)
            densitydict=layer.getDensitydict(fitpararray)           #get the density dictionary from this layer with the parameters evaluated (i.e. take the value contained in fitpararray)
            for name in atomnames:                                  #if atom with a certain name is contained within this layer, ad its density to the corresponding list in densitylistdict
                if name in densitydict:
                    densitylistdict[name][i]=densitydict[name]
        
        
    colorindex=0
    w=1
    widthstep=0.9/len(atomnames)
    for name in densitylistdict:
        if colorindex<len(colormap):
            matplotlib.pyplot.bar(range(number_of_layers), densitylistdict[name],align='center',width=w,label=name,color=colormap[colorindex],alpha=0.9)
            colorindex+=1
        else:
            matplotlib.pyplot.bar(range(number_of_layers), densitylistdict[name], align='center', width=w,label=name,alpha=0.9)
        w-=widthstep
       
    matplotlib.pyplot.xlabel("Layer number")
    matplotlib.pyplot.ylabel(r'Density in mol/cm$^3$')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlim(0,number_of_layers)
    matplotlib.pyplot.show()
    
    return densitylistdict                      #contains a dictionary, which has an entry for every atom, with its name as key and as value a list. These lists are as long as there are numbers of layers and filled with
