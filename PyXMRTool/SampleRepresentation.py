#!/usr/bin/env python
"""Deals with the sample representation for simulation of the reflectivity.

A multilayer sample is represented by a :class:`.Heterostructure` object. Its main pupose is to deliver the list of layers (Layer type of the :mod:`Pythonreflectivity` package from Martin Zwiebler) with defined susceptibilities at certain energies via :meth.`Heterostructure.getSingleEnergyStructure`. The layers within this heterostructure are represented by instances of :class:`.LayerObject` or of derived classes, which alows for a very flexibel modelling of the sample.

So far the following layer types are implemented:

* :class:`.LayerObject`: Layer with a constant (over energy) but fittable electric susceptibility tensor.

* :class:`.ModelChiLayerObject`: This layer type holds the electric susceptibility tensor as a user-defined function of energy.
    
* :class:`.AtomLayerObject`: This layer deals with compositions of atoms with different formfactors. The densities of the atoms can be varied during fitting procedures and plotted with using :func:`.plotAtomDensity`. The formfactors are represented by instances of classes which are derived from :class:`.Formfactor` (the base class is abstract and cannot be used directly).

So far the following formfactor types are implemented:

* :class:`.FFfromFile`: Reads an energy-dependent formfactor as data points from a textfile. For energies between the data points the formfactor is linearly interpolated.

* :class:`.FFfromScaledAbsorption`: Reads an absorption measurement (fitted to off-resonant tabulated values) and a theoretical/tabulated energy-dependen formfactor from textfiles. Within a given energy-range, the absorption is scaled with a fittable factor and the real part is obtained by a Kramers-Kronig transformation. See section 3.3 of Martin Zwiebler PhD-Thesis for details.
    


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
import numbers
import numpy
import ast
from  scipy import interpolate
import scipy
import matplotlib.pyplot
import Pythonreflectivity


import Parameters


class Heterostructure(object):
    """Represents a heterostructure as a stack of instances of :class:`.LayerObject` or of derived classes.
    Its main pupose is to model the sample in a very flexibel way and to get the list of layers (Layer type of the :mod:`Pythonreflectivity` package from Martin Zwiebler) with defined susceptibilities at certain energies.
    
    
    In contrast to Martin's list of Layer-type objects, this class contains all information also for different energies.
    """
    
    def __init__(self, number_of_layers=0, multilayer_structure=None):
        """Create heterostructructure object.
        
        Parameters
        ----------
        number_of_layers : int 
            gives the number of different layers
        multilayer_structure : list
            Makes it possible to define multilayers which contain identical layers several times.
            This can be done by passing a list containing the indices of layers from the lowest (e.g. substrate) to the highest (top layer, hit first by the beam).
            Default is ``[0,1,2,3, ...,number_of_layers-1]``. Multilayer syntax is e.g.``[0,1,2,[100,[3,4,5,6]],7,.,1,..]`` which repeats 100 times the sequence of
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
            self._consistency_check(number_of_layers, multilayer_structure)
        self._number_of_layers=number_of_layers
        self._multilayer_structure=multilayer_structure
        self._listoflayers=[None for i in range(number_of_layers)]
        self._updatePyReflMLString()
        
    #private members
    
    def _consistency_check(self,number_of_layers,multilayer_structure):
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
    
    
    def _updatePyReflMLString(self):
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

        
    
    def _getNumberOfLayers(self):
        """Return number of different layers (i.e. number of different indices)."""
        return self._number_of_layers
    
    
    def _getTotalNumberOfLayers(self):
        """Return total number of layers (counting also multiple use of the same layer according to \'multilayer_structure\')."""
        n=0
        for item in self._multilayer_structure:
            if isinstance(item, int):
                n+=1
            elif isinstance(item, list):
                n+=item[0]*len(item[1])
        return n
        
         
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
        
           See :meth:`.__init__` for details. Only difference is: you cannot make changes which would remove layers.
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
            numberofindices=self._consistency_check(number_of_layers, multilayer_structure)
        if numberofindices<len(self._listoflayers):
            for item in self._listoflayers[numberofindices:]:
                if item is not None:
                    raise Exception("Cannot change layout. Remove layers with index > "+str(numberofindices-1)+" first.")
            self._listoflayers=self._listoflayers[:numberofindices]
        elif numberofindices>len(self._listoflayers):
            self._listoflayers+=[None for i in range(numberofindices-len(self._listoflayers))]
        self._number_of_layers=number_of_layers
        self._multilayer_structure=multilayer_structure
        self._updatePyReflMLString()
        
    def setLayer(self,index, layer):
        """
        Place **layer** (instance of :class:`.LayerObject`) at position **index** (counting from 0, starting from the bottom or according to indices defined by **multilayer_structure**).
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
        Return the instance of :class:`.LayerObject` which is placed at position **index** (counting from 0, starting from the bottom or according to indices defined by **multilayer_structure**).
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
        Return the instance of :class:`.LayerObject` which is placed at position **index** (counting from 0, starting from the bottom, repeated layers are counted repeatedly).
        """
        return self.getLayer(self._mapTotalIndexToInternal(index))
    
    def removeLayer(self,index):
        """
        Remove the instance of :class:`.LayerObject` which is placed at position **index** (counting from 0, starting from the bottom or according to indices defined by **multilayer_structure**).
        
        
        **index** can also be a list of indices.
        BEWARE: The instance of :class:`.LayerObject` itself and the corresdonding instances of :class:`Parameters.Fitparameter` are not deleted! So in a following fitting procedure, these parameters might still be varied even though they don't have any effect on the result.
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

    
    
    def getSingleEnergyStructure(self,fitpararray,energy=None):
        """
        Return list of layers (Layer type of the :mod:`Pythonreflectivity` package from Martin Zwiebler) which can be directly used as input for :func:`Pythonreflectivity.Reflectivity`.
        
        **energy** in units of eV
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
    N=property(_getNumberOfLayers)                                             
    """(*int*) Number of different layers. Read-only."""
    N_total=property(_getTotalNumberOfLayers)                                
    """(*int*) Total number of layers counting also multiple use of the same layer according to **multilayer_structure**. Read-only."""
    

#-----------------------------------------------------------------------------------------------------------------------------
# Layer Classes

class LayerObject(object):
    """Base class for all layer objects as the common interface. Speciallized implementation should inherit from this class.
    """
    
    def __init__(self, chitensor=None, d=None,  sigma=None, magdir="0"):
        """        
        Parameters
        ----------
        d : :class:`Parameters.Parameter`
            Thickness. Unit is the same as for every other length used throughout the project and is not predefined. E.g. wavelength.
        sigma : :class:`Parameters.Parameter`
            The roughness of the upper surface of the layer. Has dimension of length. Unit: see **d**.
        chitensor : list of :class:`Parameters.Parameter`
            Electric susceptibility tensor of the layer.
            | *[chi]* sets *chi_xx = chi_yy = chi_zz = chi*
            | *[chi_xx,chi_yy,chi_z]* sets *chi_xx,chi_yy,chi_zz*, others are zero
            | *[chi_xx,chi_yy,chi_z,chi_g]* sets  *chi_xx,chi_yy,chi_zz* and depending on **magdir** *chi_yz=-chi_zy=chi_g* (if *x*), *chi_xz=-chi_zx=chi_g* (if *y*) or *chi_xz=-chi_zx=chi_g* (if *z*)
            | *[chi_xx,chi_xy,chi_xz,chi_yx,chi_yy,chi_yz,chi_zx,chi_zy,chi_zz]* sets all the corresdonding elements
         
        magdir : str
            Gives the magnetization direction for MOKE. Possible values are *\"x\"*, *\"y\"*, *\"z\"* and *\"0\"* (no magnetization).
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
                raise TypeError("Elements of \'chitensor\' must be instances of class \'Parameters.Parameter\' or of an derived class (e.g. \'Parameters.Fitparameter\'")
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
        Return the electric susceptibility tensor as a list of numbers for a certain **energy** corresponding to the parameter values in **fitpararray** (see :mod:`Parameters`).
    
        The returned list can be of length 1,3,4 or 9 (see :meth:`.__init__`).
        For the base implementation of :class:`.LayerObject` the parameter **energy** is not used. But it may be used by derived classes like :class:`.AtomLayerObject` and therefore needed for compatibility.l
        **energy** is measured in units of eV.
        """
        #check tensor bevor giving it to the outside
        if not (len(self._chitensor)==1 or len(self._chitensor)==3 or len(self._chitensor)==4 or len(self._chitensor)==9):
            raise ValueError("Chitensor must be either of length 1, 3, 4, or 9.")
        if len(self._chitensor)==4 and self._magdir=="0":
            raise ValueError("You have to define the direction of magnetization \'magdir\'.")
        #return the checked list filled with actual values
        return [item.getValue(fitpararray) for item in self._chitensor]
    
    def getD(self,fitpararray):
        """
        Return the thickness of the layer as a number corresponding to the parameter values in **fitpararray** (see :mod:`Parameters`).
        
        The thickness is given in the unit of length you chose. You are free to choose whatever unit you want, but use the same for every length troughout the project.
        """
        if self._d is None:
            return 0
        else:
            return self._d.getValue(fitpararray)
    
    def getSigma(self,fitpararray):
        """
        Return the roughness of the upper surface of the layer as a number corresponding to the parameter values in **fitpararray** (see :mod:`Parameters`).
        
        The thickness is given in the unit of length you chose. You are free to choose whatever unit you want, but use the same for every length troughout the project.
        """
        if self._sigma is None:
            return 0
        else:
            return self._sigma.getValue(fitpararray)
    
    def getMagDir(self,fitpararray=None):
        """
        Return magnetization direction
        
        **fitpararray** is not used and just there to give a common interface. Maybe a derived classes will have a benefit from it.
        """
        return self._magdir
    
    
    
    
    #exposed properties (feel like instance variables but are protected via getter and setter methods)
    #BEWARE: theese properties contain the "Parameter" objects. E.g. to get a certain value for the thicknes do this: "layer.d.getValue(parameterarray)"
    d=property(_getd_,_setd_)
    """Thickness of the layer. Unit is the same as for every other length used throughout the project and is not predefined. E.g. wavelength."""
    chitensor=property(_getChitensor_,_setChitensor_)
    """Electric susceptibility tensor of the layer. See :meth:`.__init__` for details."""
    sigma=property(_getSigma_,_setSigma_)
    """Roughness of the upper surface of the layer. Unit is the same as for every other length used throughout the project and is not predefined. E.g. wavelength."""
    magdir=property(_getMagdir_,_setMagdir_)
    """Magnetization direction for MOKE. Possible values are *\"x\"*, *\"y\"*, *\"z\"* and *\"0\"* (no magnetization)."""
    
    

class ModelChiLayerObject(LayerObject):
    """Speciallized layer to deal with an electrical suszeptibility tensor (Chi) which is modelled as function of energy.
    
    BEWARE: The inherited property :attr:`.chitensor` is now a function.
    """
    
    def __init__(self, chitensorfunction, d=None,  sigma=None, magdir="0"):
        """
        Parameters
        ----------
        d : :class:`Parameters.Parameter`
            Thickness. Unit is the same as for every other length used throughout the project and is not predefined. E.g. wavelength.
        sigma : :class:`Parameters.Parameter`
            The roughness of the upper surface of the layer. Has dimension of length. Unit: see **d**.
        chitensorfunction : callable
            Energy-dependent electric susceptibility tensor of the layer.
            It is supposed to be a function of two parameters (**fitpararray**, **energy**) which returns a list of either 1,3,4 or 9 real or complex numbers.
            See also documentation of :mod:`Pythonreflectivity`.
            
            * *[chi]* sets *chi_xx = chi_yy = chi_zz = chi*
            * *[chi_xx,chi_yy,chi_z]* sets *chi_xx,chi_yy,chi_zz*, others are zero
            * *[chi_xx,chi_yy,chi_z,chi_g]* sets  *chi_xx,chi_yy,chi_zz* and depending on **magdir**
              *chi_yz=-chi_zy=chi_g* (if *x*), *chi_xz=-chi_zx=chi_g* (if *y*) or *chi_xz=-chi_zx=chi_g* (if *z*)
            * *[chi_xx,chi_xy,chi_xz,chi_yx,chi_yy,chi_yz,chi_zx,chi_zy,chi_zz]* sets all the corresdonding elements
            
        magdir : str
            Gives the magnetization direction for MOKE. Possible values are *\"x\"*, *\"y\"*, *\"z\"* and *\"0\"* (no magnetization).
        """
        
        #check parameters
        if not callable(chitensorfunction):
            raise TypeError("\'chitensorfunction has to be callable.")
        #asign members
        self._chitensorfunction=chitensorfunction
        #call constructor of the base class
        super(type(self),self).__init__(None, d,  sigma, magdir)       
        
    def _getChitensor_(self):
        return self._chitensorfunction
    
    def _setChitensor_(self,chitensorfunction):
        if not callable(chitensorfunction):
            raise TypeError("\'chitensorfunction has to be callable.")
        self._chitensorfunction=chitensorfunction

    
    #public methods
    def getChi(self,fitpararray,energy):
        """
        Return the electric susceptibility tensor as a list of numbers for a certain **energy** corresponding to the parameter values in **fitpararray** (see :mod:`Parameters`).
    
        **energy** is measured in units of eV.
        """
        #get chitensor from chitensorfunction
        chitensor=self._chitensorfunction(fitpararray, energy)
        #check tensor bevor giving it to the outside
        if not (len(chitensor)==1 or len(chitensor)==3 or len(chitensor)==4 or len(chitensor)==9):
            raise ValueError("\'chitensorfunction\' must return an array of length 1, 3, 4, or 9.")
        if len(chitensor)==4 and self._magdir=="0":
            raise ValueError("You have to define the direction of magnetization \'magdir\'.")
        for item in chitensor:
            if not isinstance(item,numbers.Number):
                raise TypeError("Elements of the array returnd by \'chitensorfunction\' have to be numbers.")
        #return the checked list
        return chitensor
    
        
        
class AtomLayerObject(LayerObject):
    """
    Speciallized layer class to deal with compositions of atoms and their energy dependent formfactors (which can be obtained from absorption spectra).
    
    Especially usefull to deal with atomic layers, but can also be used for bulk.
    The atoms and their formfactors have to be registered a the class (with registerAtom) before they can be used to instantiate a new AtomLayerObject.
    The atom density can be plotted with :func:`.plotAtomDensity`.
    Density is measured in mol/cm$^3$ (as long as no **densityunitfactor** is applied)
    """
    
    def __init__(self, densitydict={}, d=None,  sigma=None, magdir="0", densityunitfactor=1.0):
        """
        Parameters
        ----------
        densitydict :
            a dictionary which contains atom names (strings, must agree with before registered atoms) and densities (must be instances of :class:`Parameters.Parameter` or of a derived class).
        d : :class:`Parameters.Parameter`
            Thickness. Unit is the same as for every other length used throughout the project and is not predefined. E.g. wavelength.
        sigma : :class:`Parameters.Parameter`
            The roughness of the upper surface of the layer. Has dimension of length. Unit: see **d**.
        chitensor : list of :class:`Parameters.Parameter`
            Electric susceptibility tensor of the layer.
            | *[chi]* sets *chi_xx = chi_yy = chi_zz = chi*
            | *[chi_xx,chi_yy,chi_z]* sets *chi_xx,chi_yy,chi_zz*, others are zero
            | *[chi_xx,chi_yy,chi_z,chi_g]* sets  *chi_xx,chi_yy,chi_zz* and depending on **magdir** *chi_yz=-chi_zy=chi_g* (if *x*), *chi_xz=-chi_zx=chi_g* (if *y*) or *chi_xz=-chi_zx=chi_g* (if *z*)
            | *[chi_xx,chi_xy,chi_xz,chi_yx,chi_yy,chi_yz,chi_zx,chi_zy,chi_zz]* sets all the corresdonding elements
         
        magdir : str
            Gives the magnetization direction for MOKE. Possible values are *\"x\"*, *\"y\"*, *\"z\"* and *\"0\"* (no magnetization).
        densityunitfactor : float
            If the densities in densitydict are measured in another unit than mol/cm^3, state this value which translates your generic density to the one used internally.
            I.e.::
            
                rho_in_mol_per_cubiccm = densityunitfactor * rho_in_whateverunityouwant
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
        """Return the density dictionary either with evaluated paramters (needs **fitpararray**) or with the raw :class:`Parameters.Parameter` objects (use **fitparraray** = *None*)."""
        if fitpararray==None:
            return self._densitydict.copy()
        elif not isinstance(fitpararray,list):
            raise TypeError("\fitparray\' has to be a list.")
        else:
            return dict(zip(self._densitydict.keys(),[item.getValue(fitpararray) for item in self._densitydict.values()]))              #pack new dictionary from atomnames and "unpacked" parameters (actual values instead of abstract parameter)
       
    def getChi(self,fitpararray,energy):
        """
        Return the electric susceptibility tensor as a list of numbers for a certain **energy** corresponding to the parameter values in **fitpararray** (see :mod:`Parameters`).
    
        **energy** is measured in units of eV.
        """
        #gehe alle Items in self._densitydict durch, item[1].getValue(fitpararray) liefert Dichte der Atomsorte, (type(self)._atomdict[item[0]]).getFF(fitpararray) liefert Formfaktor der Atomsorte, beides wird multipliziert und alles zusammen aufsummiert
        ffsum=sum([item[1].getValue(fitpararray)*(type(self)._atomdict[item[0]]).getFF(energy,fitpararray) for item in self._densitydict.items()])
        
        
        
        # Return the susceptibility tensor chi
        # As chi is very small, the linear approximation can be used.
        # If there is no densityunitfactor defined, the densities are assumed to be in units of mol/cm^3.
        # Energy is assumed to be in units of eV.
        # The susceptibility is therefore given as: chi= 4* pi * h_bar^2 [eV*s]^2 * c^2 [m/s]^2 * r_e [m] * N_A [1/mol] * (100cm)^3/m^3 * ffsum (mol/cm^3) / E^2 (eV)^2
        # N_A: Arvogardros number, r_e: thomson scattering length/classical electron radius
        return list(ffsum*830.3584763651544/energy**2)                
        
        
        
        
    
    #classvariables
    _atomdict={}
    
    #classmethods                                               #are not related to an instance of a class and are here used to deal with the collection of all registered atoms
    @classmethod
    def registerAtom(cls, name,formfactor):
        """
        Register an atom called **name** for later use to instantiate an AtomLayerObject.
        
        **formfactor** as to be an instance of :class:`.Formfactor` or of a derived class.
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
        """Return the :class:`.Formfactor` object registered for atom **name**."""
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
# Formfactor classes

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
        Return the formfactor for **energy** corresponding to **fitpararray** (if it depends on it) as 9-element list of complex numbers.
        
        **energy** is measured in units of eV.
        """
        raise NotImplementedError
    
    def _getMinE(self):
        raise NotImplementedError
    
    def _getMaxE(self):
        raise NotImplementedError
    
    def plotFF(self,fitpararray=None,energies=None):
        """
        Plot the energy-dependent formfactor with the energies (in eV) listed in the list/array **energies**. If this array is not given the plot covers the hole existing energy-range.
        The **fitpararray** has only to be given in cases where the formfactor depends on a fitparamter, e.g. for class:`FFfromScaledAbsorption`.
        """
        if energies is None:
            energies=numpy.linspace(self.minE, self.maxE, 1000)
        ff=[]
        for e in energies:
            ff.append(self.getFF(e,fitpararray))
        ff=numpy.array(ff)
        ff=numpy.transpose(ff)
        fig = matplotlib.pyplot.figure(figsize=(10,10))
        axes=[]
        for i in range(9):
            axes.append(fig.add_subplot(330+i+1))
        i=0
        for ax in axes:
            ax.set_xlabel('energy (eV)')
            ax.locator_params(axis='x', nbins=4)
            ax.plot(energies,ff[i].real)
            ax.plot(energies,ff[i].imag)
            i+=1
        matplotlib.pyplot.show()
        
    #properties
    maxE=property(_getMaxE)
    """Upper limit of stored energy range. Read-only."""
    minE=property(_getMinE)
    """Lower limit of stored energy range. Read-only."""
    
class FFfromFile(Formfactor):
    """
    Class to deal with energy-dependent atomic form-factors which are tabulated in files.
    """
  
    def __init__(self, filename, linereaderfunction=None, energyshift=Parameters.Parameter(0)):
        """Initializes the FFfromFile object with an energy-dependent formfactor given as file.
        
        Parameters
        ----------
        filename : str
            Path to the text file which contains the formfactor.       
        linereaderfunction : callable
            This function is used to convert one line from the text file to data.
            It should be a function which takes a string and returns a tuple or list of 10 values: ``(energy,f_xx,f_xy,f_xz,f_yx,f_yy,f_yz,f_zx,f_zy,f_zz)``,
            where `energy` is measured in units of `eV` and formfactors are complex values in units of `e/atom` (dimensionless).
            It can also return `None` if it detects a comment line.
            You can use :meth:`FFfromFile.createLinereader` to get a standard function, which just reads this array as whitespace seperated from the line.
        energyshift : :class:`Parameters.Parameter`
            Species a fittable energyshift between the energy-dependent formfactor from **filename** and the `real` one in the reflectivity measurement.
            So the formfactor delivered from :meth:`FFfromFile.getFF` will not be `formfactor_from_file(E)` but `formfactor_from_file(E+energyshift)`.
        """
        if not isinstance(filename,str):
            raise TypeError("\'filename\' needs to be a string.")
        if linereaderfunction is None:
            linereaderfunction=self.createLinereader()
        if not callable(linereaderfunction):
            raise TypeError("\'linereaderfunction\' needs to be a callable object.")
        if not os.path.isfile(filename):
            raise Exception("File \'"+filename+"\' does not exist.")
        if not isinstance(energyshift,Parameters.Parameter):
            raise TypeError("\'energyshift\' has to be of type Parameters.Parameter.")
        energies=[]
        formfactors=[]
        with open(filename,'r') as f:
            for line in f:
                linereaderoutput=linereaderfunction(line)
                if linereaderoutput is None:
                    break
                if not isinstance(linereaderoutput,(tuple,list)) :
                    raise TypeError("Linereader function has to return a list/tuple.")
                if not  len(linereaderoutput)==10:
                    raise ValueError("Linereader function hast to return a list/tuple with 10 elements.")
                for item in linereaderoutput:
                    if not isinstance(item,numbers.Number):
                        raise ValueError("Linereader function hast to return a list/tuple of numbers.")
                if isinstance(linereaderoutput[0],complex):
                    raise ValueError("Linereader function hast to return a real value for the energy.")
                energies.append(linereaderoutput[0])                                                        #store energies in one list
                formfactors.append(linereaderoutput[1:])                                                    #store corresponding formfactors in another list
        formfactors=numpy.array(formfactors)                                                                #convert list formfactors to a numpy array for convinience
        self._minE=min(energies)
        self._maxE=max(energies)
        self._energyshift=energyshift                                                                       #Attention: this is supposed to be an instance of "Parameters.Parameter". So a value can be obtained with self._energyshift.getValue(fitparraray)
        #Create an interpolation function based on the given energie-formfactor-points. The formfactors are thererfore transformed to arrays of length 18 but with real values. 
        #After that the array of N arrays of 18 element is transformed to an array of 18 arrays of N elements as needed by the interp1d function.
        #Therefore, this function will return an array of length 18 wich has to be transformed back to 9 complex valued elements.
        #Energies and formfactors don't have to be stored explicitly , because they are contained in the "self._interpolator" function.
        self._interpolator=interpolate.interp1d(energies,numpy.transpose(numpy.concatenate((formfactors.real,formfactors.imag),1))) 
    
    def _getMinE(self):
        return self._minE
    
    def _getMaxE(self):
        return self._maxE
    
    
    #public methods
    
    @staticmethod
    def createLinereader(complex_numbers=True):
        """
        Return the standard linereader function for usage with :meth:`FFfromFile.__init__`.
        
        This standard linereader function reads energy and complex elements of the formfactor tensor as a whitespace-seperated list (i.e. 10 numbers) and interpretes \"#\" as comment sign.
        If **complex_numbers** = *False* then the reader reads real and imaginary part of every element seperately, i.e. every line has to consist of 19 numbers seperated by whitespaces::
            
            energy f_xx_real ff_xx_im ... f_zy_real f_zy_im f_zz_real f_zz_im
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
        Return the (energy-shifted )formfactor for **energy** as an interpolation between the stored values from file as 9-element 1-D numpy array of complex numbers.
        
        Parameters
        ----------
        energy : float
            Measured in units of eV.
        fitpararray :
            Is actually only needed when an energyshift has been defined.
        """
        energyshift=self._energyshift.getValue(fitpararray)
        
        if energy+energyshift<self.minE or energy+energyshift>self.maxE:
            raise ValueError("\'energy + energyshift = "+str(energy)+" + "+ str(energyshift) + " = " + str(energy+energyshift) +"\' is out of range ("+str(self.minE)+","+str(self.maxE)+").")
        FFallReal=self._interpolator(energy+energyshift)
        #return directly the numpy array, it is usefull further Calculation
        return FFallReal[:9]+FFallReal[9:]*1j
    
    #properties
    maxE=property(_getMaxE)
    """Upper limit of stored energy range. Read-only."""
    minE=property(_getMinE)
    """Lower limit of stored energy range. Read-only."""
        
        
class FFfromScaledAbsorption(Formfactor):
    """
    A formfactor class which uses the imaginary part of the formfactor (experimentally determined absorption signal which has been fitted to off-resonant values) given as a file, scales it with a fittable factor and calculates the real part by the Kramers-Kronig transformation. It realizes the procedure described in section 3.3 of Martin Zwiebler PhD-Thesis.
    """
    
    def __init__(self, E1, E2, E3, scaling_factor, theoretical_filename, absorption_filename, theoretical_linereaderfunction=None, absorption_linereaderfunction=None, energyshift=Parameters.Parameter(0), minE=None, maxE=None):
        """Initializes the FFfromScaledAbsorption object with an energy-dependent imaginary part of the formfactor given as file.
        
        To perform the Kramers-Kronig transformation without integrating to infinity, also theoretical/tabulated formfactors (f0) have to be given as file. Their imaginary part differs only close to resonance from the measured absorption and should have been used before to perform the fit of the measured absorption to off-resonant values. As these tabulated form factors are usually not polarization dependent, they should be given here just as complex values (or two real ones).
            
        
        The imaginary part of each element of the formfactor is:
        
        * the value given by **Im_f0_E1**, for energy < **E1**.
        * the value given by the file scaled by **scaling_factor** (roughly, see Thesis for details), for E1 <= energy <= E2
        * linear inperpolation between the scaled value at E2 and the value given for E3 by **Im_f0_E3**, for E2 < energy < E3
        * the value given by **Im_f0_E3**, for E3 < energy       
        
        The Kramers-Kronig transformation to obtain the real part is done only once during instantiation. Therefore, it does not have to be repeated with every value of the **scaling_factor** and fitting is fast.
        
        Parameters
        ----------
        E1 : float
            Energy in eV. From this energy on the energy-dependent imaginary part of the formfactor given as file is used and scaled.
        E2 : float
            Energy in eV. From this energy on the imaginary part of the formfactor is linearly interpolated.
        E3 : float
            Energy in eV. From this energy on the imaginary part of the formactor is constant **Im_f_E3**.
        scaling_factor : :class:`Parameter.Parameter`
            Specifies the fittable scaling factor (called *a* in Martin Zwiebler PhD Thesis).
        theoretical_filename : str
            Path to the text file which contains the tabulated/theoretical formfactor for the corresponding element.
        absorption_filename : str
            Path to the text file which contains the imaginary part of the formfactor which results from an apsorption measurement and a subsequent fit to off-resonant tabulated values.
        theoretical_linereaderfunction : callable
            This function is used to convert one line from the *theoretical* text file to data.
            It should be a function which takes a string and returns a tuple or list of 10 values: ``(energy,f_xx,f_xy,f_xz,f_yx,f_yy,f_yz,f_zx,f_zy,f_zz)``,
            where `energy` is measured in units of `eV` and formfactors are complex values in units of `e/atom` (dimensionless).
            It can also return `None` if it detects a comment line.
            You can use :meth:`FFfromScaledAbsorption.createTheoreticalLinereader` to get a standard function, which just reads this array as whitespace separated from the line.
        absorption_linereaderfunction : callable
            This function is used to convert one line from the *absorption* text file to data.
            It should be a function which takes a string and returns a tuple or list of 10 values: ``(energy, Im f_xx, Im f_xy, Im f_xz, Im f_yx, Im f_yy, Im f_yz, Im f_zx, Im f_zy, Im f_zz)``,
            where `energy` is measured in units of `eV` and imaginary parts of formfactors are real values in units of `e/atom` (dimensionless).
            It can also return `None` if it detects a comment line.
            You can use :meth:`FFfromScaledAbsorption.createAbsorptionLinereader` to get a standard function, which just reads this array as whitespace seperated from the line.
        energyshift : :class:`Parameters.Parameter`
            Species a fittable energyshift between the energy-dependent formfactor calculated by the whole above mentioned procedure and the `real` one in the reflectivity measurement.
            As a consequence the peak but also E1,E2 and E3 are shifted.
        minE : float
        maxE : float
            Specify minimum and maximum energy if you don't want to use the whole energy-range given in the file **theoretical_filename**. Reducing the energy-range speeds up the Kramers-Kronig transformations significantly.
        """
        #check parameters
        if not isinstance(E1, numbers.Real):
            raise TypeError("\'E1\' needs to be a real number.")
        if not isinstance(E2, numbers.Real):
            raise TypeError("\'E2\' needs to be a real number.")
        if not isinstance(E3, numbers.Real):
            raise TypeError("\'E3\' needs to be a real number.")
        if E1<0 or E2<0 or E3<0:
            raise ValueError("Energies \'E1\', \'E2\' and \'E3\' have to be greater than zero.")
        if not [E1,E2,E3]==sorted([E1,E2,E3]):
            raise ValueError("Energies \'E1\', \'E2\' and \'E3\' have to have ascending values.")
        if not isinstance(scaling_factor,Parameters.Parameter):
            raise TypeError("\'scaling_factor\' has to be of type Parameters.Parameter.")    
        if not isinstance(theoretical_filename,str):
            raise TypeError("\'theoretical_filename\' needs to be a string.")
        if not os.path.isfile(theoretical_filename):
            raise Exception("File \'"+theoretical_filename+"\' does not exist.")
        if not isinstance(absorption_filename,str):
            raise TypeError("\'absorption_filename\' needs to be a string.")
        if not os.path.isfile(absorption_filename):
            raise Exception("File \'"+absorption_filename+"\' does not exist.")
        if theoretical_linereaderfunction is None:
            theoretical_linereaderfunction=self.createTheoreticalLinereader()
        if not callable(theoretical_linereaderfunction):
            raise TypeError("\'theoretical_linereaderfunction\' needs to be a callable object.")
        if absorption_linereaderfunction is None:
            absorption_linereaderfunction=self.createAbsorptionLinereader()
        if not callable(absorption_linereaderfunction):
            raise TypeError("\'absorption_linereaderfunction\' needs to be a callable object.")
        if not isinstance(energyshift,Parameters.Parameter):
            raise TypeError("\'energyshift\' has to be of type Parameters.Parameter.")
        if (minE is None and maxE is not None) or (minE is not None and maxE is None):
            raise Exception("You have to set both, \'minE\' and \'maxE\'.")
        if minE is not None and (not isinstance(minE,numbers.Real) or minE<0):
            raise TypeError("\'minE\' must be a positive real number.")
        if maxE is not None and (not isinstance(maxE,numbers.Real) or maxE<0):
            raise TypeError("\'maxE\' must be a positive real number.")
        if not minE<maxE:
            raise ValueError("\'minE\' must be smaller than \'maxE\'.")
        
        #store parameters
        self._E1=E1
        self._E2=E2
        self._E3=E3
        self._scaling_factor=scaling_factor       #Attention: this is supposed to be an instance of "Parameters.Parameter". So a value can be obtained with self._scaling_factor.getValue(fitparraray)
        self._energyshift=energyshift             #Attention: this is supposed to be an instance of "Parameters.Parameter". So a value can be obtained with self._energyshift.getValue(fitparraray)
        
        #read theoretica/tabulated formfactors from file
        theo_energies=[]
        theo_formfactors=[]
        with open(theoretical_filename,'r') as f:
            for line in f:
                linereaderoutput=theoretical_linereaderfunction(line)
                if linereaderoutput is None:
                    break
                if not isinstance(linereaderoutput,(tuple,list)) :
                    raise TypeError("Linereader function has to return a list/tuple.")
                if not  len(linereaderoutput)==2:
                    raise ValueError("Linereader function hast to return a list/tuple with 2 elements.")
                for item in linereaderoutput:
                    if not isinstance(item,numbers.Number):
                        raise ValueError("Linereader function hast to return a list/tuple of numbers.")
                if isinstance(linereaderoutput[0],complex):
                    raise ValueError("Linereader function hast to return a real value for the energy.")
                if (minE is not None and maxE is not None):
                    if minE<=linereaderoutput[0] and linereaderoutput[0]<=maxE:
                        theo_energies.append(linereaderoutput[0])                                                        #store energies in one list
                        theo_formfactors.append(linereaderoutput[1:])                                                    #store corresponding formfactors in another list
                else:
                    theo_energies.append(linereaderoutput[0])                                                        #store energies in one list
                    theo_formfactors.append(linereaderoutput[1:])                                                    #store corresponding formfactors in another list
        theo_formfactors=numpy.array(theo_formfactors)                                                                #convert list formfactors to a numpy array for convinience
        self._minE=min(theo_energies)
        self._maxE=max(theo_energies)
        #Create an interpolation function based on the given energie-formfactor-points. The formfactors are thererfore transformed to arrays of length 2 but with real values. 
        #After that the array of N arrays of 2 element is transformed to an array of 2 arrays of N elements as needed by the interp1d function.
        #Therefore, this function will return an array of length 2 wich has to be transformed back to 1 complex valued elements.
        self._theo_interpolator=interpolate.interp1d(theo_energies,numpy.transpose(numpy.concatenate((theo_formfactors.real,theo_formfactors.imag),1)))
                
        #read imaginary part of formfactor from file
        abs_energies=[]
        abs_im_formfactors=[]
        with open(absorption_filename,'r') as f:
            for line in f:
                linereaderoutput=absorption_linereaderfunction(line)
                if linereaderoutput is None:
                    break
                if not isinstance(linereaderoutput,(tuple,list)) :
                    raise TypeError("Linereader function has to return a list/tuple.")
                if not  len(linereaderoutput)==10:
                    raise ValueError("Linereader function hast to return a list/tuple with 10 elements.")
                for item in linereaderoutput:
                    if not isinstance(item,numbers.Real):
                        raise ValueError("Linereader function hast to return a list/tuple of real numbers.")
                abs_energies.append(linereaderoutput[0])                                                        #store energies in one list
                abs_im_formfactors.append(linereaderoutput[1:])                                                    #store corresponding formfactors in another list
        abs_im_formfactors=numpy.array(abs_im_formfactors)                                                                #convert list formfactors to a numpy array for convinience
        self._abs_minE=min(abs_energies)
        self._abs_maxE=max(abs_energies)
        if self._E1 < self._abs_minE or self._E2 > self._abs_maxE:
            raise Exception("Given absorption data does not cover needed range between E1 and E2.")
        #Create an interpolation function based on the given energie-imag_formfactor-points. 
        #Therefore, the array of N arrays of 9 element is transformed to an array of 9 arrays of N elements as needed by the interp1d function.
        #This function will return an array of length 9 which is the interpolated imaginary part of the formfactor at the requested energy.
        #Energies and formfactors don't have to be stored explicitly , because they are contained in the "self._interpolator" function.
        self._abs_interpolator=interpolate.interp1d(abs_energies,numpy.transpose(abs_im_formfactors) )
        
        
        #create array of energies and different arrays of 9-element tensor to calculate the actuall formfactors
        # "im_f1": imaginary part of formfactors for scaling_factor (a) = 1
        # "d_im_f1": derivative with respect to a of the imaginary part of formfactors
        # "diff_im_f1": difference between "im_f1" and theoretical formfactors
        # "kk_diff_im_f1": Kramers-Kronig transformed "diff_im_f1
        # "kk_d_im_f1": Kramers-Kronig transformed "d_im_f1"
        energies=[]
        im_f1=[]
        d_im_f1=[]
        diff_im_f1=[]
        im_f1_E2 = self._abs_interpolator(self._E2)        #imaginary part of formfactors for scaling_factor (a) = 1 at E2
        re,im = self._theo_interpolator(self._E1)
        im_f0_E1 = numpy.array([im,0,0,0,im,0,0,0,im])     #imaginary part of formfactors  at E1  (does not depend on "a")
        re,im = self._theo_interpolator(self._E3)
        im_f0_E3 = numpy.array([im,0,0,0,im,0,0,0,im])     #imaginary part of formfactors  at E3  (does not depend on "a")
        i=0
        for energy in theo_energies:
            if energy < self._E1:
                energies.append(energy)
                im_f1.append(numpy.array([theo_formfactors[i].imag,0,0, 0,theo_formfactors[i].imag,0, 0,0,theo_formfactors[i].imag]))      #below E1, formfactor tensor is a diagonal tensor, values are the given "theoretical" ones
                d_im_f1.append(numpy.array([0,0,0,0,0,0,0,0,0]))
                diff_im_f1.append(numpy.array([0,0,0,0,0,0,0,0,0]))                               
            i+=1
        i=0
        for energy in abs_energies:
            if energy >= self._E1 and energy <=self._E2:
                energies.append(energy)
                im_f1.append(abs_im_formfactors[i])                                #between E1 and E2, it is just the measured imaginary part of the formfactor
                d_im_f1.append(abs_im_formfactors[i]-im_f0_E1)                         #between E1 and E2, the derivative is the difference between "measured" imaginary part of the formfactor and the "theoretical" value at E1
                re,im = self._theo_interpolator(energy)
                diff_im_f1.append( abs_im_formfactors[i] - numpy.array([im,0,0,0,im,0,0,0,im]) )   #between E1 and E2, just the difference between "measured" imaginary part of the formfactor and the "theoretical" value 
            i+=1
        i=0
        for energy in theo_energies:
            if energy > self._E2:
                energies.append(energy)
            if energy > self._E2 and energy < self._E3:
                im_f1.append( im_f1_E2 + (energy-E2)/(E3-E2)*(im_f0_E3 - im_f1_E2) )             #between E2 and E3, it is a linear interpolation
                d_im_f1.append( (self._E3-energy)/(self._E3-self._E2) * (im_f1_E2-im_f0_E1) )    #between E2 and E3, the imaginary part is a linear interpolation, the derivative correspondingly
                re,im = self._theo_interpolator(energy)
                diff_im_f1.append( im_f1[i] - numpy.array([im,0,0,0,im,0,0,0,im]) )   #between E1 and E2, just the difference between linearly interpolated imaginary part of the formfactor and the "theoretical" value 
            elif energy >= self._E3:
                im_f1.append(numpy.array([theo_formfactors[i].imag,0,0, 0,theo_formfactors[i].imag,0, 0,0,theo_formfactors[i].imag]))      #above E3, formfactor tensor is a diagonal tensor, values are the given "theoretical" ones
                d_im_f1.append(numpy.array([0,0,0,0,0,0,0,0,0]))
                diff_im_f1.append(numpy.array([0,0,0,0,0,0,0,0,0])) 
            i+=1
            
        im_f1=numpy.array(im_f1)
        d_im_f1=numpy.array(d_im_f1)
        diff_im_f1=numpy.array(diff_im_f1)
        
       
        #perform Kramers-Kronig transformations to obtain "kk_diff_im_f1" and "kk_d_im_f1"
        #therefore one array for every position of the tensors is needed, which can be obtained with "transpose"
        trans_diff_im_f1=numpy.transpose(diff_im_f1)
        trans_d_im_f1=numpy.transpose(d_im_f1)
        trans_kk_diff_im_f1=[]
        trans_kk_d_im_f1=[]
        for i in range(9):
            trans_kk_diff_im_f1.append(KramersKronig(energies,trans_diff_im_f1[i]))
            trans_kk_d_im_f1.append(KramersKronig(energies,trans_d_im_f1[i]))
        trans_kk_diff_im_f1=numpy.array(trans_kk_diff_im_f1)
        trans_kk_d_im_f1=numpy.array(trans_kk_d_im_f1)
        

     
        #Create interpolation functions for the above created arrays for later use with "getFF"
        #For this the arrays of N arrays of 9 elements are transposed to arrays of 9 arras of N elements as needed by the interp1d function (trans_kk_diff_im_f1 and trans_kk_d_im_f1 are already transposed)
        #The functions will return arrays of length 9 for the corresponding energy as interpolation.
        self._im_f1_interpolator=interpolate.interp1d(energies,numpy.transpose(im_f1))
        self._d_im_f1_interpolator=interpolate.interp1d(energies,trans_d_im_f1)
        self._kk_diff_im_f1_interpolator=interpolate.interp1d(energies,trans_kk_diff_im_f1)
        self._kk_d_im_f1_interpolator=interpolate.interp1d(energies,trans_kk_d_im_f1)
        
        
    
    def _getMinE(self):
        return self._minE
    
    def _getMaxE(self):
        return self._maxE
    
    
    #public methods
    
    @staticmethod
    def createAbsorptionLinereader():
        """
        Return the standard linereader function for absorption files for usage with :meth:`FFfromScaledAbsorption.__init__`.
        
        This standard linereader function reads energy and elements of the imaginary part of the formfactor tensor as a whitespace-seperated list (i.e. 10 numbers) and interpretes \"#\" as comment sign.
        """
        commentsymbol='#'
        def linereader(line):
            if not isinstance(line,str):
                raise TypeError("\'line\' needs to be a string.")
            line=(line.split(commentsymbol))[0]                            #ignore everything behind the commentsymbol  #
            if not line.isspace():                               #ignore empty lines        
                linearray=line.split()
                if not len(linearray)==10:
                    raise Exception("Absorption file has wrong format.")
                linearray=[ast.literal_eval(item) for item in linearray]
                return linearray
            else:
                return None
        return linereader                                                               #here the FUNKTION linereader is returned
    
    @staticmethod
    def createTheoreticalLinereader(complex_numbers=True):
        """
        Return the standard linereader function for theoretical formfactor files for usage with :meth:`FFfromScaledAbsorption.__init__`.
        
        This standard linereader function reads energy and the complex formfactor as a whitespace-seperated list (i.e. 2 numbers) and interpretes \"#\" as comment sign.
        If **complex_numbers** = *False* then the reader reads real and imaginary part of the formfactor seperately, i.e. every line has to consist of 3 numbers seperated by whitespaces::
            
            energy ff_real ff_im 
        
        If you want to read the complete complex formfactor tensor, you have to build your own linereader.
        """
        commentsymbol='#'
        if complex_numbers==True:
            def linereader(line):
                if not isinstance(line,str):
                    raise TypeError("\'line\' needs to be a string.")
                line=(line.split(commentsymbol))[0]                            #ignore everything behind the commentsymbol  #
                if not line.isspace():                               #ignore empty lines        
                    linearray=line.split()
                    if not len(linearray)==2:
                        raise Exception("File for theoretical formfactor has wrong format.")
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
                    if not len(linearray)==3:
                        raise Exception("File for theoretical formfactor file has wrong format.")
                    linearray=[ast.literal_eval(item) for item in linearray]
                    return [linearray[0], linearray[1]+1j*linearray[2]]
                else:
                    return None
        else:
            raise TypeError("\'complex_numbers\' has to be boolean.")
        return linereader       
            
    def getFF(self,energy,fitpararray):
        """
        Return the (energy-shifted )formfactor for **energy** as an interpolation between the stored values from file as 9-element 1-D numpy array of complex numbers.
        
        Parameters
        ----------
        energy : float
            Measured in units of eV.
        fitpararray :
            Is needed for the scaling factor \'a\' and an energy shift if defined.
        """
        scaling_factor=self._scaling_factor.getValue(fitpararray)
        energyshift=self._energyshift.getValue(fitpararray)
        
        if energy+energyshift<self.minE or energy+energyshift>self.maxE:                                  #use this strange construction with numpy arrays to allow "energy" to be a numpy array of energies
            raise ValueError("\'energy + energyshift = "+str(energy)+" + "+ str(energyshift) + " = " + str(energy+energyshift) +"\' is out of range ("+str(self.minE)+","+str(self.maxE)+").")
        
        energy=energy+energyshift  #apply energyshift
        
        ImFF  = (scaling_factor - 1) * self._d_im_f1_interpolator(energy)  + self._im_f1_interpolator(energy)
        theo_real=self._theo_interpolator(energy)[0]
        RealFF= numpy.array([theo_real,0,0,0,theo_real,0,0,0,theo_real]) + self._kk_diff_im_f1_interpolator(energy) + (scaling_factor - 1) * self._kk_d_im_f1_interpolator(energy)
           
        return RealFF + 1j* ImFF
    
    #properties
    maxE=property(_getMaxE)
    """Upper limit of stored energy range. Read-only."""
    minE=property(_getMinE)
    """Lower limit of stored energy range. Read-only."""
    
#-----------------------------------------------------------------------------------------------------------------------------
# Density Profil classes

class DensityProfile(object):
    """
    This class can be used to generate arbitrary density profiles within a stack of several :class:`.AtomLayerObject` of equal thicknesses.
    
    The idea is to collect all information regarding the density profile in an object of this class and to generate entries for the *densitydict* of the single :class:`.AtomLayerObject` instances from it.
    This means that the class :class:`.DensityProfile` does not really talk to the layers, but is only a higher level convinience class to set up the interconnected densities of the atoms within the layers as instances of :class:`Parameters.DerivedParameter`.
    """
    
    def __init__(self, start_layer_idx, end_layer_idx, layer_thickness, profile_function, *params):
        """
        Parameters
        ----------
        start_layer_idx : int
            Index of the first layer in the scope of the density profile.
        end_layer_idx : int
            Index of the last layer in the scope of the density profile.
        layer_thickness : :class:`Parameters.Parameter`
            Thickness of the individual layers. The Unit is the same as for every other length used throughout the project and is not predefined. E.g. wavelength.
        profile_function  : callable
            The density of the corresponding atom as function of the distance **z** from the lower surface of start layer and of an arbitray number of additional numeric parameters. The function realy has to expect these parameters to be floats or complex numbers not instances of :class:`Parameters.Parameter`. E.g. ``erf(z ,center, width, max)``
        *params : :class:`Parameters.Parameter`
            Parameters for the function **profile_function** without **z**. E.g. (from above) ``*params=(center,width,max)``
        """
        
        #type checking
        if not isinstance(start_layer_idx,int):
            raise TypeError("\'start_layer_idx\' has to be an integer number.")
        elif not isinstance(end_layer_idx,int):
            raise TypeError("\'end_layer_idx\' has to be an integer number.")
        elif not isinstance(layer_thickness, Parameters.Parameter):
            raise TypeError("\'layer_thickness\' has to be an instance of class \'Parameters.Parameter\'.")
        elif not callable(profile_function):
            raise TypeError("\'profile_function\' has to be a callable.")
        for par in params:
            if not isinstance(par, Parameters.Parameter):
                raise TypeError("Each entry of *params has to be an instance of class \'Parameters.Parameter\'.") 
        
        self._start_layer_idx=start_layer_idx
        self._end_layer_idx=end_layer_idx
        self._layer_thickness=layer_thickness
        self._profile_function=profile_function
        self._params=params
    
    def getDensityPar(self, layer_idx):
        """
        Return the density parameter as instance of :class:`Parameters.DerivedParameter` for the layer with index **idx** coresponding to the defined density profile.
        """
        
        #parameter checking
        if not isinstance(layer_idx,int):
            raise TypeError("\'start_layer_idx\' has to be an integer number.")
        elif layer_idx<self._start_layer_idx or layer_idx>self._end_layer_idx:
            raise ValueError("Density Profile is only defined from layer index "+str(self._start_layer_idx)+ " to " + str(self._end_layer_idx) +". \'layer_idx="+str(layer_idx)+"\' is out of range.")
        
        #create and return the corresponding derived parameter object
        return Parameters.DerivedParameter(self._profile_function,(layer_idx-self._start_layer_idx)*self._layer_thickness, *self._params)
    
    def getDensity(self, z, fitpararray):
        """
        Return the density at a certain distance **z** from the lower surface of start layer corresponding to the fit parameter values given by **fitpararray**.
        
        Might be used for plotting the resulting density profile etc.
        """
        
        #parameter checking
        if not isinstance(z, numbers.Real):
            raise TypeError("\'z\' has to be a real number.")
        elif not isinstance(fitpararray,list):
            raise TypeError("\fitparray\' has to be a list.")
                
        return self._profile_function(z,*[par.getValue(fitpararray) for par in self._params])
    

class DensityProfile_erf(DensityProfile):
    """
    Specialized :class:`DensityProfile` class.
    Realizes a density profile with the function 
    
    ``f(z) = 0.5*maximum*(1+erf( (z-position) / (sigma*sqrt(2)) ) )``.
    """
    
    def __init__(self, start_layer_idx, end_layer_idx, layer_thickness, position, sigma, maximum):
        """
        Parameters
        ----------
        start_layer_idx : int
            Index of the first layer of the density profile.
        end_layer_idx : int
            Index of the last layer of the density profile.
        layer_thickness : :class:`Parameters.Parameter`
            Thickness of the individual layers. The Unit is the same as for every other length used throughout the project and is not predefined. E.g. wavelength.
        position : :class:`Parameters.Parameter`
            Center position of the transition. Measured from the bottom of the start layer. The Unit is the same as for every other length used throughout the project and is not predefined. E.g. wavelength.
        sigma : :class:`Parameters.Parameter`
            Width of the transition. Unit: see above.
        maximum : :class:`Parameters.Parameter`
            Maximum value of the function. Should usually be measured in mol/cm^3. For other units you have to take care with the **densityunitfactor** at :class:`.AtomLayerObject`.
        """
        
        #type checking
        if not isinstance(start_layer_idx,int):
            raise TypeError("\'start_layer_idx\' has to be an integer number.")
        elif not isinstance(end_layer_idx,int):
            raise TypeError("\'end_layer_idx\' has to be an integer number.")
        elif not isinstance(layer_thickness, Parameters.Parameter):
            raise TypeError("\'layer_thickness\' has to be an instance of class \'Parameters.Parameter\'.")
        elif not isinstance(position, Parameters.Parameter):
            raise TypeError("\'position\' has to be an instance of class \'Parameters.Parameter\'.") 
        elif not isinstance(maximum, Parameters.Parameter):
            raise TypeError("\'maximum\' has to be an instance of class \'Parameters.Parameter\'.") 
        elif not isinstance(sigma, Parameters.Parameter):
            raise TypeError("\'sigma\' has to be an instance of class \'Parameters.Parameter\'.") 
        
        self._start_layer_idx=start_layer_idx
        self._end_layer_idx=end_layer_idx
        self._layer_thickness=layer_thickness
        self._params=(position,sigma ,maximum)
        
        #define profile function
        def erf_profile(z, pos, sig, m):
            return 0.5 * m * (1+scipy.special.erf((z-pos)/(numpy.sqrt(2)*sig)))
        self._profile_function=erf_profile   
    

    
#--------------------------------------------------------------------------------------------------------------------------
# convenience functions

def plotAtomDensity(hs,fitpararray,colormap=[],atomnames=None):
    """Convenience function. Create a bar plot of the atom densities of all instances of :class:`.AtomLayerObject` contained in the :class:`.Heterostructure` object **hs** corresdonding to the **fitpararray** (see :mod:`Parameters`) and return the plotted information as dictionary.
    
    This plot is only usefull for stacks of layers with equal widths as the widths are not taken into account for the plots
    
    You can  define the colors of the bars with **colormap**. Just give a list of matplotlib color names. They will be used in the given order.
    You can define which atoms you want to plot or in which order. Give **atomnames** as a list of strings. If **atomnames** is not given, the bars will have different width, such that overlapped bars can be seen.
    """
    if not isinstance(hs,Heterostructure):
        raise TypeError("\'hs\' has to be of type \'SampleRepresentation.Heterostructure\'.")
    elif not isinstance(fitpararray,list):
            raise TypeError("\fitparray\' has to be a list.")
    elif not isinstance(colormap,list):
            raise TypeError("\'colormap\' has to be a list.")
    elif not (isinstance(atomnames,list) or atomnames is None):
            raise TypeError("\'atomnames\' has to be a list.")
    if not atomnames is None:
        for item in atomnames:
            if not isinstance(item,str):
                raise TypeError("\'atomnames\' has to be a list of strings.")
    
    number_of_layers=hs.N_total
    if atomnames is None:
        atomnames=AtomLayerObject.getAtomNames()
        widthstep=1.0/len(atomnames)                #if no order is given plot each set of bar with smaller width to not cover the underlying bar
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
    for name in atomnames:
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


def KramersKronig(energy,absorption):
    """
    Convinience funtion. Performs the Kramers Kronig transformation. It is just a wrapper for :func:`Pythonreflectivity.KramersKroning` from Martins Zwieblers :mod:`Pythonreflectivity` package.
    
    Parameters
    ----------
    energy : 
        an ordered list/array of L energies (in eV). The energies do not have to be envenly spaced, but they should be ordered.
    absorption :
        a list/array of real numbers and length L with absorption data
    """
    return Pythonreflectivity.KramersKronig(numpy.array(energy),numpy.array(absorption))