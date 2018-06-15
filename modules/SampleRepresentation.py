#!/usr/bin/env python
"""Deals with the sample representation for simulation of the reflectivity."""


__author__ = "Yannic Utz"
__copyright__ = ""
__credits__ = ["Yannic Utz and Martin Zwiebler"]
__license__ = ""
__version__ = ""
__maintainer__ = "Yannic Utz"
__email__ = "yannic.utz@tu-dresden.de"
__status__ = "Prototype"


import os.path
import Pythonreflectivity


import Parameters




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
        return self._number_of_layers
    
    def _getMultilayerStructure_(self):
        return self._multilayer_structure
        
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
        Place \'layer\' (instance of LayerObject) at position \'index\' (counting from 0, starting from the bottom).
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
        Return the instance of LayerObject which is placed at position \'index\' (counting from 0, starting from the bottom).
        """
        if not isinstance(index,int):
            raise TypeError("\'index\' must be of type int.")
        if index<0:
            raise ValueError("\'index\' must be positive.")
        if index>=self._number_of_layers:
            raise ValueError("\'index\' exceeds defined number of layers.")
        return self._listoflayers[index]
    
         
    
    def removeLayer(self,index):
        """
        Remove the instance of LayerObject which is placed at position \'index\' (counting from 0, starting from the bottom from the  heterostructructure.
        
        
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
        print self._number_of_layers
        print self._PyReflMLstring
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
    N=property(_getNumberOfLayers_)
    MLstructure=property(_getMultilayerStructure_)
    

#-----------------------------------------------------------------------------------------------------------------------------


class LayerObject(object):
    """Base class for all layer objects as the common interface. Speciallized implementation should inherit from this class.
    
       It handels the basic properties of 
    """
    
    def __init__(self, chitensor=[], d=None,  sigma=None, magdir="0"):
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
    
    

        
        
        
        
        
        
class AtomLayerObject(object):
    """Speciallized Layer to deal with compositions of Atoms and their absorption spectra."""
    pass



#-----------------------------------------------------------------------------------------------------------------------------

class Formfactor(object):
    """
    Base class to deal with energy-dependent atomic form-factors.
    
    This base class is an abstract class an cannot be used directly.
    The user should derive from this class if he wants to build his own models.
    """
    
    def __init__(self):
        raise NotImplementedError
    
    def get(self,E,fitpararray=None):
        raise NotImplementedError


class FFfromFile(Formfactor):
    """
    Class to deal with energy-dependent atomic form-factors which are tabulated in files.
    """
    
    def __init__(self, filename, linereaderfctn):
        """Initializes the FFfromFile object with the data from filename.
        
           The \'linereaderfctn\' is used to convert one line from the text file to data.
           It should be a function which takes a string and returns a list of 10 values: [energy,f_xx,f_xy,f_xz,f_yx,f_yy,f_yz,f_zx,f_zy,f_zz]
           You can use FFfromFile.getLinereader to get a standard function, which just reads this array as whitespace seperated from the line.
        """
        if not isinstance(filename,str):
            raise TypeError("\'filename\' needs to be a string.")
        if not callable(linereaderfctn):
            raise TypeError("\'linereaderfctn\' needs to be a callable object.")
        if not os.path.isfile(filename):
            raise Exception("File \'"+filename+"\' does not exist.")
        self._energydepff=[]                                    #the energy-dependent formfactors should be stored in here like this: [[E_1,[f1_xx,f1_xy,...,f1_zy,f1_zz]], ...,[E_n,[fn_xx,fn_xy,...,fn_zy,fn_zz]] 
        with open(filename,'r') as f:
        
    