# <PyXMRTool: A Python Package for the analysis of X-Ray Magnetic Reflectivity data measured on heterostructures>
#    Copyright (C) <2018>  <Yannic Utz>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


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

# Python Version 2.7


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
from scipy import interpolate
from scipy import optimize
import scipy
import matplotlib.pyplot
import inspect
import Pythonreflectivity

import Parameters

# -----------------------------------------------------------------------------------------------------------------------------
# global variables for setup

chantler_directory = "resources/ChantlerTables"  # directory which contains Chantler tables relative to this module's directory
chantler_suffix = ".cff"  # suffix of files containing a Chantler table
chantler_reader_file = "ChantlerReader.pyt"  # filename of the file containing the function chantler_linereader()

# -----------------------------------------------------------------------------------------------------------------------------
# some stuff happening at execution of the module

package_directory = os.path.dirname(os.path.abspath(__file__))  # store the absolute path of this  module
execfile(os.path.join(package_directory, chantler_directory, chantler_reader_file))


# -----------------------------------------------------------------------------------------------------------------------------
# Heterostructure

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
        if not isinstance(number_of_layers, int):
            raise TypeError("\'number_of_layers\' must be of type int.")
        if number_of_layers < 0:
            raise ValueError("\'number_of_layers\' must be positive.")
        if multilayer_structure is None:
            multilayer_structure = range(number_of_layers)
        elif not isinstance(multilayer_structure, list):
            raise TypeError("\'multilayer_structure\' has to be a list.")
        else:
            self._consistency_check(number_of_layers, multilayer_structure)
        self._number_of_layers = number_of_layers
        self._multilayer_structure = multilayer_structure
        self._listoflayers = [None for i in range(number_of_layers)]
        self._updatePyReflMLString()

    # private members

    def _consistency_check(self, number_of_layers, multilayer_structure):
        index_list = []
        for item in multilayer_structure:
            if isinstance(item, int):
                if item not in index_list:
                    index_list.append(item)
            elif isinstance(item, list):
                if not (len(item) == 2 and isinstance(item[0], int) and item[0] > 0 and isinstance(item[1], list)):
                    raise ValueError("\'multilayer_structure\' has wrong format.")
                for subitem in item[1]:
                    if not isinstance(subitem, int):
                        raise ValueError("\'multilayer_structure\' has wrong format.")
                    if subitem not in index_list:
                        index_list.append(subitem)
            else:
                raise ValueError("\'multilayer_structure\' has wrong format.")
        index_list.sort()
        if index_list[0] <> 0:
            raise ValueError("Indices in \'multilayer_structure\' have to start from 0.")
        if index_list[-1] <> len(index_list) - 1:
            raise ValueError(
                "Highest index in \'multilayer_structure\' does not agree with the number of different indices-1.")
        if len(index_list) <> number_of_layers:
            raise Exception(
                "Number of different indices in \'multilayer_structure\' does not agree with \'number_of_layers\'.")
        return len(index_list)

    def _updatePyReflMLString(self):
        self._PyReflMLstring = ""
        for item in self._multilayer_structure:
            if isinstance(item, int):
                self._PyReflMLstring += str(item) + ","
            elif isinstance(item, list):
                self._PyReflMLstring += str(item[0])
                self._PyReflMLstring += "*"
                self._PyReflMLstring += "("
                for subitem in item[1]:
                    self._PyReflMLstring += str(subitem) + ","
                self._PyReflMLstring = self._PyReflMLstring[:-1]  # remove last comma
                self._PyReflMLstring += "),"
        self._PyReflMLstring = self._PyReflMLstring[:-1]  # remove last comma

    def _getNumberOfLayers(self):
        """Return number of different layers (i.e. number of different indices)."""
        return self._number_of_layers

    def _getTotalNumberOfLayers(self):
        """Return total number of layers (counting also multiple use of the same layer according to \'multilayer_structure\')."""
        n = 0
        for item in self._multilayer_structure:
            if isinstance(item, int):
                n += 1
            elif isinstance(item, list):
                n += item[0] * len(item[1])
        return n

    def _mapTotalIndexToInternal(self, tot_ind):
        """Return the index used within \'multilayer_structure\' which corresponds to the total index of the layer counting from the bottom."""
        if tot_ind >= self.N_total:
            raise ValueError("Index out of range.")
        i = 0
        for item in self._multilayer_structure:
            if isinstance(item, int):
                if i == tot_ind:
                    return item
                i += 1
            elif isinstance(item, list):
                for loop in range(item[0]):
                    for subitem in item[1]:
                        if i == tot_ind:
                            return subitem
                        i += 1

    # public methods

    def setLayout(self, number_of_layers, multilayer_structure=None):
        """Change the layout of the heterostructure.
        
           See :meth:`.__init__` for details. Only difference is: you cannot make changes which would remove layers.
        """
        if not isinstance(number_of_layers, int):
            raise TypeError("\'number_of_layers\' must be of type int.")
        if number_of_layers < 0:
            raise ValueError("\'number_of_layers\' must be positive.")
        if multilayer_structure is None:
            multilayer_structure = range(number_of_layers)
            numberofindices = number_of_layers
        elif not isinstance(multilayer_structure, list):
            raise TypeError("\'multilayer_structure\' has to be a list.")
        else:
            numberofindices = self._consistency_check(number_of_layers, multilayer_structure)
        if numberofindices < len(self._listoflayers):
            for item in self._listoflayers[numberofindices:]:
                if item is not None:
                    raise Exception(
                        "Cannot change layout. Remove layers with index > " + str(numberofindices - 1) + " first.")
            self._listoflayers = self._listoflayers[:numberofindices]
        elif numberofindices > len(self._listoflayers):
            self._listoflayers += [None for i in range(numberofindices - len(self._listoflayers))]
        self._number_of_layers = number_of_layers
        self._multilayer_structure = multilayer_structure
        self._updatePyReflMLString()

    def setLayer(self, index, layer):
        """
        Place **layer** (instance of :class:`.LayerObject`) at position **index** (counting from 0, starting from the bottom or according to indices defined by **multilayer_structure**).
        """
        if not isinstance(index, int):
            raise TypeError("\'index\' must be of type int.")
        if index < 0:
            raise ValueError("\'index\' must be positive.")
        if index >= self._number_of_layers:
            raise ValueError("\'index\' exceeds defined number of layers.")
        if not isinstance(layer, LayerObject):
            raise TypeError("\'layer\' has to be an instance of \'LayerObject\'.")
        self._listoflayers[index] = layer

    def getLayer(self, index):
        """
        Return the instance of :class:`.LayerObject` which is placed at position **index** (counting from 0, starting from the bottom or according to indices defined by **multilayer_structure**).
        """
        if not isinstance(index, int):
            raise TypeError("\'index\' must be of type int.")
        if index < 0:
            raise ValueError("\'index\' must be positive.")
        if index >= self._number_of_layers:
            raise ValueError("\'index\' exceeds defined number of layers.")
        return self._listoflayers[index]

    def getTotalLayer(self, index):
        """
        Return the instance of :class:`.LayerObject` which is placed at position **index** (counting from 0, starting from the bottom, repeated layers are counted repeatedly).
        """
        return self.getLayer(self._mapTotalIndexToInternal(index))

    def removeLayer(self, index):
        """
        Remove the instance of :class:`.LayerObject` which is placed at position **index** (counting from 0, starting from the bottom or according to indices defined by **multilayer_structure**).
        
        
        **index** can also be a list of indices.
        BEWARE: The instance of :class:`.LayerObject` itself and the corresdonding instances of :class:`Parameters.Fitparameter` are not deleted! So in a following fitting procedure, these parameters might still be varied even though they don't have any effect on the result.
        """
        if isinstance(index, int):
            if index < 0:
                raise ValueError("\'index\' must be positive.")
            if index >= self._number_of_layers:
                raise ValueError("\'index\' exceeds defined number of layers.")
            self._listoflayers[index] = None
        elif isinstance(index, list):
            for item in index:
                if not isinstance(item, int):
                    raise "Entries of \'index\' list must be of type int."
                if item < 0:
                    raise ValueError("Entries of \'index\' list must be positive.")
                if item >= self._number_of_layers:
                    raise ValueError("Entry of \'index\' list exceeds defined number of layers.")
                self._listoflayers[item] = None
        else:
            raise TypeError("\'index\' must be of type int or list of int")

    def getSingleEnergyStructure(self, fitpararray, energy=None):
        """
        Return list of layers (Layer type of the :mod:`Pythonreflectivity` package from Martin Zwiebler) which can be directly used as input for :func:`Pythonreflectivity.Reflectivity`.
        
        **energy** in units of eV
        """
        PyReflStructure = Pythonreflectivity.Generate_structure(self._number_of_layers, self._PyReflMLstring)
        index = 0
        for layer in self._listoflayers:
            if layer is None:
                print "WARNING: Layer " + str(index) + " is undefined!"
            else:
                PyReflStructure[index].setd(layer.getD(fitpararray))
                PyReflStructure[index].setsigma(layer.getSigma(fitpararray))
                PyReflStructure[index].setmag(layer.getMagDir(fitpararray))
                PyReflStructure[index].setchi(layer.getChi(fitpararray, energy))
            index += 1
        return PyReflStructure

    # properties
    N = property(_getNumberOfLayers)
    """(*int*) Number of different layers. Read-only."""
    N_total = property(_getTotalNumberOfLayers)
    """(*int*) Total number of layers counting also multiple use of the same layer according to **multilayer_structure**. Read-only."""


# -----------------------------------------------------------------------------------------------------------------------------
# Layer Classes

class LayerObject(object):
    """Base class for all layer objects as the common interface. Speciallized implementation should inherit from this class.
    """

    def __init__(self, chitensor=None, d=None, sigma=None, magdir="0"):
        """        
        Parameters
        ----------
        d : :class:`Parameters.Parameter`
            Thickness. Unit is the same as for every other length used throughout the project and is not predefined. E.g. wavelength.
            *None* or *0* mean infinitively thick.
        sigma : :class:`Parameters.Parameter`
            The roughness of the upper surface of the layer. Has dimension of length. Unit: see **d**.
        chitensor : list of :class:`Parameters.Parameter`
            Electric susceptibility tensor of the layer.
            
            * *[chi]* sets *chi_xx = chi_yy = chi_zz = chi*
            * *[chi_xx,chi_yy,chi_z]* sets *chi_xx,chi_yy,chi_zz*, others are zero
            * *[chi_xx,chi_yy,chi_z,chi_g]* sets  *chi_xx,chi_yy,chi_zz* and depending on **magdir** *chi_yz=-chi_zy=chi_g* (if *x*), *chi_xz=-chi_zx=chi_g* (if *y*) or *chi_xz=-chi_zx=chi_g* (if *z*)
            * *[chi_xx,chi_xy,chi_xz,chi_yx,chi_yy,chi_yz,chi_zx,chi_zy,chi_zz]* sets all the corresdonding elements
         
        magdir : str
            Gives the magnetization direction for MOKE. Possible values are *\"x\"*, *\"y\"*, *\"z\"* and *\"0\"* (no magnetization).
        """

        # check parameters
        if not isinstance(d, Parameters.Parameter) and not d is None:
            raise TypeError(
                "\'d\' must be an instance of class \'Parameters.Parameter\' or of an derived class (e.g. \'Parameters.Fitparameter\'")
        if chitensor is not None:
            for item in chitensor:
                if not isinstance(item, Parameters.Parameter):
                    raise TypeError(
                        "Elements of \'d\' must be instances of class \'Parameters.Parameter\' or of an derived class (e.g. \'Parameters.Fitparameter\'")
        if not isinstance(sigma, Parameters.Parameter) and not sigma is None:
            raise TypeError(
                "\'sigma\' must be an instance of class \'Parameters.Parameter\' or of an derived class (e.g. \'Parameters.Fitparameter\'")
        if not isinstance(magdir, str) and not magdir is None:
            raise TypeError("\'magdir\' must be of type \'str\'.")
        elif not (magdir == 'x' or magdir == 'y' or magdir == 'z' or magdir == '0'):
            raise ValueError("Invalid input for \'magdir\'. Valid inputs are \'x\',\'y\',\'z\', and \'0\'")

        # asign members
        self._d = d
        self._chitensor = chitensor
        self._sigma = sigma
        self._magdir = magdir

    def _setd_(self, d):
        if not isinstance(d, Parameters.Parameter):
            raise TypeError(
                "\'d\' must be an instance of class \'Parameters.Parameter\' or of an derived class (e.g. \'Parameters.Fitparameter\'")
        self._d = d

    def _getd_(self):
        return self._d

    def _getChitensor_(self):
        return self._chitensor

    def _setChitensor_(self, chitensor):
        for item in chitensor:
            if not isinstance(item, Parameters.Parameter):
                raise TypeError(
                    "Elements of \'chitensor\' must be instances of class \'Parameters.Parameter\' or of an derived class (e.g. \'Parameters.Fitparameter\'")
        self._chitensor = chitensor

    def _getMagdir_(self):
        return self._magdir

    def _setMagdir_(self, magdir):
        if not isinstance(magdir, str):
            raise TypeError("\'magdir\' must be of type \'str\'.")
        elif not (magdir == 'x' or magdir == 'y' or magdir == 'z' or magdir == '0'):
            raise ValueError("Invalid input for \'magdir\'. Valid inputs are \'x\',\'y\',\'z\', and \'0\'")
        self._magdir = magdir

    def _getSigma_(self):
        return self._sigma

    def _setSigma_(self, sigma):
        if not isinstance(sigma, Parameters.Parameter):
            raise TypeError(
                "\'sigma\' must be an instance of class \'Parameters.Parameter\' or of an derived class (e.g. \'Parameters.Fitparameter\'")
        self._sigma = sigma

    # public methods
    def getChi(self, fitpararray, energy=None):
        """
        Return the electric susceptibility tensor as a list of numbers for a certain **energy** corresponding to the parameter values in **fitpararray** (see :mod:`Parameters`).
    
        The returned list can be of length 1,3,4 or 9 (see :meth:`.__init__`).
        For the base implementation of :class:`.LayerObject` the parameter **energy** is not used. But it may be used by derived classes like :class:`.AtomLayerObject` and therefore needed for compatibility.l
        **energy** is measured in units of eV.
        """
        # check tensor bevor giving it to the outside
        if not (len(self._chitensor) == 1 or len(self._chitensor) == 3 or len(self._chitensor) == 4 or len(
                self._chitensor) == 9):
            raise ValueError("Chitensor must be either of length 1, 3, 4, or 9.")
        if len(self._chitensor) == 4 and self._magdir == "0":
            raise ValueError("You have to define the direction of magnetization \'magdir\'.")
        # return the checked list filled with actual values
        return [item.getValue(fitpararray) for item in self._chitensor]

    def getD(self, fitpararray):
        """
        Return the thickness of the layer as a number corresponding to the parameter values in **fitpararray** (see :mod:`Parameters`).
        
        The thickness is given in the unit of length you chose. You are free to choose whatever unit you want, but use the same for every length troughout the project.
        """
        if self._d is None:
            return 0
        else:
            return self._d.getValue(fitpararray)

    def getSigma(self, fitpararray):
        """
        Return the roughness of the upper surface of the layer as a number corresponding to the parameter values in **fitpararray** (see :mod:`Parameters`).
        
        The thickness is given in the unit of length you chose. You are free to choose whatever unit you want, but use the same for every length troughout the project.
        """
        if self._sigma is None:
            return 0
        else:
            return self._sigma.getValue(fitpararray)

    def getMagDir(self, fitpararray=None):
        """
        Return magnetization direction
        
        **fitpararray** is not used and just there to give a common interface. Maybe a derived classes will have a benefit from it.
        """
        return self._magdir

    # exposed properties (feel like instance variables but are protected via getter and setter methods)
    # BEWARE: theese properties contain the "Parameter" objects. E.g. to get a certain value for the thicknes do this: "layer.d.getValue(parameterarray)"
    d = property(_getd_, _setd_)
    """Thickness of the layer. Unit is the same as for every other length used throughout the project and is not predefined. E.g. wavelength."""
    chitensor = property(_getChitensor_, _setChitensor_)
    """Electric susceptibility tensor of the layer. See :meth:`.__init__` for details."""
    sigma = property(_getSigma_, _setSigma_)
    """Roughness of the upper surface of the layer. Unit is the same as for every other length used throughout the project and is not predefined. E.g. wavelength."""
    magdir = property(_getMagdir_, _setMagdir_)
    """Magnetization direction for MOKE. Possible values are *\"x\"*, *\"y\"*, *\"z\"* and *\"0\"* (no magnetization)."""


class ModelChiLayerObject(LayerObject):
    """Speciallized layer to deal with an electrical suszeptibility tensor (Chi) which is modelled as function of energy.
    
    BEWARE: The inherited property :attr:`.chitensor` is now a function.
    """

    def __init__(self, chitensorfunction, d=None, sigma=None, magdir="0"):
        """
        Parameters
        ----------
        d : :class:`Parameters.Parameter`
            Thickness. Unit is the same as for every other length used throughout the project and is not predefined. E.g. wavelength.
            *None* or *0* mean infinitively thick.
        sigma : :class:`Parameters.Parameter`
            The roughness of the upper surface of the layer. Has dimension of length. Unit: see **d**.
        chitensorfunction : :class:`Parameters.ParametrizedFunction`
            Energy-dependent electric susceptibility tensor of the layer.
            A parametrized function of energy (see :class:`Parameters.ParametrizedFunction`) which reurns a list of either 1,3,4 or 9 real or complex numbers.
            See also documentation of :mod:`Pythonreflectivity`.
            
            * *[chi]* sets *chi_xx = chi_yy = chi_zz = chi*
            * *[chi_xx,chi_yy,chi_z]* sets *chi_xx,chi_yy,chi_zz*, others are zero
            * *[chi_xx,chi_yy,chi_z,chi_g]* sets  *chi_xx,chi_yy,chi_zz* and depending on **magdir**
              *chi_yz=-chi_zy=chi_g* (if *x*), *chi_xz=-chi_zx=chi_g* (if *y*) or *chi_xz=-chi_zx=chi_g* (if *z*)
            * *[chi_xx,chi_xy,chi_xz,chi_yx,chi_yy,chi_yz,chi_zx,chi_zy,chi_zz]* sets all the corresdonding elements
            
        magdir : str
            Gives the magnetization direction for MOKE. Possible values are *\"x\"*, *\"y\"*, *\"z\"* and *\"0\"* (no magnetization).
        """

        # check parameters
        if not isinstance(chitensorfunction, Parameters.ParametrizedFunction):
            raise TypeError("\'chitensorfunction\' has to be an instance of \`Parameters.ParametrizedFunction\`.")
        # asign members
        self._chitensorfunction = chitensorfunction
        # call constructor of the base class
        super(type(self), self).__init__(None, d, sigma, magdir)

    def _getChitensor_(self):
        return self._chitensorfunction

    def _setChitensor_(self, chitensorfunction):
        # check parameters
        if not isinstance(chitensorfunction, Parameters.ParametrizedFunction):
            raise TypeError("\'chitensorfunction\' has to be an instance of \`Parameters.ParametrizedFunction\`.")
        self._chitensorfunction = chitensorfunction

    # public methods
    def getChi(self, fitpararray, energy):
        """
        Return the electric susceptibility tensor as a list of numbers for a certain **energy** corresponding to the parameter values in **fitpararray** (see :mod:`Parameters`).
    
        **energy** is measured in units of eV.
        """
        # get chitensor from chitensorfunction
        chitensor = self._chitensorfunction.getValue(energy, fitpararray)
        # check tensor bevor giving it to the outside
        if not (len(chitensor) == 1 or len(chitensor) == 3 or len(chitensor) == 4 or len(chitensor) == 9):
            raise ValueError("\'chitensorfunction\' must return an array of length 1, 3, 4, or 9.")
        if len(chitensor) == 4 and self._magdir == "0":
            raise ValueError(
                "You have to define the direction of magnetization \'magdir\' if the \`chitensorfunction\` delivers a 4-element array.")
        for item in chitensor:
            if not isinstance(item, numbers.Number):
                raise TypeError("Elements of the array returnd by \'chitensorfunction\' have to be numbers.")
        # return the checked list
        return chitensor


class AtomLayerObject(LayerObject):
    """
    Speciallized layer class to deal with compositions of atoms and their energy dependent formfactors (which can be obtained from absorption spectra).
    
    Especially usefull to deal with atomic layers, but can also be used for bulk.
    The atoms and their formfactors have to be registered a the class (with registerAtom) before they can be used to instantiate a new AtomLayerObject.
    The atom density can be plotted with :func:`.plotAtomDensity`.
    Density is measured in mol/cm$^3$ (as long as no **densityunitfactor** is applied)
    """

    def __init__(self, densitydict={}, d=None, sigma=None, magdir="0", densityunitfactor=1.0):
        """
        Parameters
        ----------
        densitydict :
            a dictionary which contains atom names (strings, must agree with before registered atoms) and densities (must be instances of :class:`Parameters.Parameter` or of a derived class).
        d : :class:`Parameters.Parameter`
            Thickness. Unit is the same as for every other length used throughout the project and is not predefined. E.g. wavelength.
            *None* or *0* mean infinitively thick.
        sigma : :class:`Parameters.Parameter`
            The roughness of the upper surface of the layer. Has dimension of length. Unit: see **d**.
        densityunitfactor : float
            If the densities in densitydict are measured in another unit than mol/cm^3, state this value which translates your generic density to the one used internally.
            I.e.::
            
                rho_in_mol_per_cubiccm = densityunitfactor * rho_in_whateverunityouwant
        """
        if not isinstance(densitydict, dict):
            raise TypeError("\'densitydict\' has to be a dictionary.")
        for atomname in densitydict:
            if not isinstance(atomname, str):
                raise TypeError("The keys of the \'densitydict\' dictionary have to be stings.")
            if not isinstance(densitydict[atomname], Parameters.Parameter):
                raise TypeError(
                    "The values of the \'densitydict\' dictionary have to be instances of the \'Parameter\' class or of derived classes.")
            if atomname not in type(self)._atomdict:
                try:
                    type(self).registerAtom(atomname)
                except LookupError:
                    raise Exeption(
                        "Element \'" + atomname + "\' cannot be found in the Chantler tables and has not been registered yet.")
                except:
                    raise
        if not isinstance(densityunitfactor, numbers.Real):
            raise TypeError("\'densityunitfactor\' has to be a real number.")

        self._densityunitfactor = densityunitfactor

        self._densitydict = densitydict.copy()  # The usage of "copy" creates a copy of the dictionary. By this, we ensure, that changes of the original dictionary outside the object will not affect the AtomLayerObject

        # call constructor of the base class
        super(AtomLayerObject, self).__init__(None, d, sigma,
                                              magdir="0")  # magdir is not used; the magnetization and their direction should be handled within the corresponding Formfactor classes

    def getDensitydict(self, fitpararray=None):
        """Return the density dictionary either with evaluated paramters (needs **fitpararray**) or with the raw :class:`Parameters.Parameter` objects (use **fitparraray** = *None*)."""
        if fitpararray == None:
            return self._densitydict.copy()
        elif not isinstance(fitpararray, (list,tuple,numpy.ndarray)):
            raise TypeError("\'fitparray\' has to be a list, tuple or numpy array.")
        else:
            return dict(zip(self._densitydict.keys(), [item.getValue(fitpararray) for item in
                                                       self._densitydict.values()]))  # pack new dictionary from atomnames and "unpacked" parameters (actual values instead of abstract parameter)

    def getChi(self, fitpararray, energy):
        """
        Return the electric susceptibility tensor as a list of numbers for a certain **energy** corresponding to the parameter values in **fitpararray** (see :mod:`Parameters`).
    
        **energy** is measured in units of eV.
        """
        # gehe alle Items in self._densitydict durch, item[1].getValue(fitpararray) liefert Dichte der Atomsorte, (type(self)._atomdict[item[0]]).getFF(fitpararray) liefert Formfaktor der Atomsorte, beides wird multipliziert und alles zusammen aufsummiert
        # ffsum = sum_i( number_density_atom_i * ff_tensor_atom_i)
        ffsum = sum([item[1].getValue(fitpararray) * self._densityunitfactor * (type(self)._atomdict[item[0]]).getFF(energy, fitpararray) for item in self._densitydict.items()])
        

        # Return the susceptibility tensor chi
        # As chi is very small, the linear approximation can be used.
        # If there is no densityunitfactor defined, the densities are assumed to be in units of mol/cm^3.
        # Energy is assumed to be in units of eV.
        # The susceptibility is given  as chi = 4*pi/k_0^2*r_e*ffsum  (see Macke and Goering 2014, J.Pyhs.:Condens.Matter 26 363201, Eq.15)
        # With k_0=E/(h_bar*c), energy in eV, h_bar in eV*s, c in m r_e in m, densities in mol/cm^2 (and therefore also ffsum)
        # Th susceptibility can be calculated as: chi= 4* pi * h_bar^2 [eV*s]^2 * c^2 [m/s]^2 * r_e [m] * N_A [1/mol] * (100cm)^3/m^3 * ffsum (mol/cm^3) / E^2 (eV)^2
        # N_A: Arvogardros number, r_e: thomson scattering length/classical electron radius
        # With values:
        # h_bar = 6.582119514e-16
        # c = 299792458
        # r_e = 2.8179403227e-15
        # N_A= 6.02214086e+23
        return list(ffsum * 830.3582374351398 / energy ** 2)      


        # classvariables

    _atomdict = {}

    # classmethods                                               #are not related to an instance of a class and are here used to deal with the collection of all registered atoms
    @classmethod
    def registerAtom(cls, name, formfactor=None):
        """
        Register an atom called **name** for later use to instantiate an AtomLayerObject.
        
        **formfactor** as to be an instance of :class:`.Formfactor` or of a derived class.
        If no formfactor object is given, an instance of :class:`.FFfromChantler` will be created with **name** as element name. 
        This can be used for an easy registration of atoms with just their tabulated formfactors from the Chantler tables.
        
        """
        if not isinstance(name, str):
            raise TypeError("The atom \'name\' has to be a string.")
        if formfactor is not None and not isinstance(formfactor, Formfactor):
            raise TypeError("\'formfactor\' has to be an instance of \'Formfactor\' or of a derived class.")
        if name in cls._atomdict:
            print "WARNING: Atom \'" + str(name) + "\' is replaced."
        if formfactor is None:
            formfactor = FFfromChantler(name)
        cls._atomdict.update({name: formfactor})

    @classmethod
    def getAtom(cls, name):
        """Return the :class:`.Formfactor` object registered for atom **name**."""
        if not isinstance(name, str):
            raise TypeError("The atom \'name\' has to be a string.")
        if name not in cls._atomdict:
            raise ValueError("The atom \'name\' is not registered.")
        return cls._atomdict[name]

    @classmethod
    def getAtomNames(cls):
        """Return a list of names of registered atoms."""
        return cls._atomdict.keys()


# -----------------------------------------------------------------------------------------------------------------------------
# Formfactor classes

class Formfactor(object):
    """
    Base class to deal with energy-dependent atomic form-factors.
    
    This base class is an abstract class an cannot be used directly.
    The user should derive from this class if he wants to build his own models.
    
    See :doc:`/definitions/formfactors` for sign conventions.
    """

    def __init__(self):
        raise NotImplementedError

    def getFF(self, energy, fitpararray=None):
        """
        Return the formfactor for **energy** corresponding to **fitpararray** (if it depends on it) as 9-element Numpy array of complex numbers.
        
        **energy** is measured in units of eV.
        """
        raise NotImplementedError

    def _getMinE(self):
        raise NotImplementedError

    def _getMaxE(self):
        raise NotImplementedError

    def plotFF(self, fitpararray=None, energies=None):
        """
        Plot the energy-dependent formfactor with the energies (in eV) listed in the list/array **energies**. If this array is not given the plot covers the hole existing energy-range.
        The **fitpararray** has only to be given in cases where the formfactor depends on a fitparamter, e.g. for class:`FFfromScaledAbsorption`.
        """
        if energies is None:
            energies = numpy.linspace(self.minE, self.maxE, 10000)
        ff = []
        for e in energies:
            ff.append(self.getFF(e, fitpararray))
        ff = numpy.array(ff)
        ff = numpy.transpose(ff)
        fig = matplotlib.pyplot.figure(figsize=(10, 10))
        axes = []
        axes.append(fig.add_subplot(331))
        for i in range(1, 9):
            axes.append(fig.add_subplot(330 + i + 1, sharey=axes[0]))
        i = 0
        for ax in axes:
            ax.set_xlabel('energy (eV)')
            ax.locator_params(axis='x', nbins=4)
            l1 = ax.plot(energies, ff[i].real, label="real")
            l2 = ax.plot(energies, ff[i].imag, label="imaginary")
            i += 1
        axes[6].legend()  # only place a legend in the lower left element. Not nice, but works.
        matplotlib.pyplot.show()

    # properties
    maxE = property(_getMaxE)
    """Upper limit of stored energy range. Read-only."""
    minE = property(_getMinE)
    """Lower limit of stored energy range. Read-only."""


class FFfromFile(Formfactor):
    """
    Class to deal with energy-dependent atomic form-factors (entire tensor) which are tabulated in files. See :doc:`/definitions/formfactors` for sign conventions.
    """

    def __init__(self, filename, linereaderfunction=None, energyshift=Parameters.Parameter(0), minE=None, maxE=None):
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
            So the formfactor delivered from :meth:`FFfromFile.getFF` will not be `formfactor_from_file(E)` but `formfactor_from_file(E-energyshift)`.
        minE : float
        maxE : float
            State lower/upper limit energy-range which should be used. Not necessary to state, but can reduce the amount of stored data.
        """
        if not isinstance(filename, str):
            raise TypeError("\'filename\' needs to be a string.")
        if linereaderfunction is None:
            linereaderfunction = self.createLinereader()
        if not callable(linereaderfunction):
            raise TypeError("\'linereaderfunction\' needs to be a callable object.")
        if not os.path.isfile(filename):
            raise Exception("File \'" + filename + "\' does not exist.")
        if not isinstance(energyshift, Parameters.Parameter):
            raise TypeError("\'energyshift\' has to be of type Parameters.Parameter.")
        if minE is not None and (not isinstance(minE, numbers.Real) or minE < 0):
            raise TypeError("\'minE\' must be a positive real number.")
        if maxE is not None and (not isinstance(maxE, numbers.Real) or maxE < 0):
            raise TypeError("\'maxE\' must be a positive real number.")
        if maxE is not None and minE is not None and not minE < maxE:
            raise ValueError("\'minE\' must be smaller than \'maxE\'.")

        energies = []
        formfactors = []
        with open(filename, 'r') as f:
            for line in f:
                linereaderoutput = linereaderfunction(line)
                if linereaderoutput is None:
                    continue
                if not isinstance(linereaderoutput, (tuple, list)):
                    raise TypeError("Linereader function has to return a list/tuple.")
                if not len(linereaderoutput) == 10:
                    raise ValueError("Linereader function hast to return a list/tuple with 10 elements.")
                for item in linereaderoutput:
                    if not isinstance(item, numbers.Number):
                        raise ValueError("Linereader function hast to return a list/tuple of numbers.")
                if isinstance(linereaderoutput[0], complex):
                    raise ValueError("Linereader function hast to return a real value for the energy.")
                if (minE is None and maxE is None):
                    energies.append(linereaderoutput[0])  # store energies in one list
                    formfactors.append(linereaderoutput[1:])  # store corresponding formfactors in another list
                elif minE is None:
                    if linereaderoutput[0] <= maxE:
                        energies.append(linereaderoutput[0])  # store energies in one list
                        formfactors.append(linereaderoutput[1:])  # store corresponding formfactors in another list
                elif maxE is None:
                    if minE <= linereaderoutput[0]:
                        energies.append(linereaderoutput[0])  # store energies in one list
                        formfactors.append(linereaderoutput[1:])  # store corresponding formfactors in another list
                else:
                    if minE <= linereaderoutput[0] and linereaderoutput[0] <= maxE:
                        energies.append(linereaderoutput[0])  # store energies in one list
                        formfactors.append(linereaderoutput[1:])  # store corresponding formfactors in another list

        formfactors = numpy.array(formfactors)  # convert list formfactors to a numpy array for convinience
        self._minE = min(energies)
        self._maxE = max(energies)
        self._energyshift = energyshift  # Attention: this is supposed to be an instance of "Parameters.Parameter". So a value can be obtained with self._energyshift.getValue(fitparraray)
        # Create an interpolation function based on the given energie-formfactor-points. The formfactors are thererfore transformed to arrays of length 18 but with real values.
        # After that the array of N arrays of 18 element is transformed to an array of 18 arrays of N elements as needed by the interp1d function.
        # Therefore, this function will return an array of length 18 wich has to be transformed back to 9 complex valued elements.
        # Energies and formfactors don't have to be stored explicitly , because they are contained in the "self._interpolator" function.
        self._interpolator = interpolate.interp1d(energies, numpy.transpose(
            numpy.concatenate((formfactors.real, formfactors.imag), 1)))

    def _getMinE(self):
        return self._minE

    def _getMaxE(self):
        return self._maxE

    # public methods

    @staticmethod
    def createLinereader(complex_numbers=True):
        """
        Return the standard linereader function for usage with :meth:`FFfromFile.__init__`.
        
        This standard linereader function reads energy and complex elements of the formfactor tensor as a whitespace-seperated list (i.e. 10 numbers) and interpretes \"#\" as comment sign.
        If **complex_numbers** = *False* then the reader reads real and imaginary part of every element seperately, i.e. every line has to consist of 19 numbers seperated by whitespaces::
            
            energy f_xx_real ff_xx_im ... f_zy_real f_zy_im f_zz_real f_zz_im
        """
        commentsymbol = '#'
        if complex_numbers == True:
            def linereader(line):
                if not isinstance(line, str):
                    raise TypeError("\'line\' needs to be a string.")
                line = (line.split(commentsymbol))[0]  # ignore everything behind the commentsymbol  #
                if not line.isspace() and line:  # ignore empty lines
                    linearray = line.split()
                    if not len(linearray) == 10:
                        raise Exception("Formfactor file has wrong format.")
                    linearray = [ast.literal_eval(item) for item in linearray]
                    return linearray
                else:
                    return None
        elif complex_numbers == False:
            def linereader(line):
                if not isinstance(line, str):
                    raise TypeError("\'line\' needs to be a string.")
                line = (line.split(commentsymbol))[0]  # ignore everything behind the commentsymbol  #
                if not line.isspace() and line:  # ignore empty lines
                    linearray = line.split()
                    if not len(linearray) == 19:
                        raise Exception("Formfactor file has wrong format.")
                    linearray = [ast.literal_eval(item) for item in linearray]
                    return [linearray[0], linearray[1] + 1j * linearray[2], linearray[3] + 1j * linearray[4],
                            linearray[5] + 1j * linearray[6], linearray[7] + 1j * linearray[8],
                            linearray[9] + 1j * linearray[10], linearray[11] + 1j * linearray[12],
                            linearray[13] + 1j * linearray[14], linearray[15] + 1j * linearray[16],
                            linearray[17] + 1j * linearray[18]]
                else:
                    return None
        else:
            raise TypeError("\'complex_numbers\' has to be boolean.")
        return linereader  # here the FUNKTION linereader is returned

    def getFF(self, energy, fitpararray=None):
        """
        Return the (energy-shifted )formfactor for **energy** as an interpolation between the stored values from file as 9-element 1-D numpy array of complex numbers.
        
        Parameters
        ----------
        energy : float
            Measured in units of eV.
        fitpararray :
            Is actually only needed when an energyshift has been defined.
        """
        energyshift = self._energyshift.getValue(fitpararray)

        if energy + energyshift < self.minE or energy - energyshift > self.maxE:
            raise ValueError("\'energy - energyshift = " + str(energy) + " + " + str(energyshift) + " = " + str(
                energy - energyshift) + "\' is out of range (" + str(self.minE) + "," + str(self.maxE) + ").")
        FFallReal = self._interpolator(energy - energyshift)
        # return directly the numpy array, it is usefull further Calculation
        return FFallReal[:9] + FFallReal[9:] * 1j

    # properties
    maxE = property(_getMaxE)
    """Upper limit of stored energy range. Read-only."""
    minE = property(_getMinE)
    """Lower limit of stored energy range. Read-only."""


class FFfromChantler(FFfromFile):
    """
    Class to create an atomic formfactor for an element from  Database (Chantler Tables taken from https://dx.doi.org/10.18434/T4HS32).
    """

    def __init__(self, element_symbol, energyshift=Parameters.Parameter(0), minE=None, maxE=None):
        """Initializes the FFfromChantler object with an energy-dependent formfactor corresponding to the element given with **element_symbol**.
        
        Parameters
        ----------
        element_symbol : str
            Refers to the element of which the atomic formfactor should be looked up. (More specifically: name of the corresponding database file without suffix.)
        energyshift : :class:`Parameters.Parameter`
            Species a fittable energyshift between the energy-dependent formfactor from Chantler Tables and the `real` one in the reflectivity measurement.
            So the formfactor delivered from :meth:`FFfromChantler.getFF` will not be `formfactor_from_database(E)` but `formfactor_from_databayse(E-energyshift)`.
        minE : float
        maxE : float
            State lower/upper limit energy-range which should be used. Not necessary to state, but can reduce the amount of stored data.
        """
        if not isinstance(element_symbol, str):
            raise TypeError("\'element_symbol\' needs to be a string.")
        if not isinstance(energyshift, Parameters.Parameter):
            raise TypeError("\'energyshift\' has to be of type Parameters.Parameter.")

        filename = os.path.join(package_directory, chantler_directory, element_symbol + chantler_suffix)
        if not os.path.isfile(filename):
            raise LookupError("No database entry for element \'" + element_symbol + "\' existing.")
        
        f_rel= chantler_frel_reader(filename)
        f_NT= chantler_fNT_reader(filename)

        def wrapper(line):
            output = chantler_linereader(line)  # make use of the chantler_linereader deliverd with the Chantler tables
            if output is not None:
                #only diagonal elements with equal entries f
                #Re(f)= f1 + f_rel + f_NT ; Im(f)=f2
                #f1 and f2 are energy-dependent (read for each line), the corrections f_rel, f_NT are given in the header of the Chantler tables
                #See "Chantler, Journal fo Physical and Chemical Reference Data 24,71 (1995)" Eq.3 and following.
                energy,f1,f2 = output
                f=-(f1+f_rel+f_NT)+1j*f2           #convert to the PyXMRTool sign convention. See :doc:`/definitions/formfactors` for details
                return [energy, f, 0, 0, 0, f, 0, 0, 0, f]
            else:
                return None

        super(FFfromChantler, self).__init__(filename, wrapper, energyshift, minE,maxE)  # call constructor of parent class (FFfromFile)

class FFfromScaledAbsorption(Formfactor):
    """
    A formfactor class which uses the imaginary part of the formfactor (experimentally determined absorption signal which has been fitted to off-resonant values) given as a file, scales it with a fittable factor and calculates the real part by the Kramers-Kronig transformation. It realizes the procedure described in section 3.3 of Martin Zwiebler PhD-Thesis.
    It thereby deals only with the diagonal elements of the formfactor tensor. For the off-diagonal elements, the magnetic formfactors classes are used.
    
    See :doc:`/definitions/formfactors` for sign conventions.
    """

    def __init__(self, element_symbol, E1, E2, E3, scaling_factor, absorption_filename, absorption_linereaderfunction=None, energyshift=Parameters.Parameter(0), minE=None, maxE=None, autofitfunction=None, autofitrange=None, tabulated_filename=None, tabulated_linereaderfunction=None):
        """Initializes the FFfromScaledAbsorption object with an energy-dependent imaginary part of the formfactor given as file.
        
        To perform the Kramers-Kronig transformation without integrating to infinity, theoretical/tabulated formfactors (Chantler tabels from https://dx.doi.org/10.18434/T4HS32) are used. Their imaginary part differs only close to resonance from the measured absorption and should have been used before to perform the fit of the measured absorption to off-resonant values. 
                    
        
        The imaginary part of each element of the formfactor is:
        
        * the value given by **Im_f0_E1**, for energy < **E1**.
        * the value given by the file scaled by **scaling_factor** (roughly, see PhD Thesis of Martin Zwiebler,section 3.3, for details), for E1 <= energy <= E2
        * linear inperpolation between the scaled value at E2 and the value given for E3 by **Im_f0_E3**, for E2 < energy < E3
        * the value given by **Im_f0_E3**, for E3 < energy       
        
        The Kramers-Kronig transformation to obtain the real part is done only once during instantiation. Therefore, it does not have to be repeated with every value of the **scaling_factor** and fitting is fast.
        
        Parameters
        ----------
        element_symbol : string
            States the chemical element for which this formfactor is created as usual short version of its name. It is important to lookup the tabulated/theoretical reference formfactors from the Chantler tables. If you want to use your own formfactor as reference (see arguments **tabulated_filename** and **tabulated_linereaderfunction**) just enter an empty string here.
        E1 : float
            Energy in eV. From this energy on the energy-dependent imaginary part of the formfactor given as file is used and scaled.
        E2 : float
            Energy in eV. From this energy on the imaginary part of the formfactor is linearly interpolated.
        E3 : float
            Energy in eV. From this energy on the imaginary part of the formactor is constant **Im_f_E3**.
        scaling_factor : :class:`Parameter.Parameter`
            Specifies the fittable scaling factor (called *a* in Martin Zwiebler PhD Thesis).
        absorption_filename : str
            Path to the text file which contains the imaginary part of the formfactor which results from an apsorption measurement and a subsequent fit to off-resonant tabulated values.
        absorption_linereaderfunction : callable
            This function is used to convert one line from the *absorption* text file to data.
            It should be a function which takes a string and returns a tuple or list of 4 values: ``(energy, Im f_xx, Im f_yy, Im f_zz)``,
            where `energy` is measured in units of `eV` and imaginary parts of formfactors are real values in units of `e/atom` (dimensionless).
            It can also return `None` if it detects a comment line.
            You can use :meth:`FFfromScaledAbsorption.createAbsorptionLinereader` to get a standard function, which just reads this array as whitespace seperated from the line.
        energyshift : :class:`Parameters.Parameter`
            Species a fittable energyshift between the energy-dependent formfactor calculated by the whole above mentioned procedure and the `real` one in the reflectivity measurement.
            As a consequence the peak but also E1,E2 and E3 are shifted.
        minE : float
        maxE : float
            Specify minimum and maximum energy if you don't want to use the whole energy-range given in the file **tabulated_filename**. Reducing the energy-range speeds up the Kramers-Kronig transformations significantly.
        autofitfunction : callable
        autofitrange : float
            If given together with **autofitfunction**, the absorbtion from **absorbtion_filename** will be fitted to the imaginary part of the formfactors from Chantler tables or **tabulated_filename** just below/above **E1**/**E3** in a range given by **autofitrange** in eV.
            More specifically **autofifunction** will be fitted to the f2 of the Chantler tables.
            **autofitfunction** must work as following *f2=func(energy,absorbtion,a,b,c,...)*. *absorbtion* is the measured absorbtion/TEY/FY/... at a certain *energy* and *a*,*b*,*c*, ... are an arbitrary number of parameters.
            The parameters will be fitted such that the return values fit best to the f2 of the Chantler tables in the given energy-range.
            E.g. a standard fitfunction would be: *f2(E) = absorbtion*energy*a + b + c* energy. (see Martin Zwiebler PhD-Thesis, section 3.3)
        tabulated_filename : str
            Path to the text file which contains the tabulated/theoretical formfactor for the corresponding element.
            You can use this argument if you dont't want to use the standard Chantler tables. But therefore **element_symbol** has to be an empty string
        tabulated_linereaderfunction : callable
            This function is used to convert one line from the *tabulated* text file to data.
            It should be a function which takes a string and returns a tuple or list of 2 values: ``(energy,f)``,
            where `energy` is measured in units of `eV` and the formfactor `f` is a complex value in units of `e/atom` (dimensionless).
            It can also return `None` if it detects a comment line.
            You can use :meth:`FFfromScaledAbsorption.createTheoreticalLinereader` to get a standard function, which just reads this array as whitespace separated from the line.
            You can use this argument if you dont't want to use the standard Chantler tables and want to create your own linereader.
        """

        # check parameters
        if not isinstance(element_symbol, str):
            raise TypeError("\'element_symbol\' needs to be a string.")
        if not isinstance(E1, numbers.Real):
            raise TypeError("\'E1\' needs to be a real number.")
        if not isinstance(E2, numbers.Real):
            raise TypeError("\'E2\' needs to be a real number.")
        if not isinstance(E3, numbers.Real):
            raise TypeError("\'E3\' needs to be a real number.")
        if E1 < 0 or E2 < 0 or E3 < 0:
            raise ValueError("Energies \'E1\', \'E2\' and \'E3\' have to be greater than zero.")
        if not [E1, E2, E3] == sorted([E1, E2, E3]):
            raise ValueError("Energies \'E1\', \'E2\' and \'E3\' have to have ascending values.")
        if not isinstance(scaling_factor, Parameters.Parameter):
            raise TypeError("\'scaling_factor\' has to be of type Parameters.Parameter.")
        if not isinstance(absorption_filename, str):
            raise TypeError("\'absorption_filename\' needs to be a string.")
        if not os.path.isfile(absorption_filename):
            raise Exception("File \'" + absorption_filename + "\' does not exist.")
        if absorption_linereaderfunction is None:
            absorption_linereaderfunction = self.createAbsorptionLinereader()
        if not callable(absorption_linereaderfunction):
            raise TypeError("\'absorption_linereaderfunction\' needs to be a callable object.")
        if not isinstance(energyshift, Parameters.Parameter):
            raise TypeError("\'energyshift\' has to be of type Parameters.Parameter.")
        if minE is not None and (not isinstance(minE, numbers.Real) or minE < 0):
            raise TypeError("\'minE\' must be a positive real number.")
        if maxE is not None and (not isinstance(maxE, numbers.Real) or maxE < 0):
            raise TypeError("\'maxE\' must be a positive real number.")
        if maxE is not None and minE is not None and not minE < maxE:
            raise ValueError("\'minE\' must be smaller than \'maxE\'.")
        if (autofitrange is not None or autofitfunction is not None) and not (autofitrange is not None and autofitfunction is not None):
            raise ValueError("You have to give both, \'autofitfunction\' and \'autofitrange\' or none of them.")
        if autofitrange is not None and  (not isinstance(autofitrange, numbers.Real) or autofitrange < 0):
            raise TypeError("\'autofitrange\' must be a positive real number.")
        if autofitfunction is not None and not callable(autofitfunction):
            raise TypeError("\'absorption_linereaderfunction\' needs to be a callable object.")
        if not ((element_symbol <> '' and tabulated_filename is None) or (
                element_symbol == '' and tabulated_filename is not None)):
            raise ValueError(
                "Either \'tabulated_filename\' has to be None or \'element_symbol\' has to be an empty string.")
        if tabulated_filename is not None:
            if not isinstance(tabulated_filename, str):
                raise TypeError("\'tabulated_filename\' needs to be a string.")
            if not os.path.isfile(tabulated_filename):
                raise Exception("File \'" + tabulated_filename + "\' does not exist.")
            if tabulated_linereaderfunction is None:
                tabulated_linereaderfunction = self.createTheoreticalLinereader()
            if not callable(tabulated_linereaderfunction):
                raise TypeError("\'tabulated_linereaderfunction\' needs to be a callable object.")

        # store parameters
        self._E1 = float(E1)
        self._E2 = float(E2)
        self._E3 = float(E3)
        self._scaling_factor = scaling_factor  # Attention: this is supposed to be an instance of "Parameters.Parameter". So a value can be obtained with self._scaling_factor.getValue(fitparraray)
        self._energyshift = energyshift  # Attention: this is supposed to be an instance of "Parameters.Parameter". So a value can be obtained with self._energyshift.getValue(fitparraray)

        # setup acces to Chantler if neccessary
        if element_symbol <> '':
            #use Chantler tables from database and the functions delivered with it to read them
            tabulated_filename = os.path.join(package_directory, chantler_directory,element_symbol + chantler_suffix)  # filename of corresponding Chantler Table
            if not os.path.isfile(tabulated_filename):
                raise LookupError("No database entry for element \'" + element_symbol + "\' existing.")
            
            f_rel= chantler_frel_reader(tabulated_filename)
            f_NT= chantler_fNT_reader(tabulated_filename)

            def wrapper(line):
                output = chantler_linereader(line)  # make use of the chantler_linereader deliverd with the Chantler tables
                if output is not None:
                    #only diagonal elements with equal entries f
                    #Re(f)= f1 + f_rel + f_NT ; Im(f)=f2
                    #f1 and f2 are energy-dependent (read for each line), the corrections f_rel, f_NT are given in the header of the Chantler tables
                    #See "Chantler, Journal fo Physical and Chemical Reference Data 24,71 (1995)" Eq.3 and following.
                    energy,f1,f2 = output
                    f=-(f1+f_rel+f_NT)+1j*f2   #convert to the PyXMRTool sign convention. See :doc:`/definitions/formfactors`
                    return [energy, f]
                else:
                    return None            
            tabulated_linereaderfunction = wrapper


        # read theoretica/tabulated formfactors from file
        tab_energies = []
        tab_formfactors = []
        with open(tabulated_filename, 'r') as f:
            for line in f:
                linereaderoutput = tabulated_linereaderfunction(line)
                if linereaderoutput is None:
                    continue
                if not isinstance(linereaderoutput, (tuple, list)):
                    raise TypeError("Linereader function has to return a list/tuple.")
                if not len(linereaderoutput) == 2:
                    raise ValueError("Linereader function hast to return a list/tuple with 2 elements.")
                for item in linereaderoutput:
                    if not isinstance(item, numbers.Number):
                        raise ValueError("Linereader function hast to return a list/tuple of numbers.")
                if isinstance(linereaderoutput[0], complex):
                    raise ValueError("Linereader function hast to return a real value for the energy.")
                if (minE is None and maxE is None):
                    tab_energies.append(linereaderoutput[0])  # store energies in one list
                    tab_formfactors.append(linereaderoutput[1])  # store corresponding formfactors in another list
                elif minE is None:
                    if linereaderoutput[0] <= maxE:
                        tab_energies.append(linereaderoutput[0])  # store energies in one list
                        tab_formfactors.append(linereaderoutput[1])  # store corresponding formfactors in another list
                elif maxE is None:
                    if minE <= linereaderoutput[0]:
                        tab_energies.append(linereaderoutput[0])  # store energies in one list
                        tab_formfactors.append(linereaderoutput[1])  # store corresponding formfactors in another list
                else:
                    if minE <= linereaderoutput[0] and linereaderoutput[0] <= maxE:
                        tab_energies.append(linereaderoutput[0])  # store energies in one list
                        tab_formfactors.append(linereaderoutput[1])  # store corresponding formfactors in another list

        tab_formfactors = numpy.array(tab_formfactors)  # convert list formfactors to a numpy array for convinience
        self._minE = min(tab_energies)
        self._maxE = max(tab_energies)
        # Create an interpolation function based on the given energie-formfactor-points. The formfactors are thererfore transformed to arrays of length 2 but with real values.
        # After that the array of N arrays of 2 element is transformed to an array of 2 arrays of N elements as needed by the interp1d function.
        # Therefore, this function will return an array of length 2 wich has to be transformed back to 1 complex valued elements.
        self._tab_interpolator = interpolate.interp1d(tab_energies, numpy.transpose(numpy.concatenate((numpy.reshape(tab_formfactors.real, (-1, 1)), numpy.reshape(tab_formfactors.imag, (-1, 1))), 1)))

        # read imaginary part of formfactor from file
        abs_energies = []
        abs_im_formfactors = []
        with open(absorption_filename, 'r') as f:
            for line in f:
                linereaderoutput = absorption_linereaderfunction(line)
                if linereaderoutput is None:
                    continue
                if not isinstance(linereaderoutput, (tuple, list)):
                    raise TypeError("Linereader function has to return a list/tuple.")
                if not len(linereaderoutput) == 4:
                    raise ValueError("Linereader function hast to return a list/tuple with 4 elements.")
                for item in linereaderoutput:
                    if not isinstance(item, numbers.Real):
                        raise ValueError("Linereader function hast to return a list/tuple of real numbers.")
                abs_energies.append(linereaderoutput[0])  # store energies in one list
                abs_im_formfactors.append(linereaderoutput[1:])  # store corresponding formfactors in another list
        abs_im_formfactors = numpy.array(abs_im_formfactors)  # convert list formfactors to a numpy array for convinience
        self._abs_minE = min(abs_energies)
        self._abs_maxE = max(abs_energies)
        if self._E1 < self._abs_minE or self._E2 > self._abs_maxE:
            raise Exception("Given absorption data does not cover needed range between E1 and E2.")
        # Create an interpolation function based on the given energie-imag_formfactor-points.
        # Therefore, the array of N arrays of 3 element is transformed to an array of 3 arrays of N elements as needed by the interp1d function.
        # This function will return an array of length 3 which is the interpolated imaginary part of the formfactor at the requested energy.
        # Energies and formfactors don't have to be stored explicitly , because they are contained in the "self._interpolator" function.
        abs_interpolator = interpolate.interp1d(abs_energies, numpy.transpose(abs_im_formfactors))
        
        #perform autofit if autofitrange is given
        abs_energies=numpy.array(abs_energies)
        abs_im_formfactors_transposed=[]
        if autofitfunction is not None:
            autofitrange=float(autofitrange)
            N_fitfunction_arguments=len(inspect.getargspec(autofitfunction).args[2:])
            #select energy-range
            tab_energies=numpy.array(tab_energies)
            tab_formfactors=numpy.array(tab_formfactors)
            fit_indices=numpy.concatenate((numpy.nonzero( numpy.logical_and( tab_energies>=self._E1-autofitrange , tab_energies<=self._E1 ) )[0] ,numpy.nonzero( numpy.logical_and(tab_energies>=self._E3 , tab_energies<=self._E3+autofitrange))[0]))
            fit_tab_energies=tab_energies[fit_indices]
            fit_tab_formfactors_imag=tab_formfactors[fit_indices].imag
            for i in range(3):
                def f2_wrapper(energy, *args):
                    return autofitfunction(energy,abs_interpolator(energy)[i],*args) 
                popt, pcov = optimize.curve_fit(f2_wrapper,fit_tab_energies,fit_tab_formfactors_imag,p0=numpy.ones(N_fitfunction_arguments))
                a,b,c = popt
                #overwrite absorption and redo interpolation
                abs_im_formfactors_transposed.append(f2_wrapper(abs_energies, *popt))
            abs_im_formfactors_transposed = numpy.array(abs_im_formfactors_transposed)
            abs_im_formfactors=numpy.transpose(abs_im_formfactors_transposed)
            abs_interpolator = interpolate.interp1d(abs_energies, abs_im_formfactors_transposed)
            
            

        # create array of energies and different arrays of 3-element tensor to calculate the actual formfactors
        # "im_f1": imaginary part of formfactors for scaling_factor (a) = 1
        # "d_im_f1": derivative with respect to a of the imaginary part of formfactors
        # "diff_im_f1": difference between "im_f1" and tabulated formfactors
        # "kk_diff_im_f1": Kramers-Kronig transformed "diff_im_f1
        # "kk_d_im_f1": Kramers-Kronig transformed "d_im_f1"
        energies = []
        im_f1 = []
        d_im_f1 = []
        diff_im_f1 = []
        im_f1_E2 = abs_interpolator(self._E2)  # imaginary part of formfactors for scaling_factor (a) = 1 at E2
        re, im = self._tab_interpolator(self._E1)
        im_f0_E1 = numpy.array([im, im, im])  # imaginary part of formfactors  at E1  (does not depend on "a")
        re, im = self._tab_interpolator(self._E3)
        im_f0_E3 = numpy.array([im, im, im])  # imaginary part of formfactors  at E3  (does not depend on "a")
        tab_i = 0
        for energy in tab_energies:
            if energy < self._E1:
                energies.append(energy)
                im_f1.append(numpy.array([tab_formfactors[tab_i].imag, tab_formfactors[tab_i].imag, tab_formfactors[tab_i].imag]))  # below E1, formfactor tensor is a diagonal tensor, values are the given "tabulated" ones
                d_im_f1.append(numpy.array([0, 0, 0]))
                diff_im_f1.append(numpy.array([0, 0, 0]))
            tab_i += 1
        abs_i=0
        for energy in abs_energies:
            if energy >= self._E1 and energy <= self._E2:
                energies.append(energy)
                im_f1.append(abs_im_formfactors[abs_i])  # between E1 and E2, it is just the measured imaginary part of the formfactor
                d_im_f1.append(abs_im_formfactors[abs_i] - im_f0_E1)  # between E1 and E2, the derivative is the difference between "measured" imaginary part of the formfactor and the "tabulated" value at E1
                re, im = self._tab_interpolator(energy)
                diff_im_f1.append(abs_im_formfactors[abs_i] - numpy.array([im, im, im]))  # between E1 and E2, just the difference between "measured" imaginary part of the formfactor and the "tabulated" value
            abs_i += 1
        tab_i=0
        for energy in tab_energies:
            if energy > self._E2:
                energies.append(energy)
            if energy > self._E2 and energy < self._E3:
                im_f1.append(im_f1_E2 + (energy - self._E2) / float(self._E3 - self._E2) * (im_f0_E3 - im_f1_E2))  # between E2 and E3, it is a linear interpolation
                d_im_f1.append((self._E3 - energy) / float(self._E3 - self._E2) * (im_f1_E2 - im_f0_E1))  # between E2 and E3, the imaginary part is a linear interpolation, the derivative correspondingly
                diff_im_f1.append(im_f1[-1] - numpy.array([tab_formfactors[tab_i].imag, tab_formfactors[tab_i].imag,tab_formfactors[tab_i].imag]))  # between E1 and E2, just the difference between linearly interpolated imaginary part of the formfactor and the "tabulated" value
            elif energy >= self._E3:
                im_f1.append(numpy.array([tab_formfactors[tab_i].imag, tab_formfactors[tab_i].imag,tab_formfactors[tab_i].imag]))  # above E3, values are the given "tabulated" ones
                d_im_f1.append(numpy.array([0, 0, 0]))
                diff_im_f1.append(numpy.array([0, 0, 0]))
            tab_i += 1

        im_f1 = numpy.array(im_f1)
        d_im_f1 = numpy.array(d_im_f1)
        diff_im_f1 = numpy.array(diff_im_f1)

        # perform Kramers-Kronig transformations to obtain "kk_diff_im_f1" and "kk_d_im_f1"
        # therefore one array for every position of the tensors is needed, which can be obtained with "transpose"
        trans_diff_im_f1 = numpy.transpose(diff_im_f1)
        trans_d_im_f1 = numpy.transpose(d_im_f1)
        trans_kk_diff_im_f1 = []
        trans_kk_d_im_f1 = []
        for i in range(3):
            trans_kk_diff_im_f1.append(KramersKronig(energies, trans_diff_im_f1[i]))               #somehow it is necessary to use negative KramersKronig. Don't know why, but works nice.
            trans_kk_d_im_f1.append(KramersKronig(energies, trans_d_im_f1[i]))
        trans_kk_diff_im_f1 = numpy.array(trans_kk_diff_im_f1)
        trans_kk_d_im_f1 = numpy.array(trans_kk_d_im_f1)

        # inflate 3-element arrays (just diagonals of formfactor tensor) to 9-element arrays, by adding zeros as off-diogonals
        im_f1 = numpy.insert(im_f1, [1, 1, 1, 2, 2, 2], [0, 0, 0, 0, 0, 0], 1)
        trans_d_im_f1 = numpy.insert(trans_d_im_f1, [1, 1, 1, 2, 2, 2], 0, 0)
        trans_kk_diff_im_f1 = numpy.insert(trans_kk_diff_im_f1, [1, 1, 1, 2, 2, 2], 0, 0)
        trans_kk_d_im_f1 = numpy.insert(trans_kk_d_im_f1, [1, 1, 1, 2, 2, 2], 0, 0)

        # Create interpolation functions for the above created arrays for later use with "getFF"
        # For this the arrays of N arrays of 9 elements are transposed to arrays of 9 arras of N elements as needed by the interp1d function (trans_kk_diff_im_f1 and trans_kk_d_im_f1 are already transposed)
        # The functions will return arrays of length 9 for the corresponding energy as interpolation.
        self._im_f1_interpolator = interpolate.interp1d(energies, numpy.transpose(im_f1))
        self._d_im_f1_interpolator = interpolate.interp1d(energies, trans_d_im_f1)
        self._kk_diff_im_f1_interpolator = interpolate.interp1d(energies, trans_kk_diff_im_f1)
        self._kk_d_im_f1_interpolator = interpolate.interp1d(energies, trans_kk_d_im_f1)

    def _getMinE(self):
        return self._minE

    def _getMaxE(self):
        return self._maxE

    # public methods

    @staticmethod
    def createAbsorptionLinereader():
        """
        Return the standard linereader function for absorption files for usage with :meth:`FFfromScaledAbsorption.__init__`.
        
        This standard linereader function reads energy and elements of the imaginary part of the formfactor tensor as a whitespace-seperated list (i.e. 4 numbers) and interpretes \"#\" as comment sign.
        """
        commentsymbol = '#'

        def linereader(line):
            if not isinstance(line, str):
                raise TypeError("\'line\' needs to be a string.")
            line = (line.split(commentsymbol))[0]  # ignore everything behind the commentsymbol  #
            if not line.isspace() and line:  # ignore empty lines
                linearray = line.split()
                if not len(linearray) == 4:
                    raise Exception("Absorption file has wrong format.")
                linearray = [ast.literal_eval(item) for item in linearray]
                return linearray
            else:
                return None

        return linereader  # here the FUNKTION linereader is returned

    @staticmethod
    def createTabulatedLinereader(complex_numbers=True):
        """
        Return the standard linereader function for tabulated formfactor files for usage with :meth:`FFfromScaledAbsorption.__init__`.
        
        This standard linereader function reads energy and the complex formfactor as a whitespace-seperated list (i.e. 2 numbers) and interpretes \"#\" as comment sign.
        If **complex_numbers** = *False* then the reader reads real and imaginary part of the formfactor seperately, i.e. every line has to consist of 3 numbers seperated by whitespaces::
            
            energy ff_real ff_im 
        """
        commentsymbol = '#'
        if complex_numbers == True:
            def linereader(line):
                if not isinstance(line, str):
                    raise TypeError("\'line\' needs to be a string.")
                line = (line.split(commentsymbol))[0]  # ignore everything behind the commentsymbol  #
                if not line.isspace() and line:  # ignore empty lines
                    linearray = line.split()
                    if not len(linearray) == 2:
                        raise Exception("File for tabulated formfactor has wrong format.")
                    linearray = [ast.literal_eval(item) for item in linearray]
                    return linearray
                else:
                    return None
        elif complex_numbers == False:
            def linereader(line):
                if not isinstance(line, str):
                    raise TypeError("\'line\' needs to be a string.")
                line = (line.split(commentsymbol))[0]  # ignore everything behind the commentsymbol  #
                if not line.isspace() and line:  # ignore empty lines
                    linearray = line.split()
                    if not len(linearray) == 3:
                        raise Exception("File for tabulated formfactor file has wrong format.")
                    linearray = [ast.literal_eval(item) for item in linearray]
                    return [linearray[0], linearray[1] + 1j * linearray[2]]
                else:
                    return None
        else:
            raise TypeError("\'complex_numbers\' has to be boolean.")
        return linereader

    def getFF(self, energy, fitpararray):
        """
        Return the (energy-shifted )formfactor for **energy** as an interpolation between the stored values from file as 9-element 1-D numpy array of complex numbers.
        
        Parameters
        ----------
        energy : float
            Measured in units of eV.
        fitpararray :
            Is needed for the scaling factor \'a\' and an energy shift if defined.
        """
        scaling_factor = self._scaling_factor.getValue(fitpararray)
        energyshift = self._energyshift.getValue(fitpararray)

        if energy - energyshift < self.minE or energy - energyshift > self.maxE:  # use this strange construction with numpy arrays to allow "energy" to be a numpy array of energies
            raise ValueError("\'energy + energyshift = " + str(energy) + " - " + str(energyshift) + " = " + str(energy - energyshift) + "\' is out of range (" + str(self.minE) + "," + str(self.maxE) + ").")

        energy = energy - energyshift  # apply energyshift

        ImFF = (scaling_factor - 1.0) * self._d_im_f1_interpolator(energy) + self._im_f1_interpolator(energy)
        tab_real = self._tab_interpolator(energy)[0]
        #RealFF = numpy.array([tab_real, 0, 0, 0, tab_real, 0, 0, 0, tab_real]) + self._kk_diff_im_f1_interpolator(energy) + (scaling_factor - 1) * self._kk_d_im_f1_interpolator(energy)
        RealFF = numpy.array([tab_real, 0, 0, 0, tab_real, 0, 0, 0, tab_real]) + self._kk_diff_im_f1_interpolator(energy) + (scaling_factor - 1) * self._kk_d_im_f1_interpolator(energy)

        return RealFF + 1j * ImFF

    # properties
    maxE = property(_getMaxE)
    """Upper limit of stored energy range. Read-only."""
    minE = property(_getMinE)
    """Lower limit of stored energy range. Read-only."""


# -----------------------------------------------------------------------------------------------------------------------------
# Magnetic Formfactor classes

class MagneticFormfactor(Formfactor):
    """
    Base class to deal with energy-dependent magnetic form-factors, i.e. only off-diagonal elements of a formfactor tensor originating from the magnetization.
    """

    def __init__(self, m_prime, m_primeprime, theta_M, phi_M, minE, maxE, energyshift=Parameters.Parameter(0)):
        """Initializes the MagneticFormfactor with energy-dependent magnetic terms **m_prime** and **m_primeprime** and the angles **theta_M** and **phi_M** which describe the direction of the magnetization.
        
        See *Macke and Goering 2014, J.Phys.: Condens. Matter 26, 363201.* Eq. 11-14 for details.
        
        
        Parameters
        ----------
        m_prime : :class:Parameters.ParametrizedFunction
        m_primeprime : :class:Parameters.ParametrizedFunction
            Real and imaginary parts of the magnetic term. Given as parametrized functions of energy.
        theta_M : :class:Parameters.Parameter
        phi_M : :class:Parameters.Parameter
            Angles which describe the direction of the magnetization measured in degrees.
        minE : float
        maxE : float
            Lower and upper limits of the energy range for which the formfactor is defined.
        energyshift : :class:`Parameters.Parameter`
            Species a fittable energyshift between the energy-dependent formfactor created from the XMCD measurement and the `real` one in the reflectivity measurement.
        """

        # check parameters
        if not isinstance(m_prime, Parameters.ParametrizedFunction):
            raise TypeError("\'m_prime\' has to be an instance of \'Parameters.ParametrizedFunction\'.")
        if not isinstance(m_primeprime, Parameters.ParametrizedFunction):
            raise TypeError("\'m_primeprime\' has to be an instance of \'Parameters.ParametrizedFunction\'.")
        if not isinstance(theta_M, Parameters.Parameter):
            raise TypeError("\'theta_M\' has to be an instance of \'Parameters.Parameter\'.")
        if not isinstance(phi_M, Parameters.Parameter):
            raise TypeError("\'phi_M\' has to be an instance of \'Parameters.Parameter\'.")
        if not isinstance(minE, numbers.Real):
            raise TypeError("\'minE\' has to be a real number.")
        if not isinstance(maxE, numbers.Real):
            raise TypeError("\'maxE\' has to be a real number.")
        if minE < 0 or maxE < 0:
            raise ValueError("\'minE\' and \'maxE\' have to be greater than zero.")
        if minE >= maxE:
            raise ValueError("\'minE\' has to be smaller than \'maxE\'.")
        if not isinstance(energyshift, Parameters.Parameter):
            raise TypeError("\'energyshift\' has to be of type Parameters.Parameter.")

        # store parameters
        self._mp = m_prime
        self._mpp = m_primeprime
        self._theta_M = theta_M
        self._phi_M = phi_M
        self._minE = minE
        self._maxE = maxE
        self._energyshift = energyshift

    # private methods
    def _getMinE(self):
        return self._minE

    def _getMaxE(self):
        return self._maxE

    # public methods
    def getFF(self, energy, fitpararray=None):
        """
        Return the magnetic part of the formfactor for **energy** corresponding to **fitpararray** (if it depends on it) as 9-element list of complex numbers.
        
        The diagonal elements are all zero here.
        
        **energy** is measured in units of eV.
        """
        theta_M = numpy.pi / 180.0 * self._theta_M.getValue(fitpararray)
        phi_M = numpy.pi / 180.0 * self._phi_M.getValue(fitpararray)

        energyshift = self._energyshift.getValue(fitpararray)

        if energy - energyshift < self.minE or energy - energyshift > self.maxE:  # use this strange construction with numpy arrays to allow "energy" to be a numpy array of energies
            raise ValueError("\'energy + energyshift = " + str(energy) + " - " + str(energyshift) + " = " + str(energy - energyshift) + "\' is out of range (" + str(self.minE) + "," + str(self.maxE) + ").")

        energy = energy - energyshift  # apply energyshift

        return (1j * self._mp.getValue(energy, fitpararray) - self._mpp.getValue(energy, fitpararray)) * numpy.array([0, numpy.cos(theta_M), -numpy.sin(theta_M) * numpy.sin(phi_M), -numpy.cos(theta_M), 0,numpy.sin(theta_M) * numpy.cos(phi_M), numpy.sin(theta_M) * numpy.sin(phi_M), -numpy.sin(theta_M) * numpy.cos(phi_M), 0])

    # properties
    maxE = property(_getMaxE)
    """Upper limit of stored energy range. Read-only."""
    minE = property(_getMinE)
    """Lower limit of stored energy range. Read-only."""


class MFFfromXMCD(MagneticFormfactor):
    """
    Class to deal with a magnetic formfactor (MFF) derived from an XMCD measurement.
    
    BEWARE: The absolut values are only correct if you scaled the XMCD signal to tabulated absorbtion data. But usually it is enough to get relative values, which can give you magnetization profiles.
    """

    def __init__(self, theta_M, phi_M, filename, linereaderfunction=None, minE=None, maxE=None, energyshift=Parameters.Parameter(0)):
        """Initializes the MFF from an XMCD measurement given as textfile.
        
        The XMCD values are directly used to create the **m_primeprime** function.
        
        The **m_prime** function is found as Kramers-Kronig transformation of **m_primeprime**.
        

        
        
        Parameters
        ----------
        theta_M, phi_M : :class:Parameters.Parameter
            Angles which describe the direction of the magnetization measured in degrees.
        filename : str
            Path to the text file which contains the XMCD signal as function of energy.
        linereaderfunction : callable
            This function is used to convert one line from the *xmcd* text file to data.
            It should be a function which takes a string and returns a tuple or list of 2 values: ``(energy, xmcd)``,
            where `energy` is measured in units of `eV` and 'xmcd' is a real value in units of `e/atom` (dimensionless) (if it is scaled correctly).
            It can also return `None` if it detects a comment line.
            You can use :meth:`MFFfromXMCD.createLinereader` to get a standard function, which just reads this array as whitespace seperated from the line.
        minE : float
        maxE : float
            Specify minimum and maximum energy if you don't want to use the whole energy-range given in the file **xmcd_filename**. Reducing the energy-range speeds up the Kramers-Kronig transformations significantly.
        """
        # check parameters
        if not isinstance(filename, str):
            raise TypeError("\'filename\' needs to be a string.")
        if not os.path.isfile(filename):
            raise Exception("File \'" + filename + "\' does not exist.")
        if linereaderfunction is None:
            linereaderfunction = self.createLinereader()
        if not callable(linereaderfunction):
            raise TypeError("\'linereaderfunction\' needs to be a callable object.")
        if (minE is None and maxE is not None) or (minE is not None and maxE is None):
            raise Exception("You have to set both, \'minE\' and \'maxE\'.")
        if minE is not None and (not isinstance(minE, numbers.Real) or minE < 0):
            raise TypeError("\'minE\' must be a positive real number.")
        if maxE is not None and (not isinstance(maxE, numbers.Real) or maxE < 0):
            raise TypeError("\'maxE\' must be a positive real number.")
        if maxE is not None and minE is not None and not minE < maxE:
            raise ValueError("\'minE\' must be smaller than \'maxE\'.")

        # read theoretica/tabulated formfactors from file
        energies = []
        xmcd = []
        with open(filename, 'r') as f:
            for line in f:
                linereaderoutput = linereaderfunction(line)
                if linereaderoutput is None:
                    continue
                if not isinstance(linereaderoutput, (tuple, list)):
                    raise TypeError("Linereader function has to return a list/tuple.")
                if not len(linereaderoutput) == 2:
                    raise ValueError("Linereader function hast to return a list/tuple with 2 elements.")
                for item in linereaderoutput:
                    if not isinstance(item, numbers.Real):
                        raise ValueError("Linereader function hast to return a list/tuple of real numbers.")
                if (minE is None and maxE is None):
                    energies.append(linereaderoutput[0])  # store energies in one list
                    xmcd.append(linereaderoutput[1])  # store corresponding formfactors in another list
                elif minE is None:
                    if linereaderoutput[0] <= maxE:
                        energies.append(linereaderoutput[0])  # store energies in one list
                        xmcd.append(linereaderoutput[1])  # store corresponding formfactors in another list
                elif maxE is None:
                    if minE <= linereaderoutput[0]:
                        energies.append(linereaderoutput[0])  # store energies in one list
                        xmcd.append(linereaderoutput[1])  # store corresponding formfactors in another list
                else:
                    if minE <= linereaderoutput[0] and linereaderoutput[0] <= maxE:
                        energies.append(linereaderoutput[0])  # store energies in one list
                        xmcd.append(linereaderoutput[1])  # store corresponding formfactors in another list

        xmcd = numpy.array(xmcd)  # convert list xmcd to a numpy array for convinience
        minE = min(energies)
        maxE = max(energies)

        # Create an interpolation function based on the given energie-xmcd-points and plug it into a parametrized Function (even thogh theire are no parameters; but it is easy to use the base class) (buying simplicity with speed here).
        mpp = Parameters.ParametrizedFunction(interpolate.interp1d(energies, xmcd))

        # perform Kramers-Kronig transformation
        m_prime = KramersKronig(energies, xmcd)
        # Create an interpolation function based on m_prime
        mp = Parameters.ParametrizedFunction(interpolate.interp1d(energies, m_prime))

        # call constructor of base class
        super(MFFfromXMCD, self).__init__(mp, mpp, theta_M, phi_M, minE, maxE, energyshift)

    # static methods
    @staticmethod
    def createLinereader():
        """
        Return the standard linereader function for xmcd files for usage with :meth:`MFFfromXMCD.__init__`.
        
        This standard linereader function reads energy and xmcd value as a whitespace-seperated list (i.e. 2 numbers) and interpretes \"#\" as comment sign.
        """
        commentsymbol = '#'

        def linereader(line):
            if not isinstance(line, str):
                raise TypeError("\'line\' needs to be a string.")
            line = (line.split(commentsymbol))[0]  # ignore everything behind the commentsymbol  #
            if not line.isspace() and line:  # ignore empty lines
                linearray = line.split()
                if not len(linearray) == 2:
                    raise Exception("XMCD file has wrong format.")
                linearray = [ast.literal_eval(item) for item in linearray]
                return linearray
            else:
                return None

        return linereader

        # private methods

    def _getMinE(self):
        return self._minE

    def _getMaxE(self):
        return self._maxE

    # properties
    maxE = property(_getMaxE)
    """Upper limit of stored energy range. Read-only."""
    minE = property(_getMinE)
    """Lower limit of stored energy range. Read-only."""


# -----------------------------------------------------------------------------------------------------------------------------
# Density Profil classes

class DensityProfile(object):
    """
    This class can be used to generate arbitrary density profiles within a stack of several :class:`.AtomLayerObject` of equal thicknesses.
    
    The idea is to collect all information regarding the density profile in an object of this class and to generate entries for the *densitydict* of the single :class:`.AtomLayerObject` instances from it.
    This means that the class :class:`.DensityProfile` does not really talk to the layers, but is only a higher level convinience class to set up the interconnected densities of the atoms within the layers as instances of :class:`Parameters.DerivedParameter`.
    """

    def __init__(self, start_layer_idx, end_layer_idx, layer_thickness, profile_function):
        """
        Parameters
        ----------
        start_layer_idx : int
            Index of the first layer in the scope of the density profile.
        end_layer_idx : int
            Index of the last layer in the scope of the density profile.
        layer_thickness : :class:`Parameters.Parameter`
            Thickness of the individual layers. The Unit is the same as for every other length used throughout the project and is not predefined. E.g. wavelength.
        profile_function  : :class:`Parameters.ParametrizedFunction`
            The density of the corresponding atom as function of the distance **z** from the lower surface of start layer parametrized by arbitrary parameters (see :class:`Parameters.ParametrizedFunction` for details). 
        """

        # type checking
        if not isinstance(start_layer_idx, int):
            raise TypeError("\'start_layer_idx\' has to be an integer number.")
        elif not isinstance(end_layer_idx, int):
            raise TypeError("\'end_layer_idx\' has to be an integer number.")
        elif not isinstance(layer_thickness, Parameters.Parameter):
            raise TypeError("\'layer_thickness\' has to be an instance of class \'Parameters.Parameter\'.")
        elif not isinstance(profile_function, Parameters.ParametrizedFunction):
            raise TypeError("\'profile_function\' has to be an instance of \`Parameters.ParametrizedFunction\`.")

        self._start_layer_idx = start_layer_idx
        self._end_layer_idx = end_layer_idx
        self._layer_thickness = layer_thickness
        self._profile_function = profile_function

    def getDensityPar(self, layer_idx):
        """
        Return the density parameter as instance of :class:`Parameters.DerivedParameter` for the layer with index **idx** coresponding to the defined density profile.
        """

        # parameter checking
        if not isinstance(layer_idx, int):
            raise TypeError("\'start_layer_idx\' has to be an integer number.")
        elif layer_idx < self._start_layer_idx or layer_idx > self._end_layer_idx:
            raise ValueError(
                "Density Profile is only defined from layer index " + str(self._start_layer_idx) + " to " + str(
                    self._end_layer_idx) + ". \'layer_idx=" + str(layer_idx) + "\' is out of range.")

        # create and return the corresponding derived parameter object

        return self._profile_function.getParameter((layer_idx - self._start_layer_idx) * self._layer_thickness)

    def getDensity(self, z, fitpararray):
        """
        Return the density at a certain distance **z** from the lower surface of start layer corresponding to the fit parameter values given by **fitpararray**.
        
        Might be used for plotting the resulting density profile etc.
        """

        # parameter checking
        if not isinstance(z, numbers.Real):
            raise TypeError("\'z\' has to be a real number.")
        elif not isinstance(fitpararray, (list,tuple,numpy.ndarray)):
            raise TypeError("\'fitparray\' has to be a list, tuple or numpy array.")

        return self._profile_function.getValue(z, fitpararray)


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

        # type checking
        if not isinstance(start_layer_idx, int):
            raise TypeError("\'start_layer_idx\' has to be an integer number.")
        elif not isinstance(end_layer_idx, int):
            raise TypeError("\'end_layer_idx\' has to be an integer number.")
        elif not isinstance(layer_thickness, Parameters.Parameter):
            raise TypeError("\'layer_thickness\' has to be an instance of class \'Parameters.Parameter\'.")
        elif not isinstance(position, Parameters.Parameter):
            raise TypeError("\'position\' has to be an instance of class \'Parameters.Parameter\'.")
        elif not isinstance(maximum, Parameters.Parameter):
            raise TypeError("\'maximum\' has to be an instance of class \'Parameters.Parameter\'.")
        elif not isinstance(sigma, Parameters.Parameter):
            raise TypeError("\'sigma\' has to be an instance of class \'Parameters.Parameter\'.")

        self._start_layer_idx = start_layer_idx
        self._end_layer_idx = end_layer_idx
        self._layer_thickness = layer_thickness

        # define profile function
        def erf_profile(z, pos, sig, m):
            return 0.5 * m * (1 + scipy.special.erf((z - pos) / (numpy.sqrt(2) * sig)))

        self._profile_function = Parameters.ParametrizedFunction(erf_profile, position, sigma, maximum)


# --------------------------------------------------------------------------------------------------------------------------
# convenience functions

def plotAtomDensity(hs, fitpararray, colormap=[], atomnames=None):
    """Convenience function. Create a bar plot of the atom densities of all instances of :class:`.AtomLayerObject` contained in the :class:`.Heterostructure` object **hs** corresdonding to the **fitpararray** (see :mod:`Parameters`) and return the plotted information as dictionary.
    
    This plot is only usefull for stacks of layers with equal widths as the widths are not taken into account for the plots
    
    You can  define the colors of the bars with **colormap**. Just give a list of matplotlib color names. They will be used in the given order.
    You can define which atoms you want to plot or in which order. Give **atomnames** as a list of strings. If **atomnames** is not given, the bars will have different width, such that overlapped bars can be seen.
    """
    if not isinstance(hs, Heterostructure):
        raise TypeError("\'hs\' has to be of type \'SampleRepresentation.Heterostructure\'.")
    elif not isinstance(fitpararray, (list,tuple,numpy.ndarray)):
        raise TypeError("\'fitparray\' has to be a list, tuple or numpy array.")
    elif not isinstance(colormap, list):
        raise TypeError("\'colormap\' has to be a list.")
    elif not (isinstance(atomnames, list) or atomnames is None):
        raise TypeError("\'atomnames\' has to be a list.")
    if not atomnames is None:
        for item in atomnames:
            if not isinstance(item, str):
                raise TypeError("\'atomnames\' has to be a list of strings.")

    number_of_layers = hs.N_total
    if atomnames is None:
        atomnames = AtomLayerObject.getAtomNames()
        widthstep = 1.0 / len(
            atomnames)  # if no order is given plot each set of bar with smaller width to not cover the underlying bar
    else:
        widthstep = 0.0
    if atomnames == []:
        print "No atoms registered."
        return

    densitylistdict = {}
    for name in atomnames:
        densitylistdict[name] = numpy.zeros(
            number_of_layers)  # create dictionary, which has an entry for every atom, with its name as key and as value a list. These lists are as long as there are numbers of layers and filled with zeros.

    for i in range(hs.N):
        layer = hs.getLayer(i)  # go through all layers in the heterostructure
        if isinstance(layer,
                      AtomLayerObject):  # if it is an instance of AtomLayerObject (i.e. contains information about atom densities)
            densitydict = layer.getDensitydict(
                fitpararray)  # get the density dictionary from this layer with the parameters evaluated (i.e. take the value contained in fitpararray)
            for name in atomnames:  # if atom with a certain name is contained within this layer, ad its density to the corresponding list in densitylistdict
                if name in densitydict:
                    densitylistdict[name][i] = densitydict[name]

    colorindex = 0
    w = 1
    for name in atomnames:
        if colorindex < len(colormap):
            matplotlib.pyplot.bar(range(number_of_layers), densitylistdict[name], align='center', width=w, label=name,
                                  color=colormap[colorindex], alpha=0.9)
            colorindex += 1
        else:
            matplotlib.pyplot.bar(range(number_of_layers), densitylistdict[name], align='center', width=w, label=name,
                                  alpha=0.9)
        w -= widthstep

    matplotlib.pyplot.xlabel("Layer number")
    matplotlib.pyplot.ylabel(r'Density in mol/cm$^3$')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlim(0, number_of_layers)
    matplotlib.pyplot.show()

    return densitylistdict  # contains a dictionary, which has an entry for every atom, with its name as key and as value a list. These lists are as long as there are numbers of layers and filled with


def KramersKronig(energy, f_imag):
    """
    Convinience funtion. Performs the Kramers Kronig transformation
    
    .. math::
        f^\\prime(E)= - \\frac{2}{\\pi}\\mathrm{CH}\\int_0^\\infty \\frac{\\eta \\cdot f^{\\prime\\prime}(\\eta)}{\\eta^2-E^2} \\, d\\eta
        
    It is just a wrapper for :func:`Pythonreflectivity.KramersKroning` from Martins Zwieblers :mod:`Pythonreflectivity` package.
    
    
    
    Parameters
    ----------
    energy : 
        an ordered list/array of L energies (in eV). The energies do not have to be envenly spaced, but they should be ordered.
    f_imag :
        a list/array of real numbers and length L with absorption data
    """
    return Pythonreflectivity.KramersKronig(numpy.array(energy), numpy.array(absorption))
