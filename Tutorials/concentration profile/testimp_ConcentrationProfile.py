# In this test implementation  the possibility to realize a concentration profile from a stack of atomic layers 
# is tested.
# 
# The (completley arbitrary) model is: a thin layer with a fixed thickness which is basically SrO. 
# Through some method the Sr at the top is replaced by Co leading to a very thin layer of CoO on top.
# For simplicity lets assume it has the same lattice parameters as SrO.
#
# Here I will model the situation as following from bottom to top:
# - a "thick" AtomicLayer of pure SrO with variable thickness
# - a certain number of AtomicLayers containing O with fixed conentration and Sr/Co with variable concentration;
#   the thickness of the layers is fixed with the lattice spacing, and the number of layers results from the predifined thickness of the transition zone



from time import time
from scipy import constants 
import scipy.special
import math
import numpy as np
from matplotlib import pyplot as plt

import Pythonreflectivity

import sys
sys.path.append('../../')       #this statement makes it possible for the script to find the PyXMRTool package relative to the Tutorials folder. For your own projects rather copy the PyXMRTool-Folder which contains the modules to your project folder which contains the script
from PyXMRTool import SampleRepresentation
from PyXMRTool import Parameters




#############################
#Model Parameters (not fitted!!)
#############################
#all lengths in nm
transzone_thickness = 100     #thickness of the transition zone
lattice_const = 0.52            #assume cubic
#############################


pp=Parameters.ParameterPool("fitparameters.txt")
#############################
#create fitparameters
#############################
total_thickness=pp.newParameter("total_thickness")
transition_pos=pp.newParameter("transition_pos")     #position of transition from Sr to Co measured in nm from top
transition_width=pp.newParameter("transition_width")  #width of transition (sigma in error function erf(z/sqrt(2)/sigma)


#############################
pp.writeToFile("fitparameters.txt")

#some calculations
unit_cells_per_volume=1.0e-6/(0.52e-9)**3  #in mol/cm^3
density_O=Parameters.Parameter(unit_cells_per_volume/2.0)           #number density of oxigen within SrO (in mol/cm^3)
density_Sr=Parameters.Parameter(unit_cells_per_volume/2.0)         #number density of strontium within SrO (in mol/cm^3)


#calculate number of necessary transition zone layers and set up  heterostructure
number_of_trans_layers=int(math.ceil(float(transzone_thickness)/lattice_const))
hs=SampleRepresentation.Heterostructure(1+number_of_trans_layers)




#create some formfactor object and register atoms
FF_Co=SampleRepresentation.FFfromFile("Co.F",SampleRepresentation.FFfromFile.createLinereader(complex_numbers=False))
FF_Sr=SampleRepresentation.FFfromFile("Sr.F",SampleRepresentation.FFfromFile.createLinereader(complex_numbers=False))
FF_O=SampleRepresentation.FFfromFile("O.F",SampleRepresentation.FFfromFile.createLinereader(complex_numbers=False))
SampleRepresentation.AtomLayerObject.registerAtom("Co",FF_Co)
SampleRepresentation.AtomLayerObject.registerAtom("Sr",FF_Sr)
SampleRepresentation.AtomLayerObject.registerAtom("O",FF_O)

#create bottom layer and add it to heterostructure
bottom_layer=SampleRepresentation.AtomLayerObject({"O":density_O,"Sr":density_Sr},total_thickness-transzone_thickness)
hs.setLayer(0,bottom_layer)

#create transition layers and add them to the heterostructure
for i in range(number_of_trans_layers):
    Sr_concentration=0.5 * (1-scipy.special.erf((i*lattice_const-(transzone_thickness-transition_pos))/math.sqrt(2)/transition_width))  # average number of Sr atoms per Sr site (between 0 and 1)
    density_of_sr_in_layer= density_Sr * Sr_concentration
    density_of_co_in_layer= density_Sr * (1-Sr_concentration)
    l_i= SampleRepresentation.AtomLayerObject({"O":density_O,"Sr":density_of_sr_in_layer, "Co": density_of_co_in_layer},Parameters.Parameter(lattice_const))
    hs.setLayer(1+i, l_i)





cmap=['yellow','magenta','black','b','green','red','grey','magenta']
#SampleRepresentation.plotAtomDensity(hs,ar,cmap,["Al","Sr"])
SampleRepresentation.plotAtomDensity(hs,ar,cmap)

#refl=Pythonreflectivity.Reflectivity(hs.getSingleEnergyStructure(ar,850),[1+1.0*i for i in range(90)], 2*math.pi*constants.hbar/constants.e*constants.c*10**9/850)