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

#create density profile for Sr
#def erf_profile(z, maximum, pos, sigma):
#    return maximum*0.5 * (1+scipy.special.erf((z-pos)/(np.sqrt(2)*sigma)))
#density_profile_Sr=SampleRepresentation.DensityProfile(1,number_of_trans_layers,Parameters.Parameter(lattice_const),erf_profile,density_Sr,number_of_trans_layers*lattice_const-transition_pos,-transition_width)
density_profile_Sr=SampleRepresentation.DensityProfile_erf(1,number_of_trans_layers,Parameters.Parameter(lattice_const),position=number_of_trans_layers*lattice_const-transition_pos,sigma=-transition_width,maximum=density_Sr,)

#create transition layers and add them to the heterostructure
for i in range(1,number_of_trans_layers+1):
    l_i= SampleRepresentation.AtomLayerObject({"O":density_O,"Sr":density_profile_Sr.getDensityPar(i), "Co": density_Sr-density_profile_Sr.getDensityPar(i)},Parameters.Parameter(lattice_const))
    hs.setLayer(i, l_i)



#get a dummy fitpararray
fitpararray, lower, upper = pp.getStartLowerUpper()

cmap=['yellow','magenta','black','b','green','red','grey','magenta']
SampleRepresentation.plotAtomDensity(hs,fitpararray,cmap,["Co","Sr"])
#SampleRepresentation.plotAtomDensity(hs,fitpararray,cmap)

#refl=Pythonreflectivity.Reflectivity(hs.getSingleEnergyStructure(ar,850),[1+1.0*i for i in range(90)], 2*math.pi*constants.hbar/constants.e*constants.c*10**9/850)