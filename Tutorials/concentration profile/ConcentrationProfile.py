# In this test implementation  the possibility to realize a concentration profile from a stack of atomic layers 
# is tested.
# 
# The (completly arbitrary) model is: a thin layer with a fixed thickness which is basically SrO. 
# Through some method the Sr at the top is replaced by Co leading to a very thin layer of CoO on top.
# For simplicity lets assume it has the same lattice parameters as SrO.
# Due to some reason there is a oxigen deficiency at at top, leading to a linear decay in oxigen density towards the upper surface
#
# Here I will model the situation as following from bottom to top:
# - a "thick" AtomicLayer of pure SrO (not affected by deficiencies and whatsoever) with variable thickness
# - a certain number of AtomicLayers containing O and Sr/Co with variable concentration;
#   the thickness of these layers is fixed with the lattice spacing, and the number of layers results from the predifined thickness of this "transition zone"



from time import time
from scipy import constants 
import scipy.special
import math
import numpy as np
from matplotlib import pyplot as plt

import Pythonreflectivity


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
decay_depth=pp.newParameter("decay_depth")              #position at which the O density starts to decay, measured in nm from top
decay_rel_gradient=pp.newParameter("decay_rel_gradient")        #gradient of the decay (defined as positiv number)


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

#create density profile for Sr and Co (as convinient error function profile)
density_profile_Sr=SampleRepresentation.DensityProfile_erf(1,number_of_trans_layers,Parameters.Parameter(lattice_const),position=number_of_trans_layers*lattice_const-transition_pos,sigma=-transition_width,maximum=density_Sr,)

#create density profile for O (with the more flexibel base class "DensityProfile")
def lin_decay_profile_fct(z,z_start,rel_gradient,maximum):
    if z<z_start:
        return maximum
    else:
        return maximum*(1-float(z-z_start)*rel_gradient)
density_profile_O=SampleRepresentation.DensityProfile(1,number_of_trans_layers,Parameters.Parameter(lattice_const), Parameters.ParametrizedFunction(lin_decay_profile_fct, number_of_trans_layers*lattice_const-decay_depth,decay_rel_gradient, density_O) )



#create transition layers and add them to the heterostructure
for i in range(1,number_of_trans_layers+1):
    l_i= SampleRepresentation.AtomLayerObject({"O":density_profile_O.getDensityPar(i),"Sr":density_profile_Sr.getDensityPar(i), "Co": density_Sr-density_profile_Sr.getDensityPar(i)},Parameters.Parameter(lattice_const))
    hs.setLayer(i, l_i)



#get a dummy fitpararray
fitpararray, lower, upper = pp.getStartLowerUpper()

cmap=['yellow','magenta','black','b','green','red','grey','magenta']
SampleRepresentation.plotAtomDensity(hs,fitpararray,cmap,["Co","Sr"])
SampleRepresentation.plotAtomDensity(hs,fitpararray,cmap,["O"])
SampleRepresentation.plotAtomDensity(hs,fitpararray,cmap)
#SampleRepresentation.plotAtomDensity(hs,fitpararray,cmap)

#refl=Pythonreflectivity.Reflectivity(hs.getSingleEnergyStructure(ar,850),[1+1.0*i for i in range(90)], 2*math.pi*constants.hbar/constants.e*constants.c*10**9/850)