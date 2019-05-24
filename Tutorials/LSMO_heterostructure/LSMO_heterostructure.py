# This is a test implementation for a fit of reflectivity data of a LSMO heterostructure
# measured by Florian Rasch.

# The fitted model is the following (from bottom/substrate to top layer):
# - SrTiO3 layer with variable thickness, roughness and density
# - SrRuO3 layer with variable thickness, roughness and density
# - LSMO layer  with variable thickness, roughness and density
# - C layer with variable thickness, roughness and density
#
# Energy-dependent formfactors are taken from the Chantler Tables except for the Mn atom.
# For MN a measured XAS spectrum is used to create a scalelable formfactor


import math
import numpy
import timeit
import time
from matplotlib import pyplot as plt
import scipy.optimize
import pickle


from PyXMRTool import SampleRepresentation
from PyXMRTool import Parameters
from PyXMRTool import Experiment
from PyXMRTool import Fitters

### ALL LENGTHES ARE MEASURED IN Angstroem!!!! 
    
#### create ParameterPool #####################################################################
pp=Parameters.ParameterPool("parameters.txt")
#pp=Parameters.ParameterPool()


#some Parameters which are used later on
mdensity_SrTiO3=pp.newParameter("mdensity_SrTiO3")    #mass density of the material in g/cm^3
mdensity_SrRuO3=pp.newParameter("mdensity_SrRuO3")
mdensity_LSMO=pp.newParameter("mdensity_LSMO")
mdensity_C=pp.newParameter("mdensity_C")      #number density of C atoms in mol/cm^3

#some numbers
mol_mass_Sr=87.62               #molar mass of elements  g/mol
mol_mass_Ti=47.867
mol_mass_Ru=101.07
mol_mass_O=15.999
mol_mass_Mn=54.938
mol_mass_C=12.011
mol_mass_La=138.91

#derived_parameters
density_SrTiO3=mdensity_SrTiO3/(mol_mass_Sr+ mol_mass_Ti+3*mol_mass_O)   #number density of formula unit in mol/cm^3
density_SrRuO3=mdensity_SrRuO3/(mol_mass_Sr+mol_mass_Ru+3*mol_mass_O)
density_LSMO=mdensity_LSMO/(0.66667*mol_mass_La+0.33333*mol_mass_Sr+mol_mass_Mn+3*mol_mass_O)
density_C=mdensity_C/mol_mass_C
    
    
start,l,u=pp.getStartLowerUpper()           #read start values etc. to check for errors at beginnning of script
pp.writeToFile("parameters.txt")            #write parameter file to get nice layout after changing values

#####################################################################################################################################
#Creation of Model

    
    
# set up heterostructure (with 5 layers)
print("... set up heterostructure")
hs = SampleRepresentation.Heterostructure(5)


###set up formfactors
print("... set up formfactors")
   
#at first create linereader functions to read the files
commentsymbol='#'
    
def absorption_linereader(line):
    line=(line.split(commentsymbol))[0]                            #ignore everything behind the commentsymbol  #
    if not line.isspace() and line:                               #ignore empty lines        
        linearray=line.split()
        linearray=[float(item) for item in linearray]
        return (linearray[0], linearray[2], linearray[4], linearray[6])
    else:
        return None
    
    
#now create formfactor objects / register them at AtomLayerObject (only necessary for formfactors which should not be taken automatically from the Chantler tables)
f2=lambda energy,absorbtion, a, b, c: absorbtion*a+b+c*energy     #function for imaginary part of formfactor from absorption measurement, which will be fitted to the off-resonant tabulated values
#usually f2 should rather be absorption*energy*a+b+c*energy. But what I used here for "absorption" has already been fit with this procedure and should not be scaled by energy again
Mn_FF=SampleRepresentation.FFfromScaledAbsorption('Mn', E1=600,E2=700,E3=710,scaling_factor=pp.newParameter("Mn_scaling"),absorption_filename="Mn.xas_aniso",energyshift=pp.newParameter("Mn_energyshift"), absorption_linereaderfunction=absorption_linereader, autofitfunction=f2, autofitrange=20)

    
print("... plot Mn formfactor to let user check")
Mn_FF.plotFF(start,energies=numpy.arange(500,800,0.01))
   
SampleRepresentation.AtomLayerObject.registerAtom("Mn_XAS",Mn_FF)
   
### build layers from bottom
print("... build layers")
   
substrate_layer = SampleRepresentation.AtomLayerObject({"Sr":density_SrTiO3, "Ti": density_SrTiO3, "O": 3* density_SrTiO3}, d=None,sigma=pp.newParameter("substrate_roughness"))
    
layer_SrRuO3 = SampleRepresentation.AtomLayerObject({"Sr":density_SrRuO3, "Ru": density_SrRuO3, "O": 3* density_SrRuO3} ,d=pp.newParameter("SrRuO3_thickness"),sigma=pp.newParameter("SrRuO3_roughness"))
    
layer_LSMO = SampleRepresentation.AtomLayerObject({"La":0.66667*density_LSMO, "Sr":0.33333*density_LSMO, "Mn_XAS":density_LSMO, "O": 3* density_LSMO}, d=pp.newParameter("LSMO_thickness"), sigma=pp.newParameter("LSMO_roughness"))
    
#layer_MnO2 = SampleRepresentation.AtomLayerObject({"Mn_XAS":density_MnO2, "O": 2* density_MnO2}, d=pp.newParameter("MnO2_thickness"), sigma=pp.newParameter("MnO2_roughness"))
    
cap_layer=SampleRepresentation.AtomLayerObject({"Sr":density_SrTiO3, "Ti": density_SrTiO3, "O": 3* density_SrTiO3}, d=pp.newParameter("cap_thickness"),sigma=pp.newParameter("cap_roughness"))
    
carbon_contamination=SampleRepresentation.AtomLayerObject({"C":density_C}, d=pp.newParameter("contamination_thickness"),sigma=pp.newParameter("contamination_roughness"))
   
###plug layers into heterostructure
print("... plug layers into heterostructure")
hs.setLayer(0,substrate_layer)
hs.setLayer(1,layer_SrRuO3)
hs.setLayer(2,layer_LSMO)
hs.setLayer(3,cap_layer)
hs.setLayer(4,carbon_contamination)

    
    
#### set up experiment ###########################################################################
    
# instantiate experiment for linear polarization ("l") and set length scale to Angstroem
simu=Experiment.ReflDataSimulator("lL",length_scale=1e-10)
    
    
namereader=lambda string: (float(string[:-4].split("_")[-2]), None)           #liefert energy aus den Dateinamen der verwendeten dateien
    
def pointmodifier(point):        #berechnet winkel aus qz und energy und ersetzt qz dadurch. Alle anderen Werte des Datenpunktes bleiben unveraendert
    point[1]=180.0/(numpy.pi)*numpy.arcsin( point[1]*simu.hcfactor/(2*numpy.pi)/(2*point[0]))
    return point
    
print("... read experimental data")
#read measured data from files (using pointmodifier and namerreader)
# at first read pi polarization
simu.ReadData("Experiment/pi",simu.createLinereader(angle_column=0,rpi_column=1), pointmodifierfunction=pointmodifier , filenamereaderfunction=namereader)
# now read in sigma polarization
simu.ReadData("Experiment/sigma",simu.createLinereader(angle_column=0,rsigma_column=1), pointmodifierfunction=pointmodifier , filenamereaderfunction=namereader)

# connect model with experiment
b=pp.newParameter("background")
m=pp.newParameter("multiplier")
reflmodifier=lambda r, fitpararray: b.getValue(fitpararray) + r * m.getValue(fitpararray)
simu.setModel(hs,exp_energyshift=pp.newParameter("exp_Eshift"), exp_angleshift=pp.newParameter("exp_thetashift"),reflmodifierfunction=reflmodifier)
    



####################################################################################################

#### perform fit and plot status

print("... plotting according to start values")
simu.plotData(start)

input("Press any key to proceed...")

print("... performing Fit")

#scipy least_squares Fit
res=scipy.optimize.least_squares(simu.getResiduals, start, bounds=(l,u), method='trf', x_scale=numpy.array(u)-numpy.array(l), jac='3-point',verbose=2)
best=res.x

print("... plotting fitted model")
simu.plotData(best)
#write found parameter set to a file
print("... write found parameters to a file")
pp.writeToFile("parameters_best.txt",best)


#screening the whole parameter range (randomly placing start parameter vectors within the parameters space and perform least squares fit)
print("... performing parameter range screening")
out = Fitters.Explore(simu.getResiduals,pp.getStartLowerUpper(),50)    #comment out this line if you or reduce the number of seeds (last argument) if you don't want to wait for ever

#dump result of Explore to a file for later use (Explore takes very long time)
pickle.dump( out, open( "explore_out.p", "wb" ) )

#for later use of earlier runs of Explore just comment out the above to lines and uncomment the following
#out = pickle.load(open("explore_out.p"))

print("... ploting centers of found clusters for all parameters")
Fitters.plot_clusters_allpars(out,pp)

print("... ploting centers of found clusters and the fixpoints themselfs for all parameters")
Fitters.plot_fixpoints_allpars(out,pp)

