import math
import numpy
import timeit
import time
from matplotlib import pyplot as plt
import scipy.optimize


from PyXMRTool import SampleRepresentation
from PyXMRTool import Parameters
from PyXMRTool import Experiment
from PyXMRTool import Fitters

### ALL LENGTHES ARE MEASURED IN Angstroem!!!! 
    
#### create ParameterPool #####################################################################
pp=Parameters.ParameterPool("V2-parameters.txt")
#pp=Parameters.ParameterPool()


#some Parameters which are used later on
mdensity_SrTiO3=pp.newParameter("mdensity_SrTiO3")    #mass density of the material in g/cm^3
mdensity_SrRuO3=pp.newParameter("mdensity_SrRuO3")
mdensity_LSMO=pp.newParameter("mdensity_LSMO")
mdensity_MnO2=pp.newParameter("mdensity_MnO2")
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
density_LSMO=mdensity_LSMO/(0.7*mol_mass_La+0.3*mol_mass_Sr+mol_mass_Mn+3*mol_mass_O)
density_MnO2=mdensity_MnO2/(mol_mass_Mn+2*mol_mass_O)
density_C=mdensity_C/mol_mass_C
    
    
start,l,u=pp.getStartLowerUpper()           #read start values etc. to check for errors at beginnning of script
#pp.writeToFile("V2-parameters.txt")            #write parameter file to get nice layout after changing values

#####################################################################################################################################
#Creation of Model

    
    
# set up heterostructure (with 6 layers)
print "... set up heterostructure"
hs = SampleRepresentation.Heterostructure(6)


###set up formfactors
print "... set up formfactors"
   
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
    
def ff_file_linereader(line):
    line=(line.split(commentsymbol))[0]                            #ignore everything behind the commentsymbol  #
    if not line.isspace() and line:                               #ignore empty lines        
        linearray=line.split()
        linearray=[float(item) for item in linearray]
        return (linearray[0], linearray[1]+1j*linearray[2], 0,0,0, linearray[3]+1j*linearray[4], 0,0,0, linearray[5]+1j*linearray[6])
    else:
        return None
 
    
#now create formfactor objects / register them at AtomLayerObject
f2=lambda energy,absorbtion, a, b, c: absorbtion*a+b+c*energy     #function for imaginary part of formfactor from absorption measurement, which will be fitted to the off-resonant tabulated values
#usually f2 should rather be absorption*energy*a+b+c*energy. But what I used here for "absorption" has already been fit with this procedure and should not be scaled by energy again
#Mn_FF=SampleRepresentation.FFfromScaledAbsorption('Mn', E1=600,E2=700,E3=710,scaling_factor=pp.newParameter("Mn_scaling"),absorption_filename="Mn.xas_aniso",energyshift=pp.newParameter("Mn_energyshift"), absorption_linereaderfunction=absorption_linereader,minE=500,maxE=1000, autofitfunction=f2, autofitrange=20)
Mn_FF=SampleRepresentation.FFfromFile("Mn.xas_aniso", ff_file_linereader,energyshift=pp.newParameter("Mn_energyshift"))

    
print "... plot Mn formfactor to let user check"
#Mn_FF.plotFF(start)
   
SampleRepresentation.AtomLayerObject.registerAtom("Mn_XAS",Mn_FF)
   
### build layers from bottom
print "... build layers"
   
substrate_layer = SampleRepresentation.AtomLayerObject({"Sr":density_SrTiO3, "Ti": density_SrTiO3, "O": 3* density_SrTiO3}, d=None,sigma=pp.newParameter("substrate_roughness"))
    
layer_SrRuO3 = SampleRepresentation.AtomLayerObject({"Sr":density_SrRuO3, "Ru": density_SrRuO3, "O": 3* density_SrRuO3} ,d=pp.newParameter("SrRuO3_thickness"),sigma=pp.newParameter("SrRuO3_roughness"))
    
layer_LSMO = SampleRepresentation.AtomLayerObject({"La":0.7*density_LSMO, "Sr":0.3*density_LSMO, "Mn_XAS":density_LSMO, "O": 3* density_LSMO}, d=pp.newParameter("LSMO_thickness"), sigma=pp.newParameter("LSMO_roughness"))
    
layer_MnO2 = SampleRepresentation.AtomLayerObject({"Mn_XAS":density_MnO2, "O": 2* density_MnO2}, d=pp.newParameter("MnO2_thickness"), sigma=pp.newParameter("MnO2_roughness"))
    
cap_layer=SampleRepresentation.AtomLayerObject({"Sr":density_SrTiO3, "Ti": density_SrTiO3, "O": 3* density_SrTiO3}, d=pp.newParameter("cap_thickness"),sigma=pp.newParameter("cap_roughness"))
    
carbon_contamination=SampleRepresentation.AtomLayerObject({"C":density_C}, d=pp.newParameter("contamination_thickness"),sigma=pp.newParameter("cap_roughness"))
   
###plug layers into heterostructure
print "... plug layers into heterostructure"
hs.setLayer(0,substrate_layer)
hs.setLayer(1,layer_SrRuO3)
hs.setLayer(2,layer_LSMO)
hs.setLayer(3,layer_MnO2)
hs.setLayer(4,cap_layer)
hs.setLayer(5,carbon_contamination)

    
    
#### set up experiment ###########################################################################
    
# instantiate experiment for linear polarization ("l") and set length scale to Angstroem
simu=Experiment.ReflDataSimulator("lL",length_scale=1e-10)
    
    
namereader=lambda string: (float(string[:-4].split("_")[-1]), None)           #liefert energy aus den Dateinamen der verwendeten dateien
    
def pointmodifier(point):        #berechnet winkel aus qz und energy und ersetzt qz dadurch. Alle anderen Werte des Datenpunktes bleiben unveraendert
    point[1]=180.0/(numpy.pi)*numpy.arcsin( point[1]*simu.hcfactor/(2*numpy.pi)/(2*point[0]))
    return point
    
print "... read experimental data"
#read measured data from files (using pointmodifier and namerreader)
simu.ReadData("Exp_Umgeformt",simu.createLinereader(angle_column=0,rsigma_column=1,rpi_column=2),pointmodifierfunction=pointmodifier , filenamereaderfunction=namereader)
#simu.ReadData(["Exp_Umgeformt/sro_lsmo_630.0.dat"],simu.createLinereader(angle_column=0,rsigma_column=1,rpi_column=2),pointmodifierfunction=pointmodifier , filenamereaderfunction=namereader) 
    
# connect model with experiment
b=pp.newParameter("background")
m=pp.newParameter("multiplier")
reflmodifier=lambda r, fitpararray: b.getValue(fitpararray) + r * m.getValue(fitpararray)
simu.setModel(hs,reflmodifier)
    



####################################################################################################

### messe wie lange es dauert SSR 1000 mal auszurechnen
#print "... start measuring execution duration"
def wrapper():
    simu.getSSR(start)
zeitspanne=timeit.timeit(wrapper,number=1000)

print "--->Berechnung von Chi^2 dauert "+str(zeitspanne/1000)+"s"    #---->dauert ca 0.22 s pro Anfrage auf meinem Laptop

#### perform fit and plot status

print "... plotting according to start values"
simu.plotData(start)

raw_input("Druecke Taste um fortzusetzen...")

print "... performing Fit"
starttime=time.time()

#eigener Levenberg_Marquardt_Fitter
def rescost(fitpararray):
    return simu.getResidualsSSR(fitpararray)

#best, ssr = Fitters.Levenberg_Marquardt_Fitter(rescost,  ( start, l, u), 20 ,number_of_cores=4, strict=False, control_file=None,plotfunction=simu.plotData)

#eigener Evolution-Fitter
def cost(fitpararray):
    return simu.getSSR(fitpararray)
#best, ssr = Fitters.Evolution(cost, (start, l, u) , iterations=10, number_of_cores=4,mutation_strength=0.4,plotfunction=simu.plotData)

#scipy least_squares Fit
res= scipy.optimize.least_squares(simu.getResiduals, start, bounds=(l,u), method='trf', x_scale='jac',verbose=2)
best=res.x

print "---> Duration of fit procedure: "+ str(time.time()-starttime)+"s"

simu.plotData(best)

#write found parameter set to a file
pp.setStartValues(best)
pp.writeToFile("V2-best-parameters.txt")


se=hs.getSingleEnergyStructure(best,630)
for i in range(len(se)):
    print se[i].chixx()