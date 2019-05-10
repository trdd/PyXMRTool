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







    
#### create ParameterPool #####################################################################
pp=Parameters.ParameterPool("simplified-parameters.txt")
#pp=Parameters.ParameterPool()




#some Parameters which are used later on

suscept_SrTiO3=pp.newParameter("suscept_SrTiO3")    #mass density of the material in g/cm^3
suscept_SrRuO3=pp.newParameter("suscept_SrRuO3")
suscept_LSMO=pp.newParameter("suscept_LSMO")
suscept_MnO2=pp.newParameter("suscept_MnO2")
suscept_C=pp.newParameter("suscept_C")      #number density of C atoms in mol/cm^3

 
    

#pp.writeToFile("simplified-parameters.txt")            #write parameter file to get nice layout after changing values
start,l,u=pp.getStartLowerUpper()           #read start values etc. to check for errors at beginnning of script

#####################################################################################################################################
#Creation of Model

    
    
# set up heterostructure (with 6 layers)
print("... set up heterostructure")
hs = SampleRepresentation.Heterostructure(6)



### build layers from bottom
print("... build layers")
   
substrate_layer = SampleRepresentation.LayerObject([suscept_SrTiO3], d=None,sigma=pp.newParameter("substrate_roughness"))
    
layer_SrRuO3 = SampleRepresentation.LayerObject([suscept_SrRuO3],d=pp.newParameter("SrRuO3_thickness"),sigma=pp.newParameter("SrRuO3_roughness"))
    
layer_LSMO = SampleRepresentation.LayerObject([suscept_LSMO], d=pp.newParameter("LSMO_thickness"), sigma=pp.newParameter("LSMO_roughness"))

    
layer_MnO2 = SampleRepresentation.LayerObject([suscept_MnO2], d=pp.newParameter("MnO2_thickness"), sigma=pp.newParameter("MnO2_roughness"))

    
cap_layer=SampleRepresentation.LayerObject([suscept_SrTiO3], d=pp.newParameter("cap_thickness"),sigma=pp.newParameter("cap_roughness"))
    
carbon_contamination=SampleRepresentation.LayerObject([suscept_C],sigma=pp.newParameter("cap_roughness"))
   
###plug layers into heterostructure
print("... plug layers into heterostructure")
hs.setLayer(0,substrate_layer)
hs.setLayer(1,layer_SrRuO3)
hs.setLayer(2,layer_LSMO)
hs.setLayer(3,layer_MnO2)
hs.setLayer(4,cap_layer)
hs.setLayer(5,carbon_contamination)

    
    
#### set up experiment ###########################################################################
    
# instantiate experiment for linear polarization ("l")
simu=Experiment.ReflDataSimulator("lL",length_scale=1e-10)
    
    
namereader=lambda string: (float(string[:-4].split("_")[-1]), None)           #liefert energy aus den Dateinamen der verwendeten dateien
    
def pointmodifier(point):        #berechnet winkel aus qz und energy und ersetzt qz dadurch. Alle anderen Werte des Datenpunktes bleiben unveraendert
    point[1]=180.0/(numpy.pi)*numpy.arcsin( point[1]*simu.hcfactor/(2*numpy.pi)/(2*point[0]))
    return point
    
print("... read experimental data")
#read measured data from files (using pointmodifier and namerreader)
#simu.ReadData("Exp_Umgeformt",simu.createLinereader(angle_column=0,rsigma_column=1,rpi_column=2),pointmodifierfunction=pointmodifier , filenamereaderfunction=namereader)
simu.ReadData(["Exp_Umgeformt/sro_lsmo_630.0.dat"],simu.createLinereader(angle_column=0,rsigma_column=1,rpi_column=2),pointmodifierfunction=pointmodifier , filenamereaderfunction=namereader) 
    
# connect model with experiment
b=pp.newParameter("background")
m=pp.newParameter("multiplier")
reflmodifier=lambda r, fitpararray: b.getValue(fitpararray) + r * m.getValue(fitpararray)
simu.setModel(hs,reflmodifier)
    


#pp.writeToFile("simplified-parameters.txt")            #write parameter file to get a first template
####################################################################################################

### messe wie lange es dauert SSR 1000 mal auszurechnen
#print "... start measuring execution duration"
def wrapper():
    simu.getSSR(start)
zeitspanne=timeit.timeit(wrapper,number=1000)

print("--->Berechnung von Chi^2 dauert "+str(zeitspanne/1000)+"s")    #---->dauert ca 0.22 s pro Anfrage auf meinem Laptop

#### perform fit and plot status

print("... plotting according to start values")
simu.plotData(start)


print("... performing Fit")
starttime=time.time()

#eigener Levenberg_Marquardt_Fitter
def rescost(fitpararray):
    return simu.getResidualsSSR(fitpararray)

#best, ssr = Fitters.Levenberg_Marquardt_Fitter(rescost,  ( start, l, u), 20 ,number_of_cores=4, strict=False, control_file=None,plotfunction=simu.plotData)

#eigener Evolution-Fitter
def cost(fitpararray):
    return simu.getSSR(fitpararray)
#best, ssr = Fitters.Evolution(cost, (start, l, u) , iterations=1000000, number_of_cores=4,mutation_strength=0.4)#,plotfunction=simu.plotData)

#scipy least_squares Fit
res= scipy.optimize.least_squares(simu.getResiduals, start, bounds=(l,u), method='trf', x_scale='jac',verbose=2)
best=res.x

print("---> Duration of fit procedure: "+ str(time.time()-starttime)+"s")

simu.plotData(best)

#write found parameter set to a file
pp.setStartValues(best)
pp.writeToFile("simplified-best-parameters.txt")


se=hs.getSingleEnergyStructure(best,630)
for i in range(len(se)):
    print(se[i].chixx())