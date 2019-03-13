import time
import math
import numpy 
from scipy import constants 
import scipy.special

import Pythonreflectivity

from PyXMRTool import Experiment
from PyXMRTool import Parameters
from PyXMRTool import SampleRepresentation




#############################
#Model Parameters (not fitted!!)
#############################
#all lengths in nm
transzone_thickness = 100     #thickness of the transition zone
lattice_const = 0.52            #assume cubic
#############################


pp=Parameters.ParameterPool("partest_experiment_dens.txt")
#pp=Parameters.ParameterPool()
fitpararray, lower, upper = pp.getStartLowerUpper()
#############################
#create fitparameters
#############################
total_thickness=pp.newParameter("total_thickness")
transition_pos=pp.newParameter("transition_pos")     #position of transition from Sr to Co measured in nm from top
transition_width=pp.newParameter("transition_width")  #width of transition (sigma in error function erf(z/sqrt(2)/sigma)
decay_depth=pp.newParameter("decay_depth")              #position at which the O density starts to decay, measured in nm from top
decay_rel_gradient=pp.newParameter("decay_rel_gradient")        #gradient of the decay (defined as positiv number)


#############################

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
FF_Ya=SampleRepresentation.FFfromScaledAbsorption('', E1=250, E2=400, E3=500, scaling_factor=pp.newParameter("Ya_scaling"), absorption_filename="C_imag.F",energyshift=pp.newParameter("Ya_eneryshift"),tabulated_filename="C_tabul.F",tabulated_linereaderfunction=SampleRepresentation.FFfromScaledAbsorption.createTabulatedLinereader(complex_numbers=False))
SampleRepresentation.AtomLayerObject.registerAtom("Co",FF_Co)
SampleRepresentation.AtomLayerObject.registerAtom("Sr",FF_Sr)
SampleRepresentation.AtomLayerObject.registerAtom("Ya",FF_Ya)

#create bottom layer and add it to heterostructure
bottom_layer=SampleRepresentation.AtomLayerObject({"Ya":density_O,"Sr":density_Sr},total_thickness-transzone_thickness)
hs.setLayer(0,bottom_layer)

#create density profile for Sr and Co (as convinient error function profile)
density_profile_Sr=SampleRepresentation.DensityProfile_erf(1,number_of_trans_layers,Parameters.Parameter(lattice_const),position=number_of_trans_layers*lattice_const-transition_pos,sigma=-transition_width,maximum=density_Sr,)

#create density profile for Ya (with the more flexibel base class "DensityProfile")
def lin_decay_profile_fct(z,z_start,rel_gradient,maximum):
    if z<z_start:
        return maximum
    else:
        return maximum*(1-float(z-z_start)*rel_gradient)
density_profile_Ya=SampleRepresentation.DensityProfile(1,number_of_trans_layers,Parameters.Parameter(lattice_const), Parameters.ParametrizedFunction(lin_decay_profile_fct, number_of_trans_layers*lattice_const-decay_depth,decay_rel_gradient, density_O) )



#create transition layers and add them to the heterostructure
for i in range(1,number_of_trans_layers+1):
    l_i= SampleRepresentation.AtomLayerObject({"Ya":density_profile_Ya.getDensityPar(i),"Sr":density_profile_Sr.getDensityPar(i), "Co": density_Sr-density_profile_Sr.getDensityPar(i)},Parameters.Parameter(lattice_const))
    hs.setLayer(i, l_i)






cmap=['yellow','magenta','black','b','green','red','grey','magenta']
SampleRepresentation.plotAtomDensity(hs,fitpararray,cmap,["Co","Sr"])
SampleRepresentation.plotAtomDensity(hs,fitpararray,cmap,["Ya"])
SampleRepresentation.plotAtomDensity(hs,fitpararray,cmap)
#SampleRepresentation.plotAtomDensity(hs,fitpararray,cmap)


#set up experiment

simu=Experiment.ReflDataSimulator("l")
  

namereader=lambda string: (float(string[-9:-4]), None)           #liefert energy aus den Dateinamen der verwendeten dateien

def pointmodifier(point):        #berechnet winkel aus qz und energy und ersetzt qz dadurch. Alle anderen Werte des Datenpunktes bleiben unveraendert
    point[1]=360.0/(2*numpy.pi)*numpy.arcsin( point[1]*simu.hcfactor/(2*point[0]))
    return point

#read data from files
simu.ReadData("data",simu.createLinereader(energy_column=1,angle_column=0,rsigma_column=2,rpi_column=3),pointmodifierfunction=pointmodifier , filenamereaderfunction=namereader)

#test to read data via array
#create test array first
datapoints=[]
for energyseq in simu._expdata:
    energy=energyseq[0]
    n_vars=len(energyseq)-1
    for i in range(len(energyseq[1])):
        point=[energy]
        for var in range(n_vars):
            point.append(energyseq[1+var][i])
        for i in range(7-n_vars):
            point.append(None)
        datapoints.append(point)
simu.setData(datapoints)
        


b=pp.newParameter("background")
m=pp.newParameter("multiplier")
reflmodifier=lambda r, ar: b.getValue(ar) + r * m.getValue(ar)
simu.setModel(hs,reflmodifierfunction=reflmodifier)


#pp.writeToFile("partest_experiment_dens.txt")

#simulate

starttime=time.time()
simdata=simu.getSimData(fitpararray)
print time.time()-starttime

starttime=time.time()
chisqr=simu.getSSR(fitpararray)
print time.time()-starttime

simdata2=simu.getSimData(fitpararray,[[468,[1,2,3,4,5]],[468.1,[1,2,3,4,5]]])
print simdata2






