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
    
    
start,lower,upper=pp.getStartLowerUpper()           #read start values etc. to check for errors at beginnning of script    (without the parameters of the FF made of triangles)
pp.writeToFile("parameters.txt")            #write parameter file to get nice layout after changing values

#####################################################################################################################################
#Creation of Model

    
    
# set up heterostructure (with 5 layers)
print("... set up heterostructure")
hs = SampleRepresentation.Heterostructure(5)


###set up formfactors
print("... set up formfactors")
   

#--------------------------------------------------------------------------
#set up formfactor for Mn, fitable as sum of triangles (imaginary part) and the corresponding real part 

base_energies=numpy.arange(600,702,2)                   #resolution / how dense are the triangles and what is minimum and maximum

##--------------
#at first create Formfactor object from XAS measurement which can be used for start values
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
Mn_XAS_FF=SampleRepresentation.FFfromScaledAbsorption('Mn', E1=600,E2=700,E3=710,scaling_factor=Parameters.Parameter(1),absorption_filename="Mn.xas_aniso",energyshift=Parameters.Parameter(0), absorption_linereaderfunction=absorption_linereader, autofitfunction=f2, autofitrange=20)
##--------------

def ff_tensor(energy, *coeffs):
    """
    Model funtion for the formfactor tensor. Sum of ff_real and ff_imag on the diagonal elements.
    
    The imaginary part is sum of triangles with coefficients and base energies ("places where the triangles sit at").
    **base_energies** have to be sorted. **coeffs** is a list with same length as base energies and determines the hight of every triangle.
    (A simpler version of the procedure in Stone et al., PRB 86,024102 (2012) and Kuzmenko, Rev. Sci. Instrum. 76,083108.)
    """
    coeffs=numpy.array(coeffs)
    #helper function for real part
    def g(x,y):
        return (x+y)*numpy.log(numpy.abs(x+y)) + (x-y)*numpy.log(numpy.abs(x-y))
    #sum over basis functions
    f_imag = 0
    f_real = 0
    for j in range(1,len(base_energies)-1):   #go throuth every index of base_energies but not the lowest and highest  
        #imag part
        if (base_energies[j-1] < energy) and (energy <= base_energies[j]) :
            f_basis_j = ( energy - base_energies[j-1] ) / ( base_energies[j] - base_energies[j-1] )
        elif (base_energies[j] < energy) and (energy < base_energies[j+1]) :
            f_basis_j = ( base_energies[j+1] - energy ) / ( base_energies[j+1] - base_energies[j] )
        else:
            f_basis_j = 0
        f_imag = f_imag+coeffs[j]*f_basis_j
        #real part
        kk_f_basis_j= -1 /numpy.pi * ( g(energy,base_energies[j-1])/(base_energies[j]-base_energies[j-1]) - (base_energies[j+1]-base_energies[j-1])*g(energy,base_energies[j])/( (base_energies[j]-base_energies[j-1])*(base_energies[j+1]-base_energies[j]) ) + g(energy,base_energies[j+1])/(base_energies[j+1] -base_energies[j])  )
        f_real = f_real+coeffs[j]*kk_f_basis_j
    ff=f_real + 1j*f_imag
    return numpy.array([ff,0,0,0,ff,0,0,0,ff])
    
    
#generate parameters as coefficients
suffix="Mn_ff_coeff_"
coeff_pars_list=[pp.newParameter(suffix+ str(i)) for i in range(len(base_energies))] 

#generate parametrized Function
ff_tensor_parfunc = Parameters.ParametrizedFunction(ff_tensor,*coeff_pars_list)

#generate Formfactor object with parametrized funtion
Mn_triangles_FF = SampleRepresentation.FFfromFitableModel(ff_tensor_parfunc, min(base_energies), max(base_energies))

#generate parameter start values and limits for coeffients
maximum=0
for coeff_par,energy in zip(coeff_pars_list,base_energies):
    coeff_par.start_val= Mn_XAS_FF.getFF(energy, start)[0].imag         #set start values on the XAS measurement
    coeff_par.lower_lim = 0                                             #lower limit is just 0
    if coeff_par.start_val > maximum:                                   #find maximum of start values for the upper limit
        maximum=coeff_par.start_val
for coeff_par,energy in zip(coeff_pars_list,base_energies):
    coeff_par.upper_lim=2*maximum                                       #upper limit is 2 times maximum of start values

#get new complete set of start values and limits for all parameters
start, lower, upper = pp.getStartLowerUpper()

#--------------------------------------------------------------------------

print("... plot Mn formfactor to let user check")
Mn_triangles_FF.plotFF(start,energies=numpy.arange(600,700,0.01))
   
SampleRepresentation.AtomLayerObject.registerAtom("Mn_triangles",Mn_triangles_FF)
   
### build layers from bottom
print("... build layers")
   
substrate_layer = SampleRepresentation.AtomLayerObject({"Sr":density_SrTiO3, "Ti": density_SrTiO3, "O": 3* density_SrTiO3}, d=None,sigma=pp.newParameter("substrate_roughness"))
    
layer_SrRuO3 = SampleRepresentation.AtomLayerObject({"Sr":density_SrRuO3, "Ru": density_SrRuO3, "O": 3* density_SrRuO3} ,d=pp.newParameter("SrRuO3_thickness"),sigma=pp.newParameter("SrRuO3_roughness"))
    
layer_LSMO = SampleRepresentation.AtomLayerObject({"La":0.66667*density_LSMO, "Sr":0.33333*density_LSMO, "Mn_triangles":density_LSMO, "O": 3* density_LSMO}, d=pp.newParameter("LSMO_thickness"), sigma=pp.newParameter("LSMO_roughness"))
    
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
res=scipy.optimize.least_squares(simu.getResiduals, start, bounds=(lower,upper), method='trf', x_scale=numpy.array(upper)-numpy.array(lower), jac='3-point',verbose=2)
best=res.x

print("... plotting fitted model")
simu.plotData(best)
#write found parameter set to a file
print("... write found parameters to a file")
pp.writeToFile("parameters_best.txt",best)

print("... plotting fitted formfactor")
Mn_triangles_FF.plotFF(best)
