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
#pp=Parameters.ParameterPool("parameters.txt")
pp=Parameters.ParameterPool()


#some Parameters which are used later on
mdensity_LSMO=pp.newParameter("mdensity_LSMO",fixed=True,start_val=6.5)    #mass density of the material in g/cm^3



#some numbers
mol_mass_Sr=87.62               #molar mass of elements  g/mol
mol_mass_Ti=47.867
mol_mass_Ru=101.07
mol_mass_O=15.999
mol_mass_Mn=54.938
mol_mass_C=12.011
mol_mass_La=138.91

#derived_parameters
density_LSMO=mdensity_LSMO/(0.7*mol_mass_La+0.3*mol_mass_Sr+mol_mass_Mn+3*mol_mass_O)   #number density of formula unit in mol/cm^3
    
    
start,l,u=pp.getStartLowerUpper()           #read start values etc. to check for errors at beginnning of script
#pp.writeToFile("parameters.txt")            #write parameter file to get nice layout after changing values

#####################################################################################################################################
#Creation of Model

    
    
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
        return (linearray[0], -linearray[1]+1j*linearray[2], 0,0,0, -linearray[3]+1j*linearray[4], 0,0,0, -linearray[5]+1j*linearray[6])          #convert to the PyXMRTool sign convention. See :doc:`/definitions/formfactors`
    else:
        return None
 
    
#now create formfactor objects / register them at AtomLayerObject
f2=lambda energy,absorbtion, a, b, c: absorbtion*a+b+c*energy     #function for imaginary part of formfactor from absorption measurement, which will be fitted to the off-resonant tabulated values
#usually f2 should rather be absorption*energy*a+b+c*energy. But what I used here for "absorption" has already been fit with this procedure and should not be scaled by energy again
Mn_FF=SampleRepresentation.FFfromScaledAbsorption('Mn', E1=600,E2=700,E3=710,scaling_factor=Parameters.Parameter(1), absorption_filename="data_for_comparison_from_Florian/Mn.aniso2", absorption_linereaderfunction=absorption_linereader, autofitfunction=f2, autofitrange=20)
#Mn_FF=SampleRepresentation.FFfromFile("data_for_comparison_from_Florian/Mn.aniso2", ff_file_linereader)

    
#print "... plot Mn formfactor to let user check"
#Mn_FF.plotFF(start)
   
SampleRepresentation.AtomLayerObject.registerAtom("Mn_XAS",Mn_FF)
   
### build layers from bottom
print "... build layers"
   
   
layer_LSMO = SampleRepresentation.AtomLayerObject({"La":0.7*density_LSMO, "Sr":0.3*density_LSMO, "Mn_XAS":density_LSMO, "O": 3* density_LSMO})
    


#read data for comparsion 
energies=[]
comp_delta_x=[]
comp_delta_y=[]
comp_delta_z=[]
comp_beta_x=[]
comp_beta_y=[]
comp_beta_z=[]
with open("data_for_comparison_from_Florian/LSMO_deltabeta.dat") as file:
    for line in file:
        line=(line.split(commentsymbol))[0]                            #ignore everything behind the commentsymbol  #
        if not line.isspace() and line:                               #ignore empty lines        
            linearray=line.split()
            linearray=[float(item) for item in linearray]
            energies.append(linearray[0])
            comp_delta_x.append(linearray[3])
            comp_delta_y.append(linearray[4])
            comp_delta_z.append(linearray[4])
            comp_beta_x.append(linearray[1])
            comp_beta_y.append(linearray[2])
            comp_beta_z.append(linearray[2])


#create own data
delta_x=[]
delta_y=[]
delta_z=[]
beta_x=[]
beta_y=[]
beta_z=[]
for energy in energies:
    chi_tensor=layer_LSMO.getChi(start,energy)
    chi_xx=chi_tensor[0]
    chi_yy=chi_tensor[4]
    chi_zz=chi_tensor[8]
    #nach "Macke 2014 J.Phys.:Condens. Matter 26 363201" Seite 10 ist \epsilon = 1+ \chi = n^2 \approx 1 - 2 \delta + 2i \beta
    delta_x.append(chi_xx.real/(-2.0))
    delta_y.append(chi_yy.real/(-2.0))
    delta_z.append(chi_zz.real/(-2.0))
    beta_x.append(chi_xx.imag/2.0)
    beta_y.append(chi_yy.imag/2.0)
    beta_z.append(chi_zz.imag/2.0)
    
    
    
energies=numpy.array(energies)
comp_delta_x=numpy.array(comp_delta_x)
comp_delta_y=numpy.array(comp_delta_y)
comp_delta_z=numpy.array(comp_delta_z)
comp_beta_x=numpy.array(comp_beta_x)
comp_beta_y=numpy.array(comp_beta_y)
comp_beta_z=numpy.array(comp_beta_z)
delta_x=numpy.array(delta_x)
delta_y=numpy.array(delta_y)
delta_z=numpy.array(delta_z)
beta_x=numpy.array(beta_x)
beta_y=numpy.array(beta_y)
beta_z=numpy.array(beta_z)


quot_delta_x=delta_x/comp_delta_x
quot_delta_y=delta_y/comp_delta_y
quot_delta_z=delta_z/comp_delta_z
quot_beta_x=beta_x/comp_beta_x
quot_beta_y=beta_y/comp_beta_y
quot_beta_z=beta_z/comp_beta_z

diff_delta_x=delta_x-comp_delta_x
diff_delta_y=delta_y-comp_delta_y
diff_delta_z=delta_z-comp_delta_z
diff_beta_x=beta_x-comp_beta_x
diff_beta_y=beta_y-comp_beta_y
diff_beta_z=beta_z-comp_beta_z

    
#compare beta_x
plt.plot(energies, beta_x, label="beta_x")
plt.plot(energies, comp_beta_x, label="comp_beta_x")
plt.legend()
plt.show()

plt.plot(energies, quot_beta_x, label="quot_beta_x")
plt.legend()
plt.show()

plt.plot(energies, diff_beta_x, label="diff_beta_x")
plt.legend()
plt.show()



plt.plot(energies, delta_x, label="delta_x")
plt.plot(energies, comp_delta_x, label="comp_delta_x")
plt.legend()
plt.show()

plt.plot(energies, quot_delta_x, label="quot_delta_x")
plt.legend()
plt.show()

plt.plot(energies, diff_delta_x, label="diff_delta_x")
plt.legend()
plt.show()


