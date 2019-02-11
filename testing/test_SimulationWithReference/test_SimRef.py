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
pp=Parameters.ParameterPool("parameters.txt")
#pp=Parameters.ParameterPool()


#some Parameters which are used later on
mdensity_SrTiO3=pp.newParameter("mdensity_SrTiO3")    #mass density of the material in g/cm^3
mdensity_SrRuO3=pp.newParameter("mdensity_SrRuO3")
mdensity_LSMO=pp.newParameter("mdensity_LSMO")
mdensity_SrTiO3_cap=pp.newParameter("mdensity_SrTiO3_cap")
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
density_SrTiO3_cap=mdensity_SrTiO3_cap/(mol_mass_Sr+ mol_mass_Ti+3*mol_mass_O)
density_C=mdensity_C/mol_mass_C


    
    
start,l,u=pp.getStartLowerUpper()           #read start values etc. to check for errors at beginnning of script
pp.writeToFile("parameters.txt")            #write parameter file to get nice layout after changing values

#####################################################################################################################################
#Creation of Model

    
    
# set up heterostructure (with 5 layers)
print "... set up heterostructure"
hs = SampleRepresentation.Heterostructure(5)


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
        return (linearray[0], -linearray[1]+1j*linearray[2], 0,0,0, -linearray[3]+1j*linearray[4], 0,0,0, -linearray[5]+1j*linearray[6])            #BEWARE: different sign-convention
    else:
        return None
 
    
#now create formfactor objects / register them at AtomLayerObject
f2=lambda energy,absorbtion, a, b, c: absorbtion*a+b+c*energy     #function for imaginary part of formfactor from absorption measurement, which will be fitted to the off-resonant tabulated values
#usually f2 should rather be absorption*energy*a+b+c*energy. But what I used here for "absorption" has already been fit with this procedure and should not be scaled by energy again
Mn_FF=SampleRepresentation.FFfromScaledAbsorption('Mn', E1=600,E2=700,E3=710,scaling_factor=pp.newParameter("Mn_scaling"),absorption_filename="data_for_comparison_from_Florian/Mn.aniso2",energyshift=pp.newParameter("Mn_energyshift"), absorption_linereaderfunction=absorption_linereader,minE=500,maxE=1000, autofitfunction=f2, autofitrange=20)
#Mn_FF=SampleRepresentation.FFfromFile("data_for_comparison_from_Florian/Mn.aniso2", ff_file_linereader,energyshift=pp.newParameter("Mn_energyshift"))

    
print "... plot Mn formfactor to let user check"
Mn_FF.plotFF(start,energies=numpy.arange(550,750,0.01))
   
SampleRepresentation.AtomLayerObject.registerAtom("Mn_XAS",Mn_FF)
   
### build layers from bottom
print "... build layers"
   
substrate_layer = SampleRepresentation.AtomLayerObject({"Sr":density_SrTiO3, "Ti": density_SrTiO3, "O": 3* density_SrTiO3}, d=None,sigma=pp.newParameter("substrate_roughness"))
    
layer_SrRuO3 = SampleRepresentation.AtomLayerObject({"Sr":density_SrRuO3, "Ru": density_SrRuO3, "O": 3* density_SrRuO3} ,d=pp.newParameter("SrRuO3_thickness"),sigma=pp.newParameter("SrRuO3_roughness"))
    
layer_LSMO = SampleRepresentation.AtomLayerObject({"La":0.66667*density_LSMO, "Sr":0.33333*density_LSMO, "Mn_XAS":density_LSMO, "O": 3* density_LSMO}, d=pp.newParameter("LSMO_thickness"), sigma=pp.newParameter("LSMO_roughness"))
    
cap_layer=SampleRepresentation.AtomLayerObject({"Sr":density_SrTiO3_cap, "Ti": density_SrTiO3_cap, "O": 3* density_SrTiO3_cap}, d=pp.newParameter("cap_thickness"),sigma=pp.newParameter("cap_roughness"))
    
carbon_contamination=SampleRepresentation.AtomLayerObject({"C":density_C}, d=pp.newParameter("contamination_thickness"),sigma=pp.newParameter("contamination_roughness"))
   
###plug layers into heterostructure
print "... plug layers into heterostructure"
hs.setLayer(0,substrate_layer)
hs.setLayer(1,layer_SrRuO3)
hs.setLayer(2,layer_LSMO)
hs.setLayer(3,cap_layer)
hs.setLayer(4,carbon_contamination)

    
    
#### set up experiment ###########################################################################
    
# instantiate experiment for linear polarization ("l") and set length scale to Angstroem
simu=Experiment.ReflDataSimulator("lL",length_scale=1e-10)
    
    
namereader=lambda string: (float(string[:-4].split("_")[-1]), None)           #liefert energy aus den Dateinamen der verwendeten dateien
    
def pointmodifier(point):        #berechnet winkel aus qz und energy und ersetzt qz dadurch. Alle anderen Werte des Datenpunktes bleiben unveraendert
    point[1]=180.0/(numpy.pi)*numpy.arcsin( point[1]*simu.hcfactor/(2*numpy.pi)/(2*point[0]))
    return point
    
print "... read experimental data"
#read measured data from files (using pointmodifier and namerreader)
#simu.ReadData("Exp_Umgeformt",simu.createLinereader(angle_column=0,rsigma_column=1,rpi_column=2),pointmodifierfunction=pointmodifier , filenamereaderfunction=namereader)
simu.ReadData(["Exp_Umgeformt/sro_lsmo_630.0.dat"],simu.createLinereader(angle_column=0,rsigma_column=1,rpi_column=2),pointmodifierfunction=pointmodifier , filenamereaderfunction=namereader) 
    
# connect model with experiment
b=pp.newParameter("background")
m=pp.newParameter("multiplier")
reflmodifier=lambda r, fitpararray: b.getValue(fitpararray) + r * m.getValue(fitpararray)
simu.setModel(hs,exp_energyshift=pp.newParameter("exp_Eshift"), exp_angleshift=pp.newParameter("exp_thetashift"),reflmodifierfunction=reflmodifier)
#note on exp_angleshift: Florian uses a "q_z-shift" of 0.0078187. I translate it to an angleshift at an energy of 630eV (in fact an q_z-shift does not make to much sense, as the instrument works with energy-independent angles)





######################################################################################################
### Vergleichsdaten einlesen (nur sigma Polarisation)
comp_file= "data_for_comparison_from_Florian/ref"





comp_energies = []              # Liste der Energien
comp_angles = []                # Liste von Listen der Winkel (jede Subliste ein Zeile in "ref"
comp_reflectivities = []        # Reflektivitaeten nach folgendem Format  (wie die Zeilen in "ref")

#read angles and reflectivities
with open(comp_file) as file:
    for line in file:
        line=(line.split("#"))[0]                            #ignore everything behind the commentsymbol  #
        if not line.isspace() and line:                               #ignore empty lines        
            linearray=line.split()
            linearray=[float(item) for item in linearray]
            comp_angles.append(linearray[::3])                  #erste Spalte und dann jede dritte weitere (Index 0,3,6, usw.) ist q_z
            comp_reflectivities.append(linearray[2::3])           #dritte Spalte und dann jede dritte weitere (Index 2,5,8, usw.) ist berechnete sigma Reflektivitaet

#read energies
with open(comp_file) as file:
    lines=file.readlines()
    line=lines[1]
    line=line.split("#")[1]   #remove commentsymbol
    linearray=line.split("-")                                   #split entries
    linearray=[element.split(",")[0] for element in linearray]  #remove strange comma and zero
    comp_energies = [float(element) for element in linearray]



comp_energies=numpy.array(comp_energies)
comp_angles=numpy.array(comp_angles)
comp_reflectivities=numpy.array(comp_reflectivities)


#transponiere winkel und reflectivities
#comp_angles=comp_angles.transpose()
comp_reflectivities=comp_reflectivities.transpose()




#substract exp.qshift
#note on exp_angleshift: Florian uses a "exp.qshift" of 0.0078187. This cannot be translated to a global angleshift (in fact an q_z-shift does not make to much sense, as the instrument works with energy-independent angles and the mapping from angle to q_z is not linear). Therefore I substract the shift here for comparison
comp_angles=comp_angles-0.007818787485514856

#berechne winkel aus q_z
comp_angles=(180.0/(numpy.pi)*numpy.arcsin( comp_angles*simu.hcfactor/(2*numpy.pi)/(2*comp_energies))).transpose()





##########################################################
# Compare

energyAngles=[]
for i in range(len(comp_energies)):
    energyAngles.append([comp_energies[i],comp_angles[i]])
    

data=simu.getSimData(start,energyAngles)
#data=simu.getSimData(best,energyAngles)

#angle_630 = comp_angles[12]
#refl_comp_630 = numpy.log(comp_reflectivities[12])
#refl_630 = data[12][2]

#plt.plot(angle_630,refl_630,label="own")
#plt.plot(angle_630,refl_comp_630, label="ref")
#plt.legend()
#plt.show()

for i in range(len(comp_energies)):
    fig,ax = plt.subplots()
    ax.plot(comp_angles[i],data[i][2],label="sigma pol (own calc)")
    ax.plot(comp_angles[i],numpy.log(comp_reflectivities[i]),label="sigma pol (ref)")
    ax.legend()
    ax.text(0.05,0.2,str(comp_energies[i])+ " eV",transform=ax.transAxes,bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),fontsize=14)
    plt.savefig("refl_plot_"+str(comp_energies[i])+"eV.png")
    plt.close(fig)






