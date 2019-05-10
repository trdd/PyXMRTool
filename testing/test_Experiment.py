import time
import math
import numpy 

import Pythonreflectivity


from PyXMRTool import Experiment
from PyXMRTool import Parameters
from PyXMRTool import SampleRepresentation





pp=Parameters.ParameterPool("partest_Experiment.txt")
#pp=Parameters.ParameterPool()

#set up layer system
#l=SampleRepresentation.LayerObject([pp.newParameter("chi_xx"),pp.newParameter("chi_xy"),pp.newParameter("chi_xz"),pp.newParameter("chi_yx"),pp.newParameter("chi_yy"),pp.newParameter("chi_yz"),pp.newParameter("chi_zx"),pp.newParameter("chi_zy"),pp.newParameter("chi_zz")],pp.newParameter("d"),Parameters.Parameter(0))
l=SampleRepresentation.LayerObject([pp.newParameter("chi_xx"),pp.newParameter("chi_yy"),pp.newParameter("chi_zz")],pp.newParameter("d"),Parameters.Parameter(0))

#ar=[33,0.0094,-0.444]
#ar+=range(39)                       #produce some arbitrary values
ar,low,up=pp.getStartLowerUpper()

hs=SampleRepresentation.Heterostructure(13)

for i in range(7):
    hs.setLayer(i,SampleRepresentation.LayerObject([pp.newParameter("chi"+str(i))],pp.newParameter("d"+str(i))))
hs.setLayer(7,l)

print("Create Formfactors")

FF_Co=SampleRepresentation.FFfromFile("Co.F",SampleRepresentation.FFfromFile.createLinereader(complex_numbers=False))
FF_Sr=SampleRepresentation.FFfromFile("Sr.F",SampleRepresentation.FFfromFile.createLinereader(complex_numbers=False))
#FF_Ya=SampleRepresentation.FFfromScaledAbsorption("",E1=250,E2=400,E3=500,scaling_factor=pp.newParameter("Ya_scaling"),tabulated_filename="C_tabul.F",absorption_filename="C_imag.F",energyshift=pp.newParameter("Ya_eneryshift"),tabulated_linereaderfunction=SampleRepresentation.FFfromScaledAbsorption.createTabulatedLinereader(complex_numbers=False))
FF_Ya=SampleRepresentation.FFfromScaledAbsorption("C",E1=250,E2=400,E3=500,scaling_factor=pp.newParameter("Ya_scaling"),absorption_filename="C_imag.F",energyshift=pp.newParameter("Ya_eneryshift"))


SampleRepresentation.AtomLayerObject.registerAtom("Co",FF_Co)
SampleRepresentation.AtomLayerObject.registerAtom("Sr",FF_Sr)
SampleRepresentation.AtomLayerObject.registerAtom("Al",SampleRepresentation.FFfromFile("Al.F",SampleRepresentation.FFfromFile.createLinereader(complex_numbers=False)))
SampleRepresentation.AtomLayerObject.registerAtom("Yannicium", FF_Ya)


#create one atom layer object with above registered atoms and put it into the heterostructure
print("Create atom layer object with registered atoms and put it into the heterostructure")
al1=SampleRepresentation.AtomLayerObject({"Sr":pp.newParameter("al1_density_Sr"), "Al":pp.newParameter("al1_density_Al"), "Yannicium":pp.newParameter("al1_density_Ya") }, pp.newParameter("al1_d"))
hs.setLayer(8,al1)
#add some more atom layer objects
print("add some more atom layer objects")
for i in range(9,13):
    hs.setLayer(i, SampleRepresentation.AtomLayerObject({"Sr":pp.newParameter("al"+str(i-7)+"_density_Sr"), "Al":pp.newParameter("al"+str(i-7)+"_density_Al"), "Yannicium":pp.newParameter("al"+str(i-7)+"_density_Ya") }, pp.newParameter("al"+str(i-7)+"_d")))


#set up experiment

simu=Experiment.ReflDataSimulator("l")
  

namereader=lambda string: (float(string[-9:-4]), None)           #liefert energy aus den Dateinamen der verwendeten dateien

def pointmodifier(point):        #berechnet winkel aus qz und energy und ersetzt qz dadurch. Alle anderen Werte des Datenpunktes bleiben unveraendert
    point[1]=360.0/(2*numpy.pi)*numpy.arcsin( point[1]*simu.hcfactor/(2*point[0]))
    return point

#read data from files
print("... read files")
simu.ReadData("data",simu.createLinereader(energy_column=1,angle_column=0,rsigma_column=2,rpi_column=3),pointmodifierfunction=pointmodifier , filenamereaderfunction=namereader)

print("... read from array")
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
reflmodifier=lambda r, fitpararray: b.getValue(fitpararray) + r * m.getValue(fitpararray)
simu.setModel(hs,reflmodifierfunction=reflmodifier)

#pp.setStartValues(ar)
#pp.writeToFile("partest_Experiment.txt")

#simulate
print("... simulate")
starttime=time.time()
simdata=simu.getSimData(ar)
print(time.time()-starttime)

starttime=time.time()
chisqr=simu.getSSR(ar)
print(time.time()-starttime)

simdata2=simu.getSimData(ar,[[468,[1,2,3,4,5]],[468.1,[1,2,3,4,5]]])
print(simdata2)



