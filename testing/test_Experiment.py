import time
import math
import numpy 

import Pythonreflectivity

import sys
sys.path.append('../')       #this statement makes it possible for the script to find the PyXMRTool package relative to the Tutorials folder. For your own projects rather copy the PyXMRTool-Folder which contains the modules to your project folder which contains the script
from PyXMRTool import Experiment
from PyXMRTool import Parameters
from PyXMRTool import SampleRepresentation





#pp=Parameters.ParameterPool("partest_Experiment.txt")
pp=Parameters.ParameterPool()

#set up layer system
#l=SampleRepresentation.LayerObject([pp.newParameter("chi_xx"),pp.newParameter("chi_xy"),pp.newParameter("chi_xz"),pp.newParameter("chi_yx"),pp.newParameter("chi_yy"),pp.newParameter("chi_yz"),pp.newParameter("chi_zx"),pp.newParameter("chi_zy"),pp.newParameter("chi_zz")],pp.newParameter("d"),Parameters.Parameter(0))
l=SampleRepresentation.LayerObject([pp.newParameter("chi_xx"),pp.newParameter("chi_yy"),pp.newParameter("chi_zz")],pp.newParameter("d"),Parameters.Parameter(0))

ar=[33,0.0094,-0.444]
ar+=range(21)

hs=SampleRepresentation.Heterostructure(8,[0,1,2,[10,[3,4,5,6]],7])

for i in range(7):
    hs.setLayer(i,SampleRepresentation.LayerObject([pp.newParameter("chi"+str(i))],pp.newParameter("d"+str(i))))
hs.setLayer(7,l)

FF_Co=SampleRepresentation.FFfromFile("Co.F",SampleRepresentation.FFfromFile.createLinereader(complex_numbers=False))
FF_Sr=SampleRepresentation.FFfromFile("Sr.F",SampleRepresentation.FFfromFile.createLinereader(complex_numbers=False))


SampleRepresentation.AtomLayerObject.registerAtom("Co",FF_Co)
SampleRepresentation.AtomLayerObject.registerAtom("Sr",FF_Sr)
SampleRepresentation.AtomLayerObject.registerAtom("Al",SampleRepresentation.FFfromFile("Al.F",SampleRepresentation.FFfromFile.createLinereader(complex_numbers=False)))


al1=SampleRepresentation.AtomLayerObject({"Sr":pp.newParameter("al1_density_Sr"),"Al":pp.newParameter("al1_density_Al"),"Co":pp.newParameter("al1_density_Co")},pp.newParameter("al1_d"))



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
reflmodifier=lambda r, fitpararray: b.getValue(fitpararray) + r * m.getValue(fitpararray)
simu.setModel(hs,reflmodifier)

#pp.setStartValues(ar)
#pp.writeToFile("partest_Experiment.txt")

#simulate

starttime=time.time()
simdata=simu.getSimData(ar)
print time.time()-starttime

starttime=time.time()
chisqr=simu.getSSR(ar)
print time.time()-starttime

simdata2=simu.getSimData(ar,[[468,[1,2,3,4,5]],[468.1,[1,2,3,4,5]]])
print simdata2



