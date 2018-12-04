import numpy
import time
import types

import sys
sys.path.append('../')       #this statement makes it possible for the script to find the PyXMRTool package relative to the Tutorials folder. For your own projects rather copy the PyXMRTool-Folder which contains the modules to your project folder which contains the script
from PyXMRTool import Experiment
from PyXMRTool import Parameters
from PyXMRTool import SampleRepresentation
from PyXMRTool import Fitters



if __name__ == '__main__':

    pp=Parameters.ParameterPool("partest.txt")

    #set up layer system
    #l=SampleRepresentation.LayerObject([pp.newParameter("chi_xx"),pp.newParameter("chi_xy"),pp.newParameter("chi_xz"),pp.newParameter("chi_yx"),pp.newParameter("chi_yy"),pp.newParameter("chi_yz"),pp.newParameter("chi_zx"),pp.newParameter("chi_zy"),pp.newParameter("chi_zz")],pp.newParameter("d"),Parameters.Parameter(0))
    l=SampleRepresentation.LayerObject([pp.newParameter("chi_xx"),pp.newParameter("chi_yy"),pp.newParameter("chi_zz")],pp.newParameter("d"),Parameters.Parameter(0))

    ar=[33,0.0094,-0.444]
    ar+=range(36)

    hs=SampleRepresentation.Heterostructure(9,[0,1,2,[10,[3,4,5,6]],7,8])

    for i in range(7):
        hs.setLayer(i,SampleRepresentation.LayerObject([pp.newParameter("chi"+str(i))],pp.newParameter("d"+str(i))))
    hs.setLayer(7,l)

    FF_Co=SampleRepresentation.FFfromFile("Co.F",SampleRepresentation.FFfromFile.createLinereader(complex_numbers=False))
    FF_Sr=SampleRepresentation.FFfromFile("Sr.F",SampleRepresentation.FFfromFile.createLinereader(complex_numbers=False))


    SampleRepresentation.AtomLayerObject.registerAtom("Co",FF_Co)
    SampleRepresentation.AtomLayerObject.registerAtom("Sr",FF_Sr)
    SampleRepresentation.AtomLayerObject.registerAtom("Al",SampleRepresentation.FFfromFile("Al.F",SampleRepresentation.FFfromFile.createLinereader(complex_numbers=False)))


    al1=SampleRepresentation.AtomLayerObject({"Sr":pp.newParameter("al1_density_Sr"),"Al":pp.newParameter("al1_density_Al"),"Co":pp.newParameter("al1_density_Co")},pp.newParameter("al1_d"))
    
    hs.setLayer(8,al1)


    #set up experiment

    simu=Experiment.ReflDataSimulator("l")
  

    namereader=lambda string: (float(string[-9:-4]), None)           #liefert energy aus den Dateinamen der verwendeten dateien

    def pointmodifier(point):        #berechnet winkel aus qz und energy und ersetzt qz dadurch. Alle anderen Werte des Datenpunktes bleiben unveraendert
        point[1]=360.0/(2*numpy.pi)*numpy.arcsin( point[1]*simu.hcfactor/(2*point[0]))
        return point
    
    simu.ReadData("data",simu.createLinereader(energy_column=1,angle_column=0,rsigma_column=2,rpi_column=3),pointmodifierfunction=pointmodifier , filenamereaderfunction=namereader)

    b=pp.newParameter("background")
    m=pp.newParameter("multiplier")
    reflmodifier=lambda r, fitpararray: b.getValue(fitpararray) + r * m.getValue(fitpararray)
    simu.setModel(hs,reflmodifier)

    #pp.WriteToFile("tmp.txt")

    #fitting
    start, lower_limits, upper_limits=pp.getStartLowerUpper()
    
    simu.plotData(start)
    
    def costfunction(fitpararray):
        return simu.getSSR(fitpararray)
    best,ssr=Fitters.Evolution(costfunction,pp.getStartLowerUpper(), iterations=5, number_of_cores=3,mutation_strength=0.4,plotfunction=simu.plotData)
    
    simu.plotData(best)

    
    
    def rescost(fitpararray):
        return simu.getResidualsSSR(fitpararray)
    best, ssr = Fitters.Levenberg_Marquardt_Fitter(rescost,  ( best, lower_limits, upper_limits), 20 ,number_of_cores=3, strict=False, control_file=None)
    #best, ssr = Fitters.Levenberg_Marquardt_Fitter(rescost,  ( start, lower_limits, upper_limits), 20 ,number_of_cores=3, strict=False, control_file=None)
    
    simu.plotData(best)
    
    print "Best Parameters"
    i=0
    for name in pp.getNames():
        print str(i)+": "+ name + "=" +str(best[i])
        i+=1
    
    

    