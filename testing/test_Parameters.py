import time
import math

import sys
sys.path.append('../')       #this statement makes it possible for the script to find the PyXMRTool package relative to the Tutorials folder. For your own projects rather copy the PyXMRTool-Folder which contains the modules to your project folder which contains the script
from PyXMRTool import Parameters

pp=Parameters.ParameterPool("partest.txt")


parh=pp.newParameter("h",False,56,10,100)



#pp.writeToFile("partest2.txt")

(s,l,u)=pp.getStartLowerUpper()



starttime=time.clock()
for i in range(1000):
    pp.newParameter("real"+str(i),False,i*0.3,-5000,5000)
for i in range(1000):
    pp.newParameter("complex"+str(i),False,i*0.3j,-5000*(1+1j),5000*(1+1j)) 
print "Creating paramters took "+str((time.clock()-starttime)/5000)+" per parameter."




#pp.writeToFile("partest3.txt")

starttime=time.clock()
(s,l,u)=pp.getStartLowerUpper()

print "Reading took "+str(time.clock()-starttime)



fitpararray=[i*0.1 for i in range(3023)]

starttime=time.clock()
pp.setStartValues(fitpararray)

print "Writing took "+str(time.clock()-starttime)


#pp.writeToFile("partest4.txt")

starttime=time.clock()
value=parh.getValue(fitpararray)

print "Reading value took "+str(time.clock()-starttime)+"s. Value is "+ str(value)


summe=4+parh

print "Summe ist: "+str(summe.getValue(fitpararray))



def sinus(A,w,t): 
    return A*math.sin(w*t)

auslenkung=Parameters.DerivedParameter(sinus,summe, parh, pp.getParameter("real456"))

print "Die Auslenkung wird repraesentiert von: " + str(auslenkung)

print "Die Auslenkung ist: "+ str(auslenkung.getValue(fitpararray))

intensity=auslenkung**2

print "Die Intensitaet ist "+ str(intensity.getValue(fitpararray))


def sinus2(t,A,w): 
    return A*math.sin(w*t)


sinusfunktion=Parameters.ParametrizedFunction(sinus2, summe, parh)

print "Die Sinusfunktion wird repreasentiert von: " + str(sinusfunktion)

print "Der Funktionswert an der Stelle \'t=pi\' ist mit den angegebenen Parametern:  " + str(sinusfunktion.getValue(math.pi,fitpararray))
                                                           
sinus_fest = sinusfunktion.getFunction(fitpararray)

print "Der Funktionswert fuer \'t=pi'\ ist: " + str(sinus_fest(math.pi))

auslenkung_bei_pi = sinusfunktion.getParameter(math.pi)

print "Die parametrisierte Auslenkung an der festen Stelle \'t=pi\' ist mit den angegebenen Parametern:  " + str(auslenkung_bei_pi.getValue(fitpararray))