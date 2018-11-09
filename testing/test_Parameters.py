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



array=[i*0.1 for i in range(3023)]

starttime=time.clock()
pp.setStartValues(array)

print "Writing took "+str(time.clock()-starttime)


#pp.writeToFile("partest4.txt")

starttime=time.clock()
value=parh.getValue(array)

print "Reading value took "+str(time.clock()-starttime)+"s. Value is "+ str(value)


summe=4+parh

print "Summe ist: "+str(summe.getValue(array))



def sinus(A,w,t): 
    return A*math.sin(w*t)

auslenkung=Parameters.DerivedParameter(sinus,summe, parh, pp.getParameter("real456"))

print "Die Auslenkung wird repraesentiert von: " + str(auslenkung)

print "Die Auslenkung ist: "+ str(auslenkung.getValue(array))

intensity=auslenkung**2

print "Die Intensitaet ist "+ str(intensity.getValue(array))