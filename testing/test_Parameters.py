import time
import Parameters

pp=Parameters.ParameterPool()
pp.ReadFromFile("partest.txt")

parh=pp.NewParameter("h",0,56,10,100)



pp.WriteToFile("partest2.txt")

(s,l,u)=pp.GetStartLowerUpper()



starttime=time.clock()
for i in range(1000):
    pp.NewParameter("real"+str(i),0,i*0.3,-5000,5000)
for i in range(1000):
    pp.NewParameter("complex"+str(i),0,i*0.3j,-5000*(1+1j),5000*(1+1j)) 
print "Creating paramters took "+str((time.clock()-starttime)/5000)+" per parameter."




pp.WriteToFile("partest3.txt")

starttime=time.clock()
(s,l,u)=pp.GetStartLowerUpper()

print "Reading took "+str(time.clock()-starttime)



array=[i*0.1 for i in range(3004)]

starttime=time.clock()
pp.SetStartValues(array)

print "Writing took "+str(time.clock()-starttime)


pp.WriteToFile("partest4.txt")

starttime=time.clock()
value=parh.getValue(array)

print "Reading value took "+str(time.clock()-starttime)+"s. Value is "+ str(value)


summe=4+parh

print "Summe ist: "+str(summe.getValue(array))