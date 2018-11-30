from time import time
from scipy import constants 
import math
import numpy as np
from matplotlib import pyplot as plt

import Pythonreflectivity

import sys
sys.path.append('../')       #this statement makes it possible for the script to find the PyXMRTool package relative to the Tutorials folder. For your own projects rather copy the PyXMRTool-Folder which contains the modules to your project folder which contains the script
from PyXMRTool import SampleRepresentation
from PyXMRTool import Parameters




pp=Parameters.ParameterPool("partest-samplerep.txt")


#one LayerObject

l=SampleRepresentation.LayerObject([pp.newParameter("chi_xx"),pp.newParameter("chi_xy"),pp.newParameter("chi_xz"),pp.newParameter("chi_yx"),pp.newParameter("chi_yy"),pp.newParameter("chi_yz"),pp.newParameter("chi_zx"),pp.newParameter("chi_zy"),pp.newParameter("chi_zz")],pp.newParameter("d"),Parameters.Parameter(0))




print "One Layer"
ar,lower,upper=pp.getStartLowerUpper()

print "Sigma: " +str(l.getSigma(ar))

print "D: "+str(l.getD(ar))

print "Chi: "+str(l.getChi(ar,456))

print "MagDir: " + l.getMagDir(ar)


#create complicated Heterostructure

print "hs=SampleRepresentation.Heterostructure(9,[0,1,2,[10,[3,4,5,6]],7,8,9,10,11,12])"
hs=SampleRepresentation.Heterostructure(13,[0,1,2,[10,[3,4,5,6]],7,8,9,10,11,12])


#populate Heterostructure with simple LayerObjects

for i in range(7):
    hs.setLayer(i,SampleRepresentation.LayerObject([pp.newParameter("chi"+str(i))],pp.newParameter("d"+str(i))))
hs.setLayer(7,l)


#hs.removeLayer([3,4,5,6])
#hs.removeLayer(7)

#print "hs.setLayout(3,[0,[10,[1,2]]])"
#hs.setLayout(3,[0,[10,[1,2]]])

#structur = hs.getSingleEnergyStructure(ar,35)


#create some formfactor object and register atoms
print "Creating formfactor objects..."
print "... FF_Co"
FF_Co=SampleRepresentation.FFfromFile("Co.F",SampleRepresentation.FFfromFile.createLinereader(complex_numbers=False))
print "... FF_Sr"
FF_Sr=SampleRepresentation.FFfromFile("Sr.F",SampleRepresentation.FFfromFile.createLinereader(complex_numbers=False))
#fake atom zum testen (sollte so eigentlich nicht genutzt werden)
#Die Dateien "C_imag.F" und "C_tabul.F" sind beide selbst gebastelt aus den Daten von "https://physics.nist.gov/PhysRefData/FFast/html/form.html" fuer Kohlenstoff.
#"C_Imag.F" enthaelt nur den Imaginaerteil des Formfaktors, dafuer aber als Matrix ausgebreitet (allerdings sind nur die Diagonaleintraege ungleich Null und alle gleich).
#"C_Theo.F" enthaelt Real- und Imaginaerteil des Formfaktors, aber jeweils nur als Skalar.
print "... FF_Ya"
FF_Ya=SampleRepresentation.FFfromScaledAbsorption(E1=250,E2=400,E3=500,scaling_factor=pp.newParameter("Ya_scaling"),tabulated_filename="C_tabul.F",absorption_filename="C_imag.F",energyshift=pp.newParameter("Ya_eneryshift"),tabulated_linereaderfunction=SampleRepresentation.FFfromScaledAbsorption.createTabulatedLinereader(complex_numbers=False),minE=150,maxE=650)
print "... MFF_Ya"
MFF_Ya=SampleRepresentation.MFFfromXMCD(pp.newParameter("theta_M"), pp.newParameter("phi_M"), "generated_xmcd.txt", minE=50, maxE=700, energyshift=pp.newParameter("MFF_energyshift"))

print "Register atoms"
SampleRepresentation.AtomLayerObject.registerAtom("Co",FF_Co)
SampleRepresentation.AtomLayerObject.registerAtom("Sr",FF_Sr)
SampleRepresentation.AtomLayerObject.registerAtom("Al",SampleRepresentation.FFfromFile("Al.F",SampleRepresentation.FFfromFile.createLinereader(complex_numbers=False)))
SampleRepresentation.AtomLayerObject.registerAtom("Yannicium", FF_Ya)
SampleRepresentation.AtomLayerObject.registerAtom("Yannicium_Magnetization", MFF_Ya)


#test ploting
print "Plot Formfactor FF_Ya"
FF_Ya.plotFF(ar,np.linspace(200,600,10000))
print "Plot Magnetic Formfactor MFF_Ya"
MFF_Ya.plotFF(ar,np.linspace(200,600,10000))

#test behavior of FF_Ya
print "Test behavior of FF_Ya"
print "...Therefore plot FF_Ya (real and imag) with scaling_factor=1, scaling_factor=3 and the stored theoretical/tabulated values."
en=np.arange(150,600)
ar[pp.getIndex("Ya_scaling")]=1  #setze scaling_factor auf 1
one=[]
for e in en:
    one.append(FF_Ya.getFF(e,ar)[0])
one=np.array(one)
ar[pp.getIndex("Ya_scaling")]=3  #setze scaling_factor auf 3
ten=[]
for e in en:
    ten.append(FF_Ya.getFF(e,ar)[0])
ten=np.array(ten)
tab=[]
for e in en:
    x=FF_Ya._tab_interpolator(e)
    tab.append(x[0]+x[1]*1j)
tab=np.array(tab)
fig, ax = plt.subplots()
ax.plot(en,one.real, label="scaling=1, real") 
ax.plot(en,one.imag, label="scaling=1, imag")
ax.plot(en,ten.real, label="scaling=3, real")
ax.plot(en,ten.imag, label="scaling=3, imag")
ax.plot(en,tab.real, label="tabulated, real")
ax.plot(en,tab.imag, label="tabulated, imag")
legend = ax.legend(loc='upper right', shadow=True)
plt.show()


#create one atom layer object with above registered atoms and put it into the heterostructure
print "Create atom layer object with registered atoms and put it into the heterostructure"
al1=SampleRepresentation.AtomLayerObject({"Sr":pp.newParameter("al1_density_Sr"), "Al":pp.newParameter("al1_density_Al"), "Yannicium":pp.newParameter("al1_density_Ya") , "Yannicium_Magnetization":pp.newParameter("al1_magnetization")}, pp.newParameter("al1_d"))
hs.setLayer(8,al1)
#add some more atom layer objects
print "add some more atom layer objects"
for i in range(9,13):
    hs.setLayer(i, SampleRepresentation.AtomLayerObject({"Sr":pp.newParameter("al"+str(i-7)+"_density_Sr"), "Al":pp.newParameter("al"+str(i-7)+"_density_Al"), "Yannicium":pp.newParameter("al"+str(i-7)+"_density_Ya"), "Yannicium_Magnetization":pp.newParameter("al"+str(i-7)+"_magnetization") }, pp.newParameter("al"+str(i-7)+"_d")))


#pp.writeToFile("partest-samplerep.txt")

print "Densitydict: " + str(al1.getDensitydict(ar))


x= al1.getChi(ar,300)
print "Chi of Atomlayer at energy 300 eV: " +str(x)



print hs._listoflayers
print hs.N
print hs._multilayer_structure


cmap=['yellow','magenta','black','b','green','red','grey','magenta']
#SampleRepresentation.plotAtomDensity(hs,ar,cmap,["Al","Sr"])
SampleRepresentation.plotAtomDensity(hs,ar,cmap)

#refl=Pythonreflectivity.Reflectivity(hs.getSingleEnergyStructure(ar,850),[1+1.0*i for i in range(90)], 2*math.pi*constants.hbar/constants.e*constants.c*10**9/850)