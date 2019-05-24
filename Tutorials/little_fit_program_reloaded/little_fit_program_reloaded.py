#!/usr/bin/env python
"""Should do the same as Martins \'my_little_fitprogramm.py\' using PyXMRTool"""

#Python Version 2.7


import numpy
import scipy
import matplotlib
from matplotlib import pyplot

import Pythonreflectivity


from PyXMRTool import Parameters
from PyXMRTool import SampleRepresentation
from PyXMRTool import Experiment
from PyXMRTool import Fitters


#parameters
contribution_of_xmcd_to_fit=20
used_cores=3


#numbers
atom_density = 0.2      # mol per cubic centimeter
re=2.8179e-5            # Thomson scattering length in Angstroem 
hbar_times_c=scipy.constants.physical_constants["Planck constant over 2 pi in eV s"][0]*scipy.constants.physical_constants["speed of light in vacuum"][0]/1e-10         #ev times Angstroem

#helper functions
def Lorentzian(a,s,t,x):
    return a*s/(s*s+(t-x)**2 )




##This one here is important if you want to parallelize python in windows.
##This code will only be executed in the mother process.
if __name__ == '__main__':
    
    #####################################################################################
    #create a pool which handles all fitparameters (and connect it to a Parameter File)
    pp = Parameters.ParameterPool("parameters.txt")
    
    #####################################################################################
    ####  set up the MODEL for the sample 
    # it should consist of three layers: substrate, layer and cap
    # substrate has a fixed susceptibility, the layers suscept is energy-dependent and fitted and cap has a constant but fitted suscept. For further paramters see below.
    
    #create a Heterostructur object as sample representation, with 3 layers
    hs = SampleRepresentation.Heterostructure(3)
    
    ##substrate
    chi_substrate = Parameters.Parameter(-0.0005+0.0005j)                 #set fixed,isotropic and constant chi as instance of the Parameter class
    substrate_roughness = pp.newParameter("substrate_roughness")          #set substrate roughness as fittable parameter. Therefor create an instance of the Fitparameter class inside the ParameterPool pp
    substrate = SampleRepresentation.LayerObject(chitensor=[chi_substrate],sigma=substrate_roughness)  #create the layer object
    hs.setLayer(0,substrate)                                                                            #add layer to Heterostructure
    
    ##layer
    layer_roughness = pp.newParameter("layer_roughness")                  #set layer roughness as fittable parameter. Therefore create an instance of the Fitparameter class inside the ParameterPool pp
    layer_thickness = pp.newParameter("layer_thickness")                  #set layer thickness as fittable parameter. Therefore create an instance of the Fitparameter class inside the ParameterPool pp
    
    #model (magnetic) chi as Lorentzian absorption peak with fitparameters
    peak_height = pp.newParameter("peak_height")
    peak_width = pp.newParameter("peak_width")
    peak_position = pp.newParameter("peak_position")
    peak_step = pp.newParameter("peak_step")
    ffre_offset =pp.newParameter("ffre_offset")
    mag_height = pp.newParameter("mag_height")
    mag_width = pp.newParameter("mag_width")
    mag_position = pp.newParameter("mag_position")
    
    def layer_chi(energy, p_height, p_width, p_pos, p_step, ffre_os, m_height, m_width, m_pos ):
        en = numpy.linspace(600,750,301)                                    #define energy-range to model (from 600eV to 750eV in 301 steps), array of energies
        ffim = Lorentzian(p_height,p_width,p_pos,en) + p_step * ( numpy.arctan((en-p_pos)/p_width) + numpy.pi/2)      #model imaginary part of the formfactor as Lorentzian peak plus edge jump. FOR EXPLANATION of the strange constructions: You can perform arithmetic operations with Parameter type object. The result is another Parameter object. Here this is used to not evaluate only the result of Lorentzian and not every parameter individually.
        ffre = SampleRepresentation.KramersKronig(en,ffim)+ffre_os     #real part of formfactor as Kramers Kronig transformation of imag part + ffre_offset
        ff=ffre+ffim*1j
        ff_func = scipy.interpolate.interp1d(en,ff)                 #create function for formfactor as interpolation of the calculated values
        chi_diag= ff_func(energy)*atom_density*re*4*numpy.pi*energy**2/hbar_times_c**2
        #the same for magnetic contribution
        ffmag_im=Lorentzian(m_height,m_width,m_pos,en)
        ffmag_re=SampleRepresentation.KramersKronig(en,ffmag_im)
        ffmag=-1j*(ffmag_re+ffmag_im*1j)                                        #no idea why there is the factor -1j, Martin did it like this
        ffmag_func=scipy.interpolate.interp1d(en,ffmag)
        chi_mag=ffmag_func(energy)*atom_density*re*4*numpy.pi*energy**2/hbar_times_c**2
        return [chi_diag,chi_diag,chi_diag,chi_mag]                                         #return array for chi meaning a tensor with chi_diag on the diagonals and chi_mag and -chi_mag on two off-diagonal places depending on the magnetization direction
    
    layer_chi_pf =  Parameters.ParametrizedFunction(layer_chi, peak_height, peak_width, peak_position, peak_step, ffre_offset, mag_height, mag_width, mag_position)
    layer = SampleRepresentation.ModelChiLayerObject(layer_chi_pf, d=layer_thickness,sigma=layer_roughness,magdir="y")   #create the layer object  (this time a ModelChiLayer), magnetization points along "y"
    #---------
    
    hs.setLayer(1,layer)                                                                                         #add layer to Heterostructure
    
    ##cap
    cap_thickness=pp.newParameter("cap_thickness")
    cap_roughness=pp.newParameter("cap_roughness")
    cap_chi=(-0.0002+0.0002j)*pp.newParameter("cap_chi_factor")                                                        #set a fitable chi (Martin used this method with value and fitted factor, because he didn't have the oportunity to fit complex values, I just reproduce)
    #cap_chi=pp.newParameter("cap_chi_factor")                                                                         #variant with a complex fit parameter (to realy use a complex parameter, set complex start value and limits in the parameter file)
    cap = SampleRepresentation.LayerObject(chitensor=[cap_chi],d=cap_thickness,sigma=cap_roughness)                    #create the layer object     
    hs.setLayer(2,cap)                                                                                                 #add layer to Heterostructure
    
    
    #####################################################################################
    ####  set up the EXPERIMENT and how the data files are interpreted
    
    # create therefore the ReflDatSimulator object
    # set the mode to "cLx" with a xfactor given by contribution_of_xmcd_to_fit (meaning that the logarithm of circular polarized reflection
    # and xmcd are simultaniously handeled by the object and the xmcd is multiplied by the xfactor contribution_of_xmcd_to_fit, to make it more important for the fit)
    # length_scale=1e-9 means every length (thickness, wavelength, rouughness, ...) is measured in nm (this is also the default), for Angstroem you should enter length_scale=1e-10 
    simu = Experiment.ReflDataSimulator(mode="cLx"+str(contribution_of_xmcd_to_fit), length_scale=1e-10)      

    
    #define the function namereader which delivers the energy belonging to a certain data file from its name as tupel (energy, angle) where angle is set to None because there are different angles in every data file
    namereader = lambda string: (float(string[4:9]), None)           
    
    #create a function "linereader" which delivers angle,reflectivities and xmcd signal from one line of the data files
    #use for this the "ReflDataSimulator.createLinereader" method
    #you can also define a function by yourself. It has to take a string and give a tupel/list [energy,angle,rsigma,rpi,rleft,rright,xmcd]. Unused entries set to "None". 
    linereader=simu.createLinereader(angle_column=0,rleft_column=1,rright_column=2,xmcd_column=3)
    
    #read the experimental data from all files located in the folder "little_fit_program_reloaded_data" (instead you could also specify a list of filenames)
    #energies, anlges, reflectivities and xmcd signals are obtained from filename and the lines of the textfiles using namereader and linereader
    #in general the obtained information can afterwards be transformed with the pointmodifierfuntion (e.g. if files contain k_z instead of angles), but here it is set to "None" (which is also the default)
    simu.ReadData(files="data",linereaderfunction=linereader,pointmodifierfunction=None , filenamereaderfunction=namereader)
    
    #create a function "reflmodifier", it takes a reflectivity value and a list of fitparameter values (fitpararray) and gives back a modified reflectivity
    #it is used to modify the simulated reflectivities, here by multiplying a constant and adding a background, which are both varied while fitting
    b=pp.newParameter("background")
    m=pp.newParameter("multiplier")
    #reflmodifier=lambda r, fitpararray: b.getValue(fitpararray) + r * m.getValue(fitpararray)
    reflmodifier=lambda r, fitpararray: m.getValue(fitpararray)*( r + b.getValue(fitpararray) )                 #Martin used this background/multiplier combination instead of the above one
    
    
    #set the model for the experiment: connect the before defined Heterostructure model "hs" and the above created "reflmodifier" function to "simu"  
    simu.setModel(hs,reflmodifierfunction=reflmodifier)

    
    #####################################################################################
    ####  write TEMPLATE PARAMETER FILE
    #write a template parameter file containing all above defined parameters (so you dont have to do this by yourself)
    #this is usually done only once and commented out later
    
    #pp.writeToFile("parameters.txt")
    
    
    
    
    #####################################################################################
    #####  do the FITTING and PLOTTING
    
    start,lower_limits,upper_limits = pp.getStartLowerUpper()                   #get start values, lower and upper limits of the fit parameters as lists of values
    
    simu.plotData(start)                                                        #plot experimental data and simulated data according to the start values of the fit paramters
    
   
   
   
    #you can also use the evolutionary fitting algorithm (but usually it is better for searching good start values for the Levenberg_Marquardt_Fitter
    #it needs only a costfunction (usually sum of squared residuals), not the residuals
    def cost(fitpararray):
        return simu.getSSR(fitpararray)
    
    #best, ssr = Fitters.Evolution(cost, pp.getStartLowerUpper(), iterations=10, number_of_cores=3, generation_size=300, mutation_strength=0.01, elite=3, parent_percentage=0.25, control_file=None)# plotfunction=simu.plotData)
    #set best as new start values
    #pp.setStartValues(best)
   
    
   
   
   #the fit itself is done with the Levenberg_Marquardt_Fitter (there are also others).
    #It does not know about the data. It just varies the list of parameter values within the limits and gets the residuals and the sum of squared residuals
    #from the method "simu.getResidualsSSR". However there is an issue due to parallization, which does not allow to give this instance method directly to the fitter function.
    #Instead is has to be wrapped into another function "rescost" which is defined here. (Usual functions are ok instead of instance methods.)
    #plotfunction=simu.plotData leads to a plot of the current state after every iteration. Can also be set to \'None\' (default) if not needed.
    def rescost(fitpararray):
        return simu.getResidualsSSR(fitpararray)

    best, ssr = Fitters.Levenberg_Marquardt_Fitter(rescost,pp.getStartLowerUpper(), parallel_points=20 ,number_of_cores=used_cores, strict=False, control_file=None,plotfunction=simu.plotData)

    
    
        
    #plot experimental data and simulated data according to best fitted values
    simu.plotData(best)                                                         
    
        
    
    #output of fitted parameters and comparison with parameters which were originally used to create/simulate the data
    simulationparameters=numpy.array([261, 7, 651, 0.8, 32, 4, 649, -6, 1.2, 103, 15,2.5, 3.5, 4.1, 3.0e-7, 0.34])
    print("######")
    print("Result")
    print("######")
    print("parameter name".ljust(20)+"fitted parameters".ljust(30)+"original model parameters".ljust(30))
    i=0
    for item in best:
        print(pp.getNames()[i].ljust(20)+str(item).ljust(30)+str(simulationparameters[i]).ljust(30))
        i+=1
    print("######")
    
    #write fitted parameters to file (the fitted values are stored as 'start values')
    pp.writeToFile("parameter-output.txt",best) 
    
    #obtain and plot simulated reflectivity data instead of logarithm
    print("--> Plot result")
    simu.setMode("cx")
    simdata=simu.getSimData(best)
    simu.plotData(best)
    
    
    #plot obtained suszeptibility tensor of the magnetic layer
    energies=numpy.linspace(600,750,1000)
    chi_array=[]
    for e in energies:
        chi=layer_chi_pf.getValue(e,best)
        chi_array.append([chi[0],0,-chi[3],0,chi[1],0,chi[3],0,chi[2]])
    chi_array=numpy.array(chi_array)
    chi_array=numpy.transpose(chi_array)
    fig = matplotlib.pyplot.figure(figsize=(10,10))
    axes=[]
    for i in range(9):
        axes.append(fig.add_subplot(330+i+1))
    i=0
    for ax in axes:
        ax.set_xlabel('energy (eV)')
        ax.locator_params(axis='x', nbins=4)
        ax.plot(energies,chi_array[i].real)
        ax.plot(energies,chi_array[i].imag)
        i+=1
    matplotlib.pyplot.show()
        


    
    
    
    
    
    
    
    #########################################################
    # Comparison with Martins "My_little_fitprogramm
    #
    # Martins program converges to slightly different paramters (see below "aite"). The relative differences are of the order of 1e-5.
    # The reason is most probably that I use the residuals in the fitting procedure, whereas Martin uses directly the simulated reflectivity.
    # In principle, this should not make any difference, because the residuals are just simulated reflectivities substracted by the constant "measured" reflectivities.
    # But due to the approximative nature of floating point arithmetic, differences occur, which might lead to the slightly different convergence.
    # 
    # If I take Martins parameters in convergence (see below "aite"), I get exactly the same reflectivity values as long as I set the used value for the Boltzmann constant times c to the the same approximation as he does (hbar_times_c=1973.16).
    # Not only in this script here but also within the module "Experiment.py".
    # This proves that the above mentioned differences have nothing to do with bugs within the simulation.
    # The usage of more exact values given with the scipy package lead to a maximum relative differenc in the reflectivities of about 1e-3.
    # But the maximal relative difference of the parameters in convergence is still only 1e-4.
    
    #Vergleich mit Martins Reflektivitaets Werten
    #Fuer exakten Vergleich kopiere ich die hexadezimalen Werte als Strings hier her.
    #Martins gefittete Parameter
    aite_hex=['0x1.065b012fed054p+8', '0x1.c1b1405f8f3a4p+2', '0x1.457e71aec5864p+9', '0x1.98c2de016fa6cp-1', '0x1.002c4c9fe9859p+5', '0x1.fdfaae3872913p+1', '0x1.447e46796b9b8p+9', '-0x1.7902f2dab3f14p+2', '0x1.32c1da5eb8036p+0', '0x1.9c07a1fbb4c7dp+6', '0x1.df9d99fe6ba5cp+3', '0x1.3f7eed8e76457p+1', '0x1.bedc0166d6c04p+1', '0x1.04f85b29737cap+2', '0x1.42d56eb484ee9p-22', '0x1.5b61473543f67p-2']
    aite=numpy.array([float.fromhex(i) for i in aite_hex])
    #Reflectivitaet bei 650eV und rechts-polarisierten Roentgenstrahlen
    martin_hex=['0x1.96e19aaf32ccfp-17', '0x1.b9437b0629666p-19', '0x1.b513fc02ead81p-23', '0x1.524d1708f0973p-20', '0x1.f9da754b8c744p-19', '0x1.736caf5830899p-18', '0x1.7984bae21690fp-18', '0x1.20b8e0c3c7296p-18', '0x1.488594719bb5ap-19', '0x1.f0df2e338648dp-21', '0x1.851231ef25187p-23', '0x1.798c4051ec125p-23', '0x1.2fe5e6921698dp-21', '0x1.023d97ebb2781p-20', '0x1.2c38e5ffe7616p-20', '0x1.0af7dd2099c82p-20', '0x1.782dc90402104p-21', '0x1.a783bac9cec5ap-22', '0x1.9a943c8e49d76p-23', '0x1.0cc734016caddp-23', '0x1.5983727a880b9p-23', '0x1.f3e66c8a163c8p-23', '0x1.31a2e02e07b22p-22', '0x1.38b78783757e5p-22', '0x1.1548330290c58p-22', '0x1.bf246720d85e6p-23', '0x1.6047d88e78baep-23', '0x1.28d20fa0b913dp-23', '0x1.1920a24975389p-23', '0x1.20bdf0ceab9f0p-23', '0x1.2c52ff8533011p-23', '0x1.2fc073457add0p-23', '0x1.28f7005042290p-23', '0x1.1cd344682330ep-23', '0x1.11af648dbf2acp-23', '0x1.0b2ef2494ae3dp-23', '0x1.08fac46a859f7p-23', '0x1.081baec56883ap-23', '0x1.056db4d7cc131p-23', '0x1.ff20590f6dbd4p-24', '0x1.eee0d38fc0836p-24', '0x1.def8ca87de924p-24', '0x1.d460a22283fd2p-24', '0x1.d1ac4cd1df4d0p-24', '0x1.d60a66e4c2268p-24', '0x1.ddf1557faa07bp-24', '0x1.e4e063767756ep-24', '0x1.e745353e29fc1p-24', '0x1.e3b90abe11e1ap-24', '0x1.db37908148401p-24', '0x1.d06af39952ff3p-24', '0x1.c67bcb4363f58p-24', '0x1.bfe98235b4670p-24', '0x1.bdd06fe678f76p-24', '0x1.bfc6d4baf8759p-24', '0x1.c43dea121ef7cp-24', '0x1.c92ce40ac9d67p-24', '0x1.ccc105005784fp-24', '0x1.cddbd819eab3ep-24', '0x1.cc44669895338p-24', '0x1.c88d756e52cddp-24', '0x1.c3c838ed27710p-24', '0x1.bf242366bbbb7p-24', '0x1.bb9a0cfea86d2p-24', '0x1.b9b6b9f2cf119p-24', '0x1.b98bae8b0c2d7p-24', '0x1.bac1a601564ddp-24', '0x1.bcc096c25f1c6p-24', '0x1.bedebf7836848p-24', '0x1.c08a0816b4ceap-24', '0x1.c1637251be5eap-24', '0x1.c14995c6b00fbp-24', '0x1.c053b93e4cd18p-24', '0x1.bec21fcb4b232p-24', '0x1.bce9539dfcb50p-24', '0x1.bb1dd67530510p-24', '0x1.b9a43003dc68dp-24', '0x1.b8a770f33e893p-24', '0x1.b83673a1e5e83p-24', '0x1.b846c44ec7c7dp-24', '0x1.b8bb50dd0ccd9p-24', '0x1.b96cc9be7848bp-24', '0x1.ba31da0572fc9p-24', '0x1.bae5df2da19acp-24', '0x1.bb6d64f414427p-24', '0x1.bbb83ab4cfe9cp-24', '0x1.bbc1675bb09f2p-24', '0x1.bb8d856d1d098p-24', '0x1.bb282b22b620ep-24', '0x1.baa0fb3edbdbep-24', '0x1.ba08dd80bad36p-24', '0x1.b96fb52393416p-24', '0x1.b8e2c23f7c466p-24', '0x1.b86bb6f985e53p-24', '0x1.b8106ee0525a1p-24', '0x1.b7d3269dcbfa6p-24', '0x1.b7b30b6717816p-24', '0x1.b7acfa429cd14p-24', '0x1.b7bc4ce5064f2p-24', '0x1.b7db9a2e407c3p-24']
    martin=numpy.array([float.fromhex(i) for i in martin_hex])
    
    ich=numpy.array(simu.getSimData(aite)[10][3])
    
    #print (martin-ich)/(martin+ich)
    
    #simu.plotData(fitparameters)
