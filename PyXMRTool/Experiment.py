#!/usr/bin/env python
"""Deals with the description of the experiment and brings experimental and simulated data together.
It contains currently only the class :class:`.ReflDataSimulator`, which does this job.
"""

#Python Version 2.7

__author__ = "Yannic Utz"
__copyright__ = ""
__credits__ = ["Yannic Utz and Martin Zwiebler"]
__license__ = "GNU General Public License v3.0"
__version__ = "0.9"
__maintainer__ = "Yannic Utz"
__email__ = "yannic.utz@tu-dresden.de"
__status__ = "beta"



import numbers
import os.path
import os
import numpy
import scipy.constants
import copy
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import Pythonreflectivity

import SampleRepresentation





class ReflDataSimulator(object):
    """Holds the experimental data, simulates it according to the settings and fitparameters and can directly deliver the sum of squared residuals (:meth:`.getSimData`) and the residuals themselfs (:meth:`.getResidualsSSR`), which both describe the difference between data and simulation at a certain parameter set. It can be in different modes which determins which data or which derived data is stored and simulated."""
    
    def __init__(self, mode, length_scale=1e-9):
        """
        Initialize the ReflDataSimulator with a certain mode.
        
        Parameters
        ----------
        mode : string
            The following modes are implemented so far:
                
            * \'l\'             - for linear polarized light, only reflectivity for sigma and pi polarization will be stored and simulated
            * \'c\'             - for circular polarized light, only reflectivity for left circular and right circular polarization will be stored and simulated
            * \'s\'             - only the sum of the reflectivities of left and right polarized light will be stored and simulated (contains only structural information)
            * \'x\'             - for xmcd, only the difference between the reflectivity for right circular and left circular polarization will be stored and simulated (contains only magnetic information). Actually, it is the normalized XMCD or asymmetry *(rleft-rright)/(rleft+rright)*.
            * \'cx<xfactor>\'   - for the reflections of circular pol. light and the xmcd signal (which should usually been calculated from the left and right circ. pol.) simultaniously \'<xfactor>\' is optional and can be used to multiply the xmcd signal with this value. This can be usefull to give the xmcd more or less weight during fitting e.g.\'cx20\' or \'cx0.1\'
            * \'lL\', \'cL\', \'sL\', \'xL\', \'cLx<xfactor>\', - as before, but instead of the corresponding reflectivities (or derived values) themselfs their logarithms are stored and simulated. This is usefull for fitting as with the logarithm the errors of different data points are weighted in a comparable way, in spite of the strongly decaying intensitiy for higher angles (see J.Pyhs.: Condens. Matter 26 (2014) 363201, page 16).
        length_scale : float
            Defines in which unit lengths are measured in your script. The unit is then **length_scale** * meters. Default is **length_scale** = *1e-9* which means *nm*. It is important to define it here due to conversion between energies and wavelength.
        """
        
        if not isinstance(mode,str):
            raise TypeError("\'mode\' has to be a string.")
        if not (mode=="l" or mode=="c" or mode=="s" or mode=="sL" or mode=="x" or mode[0:2]=="cx" or mode=="lL" or mode=="cL" or mode[0:3]=="cLx"):
            raise ValueError("\'mode\' can only take the values \'l\', \'lL\', \'c\', \'cL\', \'s\', \'sL\', \'x\', \'cx<xfactor>\' or \'cLx<xfactor>\'.")      
        if not isinstance(length_scale, numbers.Real):
            raise ValueError("\'length_scale\' has to be a real number.")
        
        
        if mode[0:2]=='cx':
            self._mode=mode[0:2]                                #save only the leading letters of the mode string
            if mode[2:]=='':
                self._xmcdfactor=1.0
            else:
                self._xmcdfactor=eval(mode[2:])               #save the xmcdfactor as an extra private property for later use
        elif mode[0:3]=='cLx':
            self._mode=mode[0:3]                                #save only the leading letters of the mode string
            if mode[3:]=='':
                self._xmcdfactor=1.0
            else:
                self._xmcdfactor=eval(mode[3:])               #save the xmcdfactor as an extra private property for later use
        else:
            self._mode=mode                                   #for the other modes the mode-string can stay as it is
        
        #set length scale. Has direct impact for the calculation of wavelengths from energies.
        self._lengthscale=length_scale                                                                  #need this property only for the methode setMode
        self._hcfactor=scipy.constants.physical_constants["Planck constant in eV s"][0]*scipy.constants.physical_constants["speed of light in vacuum"][0]/length_scale   
        
        
    #private methods
    
    def _getMode(self):
        if self._mode=='cx' or self._mode=='cLx':
            return self._mode+str(self._xmcdfactor)
        else:
            return self._mode
    
    
    def _getExpDataFlat(self):
        flatexpdata=[]
        if self._mode=="l" or self._mode=="lL" or self._mode=="c" or self._mode=="cL":
            for item in self._expdata:
                flatexpdata.extend(item[2])
                flatexpdata.extend(item[3])
        elif self._mode=="x" or self._mode=="s" or self._mode=="sL":
            for item in self._expdata:
                flatexpdata.extend(item[2])
        elif self._mode=='cx' or self._mode=='cLx' :
            for item in self._expdata:
                flatexpdata.extend(item[2])
                flatexpdata.extend(item[3])
                flatexpdata.extend(item[4])                
        return flatexpdata
    
    def _getSimDataFlat(self,fitpararray):
        """
        Return simulated data according to the bevor set-up model and the parameter values given with fitpararray as flat array.
        """
        # leave out parameter test, and test for existance of heterostructure to speed things up (this function will be called often in fit routines)
        flatsimdata=[]
        for item in self._expdata:
            #item[0] is the energy, item[1] is the list of angles at this energy
            singeE_HS=self._hs.getSingleEnergyStructure(fitpararray,item[0])      
            wavelength=self._hcfactor/item[0]
            rcalc=Pythonreflectivity.Reflectivity(singeE_HS, item[1],wavelength, Output="T", MultipleScattering=self._multiplescattering, MagneticCutoff=self._magneticcutoff)
            
            #if non-magnetic, Pythonreflectivity.Reflectivity delivers only pi and sigma polarization!
            #check for this case and get left and right circular as average of pi and sigma if necessary
            if len(rcalc)==2 and (self._mode=="c" or self._mode=="cL" or self._mode=="s" or self._mode=="sL"or self._mode=="x" or self._mode=="cx" or self._mode=="cLx"):
                average=(rcalc[0]+rcalc[1])/2.0
                circular=numpy.empty((2,len(average)))
                circular[0]=average         #left circular polarization
                circular[1]=average         #right circular polarization
                rcalc=numpy.append(rcalc,circular,0)           
                #dedug
                print "magnetic zero"
            
            
            if self._mode=="l":  #linear polarization
                flatsimdata.extend(self._reflmodifierfunction(rcalc[0],fitpararray))
                flatsimdata.extend(self._reflmodifierfunction(rcalc[1],fitpararray))
            elif self._mode=="lL":  #logarithm of linear polarization
                flatsimdata.extend(numpy.log(self._reflmodifierfunction(rcalc[0],fitpararray)))               #calculate logarithms of reflectivities
                flatsimdata.extend(numpy.log(self._reflmodifierfunction(rcalc[1],fitpararray)))
            elif self._mode=="c": #circular polarization
                flatsimdata.extend(self._reflmodifierfunction(rcalc[2],fitpararray))
                flatsimdata.extend(self._reflmodifierfunction(rcalc[3],fitpararray))
            elif self._mode=="cL": #logarithm of circular polarization
                flatsimdata.extend(numpy.log(self._reflmodifierfunction(rcalc[2],fitpararray)))                 #calculate logarithms of reflectivities
                flatsimdata.extend(numpy.log(self._reflmodifierfunction(rcalc[3],fitpararray)))
            elif self._mode=="s": # sum of circular polarizations
                flatsimdata.extend(self._reflmodifierfunction(rcalc[2],fitpararray)+self._reflmodifierfunction(rcalc[3],fitpararray))
            elif self._mode=="sL": #logarithm of sum of circular polarizations
                flatsimdata.extend(numpy.log(self._reflmodifierfunction(rcalc[2],fitpararray)+self._reflmodifierfunction(rcalc[3],fitpararray)))                 #calculate logarithms of sum of reflectivities
            elif self._mode=="x": #xmcd: normalized difference between circular polarizations. 
                rleft=self._reflmodifierfunction(rcalc[2],fitpararray)
                rright=self._reflmodifierfunction(rcalc[3],fitpararray)
                xmcd=(rleft-rright)/(rleft+rright)                                                         #calculate xmcd as (rleft-rright)/(rleft+rright). Does not follow any sign convention, if there is one.
                flatsimdata.extend(xmcd)                           
            elif self._mode=='cx': #circular polarizations and xmcd simultaniously and xmcd multiplied with a factor
                rleft=self._reflmodifierfunction(rcalc[2],fitpararray)
                rright=self._reflmodifierfunction(rcalc[3],fitpararray)
                xmcd=(rleft-rright)/(rleft+rright)                                                         #calculate xmcd as (rleft-rright)/(rleft+rright). Does not follow any sign convention, if there is one.
                flatsimdata.extend(rleft)
                flatsimdata.extend(rright)
                flatsimdata.extend(self._xmcdfactor*xmcd)                                   #here the simulated xmcd signal is multiplied with a user defined factor. This is usefull if you want to give more or less weight to the xmcd while fitting
            elif self._mode=='cLx': #logarithm of polarizations and xmcd simultaniously and xmcd multiplied with a factor
                rleft=self._reflmodifierfunction(rcalc[2],fitpararray)
                rright=self._reflmodifierfunction(rcalc[3],fitpararray)
                xmcd=(rleft-rright)/(rleft+rright)                                                         #calculate xmcd as (rleft-rright)/(rleft+rright). Does not follow any sign convention, if there is one.
                flatsimdata.extend(numpy.log(rleft))
                flatsimdata.extend(numpy.log(rright))
                flatsimdata.extend(self._xmcdfactor*xmcd)                                   #here the simulated xmcd signal is multiplied with a user defined factor. This is usefull if you want to give more or less weight to the xmcd while fitting
        return flatsimdata
            
    
    def _getHCFactor(self):
        return self._hcfactor
    
    
    #public methods
    
    def ReadData(self,files,linereaderfunction, energies=None, angles=None, filenamereaderfunction=None, pointmodifierfunction=None, headerlines=0, skipzeroreflectivities=True):
        """
        Read the data files and store the data corresponding to the **mode** specified with instanciation (see :meth:`ReflDataSimulator.__init__`)
        
        This function enables a very flexible reading of the data files.
        Logically, this function uses data points which consist of the independent variables energy and angle, and the reflectivities as dependent variables (rsigmag,rpi,rleft,rright,xmcd).
        So one point is specified by (energy,angle,rsigmag,rpi,rleft,rright,xmcd)  with energies in eV and angles in degrees.
        Where the values for the independent variables comes from can differ: either from lists (**energies**, **angles**), from the filenames (**filenamereaderfunction**) or from the lines in the data file (**linereaderfunction**).
        
        Parameters
        ---------
        files : str or list of str
            Specifies the set of data files. Either a list of filenames or one foldername of a folder containing all the data files (and only them!).
        linereaderfunction : callable
            A function given by the user which takes one line of an input file as string and returns a list/tuple of real numbers *(energy,angle,rsigma,rpi,rleft,rright,xmcd)*. Entries can also be \'None\'. Exceptions will only be trown if the needed information for the specified **mode** is not delivered. An easy way to create such a function is to use the method :meth:`.createLinereader`.
        energies : list of floats
            Only possible to be different from *None* if **files** is a list of filenames and **angles** is `None`. Gives the energies which belong to the corresponding files (same order) as floats.
        angles : list of floats
            Only possible to be different from *None* if **files** is a list of filenames and **energies** is `None`. Gives the angles which belong to the corresponding files (same order) as floats.
        filenamereaderfunction : callable
            A user-defined function which reads energies and/or angles from the filenames of the data files. This function should take a string (a filename without path), extract energy and/or angle out of it and return this as a tuple/list *(energy,angle)*. Both entries can also be set to *None*, but their will be an exception if the needed information for the data points can also not be obtained from the **linereaderfunction**.
        pointmodifierfunction : callable
            A user-definde function which is used to modify the obtained information. It takes the tuple/list of independent and dependent variables of a single data point and returns a modified one. It can be used for example if the data file contains qz values instead of angles. In this case you can read the qz values first as angles and replace them afterwards with the angles calculated out of it with the **pointmodifierfunction**. Of course you can also use a adopted **linereaderfunction** for this purpose (if all necessary information can be found in one line of the data files).
        headerlines : int
            specifies the number of lines which should be ignored at the top of each file.
        skipzeroreflectivities : bool
            Measured reflectivities should usually not be exactly zero. Sometimes zeros arise in a measurement file, when only one of the polarizations was measured at a certain angle-energy-combination. Usually this happens if this angle-energy-combination turns out not to be interesting after measuring one polarization and therefor the second one was not measured. For such cases you can set **skipzeroreflectivities** to *True* (default). Datapoints which contain reflectivities which are exactly zero will then be removed. If you don't want to have this behavior, set **skipzeroreflectivities** to *False*. But be aware that this can lead to problems. The program will try to fit these zero-values and if you use a logarithmic mode there will be errors.
        """
        
        #Parameter checking
        if not isinstance(files,(tuple,list,str)):
            raise TypeError("\'files\' has to be a list of filenames or a folder name.")
        if isinstance(files,(tuple,list)):
            for name in files:
                if not isinstance(name,str):
                    raise TypeError("Entries of \'files\' have to be strings.")
                elif not os.path.isfile(name):
                    raise ValueError("\'"+name+"\' (entry of \'files\' is not an existing file.")
        if isinstance(files,str):
            if not os.path.isdir(files):
                raise ValueError("\'"+name+"\' is not an existing directory.")
        if not callable(linereaderfunction):
            raise TypeError("\'linereaderfunction\' has to be callable.")
        if not (isinstance(energies,(list,tuple)) or energies is None):
            raise TypeError("\energies\' has to be a list of numbers.")
        if isinstance(energies,(tuple,list)):
            for en in energies:
                if not isinstance(en,numbers.Real):
                    raise TypeError("Entries of \'energies\' have to be real numbers.")
        if isinstance(angles,(tuple,list)):
            for an in angles:
                if not isinstance(an,numbers.Real):
                    raise TypeError("Entries of \'angles\' have to be real numbers.")
        if not (callable(filenamereaderfunction) or filenamereaderfunction is None):
            raise TypeError("\'filenamereaderfunction\' has to be callable.")
        if pointmodifierfunction is not None and not callable(pointmodifierfunction):
            raise TypeError("\'pointmodifierfunction\' has to be callable.")
        if not isinstance(headerlines,int):
            raise TypeError("\headerlines\' has to be an integer number.")
        if headerlines<0:
            raise ValueError("\headerlines\' has to be a positive number.")
        if (energies is not None and angles is not None and filenamereaderfunction is not None) or (energies is not None and angles is not None) or (energies is not None and filenamereaderfunction is not None)  or (angles is not None and filenamereaderfunction is not None):
            raise ValueError("Either use \'energies\', \'angles\' or \'filenamereaderfunction\' but not several of them.")
        if (energies is not None or angles is not None) and isinstance(files,str):
            raise ValueError("\'energies\' or \'angles\' can only be used if \'files\' is an array of filenames.")
        
        #store parameters for later use with setMode
        self._datafiles=files
        self._datalinereaderfunction=linereaderfunction
        self._dataenergies=energies
        self._dataangles=angles
        self._datafilenamereaderfunction=filenamereaderfunction
        self._datapointmodifierfunction=pointmodifierfunction
        self._dataheaderlines=headerlines
        
        
        #get filenames of files in directory
        if isinstance(files,str):
            files=[files+"/"+name for name in os.listdir(files)]
        
       
        #go trough all files
        datapoints=[]
        i=0
        file_energy=None
        file_angle=None
        for fname in files:
            if energies is not None:                                                    #if "file-wide" independent variables are defined set them here
                file_energy=energies[i]
            elif angles is not None:
                file_angles=angles[i]
            elif filenamereaderfunction is not None:
                file_energy,file_angle=filenamereaderfunction(os.path.basename(fname))      #give only the filename without the path to the filenamereaderfunction
            f=open(fname,'r')
            lines=f.readlines()[headerlines:]                                                #read file, skip headerlines, store as array "lines"
            f.close()
            for line in lines:
                output=linereaderfunction(line)
                if output is not None:
                    energy,angle,rsigma,rpi,rleft,rright,xmcd=output
                    if file_energy is not None:                                                  #overwrite energy and/or angle if file-wide energy and/or angle is given
                        energy=file_energy
                    if file_angle is not None:
                        angle=file_angle
                    if energy is None or angle is None:
                        raise Exception("Needed data not in line")
                    if pointmodifierfunction is not None:
                        energy,angle,rsigma,rpi,rleft,rright,xmcd=pointmodifierfunction([energy,angle,rsigma,rpi,rleft,rright,xmcd])   #apply pointmodifierfunction
                    if self._mode=='l':
                        if rsigma is None or rpi is None:
                            raise Exception("Needed data not in line")
                        if skipzeroreflectivities==True and (rsigma==0 or rpi==0):
                            print "  *** Zero reflectivity at energy="+str(energy)+ " and angle="+str(angle)+". Skiping datapoint."
                            print "      See option \'skipzeroreflectivities\' of \'PyXMRTool.Experiment.ReflDataSimulator.ReadData()\'. ***"
                        else:
                            datapoints.append([energy,angle,rsigma,rpi]) 
                    elif self._mode=='lL':
                        if rsigma is None or rpi is None:
                            raise Exception("Needed data not in line")
                        if skipzeroreflectivities==True and (rsigma==0 or rpi==0):
                            print "  *** Zero reflectivity at energy="+str(energy)+ " and angle="+str(angle)+". Skiping datapoint."
                            print "      See option \'skipzeroreflectivities\' of \'PyXMRTool.Experiment.ReflDataSimulator.ReadData()\'. ***"
                        else:
                            datapoints.append([energy,angle,numpy.log(rsigma),numpy.log(rpi)])                        #store logarithms of reflectivities
                    elif self._mode=='c':
                        if rleft is None or rright is None:
                            raise Exception("Needed data not in line")
                        if skipzeroreflectivities==True and (rleft==0 or rright==0):
                            print "  *** Zero reflectivity at energy="+str(energy)+ " and angle="+str(angle)+". Skiping datapoint."
                            print "      See option \'skipzeroreflectivities\' of \'PyXMRTool.Experiment.ReflDataSimulator.ReadData()\'. ***"
                        else:
                            datapoints.append([energy,angle,rleft,rright])                         
                    elif self._mode=='cL':
                        if rleft is None or rright is None:
                            raise Exception("Needed data not in line")
                        if skipzeroreflectivities==True and (rleft==0 or rright==0):
                            print "  *** Zero reflectivity at energy="+str(energy)+ " and angle="+str(angle)+". Skiping datapoint."
                            print "      See option \'skipzeroreflectivities\' of \'PyXMRTool.Experiment.ReflDataSimulator.ReadData()\'. ***"
                        else:
                            datapoints.append([energy,angle,numpy.log(rleft),numpy.log(rright)])                       #store logarithms of reflectivities
                    elif self._mode=='s':
                        if rleft is None or rright is None:
                            raise Exception("Needed data not in line")
                        if skipzeroreflectivities==True and (rleft==0 or rright==0):
                            print "  *** Zero reflectivity at energy="+str(energy)+ " and angle="+str(angle)+". Skiping datapoint."
                            print "      See option \'skipzeroreflectivities\' of \'PyXMRTool.Experiment.ReflDataSimulator.ReadData()\'. ***"
                        else:
                            datapoints.append([energy,angle,rleft+rright])                         
                    elif self._mode=='sL':
                        if rleft is None or rright is None:
                            raise Exception("Needed data not in line")
                        if skipzeroreflectivities==True and (rleft==0 or rright==0):
                            print "  *** Zero reflectivity at energy="+str(energy)+ " and angle="+str(angle)+". Skiping datapoint."
                            print "      See option \'skipzeroreflectivities\' of \'PyXMRTool.Experiment.ReflDataSimulator.ReadData()\'. ***"
                        else:
                            datapoints.append([energy,angle,numpy.log(rleft+rright)])                               #store logarithms of sum of reflectivities
                    elif self._mode=='x':
                        if xmcd is None:
                            raise Exception("Needed data not in line")
                        datapoints.append([energy,angle,xmcd]) 
                    elif self._mode=='cx':
                        if rleft is None or rright is None or xmcd is None:
                            raise Exception("Needed data not in line")
                        if skipzeroreflectivities==True and (rleft==0 or rright==0):
                            print "  *** Zero reflectivity at energy="+str(energy)+ " and angle="+str(angle)+". Skiping datapoint."
                            print "      See option \'skipzeroreflectivities\' of \'PyXMRTool.Experiment.ReflDataSimulator.ReadData()\'. ***"
                        else:
                            datapoints.append([energy,angle,rleft,rright,self._xmcdfactor*xmcd])                   #here the measured xmcd signal is multiplied with a user defined factor. This is usefull if you want to give more or less weight to the xmcd while fitting
                    elif self._mode=='cLx':
                        if rleft is None or rright is None or xmcd is None:
                            raise Exception("Needed data not in line")
                        if skipzeroreflectivities==True and (rleft==0 or rright==0):
                            print "  *** Zero reflectivity at energy="+str(energy)+ " and angle="+str(angle)+". Skiping datapoint."
                            print "      See option \'skipzeroreflectivities\' of \'PyXMRTool.Experiment.ReflDataSimulator.ReadData()\'. ***"
                        else:
                            datapoints.append([energy,angle,numpy.log(rleft),numpy.log(rright),self._xmcdfactor*xmcd])  #store logarithms of reflectivities
            i+=1
        
        # up to now there is an intermediate data structure: a list of complete datapoints eg. [ [energy1,angle1,rsigma1,rpi1], ...., [energyN,angleN,rsigmaN,rpiN]]
        # So there are many datapoints with the same energies
        
        #create now the data structure self._expdata for internal storage. It should be fast for delivering data belonging to single energies.
        #Therefore it looks like this:  self._expdata=[[energy1,[angle1,....angleN], [rsigma1, .... rsigmaN], [rpi1,...rpiN]], ...[energyL,[angle1,....angleK], [rsigma1, .... rsigmaK], [rpi1,...rpiK]] 
        #                          or:  self._expdata=[[energy1,[angle1,....angleN], [rleft1, .... rleftN], [rright1,...rrightN]], ...[energyL,[angle1,....angleK], [rleft1, .... rleftK], [rright1,...rrightK]] 
        #                          or:  self._expdata=[[energy1,[angle1,....angleN], [xmcd1, .... xmcdN]], ...[energyL,[angle1,....angleK], [xmcd1, .... xmcdK]] 
        #                          or:  self._expdata=[[energy1,[angle1,....angleN], [rleft1, .... rleftN], [rright1,...rrightN], [xmcd1, .... xmcdN]]], ...[energyL,[angle1,....angleK], [rleft1, .... rleftK], [rright1,...rrightK],[xmcd1, .... xmcdK]] 
        
        self._expdata=[]
        while(len(datapoints)>0):
            element=datapoints.pop(0)
            energy=element[0]
            single_energy_datapoints=[element]
            #fill single_energy_datapoints array
            while(1):
                try:
                    index=[item[0] for item in datapoints].index(energy)
                except:
                    break
                single_energy_datapoints.append(datapoints.pop(index))
            single_energy_datapoints=[item[1:] for item in single_energy_datapoints]        #remove energy
            single_energy_datapoints.sort(key=lambda item: item[0])                         #sort for increasing angles
            single_energy_datapoints=((numpy.array(single_energy_datapoints)).transpose()).tolist()                  #make a numpy array out of it,transpose it, and transform it again to a list of lists
            item=[energy]                                                                    #create one item for the list self._expdata
            item.extend(single_energy_datapoints)                                            #extend it, such that it looks like this [energyL,[angle1,....angleK], [rsigma1, .... rsigmaK], [rpi1,...rpiK]  or equivalent
            for ang in item[1]:                                                  #check if every angle occurs only once for this energy
                if item[1].count(ang)>1:
                    raise Exception("There is more than one datapoint with energy="+str(energy)+"eV and angle="+ str(ang)+"degrees.")
            self._expdata.append(item)
        self._expdata.sort(key=lambda item:item[0])                         #sort with ascending energy
        
        
      
    
    def setModel(self, heterostructure, reflmodifierfunction=None, MultipleScattering=True, MagneticCutoff=1e-50):
        """
        Set up the model for the simulation of the reflectivity data. 
        
        The simulation of the reflectivities is in prinicple done by using the information about the sample stored in **heterostructure** (of type :class:`SampleRepresentation.Heterostructure`).
        The calculated reflectivities are then given to the **reflmodifierfunction** (takes one number or numpy array and the fitpararray; returns one number or a numpy array). This function has to be defined 
        by the user and can be used for example to multiply the reflectivity by a global number and/or to add a common background. To make these numbers fittable, use the fitparameters registerd at an instance of :class:`Paramters.ParamterPool`.
        Example::
        
            pp=Paramters.ParameterPool("any_parameterfile")
            ...
            b=pp.NewParameter("background")
            m=pp.NewParameter("multiplier")
            reflmodifierfunction=lambda r, fitpararray: b.getValue(fitpararray) + r * m.getValue(fitpararray)
        
        and give this function to :meth:`.setModel`.
        
        BEWARE: The reflmodifierfunction is called very often during fitting procedures. Make it performant!
               
        With **MultipleScattering** you can switch on (*True*) and off (*False*) the simulation of multiple scattering. *False* is 20 percent faster. Default is *True*. Has no effect on calculations that require the full matrix.
        
        **MagneticCutoff**: If an off-diagonal element of chi (chi_g) fulfills abs(chi_g)<MagneticCutoff, it is set to zero. It defaults to 10e-50.
        
        The last two parameters are directly passed to :func:`Pythonreflectivity.Reflectivity`. See also the Documentation of :mod:`Pythonreflectivity`.
        """
        
        if not isinstance(heterostructure,SampleRepresentation.Heterostructure):
            raise TypeError("\'heterostructure\' must be of type \'SampleRepresentation.Heterostructure\'.")
        if reflmodifierfunction is not None and not callable(reflmodifierfunction):
            raise TypeError("\'reflmodifierfunction\' has to be callable.")
        if not isinstance(MultipleScattering, bool):
            raise TypeError("\'MultipleScattering\' has to be a boolean value.")
        if not isinstance(MagneticCutoff, numbers.Real):
            raise TypeError("MagneticCutoff has to be a real number.")            
        self._hs=heterostructure
        if reflmodifierfunction is None:                                            #if no reflmodifierfunction is given, set hier the "identity function" to avoid testing for None in getSimData
            self._reflmodifierfunction = lambda r,fitpararray: r
        else:
            self._reflmodifierfunction=reflmodifierfunction
        self._multiplescattering=MultipleScattering
        self._magneticcutoff=MagneticCutoff
        
    def getLenDataFlat(self):
        """
        Return length of the flat data representation. 
        
        It will be the number of measured data points times 2 for mode "l" and "c", only the number of measured data points for mode "x" and "s", and the number of measured data points times 3 for mode "cx"
        """
        return len(self._getExpDataFlat())
        
        
    def getSimData(self,fitpararray):
        """
        Return simulated data according to the bevor set-up model and the parameter values given with **fitpararray** (see also :mod:`Parameters`).
        
        The retured data is a list and has on of the following or similar shapes::
            
            [[energy1,[angle1,....angleN], [rsigma1, .... rsigmaN], [rpi1,...rpiN]], ...[energyL,[angle1,....angleK], [rsigma1, .... rsigmaK], [rpi1,...rpiK]] 
            [[energy1,[angle1,....angleN], [rleft1, .... rleftN], [rright1,...rrightN]], ...[energyL,[angle1,....angleK], [rleft1, .... rleftK], [rright1,...rrightK]] 
            [[energy1,[angle1,....angleN], [xmcd1, .... xmcdN]], ...[energyL,[angle1,....angleK], [xmcd1, .... xmcdK]] 
            [[energy1,[angle1,....angleN], [rleft1, .... rleftN], [rright1,...rrightN], [xmcd1, .... xmcdN]]], ...[energyL,[angle1,....angleK], [rleft1, .... rleftK], [rright1,...rrightK],[xmcd1, .... xmcdK]] 
        """
        # leave out parameter test, and test for existance of heterostructure to speed things up (this function will be called often in fit routines)
        simdata=copy.deepcopy(self._expdata)                                    #copy experimental data to get energies and angles
        simdata_flat=self._getSimDataFlat(fitpararray)
        #replace the copied experimental data with simulated values
        startindex=0
        if self._mode=="l" or self._mode=="lL" or self._mode=="c" or self._mode=="cL":
            for item in simdata:
                datalen=len(item[1])
                item[2]=simdata_flat[startindex:startindex+datalen]                                     #rsigma or rleft
                item[3]=simdata_flat[startindex+datalen:startindex+2*datalen]                           #rpi or rright
                startindex=startindex+2*datalen
        elif self._mode=="x" or self._mode=="s" or self._mode=="sL":
            for item in simdata:
                datalen=len(item[1])
                item[2]=simdata_flat[startindex:startindex+datalen]                                     #xmcd or sum of rleft and rright
                startindex=startindex+datalen
        elif self._mode=="cx" or self._mode=="cLx":
            for item in simdata:
                datalen=len(item[1])
                item[2]=simdata_flat[startindex:startindex+datalen]                                     #rleft
                item[3]=simdata_flat[startindex+datalen:startindex+2*datalen]                           #rright
                item[4]=simdata_flat[startindex+2*datalen:startindex+3*datalen]                         #xmcd
                startindex=startindex+3*datalen
        return simdata
    
    def getExpData(self):
        """
        Return stored experimental data.
        
        
        The retured data is a list and has on of the following or similar shapes::
            
            [[energy1,[angle1,....angleN], [rsigma1, .... rsigmaN], [rpi1,...rpiN]], ...[energyL,[angle1,....angleK], [rsigma1, .... rsigmaK], [rpi1,...rpiK]] 
            [[energy1,[angle1,....angleN], [rleft1, .... rleftN], [rright1,...rrightN]], ...[energyL,[angle1,....angleK], [rleft1, .... rleftK], [rright1,...rrightK]] 
            [[energy1,[angle1,....angleN], [xmcd1, .... xmcdN]], ...[energyL,[angle1,....angleK], [xmcd1, .... xmcdK]] 
            [[energy1,[angle1,....angleN], [rleft1, .... rleftN], [rright1,...rrightN], [xmcd1, .... xmcdN]]], ...[energyL,[angle1,....angleK], [rleft1, .... rleftK], [rright1,...rrightK],[xmcd1, .... xmcdK]] 
        """
        return self._expdata
    
    def getSSR(self,fitpararray):
        """
        Return sum of squared residuals as float according to the parameterset given by **fitpararray** (see also :mod:`Parameters`).
        """
        return numpy.sum( numpy.square( numpy.array(self._getSimDataFlat(fitpararray)) -  numpy.array(self._getExpDataFlat()) )  )
    
    def getResidualsSSR(self,fitpararray):
        """
        Return the residuals and the sum of squared residuals according to the parameterset given by **fitpararray** (see also :mod:`Parameters`).
        
        The information is returned as tuple: array of differences between simulated and measured data, sum of squared residuals.
        """
        residuals = numpy.array(self._getSimDataFlat(fitpararray)) - numpy.array(self._getExpDataFlat())
        ssr = numpy.sum( numpy.square( residuals  ))
        return residuals,ssr
    
    def plotData(self, fitpararray,simcolor='r',expcolor='b',simlabel='simulated',explabel='experimental'):
        """
        Plot simulated and experimental data.
        
        This function generates a plot at the first call and refreshes it if called again.
        
        Parameters
        ---------
            simcolor : str
                Specifies the color of the simulated data for the plotting with pyplot (see https://matplotlib.org/users/colors.html). Default is red.
            expcolor : str  are supposed to be strings which specify a color for the plotting with pyplot (see https://matplotlib.org/users/colors.html).
                Specifies the color of the experimental data for the plotting with pyplot (see https://matplotlib.org/users/colors.html). Default is blue.
            simlabel : str 
                Label shown in the legend of the plot for the simulated data. Default is *"simulated"*.
            explabel : str 
                Label shown in the legend of the plot for the experimental data. Default is *"experimental"*.
        """
        
        #check parameters
        if not isinstance(simcolor,str):
            raise TypeError("\'simcolor\' must be of string type.")
        if not isinstance(expcolor,str):
            raise TypeError("\'expcolor\' must be of string type.")
        if not isinstance(simlabel,str):
            raise TypeError("\'simlabel\' must be of string type.")
        if not isinstance(explabel,str):
            raise TypeError("\'explabel\' must be of string type.")
        
        
        
        #get data
        simdata=self.getSimData(fitpararray)
        expdata=self.getExpData()
        
        
       
           
        if hasattr(self,'_fig'):            #close previous figure
            plt.close(self._fig)
        
        #create figure and subplots
        if self._mode == "l" or self._mode == "lL" or self._mode == "c" or self._mode == "cL":      #linear and circular polarization (or logarithm of it)       
            self._fig = plt.figure(figsize=(10,5))
            self._ax1 = self._fig.add_subplot(121,projection='3d')                  
            self._ax2 = self._fig.add_subplot(122,projection='3d')                  
        elif self._mode == "x" or self._mode == "s" or self._mode == "sL":          #xmcd and sum of circular polarizations
            self._fig = plt.figure(figsize=(10,5))
            self._ax1 = self._fig.add_subplot(111,projection='3d')              #for xmcd
        elif self._mode == "cx" or self._mode == "cLx" :    #circular polarization and xmcd
            self._fig = plt.figure(figsize=(15,5))
            self._ax1 = self._fig.add_subplot(131,projection='3d')              #for left circular pol 
            self._ax2 = self._fig.add_subplot(132,projection='3d')              #for right circular pol
            self._ax3 = self._fig.add_subplot(133,projection='3d')              #for xmcd
        
                
        if self._mode == "l":            #linear polarization
            self._ax1.clear()
            self._ax2.clear()
            self._ax1.set_xlabel('angle')
            self._ax1.set_ylabel('energy')
            self._ax1.set_zlabel('sigma refl.')
            self._ax2.set_xlabel('theta')
            self._ax2.set_ylabel('energy')
            self._ax2.set_zlabel('pi refl.')
            for item in expdata:                                                            #go trough energies of experimental data
                self._ax1.plot(item[1],item[0]*numpy.ones(len(item[1])),item[2],color=expcolor)   #angles on the x axis, energies on the y axis and intensities on the z axis
                self._ax2.plot(item[1],item[0]*numpy.ones(len(item[1])),item[3],color=expcolor)   #angles on the x axis, energies on the y axis and intensities on the z axis
            for item in simdata:                                                            #go trough energies of simulated data
                self._ax1.plot(item[1],item[0]*numpy.ones(len(item[1])),item[2],color=simcolor)   #angles on the x axis, energies on the y axis and intensities on the z axis
                self._ax2.plot(item[1],item[0]*numpy.ones(len(item[1])),item[3],color=simcolor)   #angles on the x axis, energies on the y axis and intensities on the z axis
        
        elif self._mode == "lL":            #log of linear polarization
            self._ax1.clear()
            self._ax2.clear()
            self._ax1.set_xlabel('angle')
            self._ax1.set_ylabel('energy')
            self._ax1.set_zlabel('log( sigma refl. )')
            self._ax2.set_xlabel('theta')
            self._ax2.set_ylabel('energy')
            self._ax2.set_zlabel('log( pi refl. )')
            for item in expdata:                                                            #go trough energies of experimental data
                self._ax1.plot(item[1],item[0]*numpy.ones(len(item[1])),item[2],color=expcolor)   #angles on the x axis, energies on the y axis and log of intensities on the z axis
                self._ax2.plot(item[1],item[0]*numpy.ones(len(item[1])),item[3],color=expcolor)   #angles on the x axis, energies on the y axis and log of intensities on the z axis
            for item in simdata:                                                            #go trough energies of simulated data
                self._ax1.plot(item[1],item[0]*numpy.ones(len(item[1])),item[2],color=simcolor)   #angles on the x axis, energies on the y axis and log of intensities on the z axis
                self._ax2.plot(item[1],item[0]*numpy.ones(len(item[1])),item[3],color=simcolor)   #angles on the x axis, energies on the y axis and log of intensities on the z axis
        
        elif self._mode == "c":            #circular polarization
            self._ax1.clear()
            self._ax2.clear()
            self._ax1.set_xlabel('angle')
            self._ax1.set_ylabel('energy')
            self._ax1.set_zlabel('left circ. refl.')
            self._ax2.set_xlabel('theta')
            self._ax2.set_ylabel('energy')
            self._ax2.set_zlabel('right circ. refl.')
            for item in expdata:                                                            #go trough energies of experimental data
                self._ax1.plot(item[1],item[0]*numpy.ones(len(item[1])),item[2],color=expcolor)   #angles on the x axis, energies on the y axis and intensities on the z axis
                self._ax2.plot(item[1],item[0]*numpy.ones(len(item[1])),item[3],color=expcolor)   #angles on the x axis, energies on the y axis and intensities on the z axis
            for item in simdata:                                                            #go trough energies of simulated data
                self._ax1.plot(item[1],item[0]*numpy.ones(len(item[1])),item[2],color=simcolor)   #angles on the x axis, energies on the y axis and intensities on the z axis
                self._ax2.plot(item[1],item[0]*numpy.ones(len(item[1])),item[3],color=simcolor)   #angles on the x axis, energies on the y axis and intensities on the z axis
        
        elif self._mode == "cL":            #log of circular polarization
            self._ax1.clear()
            self._ax2.clear()
            self._ax1.set_xlabel('angle')
            self._ax1.set_ylabel('energy')
            self._ax1.set_zlabel('log( left circ. refl. )')
            self._ax2.set_xlabel('theta')
            self._ax2.set_ylabel('energy')
            self._ax2.set_zlabel('log( right circ. refl. )')
            for item in expdata:                                                            #go trough energies of experimental data
                self._ax1.plot(item[1],item[0]*numpy.ones(len(item[1])),item[2],color=expcolor)   #angles on the x axis, energies on the y axis and log of intensities on the z axis
                self._ax2.plot(item[1],item[0]*numpy.ones(len(item[1])),item[3],color=expcolor)   #angles on the x axis, energies on the y axis and log of intensities on the z axis
            for item in simdata:                                                            #go trough energies of simulated data
                self._ax1.plot(item[1],item[0]*numpy.ones(len(item[1])),item[2],color=simcolor)   #angles on the x axis, energies on the y axis and log of intensities on the z axis
                self._ax2.plot(item[1],item[0]*numpy.ones(len(item[1])),item[3],color=simcolor)   #angles on the x axis, energies on the y axis and log of intensities on the z axis
        
        elif self._mode == "s":            #sum of circular polarizations
            self._ax1.clear()
            self._ax1.set_xlabel('angle')
            self._ax1.set_ylabel('energy')
            self._ax1.set_zlabel('sum of circ. refl.')
            for item in expdata:                                                            #go trough energies of experimental data
                self._ax1.plot(item[1],item[0]*numpy.ones(len(item[1])),item[2],color=expcolor)   #angles on the x axis, energies on the y axis and xmcd on the z axis
            for item in simdata:                                                            #go trough energies of simulated data
                self._ax1.plot(item[1],item[0]*numpy.ones(len(item[1])),item[2],color=simcolor)   #angles on the x axis, energies on the y axis and xmcd on the z axis
                
        elif self._mode == "sL":            #log of sum of circular polarizations
            self._ax1.clear()
            self._ax1.set_xlabel('angle')
            self._ax1.set_ylabel('energy')
            self._ax1.set_zlabel('log( sum of circ. refl. )')
            for item in expdata:                                                            #go trough energies of experimental data
                self._ax1.plot(item[1],item[0]*numpy.ones(len(item[1])),item[2],color=expcolor)   #angles on the x axis, energies on the y axis and xmcd on the z axis
            for item in simdata:                                                            #go trough energies of simulated data
                self._ax1.plot(item[1],item[0]*numpy.ones(len(item[1])),item[2],color=simcolor)   #angles on the x axis, energies on the y axis and xmcd on the z axis
        
        elif self._mode == "x":            #xmcd#log of sum of circular polarizations
            self._ax1.clear()
            self._ax1.set_xlabel('angle')
            self._ax1.set_ylabel('energy')
            self._ax1.set_zlabel('xmcd')
            for item in expdata:                                                            #go trough energies of experimental data
                self._ax1.plot(item[1],item[0]*numpy.ones(len(item[1])),item[2],color=expcolor)   #angles on the x axis, energies on the y axis and xmcd on the z axis
            for item in simdata:                                                            #go trough energies of simulated data
                self._ax1.plot(item[1],item[0]*numpy.ones(len(item[1])),item[2],color=simcolor)   #angles on the x axis, energies on the y axis and xmcd on the z axis
                        
        elif self._mode == "cx":          #circular polarization and xmcd
            self._ax1.clear()
            self._ax2.clear()
            self._ax3.clear()
            self._ax1.set_xlabel('angle')
            self._ax1.set_ylabel('energy')
            self._ax1.set_zlabel('left circ. refl.')
            self._ax2.set_xlabel('theta')
            self._ax2.set_ylabel('energy')
            self._ax2.set_zlabel('right circ. refl.')
            self._ax3.set_xlabel('angle')
            self._ax3.set_ylabel('energy')
            if self._xmcdfactor==1:
                self._ax3.set_zlabel('xmcd')
            else:
                self._ax3.set_zlabel('xmcd * '+str(self._xmcdfactor))
            for item in expdata:                                                            #go trough energies of experimental data
                self._ax1.plot(item[1],item[0]*numpy.ones(len(item[1])),item[2],color=expcolor)   #angles on the x axis, energies on the y axis and intensities on the z axis
                self._ax2.plot(item[1],item[0]*numpy.ones(len(item[1])),item[3],color=expcolor)   #angles on the x axis, energies on the y axis and intensities on the z axis
                self._ax3.plot(item[1],item[0]*numpy.ones(len(item[1])),numpy.array(item[4]),color=expcolor)   #angles on the x axis, energies on the y axis and xmcd on the z axis
            for item in simdata:                                                            #go trough energies of simulated data
                self._ax1.plot(item[1],item[0]*numpy.ones(len(item[1])),item[2],color=simcolor)   #angles on the x axis, energies on the y axis and intensities on the z axis
                self._ax2.plot(item[1],item[0]*numpy.ones(len(item[1])),item[3],color=simcolor)   #angles on the x axis, energies on the y axis and intensities on the z axis
                self._ax3.plot(item[1],item[0]*numpy.ones(len(item[1])),numpy.array(item[4]),color=simcolor)   #angles on the x axis, energies on the y axis and xmcd on the z axis
        
        elif self._mode == "cLx":          #log of circular polarization and xmcd
            self._ax1.clear()
            self._ax2.clear()
            self._ax3.clear()
            self._ax1.set_xlabel('angle')
            self._ax1.set_ylabel('energy')
            self._ax1.set_zlabel('log( left circ. refl. )')
            self._ax2.set_xlabel('theta')
            self._ax2.set_ylabel('energy')
            self._ax2.set_zlabel('log( right circ. refl. )')
            self._ax3.set_xlabel('angle')
            self._ax3.set_ylabel('energy')
            if self._xmcdfactor==1:
                self._ax3.set_zlabel('xmcd')
            else:
                self._ax3.set_zlabel('xmcd * '+str(self._xmcdfactor))
            for item in expdata:                                                            #go trough energies of experimental data
                self._ax1.plot(item[1],item[0]*numpy.ones(len(item[1])),item[2],color=expcolor)   #angles on the x axis, energies on the y axis and intensities on the z axis
                self._ax2.plot(item[1],item[0]*numpy.ones(len(item[1])),item[3],color=expcolor)   #angles on the x axis, energies on the y axis and intensities on the z axis
                self._ax3.plot(item[1],item[0]*numpy.ones(len(item[1])),numpy.array(item[4]),color=expcolor)   #angles on the x axis, energies on the y axis and xmcd on the z axis
            for item in simdata:                                                            #go trough energies of simulated data
                self._ax1.plot(item[1],item[0]*numpy.ones(len(item[1])),item[2],color=simcolor)   #angles on the x axis, energies on the y axis and intensities on the z axis
                self._ax2.plot(item[1],item[0]*numpy.ones(len(item[1])),item[3],color=simcolor)   #angles on the x axis, energies on the y axis and intensities on the z axis
                self._ax3.plot(item[1],item[0]*numpy.ones(len(item[1])),numpy.array(item[4]),color=simcolor)   #angles on the x axis, energies on the y axis and xmcd on the z axis
        
        #create legend
        exp_patch = mpatches.Patch(color=expcolor,label=explabel)
        sim_patch = mpatches.Patch(color=simcolor,label=simlabel)
        plt.legend(handles=[exp_patch,sim_patch])
        
        plt.show(block=False)
        plt.pause(1)
    
    def setMode(self,mode):
        """Change the mode after instantiation.
           
        Be carefull with this function. Errors can occur if the mode does not fit to the available information in the data files.
           
        Parameters
        ----------
        mode : string
            The following modes are implemented so far:
                
            * \'l\'             - for linear polarized light, only reflectivity for sigma and pi polarization will be stored and simulated
            * \'c\'             - for circular polarized light, only reflectivity for left circular and right circular polarization will be stored and simulated
            * \'s\'             - only the sum of the reflectivities of left and right polarized light will be stored and simulated (contains only structural information)
            * \'x\'             - for xmcd, only the difference between the reflectivity for right circular and left circular polarization will be stored and simulated. Actually, it is the normalized XMCD or asymmetry *(rleft-rright)/(rleft+rright)*.
            * \'cx<xfactor>\'   - for the reflections of circular pol. light and the xmcd signal (which should usually been calculated from the left and right circ. pol.) simultaniously \'<xfactor>\' is optional and can be used to multiply the xmcd signal with this value. This can be usefull to give the xmcd more or less weight during fitting e.g.\'cx20\' or \'cx0.1\'
            * \'lL\', \'cL\', \'sL\', \'xL\', \'cLx<xfactor>\', - as before, but instead of the corresponding reflectivities themselfs their logarithms are stored and simulated. This is usefull for fitting as with the logarithm the errors of different data points are weighted in a comparable way, in spite of the strongly decaying intensitiy for higher angles (see J.Pyhs.: Condens. Matter 26 (2014) 363201, page 16).
        """
        
        #use the __init__ method to change mode to be sure to treat mode in the same way, even if changes occur in the futur
        self.__init__(mode,self._lengthscale)
        #read data again if already read
        if hasattr(self,"_expdata"):
            self.ReadData(self._datafiles,self._datalinereaderfunction, self._dataenergies, self._dataangles, self._datafilenamereaderfunction, self._datapointmodifierfunction, self._dataheaderlines)

        
               
        
    @staticmethod
    def createLinereader(energy_column=None,angle_column=None,rsigma_column=None,rpi_column=None,rleft_column=None,rright_column=None,xmcd_column=None,commentsymbol='#'):
        """
        Return a linereader function which can read lines from whitespace-seperated files and returns lists of real numbers *[energy,angle,rsigma,rpi,rleft,rright,xmcd]* (or *None* for a uncommented line).
        
        With the parameters *..._column* you can determin wich column is interpreted how.
        Column numbers are starting from 0.
        """
        #check parameters
        parameterlist=[energy_column,angle_column,rsigma_column,rpi_column,rleft_column,rright_column,xmcd_column]
        for item in parameterlist:
            if not (isinstance(item, int) or item is None):
                raise TypeError("Columns have to be given as integer numbers.")
            if item is not None:
                if item<0:
                    raise ValueError("Columns have to be positive numbers.")
        if not isinstance(commentsymbol,str):
            raise TypeError("\'commentsymbol\' has to be a string.")
        #define the linereader function
        def linereader(line):
                if not isinstance(line,str):
                    raise TypeError("\'line\' needs to be a string.")
                line=(line.split(commentsymbol))[0]                            #ignore everything behind the commentsymbol  #
                if not line.isspace():                               #ignore empty lines        
                    linearray=line.split()
                    linelist=[]
                    i=0
                    for item in parameterlist:
                        if item is None:
                            linelist.append(None)
                        else:
                            linelist.append(float(linearray[item]))
                    return linelist
                else:
                    return None
        return linereader
    
    
    #public properties
    mode = property(_getMode)
    """The current mode. See :meth:`.__init__` for possible modes. Read-only."""
    hcfactor = property(_getHCFactor)
    """Planck constant times the speed of light in units of *eV* times the unit of length which was defined by **length_scale** with :meth:`.__init__`. Read-only."""