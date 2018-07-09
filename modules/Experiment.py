#!/usr/bin/env python
"""Deals the description of the experiment and brings experimental and simulated data together."""

#Python Version 2.7

__author__ = "Yannic Utz"
__copyright__ = ""
__credits__ = ["Yannic Utz and Martin Zwiebler"]
__license__ = ""
__version__ = ""
__maintainer__ = "Yannic Utz"
__email__ = "yannic.utz@tu-dresden.de"
__status__ = "Prototype"



import numbers
import os.path
import os
import numpy
import scipy.constants
import copy
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

import Pythonreflectivity

import SampleRepresentation





class ReflDataSimulator(object):
    """Holds the experimental data, simulates it according to the settings and fitparameters and can directly deliver chi_square (which measures the difference between the data and the simulation with a certain parameter set."""
    
    def __init__(self, mode, length_scale=10e-9):
        """
        Initialize the ReflDataSimulator with a certain mode.
        
        \'mode\' can be:    \'l\' - for linear polarized light, only reflectivity for sigma and pi polarization will be stored and simulated
                            \'c\' - for circular polarized light, only reflectivity for left circular and right circular polarization will be stored and simulated
                            \'x\' - for xmcd, only the difference between the reflectivity for right circular and left circular polarization will be stored and simulated
        
        
        Define with \'length_scale\' in which units lengths are measured in your script. The unit is then \'length_scale\'*meters. Standard is \'length_scale=10e-9\' which means nm.
        It is important to define it here due to conversion between energies and wavelength.
        """
        
        if not isinstance(mode,str):
            raise TypeError("\'mode\' has to be a string.")
        if not (mode=="l" or mode=="c" or mode=="x"):
            raise TypeError("\'mode\' can only take the values \'l\', \'c\' or \'x\'.")      
        if not isinstance(length_scale, numbers.Real):
            raise ValueError("\'length_scale\' has to be a real number.")
        self._mode=mode
        self._hcfactor=scipy.constants.physical_constants["Planck constant in eV s"][0]*scipy.constants.physical_constants["speed of light in vacuum"][0]/length_scale
        
        
    
    def _getMode_(self):
        return self._mode
    
    
    #public methods
    
    def ReadData(self,files,linereaderfkt, energies=None, angles=None, filenamereaderfkt=None, pointmodifierfkt=None, headerlines=0):
        """
        Read the data files and store the data corresponding to the \'mode\' specified with instanciation.
        
        This function enables a very flexible reading of the data files.
        Logically this function uses data point which consist of the independent variables energy and angle, and the reflectivities as dependent variables (rsigmag,rpi,rleft,rright,xmcd).
        So one point is specified by (energy,angle,rsigmag,rpi,rleft,rright,xmcd)  with energies in eV and angles in degrees.
        Where this information comes from can differ.
        
        At first, there are two different ways to specify the data files: Either a list of filenames (strings) or one foldername (string) of a folder containing all the data files (and only them!).
        
        One possibility is that all information comes from the \'linereaderfkt\'. This function can be defined by the user (or created with createLinereader()).
        It takes one line as a string and returns a list/tuple of real numbers (energy,angle,rsigma,rpi,rleft,rright,xmcd). Entries can also be \'None\'. The function will complain only if the needed information for \'mode\' is not delivered.
        
        Sometimes, not all the information on independent variables can be obtained from single lines of the file. To specify an independent variable which is valid for complete files there are 3 different possibilities, which cannot be mixed:
        Set the list \'energies\': Only possible if \'files\' is a list of filenames. Gives the energies which belong to the corresponding files (same order) as floats.
        Set the list \'angles\': Only possible if \'files\' is a list of filenames. Gives the angles which belong to the corresponding files (same order) as floats.
        Set \'filenamereaderfkt\': Give a user-specified function to the function \'ReadData()\'. It should take a string (a filename without path), extract energy and/or angle out of it and return this as a tuple/list (energy,angle). Both entries can also be set to \'None\', but their will be an exception if the information for the data points can also not be obtained from the linereaderfkt.
        
        
        With the parameter \'pointmodifierfkt\' you can hand over a functions which takes the list of independent and dependent variables of a single data point and returns a modified one.
        Can be used e.g. if the data file contains qz values instead of angles. The \'pointmodifierfkt\' can calculate the angles.
        Of course you can also use a adopted linereaderfkt for this (if all necessary information can be found in one line of the data files).
        
        \'headerlines\' specifies the number of lines which should be ignored in each file.
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
        if not callable(linereaderfkt):
            raise TypeError("\'linereaderfkt\' has to be callable.")
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
        if not (callable(filenamereaderfkt) or filenamereaderfkt is None):
            raise TypeError("\'filenamereaderfkt\' has to be callable.")
        if pointmodifierfkt is not None and not callable(pointmodifierfkt):
            raise TypeError("\'pointmodifierfkt\' has to be callable.")
        if not isinstance(headerlines,int):
            raise TypeError("\headerlines\' has to be an integer number.")
        if headerlines<0:
            raise ValueError("\headerlines\' has to be a positive number.")
        if (energies is not None and angles is not None and filenamereaderfkt is not None) or (energies is not None and angles is not None) or (energies is not None and filenamereaderfkt is not None)  or (angles is not None and filenamereaderfkt is not None):
            raise ValueError("Either use \'energies\', \'angles\' or \'filenamereaderfkt\' but not several of them.")
        if (energies is not None or angles is not None) and isinstance(files,str):
            raise ValueError("\'energies\' or \'angles\' can only be used if \'files\' is an array of filenames.")
        
        
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
            elif filenamereaderfkt is not None:
                file_energy,file_angle=filenamereaderfkt(os.path.basename(fname))      #give only the filename without the path to the filenamereaderfkt
            f=open(fname,'r')
            lines=f.readlines()[headerlines:]                                                #read file, skip headerlines, store as array "lines"
            f.close()
            for line in lines:
                output=linereaderfkt(line)
                if output is not None:
                    energy,angle,rsigma,rpi,rleft,rright,xmcd=output
                    if file_energy is not None:                                                  #overwrite energy and/or angle if file-wide energy and/or angle is given
                        energy=file_energy
                    if file_angle is not None:
                        angle=file_angle
                    if energy is None or angle is None:
                        raise Exception("Needed data not in line")
                    if pointmodifierfkt is not None:
                        energy,angle,rsigma,rpi,rleft,rright,xmcd=pointmodifierfkt([energy,angle,rsigma,rpi,rleft,rright,xmcd])   #apply pointmodifierfkt
                    if self.mode=='l':
                        if rsigma is None or rpi is None:
                            raise Exception("Needed data not in line")
                        datapoints.append([energy,angle,rsigma,rpi]) 
                    elif self.mode=='c':
                        if rleft is None or rright is None:
                            raise Exception("Needed data not in line")
                        datapoints.append([energy,angle,rleft,rright]) 
                    elif self.mode=='x':
                        if xmcd is None:
                            raise Exception("Needed data not in line")
                        datapoints.append([energy,angle,xmcd]) 
            i+=1
        
        #create now the data structure self._expdata for internal storage. It should be fast for delivering data belonging to single energies.
        #Therefore it looks like this:  self._expdata=[[energy1,[angle1,....angleN], [rsigma1, .... rsigmaN], [rpi1,...rpiN]], ...[energyL,[angle1,....angleK], [rsigma1, .... rsigmaK], [rpi1,...rpiK]] 
        #                          or:  self._expdata=[[energy1,[angle1,....angleN], [rleft1, .... rleftN], [rright1,...rrightN]], ...[energyL,[angle1,....angleK], [rleft1, .... rleftK], [rright1,...rrightK]] 
        #                          or:  self._expdata=[[energy1,[angle1,....angleN], [xmcd1, .... xmcdN]], ...[energyL,[angle1,....angleK], [xmcd1, .... xmcdK]] 
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
        
        
      
    def _getExpDataFlat_(self):
        flatexpdata=[]
        if self._mode=="l" or self._mode=="c":
            for item in self._expdata:
                flatexpdata.extend(item[2])
                flatexpdata.extend(item[3])
        elif self._mode=="x":
            for item in self._expdata:
                flatexpdata.extend(item[2])
        return flatexpdata
    
    def _getSimDataFlat_(self,fitpararray):
        """
        Return simulated data according to the bevor set-up model and the parameter values given with fitpararray as tuple of one or two numpy array (depending on mode) as flat array.
        """
        # leave out parameter test, and test for existance of heterostructure to speed things up (this function will be called often in fit routines)
        flatsimdata=[]
        for item in self._expdata:
            #item[0] is the energy, item[1] is the list of angles at this energy
            singeE_HS=self._hs.GetSingleEnergyStructure(fitpararray,item[0])
            rcalc=Pythonreflectivity.Reflectivity(singeE_HS, item[1], self._hcfactor/item[0], Output="T", MultipleScattering=self._multiplescattering, MagneticCutoff=self._magneticcutoff)
            if self._mode=="l":  #linear polarization
                flatsimdata.extend(self._reflmodifierfkt(rcalc[0],fitpararray))
                flatsimdata.extend(self._reflmodifierfkt(rcalc[1],fitpararray))
            elif self._mode=="c": #circular polarization
                flatsimdata.extend(self._reflmodifierfkt(rcalc[2],fitpararray))
                flatsimdata.extend(self._reflmodifierfkt(rcalc[3],fitpararray))
            elif self._mode=="x": #xmcd: difference between circular polarizations. 
                flatsimdata.extend(self._reflmodifierfkt(rcalc[2],fitpararray)-self._reflmodifierfkt(rcalc[3],fitpararray))                            #calculate xmcd as rleft-rright. Does not follow any sign convention, if there is one.
        return flatsimdata
            
    
    def _getHCFactor_(self):
        return self._hcfactor
    
    
    
    
    #public methods
    
    def setModel(self, heterostructure, reflmodifierfkt=None, MultipleScattering=True, MagneticCutoff=10e-50):
        """
        Set up the model for the simulation of the reflectivity data. 
        
        The simulation of the reflectivities is in prinicple done by using the information about the sample stored in heterostructure (of type SampleRepresentation.Heterostructure).
        The reflectivities calculated by this are then given to the \'reflmodifierfkt\' (takes one number or numpy array and the fitpararray; returns one number or numpy array). This funktion has to be defined 
        by the user and can be used e.g. to multiply the reflectivity by a global number and/or to add a common background. To make these numbers fittable, use the fitparameters registerd at the ParamterPool
        e.g
          pp=Paramters.ParameterPool("any_parameterfile")
          ...
          b=pp.NewParameter("background")
          m=pp.NewParameter("multiplier")
          reflmodifierfkt=lambda r, fitpararray: b.getValue(fitpararra) + r * m.getValue(fitpararray)
        and give this function to \'setMode\'.
        BEWARE: The reflmodifierfkt is called very often during fitting procedures. Make it performant!
        
        
        With \'MultipleScattering\' you can switch on (True) and off (False) the simulation of multiple scattering. False is 20 percent faster. Default is True. Has no effect on calculations that require the full matrix.
        
        MagneticCutoff: If an off-diagonal element of chi (chi_g) fulfills abs(chi_g)<MagneticCutoff, it is set to zero. It defaults to 10e-50.
        
        The last to parameters are directly passed to Pythonreflectivity.Reflectivity. See also the Documentation of Pythonreflectivity.
        """
        
        if not isinstance(heterostructure,SampleRepresentation.Heterostructure):
            raise TypeError("\'heterostructure\' must be of type \'SampleRepresentation.Heterostructure\'.")
        if reflmodifierfkt is not None and not callable(reflmodifierfkt):
            raise TypeError("\'reflmodifierfkt\' has to be callable.")
        if not isinstance(MultipleScattering, bool):
            raise TypeError("\'MultipleScattering\' has to be a boolean value.")
        if not isinstance(MagneticCutoff, numbers.Real):
            raise TypeError("MagneticCutoff has to be a real number.")            
        self._hs=heterostructure
        if reflmodifierfkt is None:                                            #if no reflmodifierfkt is given, set hier the "identity function" to avoid testing for None in getSimData
            self._reflmodifierfkt = lambda r,fitpararray: r
        else:
            self._reflmodifierfkt=reflmodifierfkt
        self._multiplescattering=MultipleScattering
        self._magneticcutoff=MagneticCutoff
        
    def getLenDataFlat(self):
        """
        Return length of the flat data representation. 
        
        It will be the number of measured data points times 2 for mode "l" and "c" and only the number of measured data points for mode "x"
        """
        return len(self._getExpDataFlat_())
        
        
    def getSimData(self,fitpararray):
        """
        Return simulated data according to the bevor set-up model and the parameter values given with fitpararray.
        """
        # leave out parameter test, and test for existance of heterostructure to speed things up (this function will be called often in fit routines)
        simdata=copy.deepcopy(self._expdata)
        simdata_flat=self._getSimDataFlat_(fitpararray)
        startindex=0
        if self._mode=="l" or self._mode=="c":
            for item in simdata:
                datalen=len(item[1])
                item[2]=simdata_flat[startindex:startindex+datalen]
                item[3]=simdata_flat[startindex+datalen:startindex+2*datalen]
                startindex=startindex+2*datalen
        elif self._mode=="x":
            for item in simdata:
                datalen=len(item[1])
                item[2]=simdata_flat[startindex:startindex+datalen]
                startindex=startindex+datalen
        return simdata
    
    def getExpData(self):
        """
        Return experimental data in the shape in which it is stored internally.
        """
        return self._expdata
    
    def getSSR(self,fitpararray):
        """
        Return sum of squared residuals.
        
        For some reason (I don't know why) Martin calculated the SSR for the evolutionary fit algorithm from the logarithm of the reflectivities. Use \'getSSRLog\' to do the same.
        """
        return numpy.sum( numpy.square( numpy.array(self._getSimDataFlat_(fitpararray)) -  numpy.array(self._getExpDataFlat_()) )  )
    
    def getSSRLog(self,fitpararray):
        """
        Return sum of squared residuals.
        
        For some reason (I don't know why) Martin calculated the SSR for the evolutionary fit algorithm from the logarithm of the reflectivities. I just do the same.
        Martin is calling this value "chi_square".
        """
        return numpy.sum( numpy.square( numpy.log(numpy.array(self._getSimDataFlat_(fitpararray))) -  numpy.log(numpy.array(self._getExpDataFlat_()) ))  )
    
    def getResidualsSSR(self,fitpararray):
        """
        Return tuple: array of differences between simulated and measured data, sum of squared residuals.
        """
        residuals = numpy.array(self._getSimDataFlat_(fitpararray)) - numpy.array(self._getExpDataFlat_())
        ssr = numpy.sum( numpy.square( residuals  ))
        return residuals,ssr
    
    def plotData(self, fitpararray,simcolor='r',expcolor='b'):
        """
        Plot simulated and experimental Data.
        
        \'simcolor\' and \'expcolor\' are supposed to be strings which specify a color for the plotting with pyplot (see https://matplotlib.org/users/colors.html).
        Defaults are red and blue.
        """
        simdata=self.getSimData(fitpararray)
        expdata=self.getExpData()
        
        if self.mode == "l":            #linear polarization
            fig = plt.figure(figsize=(20,10))
            ax1 = fig.add_subplot(121,projection='3d')                  #for sigma polarization
            ax2 = fig.add_subplot(122,projection='3d')                  #for pi polarization
            ax1.set_xlabel('angle')
            ax1.set_ylabel('energy')
            ax1.set_zlabel('log sigma pol. reflectivity')
            ax2.set_xlabel('theta')
            ax2.set_ylabel('energy')
            ax2.set_zlabel('log pi pol. reflectivity')
            for item in expdata:                                                            #go trough energies of experimental data
                ax1.plot(item[1],item[0]*numpy.ones(len(item[1])),numpy.log(item[2]),color=expcolor)   #angles on the x axis, energies on the y axis and logarithms of intensities on the z axis
                ax2.plot(item[1],item[0]*numpy.ones(len(item[1])),numpy.log(item[3]),color=expcolor)   #angles on the x axis, energies on the y axis and logarithms of intensities on the z axis
            for item in simdata:                                                            #go trough energies of simulated data
                ax1.plot(item[1],item[0]*numpy.ones(len(item[1])),numpy.log(item[2]),color=simcolor)   #angles on the x axis, energies on the y axis and logarithms of intensities on the z axis
                ax2.plot(item[1],item[0]*numpy.ones(len(item[1])),numpy.log(item[3]),color=simcolor)   #angles on the x axis, energies on the y axis and logarithms of intensities on the z axis
            
        elif self.mode == "c":          #circular polarization
            fig = plt.figure(figsize=(20,10))
            ax1 = fig.add_subplot(121,projection='3d')              #for left circular pol 
            ax2 = fig.add_subplot(122,projection='3d')              #for right circular pol
            ax1.set_xlabel('angle')
            ax1.set_ylabel('energy')
            ax1.set_zlabel('log left circular reflectivity')
            ax2.set_xlabel('theta')
            ax2.set_ylabel('energy')
            ax2.set_zlabel('log right circular reflectivity')
            for item in expdata:                                                            #go trough energies of experimental data
                ax1.plot(item[1],item[0]*numpy.ones(len(item[1])),numpy.log(item[2]),color=expcolor)   #angles on the x axis, energies on the y axis and logarithms of intensities on the z axis
                ax2.plot(item[1],item[0]*numpy.ones(len(item[1])),numpy.log(item[3]),color=expcolor)   #angles on the x axis, energies on the y axis and logarithms of intensities on the z axis
            for item in simdata:                                                            #go trough energies of simulated data
                ax1.plot(item[1],item[0]*numpy.ones(len(item[1])),numpy.log(item[2]),color=simcolor)   #angles on the x axis, energies on the y axis and logarithms of intensities on the z axis
                ax2.plot(item[1],item[0]*numpy.ones(len(item[1])),numpy.log(item[3]),color=simcolor)   #angles on the x axis, energies on the y axis and logarithms of intensities on the z axis
            
        elif self.mode == "x":          #xmcd
            fig = plt.figure(figsize=(10,10))
            ax1 = fig.add_subplot(111,projection='3d')
            ax1.set_xlabel('angle')
            ax1.set_ylabel('energy')
            ax1.set_zlabel('log xmcd')
            for item in expdata:                                                            #go trough energies of experimental data
                ax1.plot(item[1],item[0]*numpy.ones(len(item[1])),numpy.log(item[2]),color=expcolor)   #angles on the x axis, energies on the y axis and logarithms of intensities on the z axis
            for item in simdata:                                                            #go trough energies of simulated data
                ax1.plot(item[1],item[0]*numpy.ones(len(item[1])),numpy.log(item[2]),color=simcolor)   #angles on the x axis, energies on the y axis and logarithms of intensities on the z axis
        plt.show(block=False)
        
        
                
        
    @staticmethod
    def createLinereader(energy_column=None,angle_column=None,rsigma_column=None,rpi_column=None,rleft_column=None,rright_column=None,xmcd_column=None,commentsymbol='#'):
        """
        Return a linereader function which can read lines from whitespace-seperated files and returns lists of real numbers [energy,angle,rsigma,rpi,rleft,rright,xmcd] (or None).
        
        With the parameters \'..._column\' you can determin wich column is interpreted how.
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
    mode = property(_getMode_)
    hcfactor = property(_getHCFactor_)