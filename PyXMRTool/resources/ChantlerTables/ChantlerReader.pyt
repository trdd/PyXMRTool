#<PyXMRTool: A Python Package for the analysis of X-Ray Magnetic Reflectivity data measured on heterostructures>
#    Copyright (C) <2018>  <Yannic Utz>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.



### In this file, only the function *chantler_linerreader()* is created.
### It will be used by every piece of code which uses the Chantler tables within this folder.
### So if the tables change their format, the function has to be changed too.
### The function will get one line of the text file and should return a tuple
### *(energy, formfactor)*. With *energy* is a real number and *formfactor* is a complex number.
### If the line is commented out (with whatever symbol is chosen for that), the function should return *None*.

### This file will not be used as module but just execute with *execfile()*. So be carefull. Whatever you state
### here will be in the namespace of the executing module!!!



#Python Version 3.6


__author__ = "Yannic Utz"
__copyright__ = ""
__credits__ = ["Yannic Utz and Martin Zwiebler"]
__license__ = "GNU General Public License v3.0"
__version__ = "0.9"
__maintainer__ = "Yannic Utz"
__email__ = "yannic.utz@tu-dresden.de"
__status__ = "beta" 


chantler_commentsymbol='#'

def chantler_linereader(line):
    """Return the tuple (energy, f1, f2) from one line of a Chantler Table.
       BEWARE: While the imaginary part of the formfactor Im(f)=f2, the real part contains also corrections
       Re(f)=f1+f_rel+f_NT   for the forward direction. See "Chantler, Journal fo Physical and Chemical Reference Data 24,71 (1995)" Eq.3 and following.
       f_rel and f_NT are small corrections for light atoms but get relevant with increasing mass.
       BEWARE: The sign convention differs from the one used within PyXMRTool. See also :doc:`/definitions/formfactors`. This linereader still delivers the "raw data" without sign conversion.
    """
    if not isinstance(line,str):
        raise TypeError("\'line\' needs to be a string.")
    line=(line.split(chantler_commentsymbol))[0]                            #ignore everything behind the commentsymbol  #
    if not line.isspace() and line:                               #ignore empty lines        
        linearray=line.split()
        if not len(linearray)==3:
            raise Exception("File for tabulated formfactor file has wrong format.")
        linearray=[ast.literal_eval(item) for item in linearray]
        return [linearray[0], linearray[1], linearray[2]]
    else:
        return None
    
def chantler_frel_reader(filename):
    """Read the relativistic correction to the real part of the formfactor from the chantler table given with **filename**.
       Re(f)=f1+f_rel+f_NT   for the forward direction. See "Chantler, Journal fo Physical and Chemical Reference Data 24,71 (1995)" Eq.3 and following.
       f_rel and f_NT are small corrections for light atoms but get relevant with increasing mass.
    """
    with open(filename) as file:
        for line in file:
            if "f_rel" in line and "=" in line and "(Relativistic correction estimate)" in line: 
                return float(line.split()[3])
    raise Exception("Relativistic correction estimate not found!!")

def chantler_fNT_reader(filename):
    """Read the nuclear Thomson correction to the real part of the formfactor from the chantler table given with **filename**.
       Re(f)=f1+f_rel+f_NT   for the forward direction. See "Chantler, Journal fo Physical and Chemical Reference Data 24,71 (1995)" Eq.3 and following.
       f_rel and f_NT are small corrections for light atoms but get relevant with increasing mass.
    """
    with open(filename) as file:
        for line in file:
            if "f_NT" in line and "=" in line and "(Nuclear Thomson correction)" in line: 
                return float(line.split()[3])
    raise Exception("Nuclear Thomson correction not found!!")
