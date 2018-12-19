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


"""
In this file, only the function *chantler_linerreader()* is created. It will be used by every piece of code which uses the Chantler tables within this folder.
So if the tables change their format, the function has to be changed too.
The function will get one line of the text file and should return a tuple *(energy, formfactor)*. With *energy* is a real number and *formfactor* is a complex number.
If the line is commented out (with whatever symbol is chosen for that), the function should return *None*.

This file will not be used as module but just execute with *execfile()*. So be carefull. Whatever you state here will be in the namespace of the executing module!!!
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


chantler_commentsymbol='#'

def chantler_linereader(line):
    if not isinstance(line,str):
        raise TypeError("\'line\' needs to be a string.")
    line=(line.split(chantler_commentsymbol))[0]                            #ignore everything behind the commentsymbol  #
    if not line.isspace() and line:                               #ignore empty lines        
        linearray=line.split()
        if not len(linearray)==3:
            raise Exception("File for tabulated formfactor file has wrong format.")
        linearray=[ast.literal_eval(item) for item in linearray]
        return [linearray[0], linearray[1]+1j*linearray[2]]
    else:
        return None