#<PyXMRTool: A Python Package for the analysis of X-Ray Magnetic Reflectivity data measured on heterostructures>
#    Copyright (C) <2019>  <Yannic Utz>
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
    Retrive the Chantler Tables from the NIST Homepage automatically.
"""


import requests  
from lxml import html
import os.path
import datetime

commentsymbol='#'

NIST_URL="https://physics.nist.gov/cgi-bin/ffast/ffast.pl"

header_text= commentsymbol+" Form Factors from Chantler Tables\n"+commentsymbol+" DOI: https://dx.doi.org/10.18434/T4HS32\n" \
            +commentsymbol+" automatically retrived "+ datetime.date.today().isoformat()+ " using URL "+NIST_URL + "\n\n" \
            +commentsymbol+" For f_rel the second term, which is 3/5th of the Cromer-Liberman value, is used as suggested by Chantler.\n" \
            +commentsymbol+" Real and imaginary part of the formfactors are: f'(E)=f1(E)+f_rel+f_NT and f''(E)=f2(E)\n" \
            +commentsymbol+" see also original publication:  https://doi.org/10.1063/1.555974\n\n"




for i in range(92):             #looping through the peridic table
    Z=i+1
    pars={'Z' : str(Z), 'Formula' : '', 'gtype' : '4', 'range' : 'S', 'lower': '0.001', 'upper': '433', 'density': ''}
    r=requests.get(NIST_URL, params=pars)           #get html file
    tree=html.fromstring(r.content)                 #pars html file
    
    #extract Element Symbol
    bolds=tree.xpath("//b")
    elsymbol=str(bolds[0].text.split()[0])
    
    #extract f_rel and f_NT
    texts=tree.xpath("//text()")
    inds=[j for j, s in enumerate(texts) if "Relativistic correction estimate" in s]
    if len(inds)==0:
        raise Exception("Relativistic correction estimate not found.")
    elif len(inds)>1:
        raise Exception("Several entries Relativistic correction estimate found.")
    else:
        ind=inds[0]
        sub_element=texts[ind+3].getparent()
        if texts[ind+1]=='f' and texts[ind+2]=='rel' and sub_element.tag=='sub' and texts[ind+3].split()[0]=='(H82,3/5CL)':
            f_rel=float( texts[ind+3].split()[3] )                 #use the second value for f_rel (3/5CL), which is recommended, see original publication
        else:
            raise Exception("Relativistic correction estimate not found: Unexpected HTML structur.")
    inds=[j for j, s in enumerate(texts) if "Nuclear Thomson correction" in s]
    if len(inds)==0:
        raise Exception("Nuclear Thomson correction not found.")
    elif len(inds)>1:
        raise Exception("Several entries Nuclear Thomson correction found.")
    else:
        ind=inds[0]
        sub_element=texts[ind+3].getparent()
        if texts[ind+1]=='f' and texts[ind+2]=='NT' and sub_element.tag=='sub':
            f_NT=float( texts[ind+3].split()[1] )                 #use first value for f_rel (there is one alternative!!!)
        else:
            raise Exception("Nuclear Thomson correction not found: Unexpected HTML structur.")
    
    #check if header line fits my format expectation
    inds=[j for j, s in enumerate(texts) if "Form Factors" in s]
    ind=inds[-1]
    if not (texts[ind+5]=='E' and  texts[ind+7]=='f' and  texts[ind+8]=='1' and  texts[ind+10]=='f' and texts[ind+11]=='2'):
        raise Exception("Table header does not fit format expectation.")
    lines=texts[-1].split('\n') 
    del lines[0]   #first line is end of header
    energies=[]
    f1s=[]
    f2s=[]
    for line in lines:
        if not line=='': 
            line=line.split()
            energies.append(line[0])
            f1s.append(line[1])
            f2s.append(line[2])
            
    #save to file
    filename=elsymbol+".cff"
    if os.path.isfile(filename):
        answer=raw_input("Do you realy want to replace \'"+filename+"\'? (Y/n)")
        if not(answer=="Y" or answer=="y" or answer==''):
            continue            
    with open(filename,"w") as file:
        file.write(commentsymbol+" "+ elsymbol+"\n")
        file.write(header_text)
        file.write(commentsymbol+ " Z = "+ str(Z)+"\n")
        file.write(commentsymbol+ " f_rel = "+ str(f_rel)+" e/atom  (Relativistic correction estimate)\n")
        file.write(commentsymbol+ " f_NT = "+ str(f_NT)+" e/atom (Nuclear Thomson correction)\n")
        file.write("\n\n")
        
        file.write(commentsymbol+ " "+"E (eV)".center(20)+" "+"f1 (e/atom)".center(20)+ " " + "f2 (e/atom)".center(20)+"\n")
        for energy,f1,f2 in zip(energies,f1s,f2s):
            file.write("  "+str(float(energy)*1000).center(20)+" "+str(float(f1)).center(20)+ " " + str(float(f2)).center(20)+"\n")
        
        
    print "...got "+ elsymbol
    
    
    