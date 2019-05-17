import numpy

from PyXMRTool import Parameters
from PyXMRTool import SampleRepresentation






base_energies=numpy.arange(500,600,2)

def ff_imag(energy, coeffs,base_energies):
    """
    Model function for the imaginary part of formfactor (resp. one entry of the formfactor tensor.)
    It is sum of triangles with coefficients and base energies ("places where the triangles sit at").
    **base_energies** have to be sorted. **coeffs** is a list with same length as base energies and determines the hight of every triangle.
    (See Stone et al., PRB 86,024102 (2012) and Kuzmenko, Rev. Sci. Instrum. 76,083108 for details.)
    """
    base_energies=numpy.array(base_energies)
    coeffs=numpy.array(coeffs)
    #sum over basis functions (triangles) for imaginary part
    f_imag = 0
    for j in range(1,len(base_energies)-1):   #go throuth every index of base_energies but not the lowest and highest  
        if (base_energies[j-1] < energy) and (energy <= base_energies[j]) :
            f_basis_j = ( energy - base_energies[j-1] ) / ( base_energies[j] - base_energies[j-1] )
        elif (base_energies[j] < energy) and (energy < base_energies[j+1]) :
            f_basis_j = ( base_energies[j+1] - energy ) / ( base_energies[j+1] - base_energies[j] )
        else:
            f_basis_j = 0
        f_imag = f_imag+coeffs[j]*f_basis_j
    return f_imag

def ff_real(energy, coeffs,base_energies):
    """
    Model function for the real part of formfactor (resp. one entry of the formfactor tensor.)
    It is the Kramers-Kronig transform of the sum of the triangles.
    **base_energies** have to be sorted. **coeffs** is a list with same length as base energies and determines the hight of every triangle.
    Use the same lists **coeffs** and **base_energies** as in **f_imag**!!
     (See Stone et al., PRB 86,024102 (2012) and Kuzmenko, Rev. Sci. Instrum. 76,083108 for details.)
    """
    base_energies=numpy.array(base_energies)
    coeffs=numpy.array(coeffs)
    #helper function
    def g(x,y):
        return (x+y)*numpy.log(numpy.abs(x+y)) + (x-y)*numpy.log(numpy.abs(x-y))
    #sum over KK transformed basis functions for real part
    f_real = 0
    for j in range(1,len(base_energies)-1):   #go throuth every index of base_energies but not the lowest and highest 
        kk_f_basis_j= -1 /numpy.pi * ( g(energy,base_energies[j-1])/(base_energies[j]-base_energies[j-1]) - (base_energies[j+1]-base_energies[j-1])*g(energy,base_energies[j])/( (base_energies[j]-base_energies[j-1])*(base_energies[j+1]-base_energies[j]) ) + g(energy,base_energies[j+1])/(base_energies[j+1] -base_energies[j])  )
        f_real = f_real+coeffs[j]*kk_f_basis_j
    return f_real


base_energies=numpy.arange(500,600,2)



def ff_tensor(energy, *coeffs):
    """
    Model funtion for the formfactor tensor. Sum of ff_real and ff_imag on the diagonal elements.
    """
    ff=ff_real(energy,coeffs,base_energies) + 1j*ff_imag(energy,coeffs,base_energies)
    return numpy.array([ff,0,0,0,ff,0,0,0,ff])
    
    
#generate parameters as coefficients
pp=Parameters.ParameterPool()
suffix="ff_coeff_"
coeff_pars_list=[pp.newParameter(suffix+ str(i)) for i in range(len(base_energies))] 

#generate parametrized Function
ff_tensor_parfunc = Parameters.ParametrizedFunction(ff_tensor,*coeff_pars_list)

#generate Formfactor object with parametrized funtion
FF_Mn_fit = SampleRepresentation.FFfromFitableModel(ff_tensor_parfunc, min(base_energies), max(base_energies))

#generate parameter values for coeffients
fitpararray = [numpy.exp(-(energy-550)**2/(2*5)**2) for energy in base_energies]


FF_Mn_fit.plotFF(fitpararray)
