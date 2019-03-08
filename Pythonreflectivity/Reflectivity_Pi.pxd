
cdef extern from "math.h":
    double complex sqrt(double complex)  nogil
    double complex exp(double complex) nogil
    double cos(double) nogil
    double sin(double) nogil

from Mathematical_Functions_Reflectivity cimport *
from Structural cimport CLayer, Heterostructure
from Multilayer_Functions_Reflectivity cimport Calculate_Multilayer


cdef inline double complex CalculateVZpi(double vyvy, double complex cy, double complex cz):
    return sqrt((1.-vyvy/(1.+cz))*(1+cy))

cdef double complex Calculate_rpi_precisely(double vyvy, double complex vz1, double complex vz2, double complex cy1,double complex cy2, double complex cz1, double complex cz2)

cdef double complex LinDicParatt_Pi(Heterostructure* HS, double th, double wavelength)

cdef double complex LinDicParatt_Pi_MS(Heterostructure* HS, double th, double wavelength)
