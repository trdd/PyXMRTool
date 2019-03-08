
cdef extern from "math.h":
    double complex sqrt(double complex)  nogil
    double complex exp(double complex) nogil
    double cos(double) nogil
    double sin(double) nogil

from Mathematical_Functions_Reflectivity cimport *
from Structural cimport CLayer, Heterostructure
from Multilayer_Functions_Reflectivity cimport Calculate_Multilayer


cdef inline double complex CalculateVZsigma(double vyvy, double complex cx):
    return sqrt(1.+cx-vyvy)

cdef inline double complex Calculate_rsigma_precisely(double complex vz1, double complex vz2, double complex cx1, double complex cx2):
    return (cx1-cx2)/cquadr(vz1+vz2)


cdef double complex LinDicParatt_Sigma(Heterostructure* HS, double th, double wavelength)

cdef double complex LinDicParatt_Sigma_MS(Heterostructure* HS, double th, double wavelength)
