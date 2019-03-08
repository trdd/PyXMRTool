
cdef extern from "math.h":
    double complex sqrt(double complex)  nogil
    double complex exp(double complex) nogil
    double cos(double) nogil
    double sin(double) nogil

from Mathematical_Functions_Reflectivity cimport *
from Structural cimport CLayer, Heterostructure
from Multilayer_Functions_Reflectivity cimport Calculate_Multilayer

cdef inline double complex CalculateVZpi_m(double vyvy, double complex cy, double complex cz, double complex cg):
    return sqrt((1.-vyvy/(1.+cz))*(1+cy)+cg*cg/(1+cz))

cdef void Relevant_Stuff_for_xmag_precisely(double complex cy1, double complex cy2, double complex cz1, double complex cz2, \
                                  double complex cg1, double complex cg2, double complex *vz1, double complex *vz2, \
                                  double vy, double vyvy, double k0, double sigma, int IsMagnetic1, int IsMagnetic2, \
                                  double complex *r, double complex *rp, double complex *t, double complex *tp)

cdef double complex LinDicParatt_Pi_xmag(Heterostructure* HS, double th, double wavelength)

cdef double complex LinDicParatt_Pi_xmag_MS(Heterostructure* HS, double th, double wavelength)
