from Mathematical_Functions_Reflectivity cimport *
cdef extern from "math.h":
    double complex sqrt(double complex)  nogil
    double complex exp(double complex) nogil
    double cos(double) nogil
    double sin(double) nogil

from Structural cimport *
from Mathematical_Functions_Reflectivity cimport *
from Multilayer_Functions_Reflectivity cimport *
from Reflectivity_Sigma cimport *
from Reflectivity_Pi cimport *

cdef void Calculate_rt_z(rMemory *Mem1, rMemory *Mem2, double vy, double vyvy, double omvyvy, double complex chix1, double complex chiy1, double complex chiz1, double complex chig1, double complex chix2, double complex chiy2, double complex chiz2, double complex chig2, \
                    int IsMagnetic1, int IsMagnetic2, double complex (*r)[2][2], double complex (*rprime)[2][2], double complex (*t)[2][2], double complex (*tprime)[2][2], double sigma, double k0)

cdef void Fill_rMemory_z(rMemory *Mem, double vy, double vyvy, double omvyvy, double complex chix, double complex chiy, double complex chiz, double complex chig)

cdef void Paratt_magnetic_z(Heterostructure* HS, double th, double wavelength, double complex (*rtot)[2][2])

cdef void Paratt_magnetic_z_MS(Heterostructure* HS, double th, double wavelength, double complex (*rtot)[2][2])
