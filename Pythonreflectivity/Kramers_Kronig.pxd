cdef extern from "math.h":
    double log(double)  nogil

from Mathematical_Functions_Reflectivity cimport dabsvalue
from numpy cimport ndarray


cdef double KKT_Onepoint(  int j,  double *e,  double *im, int Lred )

cdef void Kramers_Kronig_Transformation_internal( ndarray[double, ndim=1, mode="c"] e, \
                                         ndarray[double, ndim=1, mode="c"] im,
                                         ndarray[double, ndim=1, mode="c"] re, int L)
