cdef extern from "math.h":
    double dsqrt "sqrt" (double) nogil

cdef double Cmaxnorm(double complex x)

cdef inline double complex cquadr(double complex x):
    return x*x

cdef inline double complex ccube(double complex x):
    return x*x*x

cdef inline double quadr(double x):
    return x*x
cdef inline double cabsquadr(double complex x):
    return quadr(x.real)+quadr(x.imag)

cdef inline double cabsvalue(double complex x):
    return dsqrt( quadr(x.real)+quadr(x.imag) )

cdef double dabsvalue(double x) nogil

cdef inline double complex cconj(double complex x):
    return x.real -1.j*x.imag

cdef inline double two_pi_div_360():
    return 0.017453292519943295


cdef double complex ipow(double complex base, int exp)

cdef void Mult2x2_rightside(double complex (*A)[2][2], double complex (*B)[2][2])

cdef void Mult2x2_leftside(double complex (*A)[2][2], double complex (*B)[2][2])

cdef void Elimination_4x4(double complex (*A)[4][4], double complex (*B)[4])

cdef void Invert2x2(double complex (*M)[2][2])

cdef double twobytwocondi(double complex (*Matr)[2][2] )

cdef void Normalize4vector( double complex (*vec)[4] )

cdef void Elimination2x2( double complex (*Matr)[2][2], double complex (*x)[2], double complex (*b)[2]  )

cdef void Matrix4Invert(double complex (*a)[4][4], double complex (*inva)[4][4] )

cdef void Mult4x4_leftside(double complex (*A)[4][4], double complex (*B)[4][4])

cdef void Mult4x4_leftside_diag(double complex (*A)[4][4], double complex (*B)[4])

cdef double complex Maxcomplex(double complex a, double complex b)
