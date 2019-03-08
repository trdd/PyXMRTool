from libc.stdlib cimport malloc, free

cdef struct CLayer:
    double Thickness, Roughness
    double complex cx, cy, cz, cg
    double complex cxy, cyx, cxz, czx, cyz, czy
    int type, magdir



cdef struct Heterostructure:
    int NLayers
    int NLayers_types
    int *MLLENGTH
    int *MLREP
    int **MLCOMP
    CLayer *LR

cdef int FindLayerNumber(string, MaxLayer)

cdef int MakeConsistencyCheck(string, Heterostructure *STR, int MaxLayer)

cdef int FindComposition(string, int MaxLayer, int LayerNumber, Heterostructure *STR)


cdef struct rMemory:
    double complex cgcg
    double complex epsy
    double complex epsz
    double complex Delta31
    double complex C1
    double complex C2
    double complex C3
    double complex B1
    double complex B2
    double complex B
    double complex root
    double complex vz1
    double complex vz2
    double complex PHI1
    double complex PHI2
    double complex PHI3
    int IsFilled

cdef struct MatrixSafer:
    int IsFilled
    double complex exx
    double complex eyy
    double complex ezz

    double complex Mx
    double complex exyyx
    double complex exzzx
    double complex eyzzy
    double complex crossmag
    double complex summag
    double complex mixmag
    double complex inverseezz

    double complex D21ic
    double complex D23
    double complex D24ic

    double complex D31ic
    double complex D33ic

    double complex D41
    double complex D43
    double complex D44ic
