cdef void FillC0(double complex (*C0)[2][2], double complex  (*rprime)[2][2], double complex (*rtot)[2][2],double complex  (*p)[2][2])


cdef void Calculate_ANXBN(double complex (*A)[2][2], double complex (*B)[2][2], double complex (*X)[2][2], int N)

cdef void Calculate_Multilayer_equation(double complex  (*A)[2][2], double complex  (*B)[2][2], double complex (*X)[2][2], double complex  (*result)[2][2], int N)

cdef void Calculate_Multilayer(double complex *t_comp1_up, double complex *t_comp2_up, double complex *t_comp1_do, double complex *t_comp2_do, double complex *r_ML_in1, double complex *r_ML_in2, double complex *r_ML_ba1, double complex *r_ML_ba2, int N)

cdef void Calculate_Multilayer_with_Matrices(double complex (*t_comp1_up)[2][2], double complex (*t_comp2_up)[2][2], double complex (*t_comp1_do)[2][2], double complex (*t_comp2_do)[2][2], double complex (*r_ML_in1)[2][2], double complex (*r_ML_in2)[2][2], double complex (*r_ML_ba1)[2][2], double complex (*r_ML_ba2)[2][2], int N)

cdef void Matrixexp( double complex (*A)[4][4], int N)
