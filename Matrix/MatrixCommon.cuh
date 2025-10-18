#include <curand_kernel.h>

namespace MatrixCommon
{
    __host__ void add(double* a, double* b, double* c, int a_m, int a_n, int b_m, int b_n);
    __host__ void subtract(double* a, double* b, double* c, int a_m, int a_n, int b_m, int b_n);
    __host__ void multiply(double* a, double* b, double* c, int a_m, int a_n, int b_m, int b_n);
    __host__ void divide(double* a, double* b, double* c, int a_m, int a_n, int b_m, int b_n);

    __host__ void add(double* a, double num, double* b, int m, int n);
    __host__ void subtract(double* a, double num, double* b, int m, int n);
    __host__ void multiply(double* a, double num, double* b, int m, int n);
    __host__ void divide(double* a, double num, double* b, int m, int n);
}