#include <curand_kernel.h>

namespace MatrixCommon
{
    __host__ void add(double* a, double* b, double* c, int a_m, int a_n, int b_m, int b_n);
    __host__ void subtract(double* a, double* b, double* c, int a_m, int a_n, int b_m, int b_n);
    __host__ void multiply(double* a, double* b, double* c, int a_m, int a_n, int b_m, int b_n);
    __host__ void hadamardProduct(double* a, double* b, double* c, int a_m, int a_n, int b_m, int b_n);
    __host__ void divide(double* a, double* b, double* c, int a_m, int a_n, int b_m, int b_n);

    __host__ void add(double* a, double num, double* b, int m, int n);
    __host__ void subtract(double* a, double num, double* b, int m, int n);
    __host__ void multiply(double* a, double num, double* b, int m, int n);
    __host__ void divide(double* a, double num, double* b, int m, int n);
    
    __host__ void transpose(double* a, double* b, int m, int n);
    __host__ void sum(double* a, double* b, int m, int n, int axis);

    __host__ void cross_entropy(double* a, double* b, double* c, int a_m, int a_n, int b_m, int b_n);
    
    __host__ void sigmoid(double* a, double* b, int m, int n);
    __host__ void d_sigmoid(double* a, double* b, int m, int n);
    __host__ void tanh(double* a, double* b, int m, int n);
    __host__ void d_tanh(double* a, double* b, int m, int n);
    __host__ void relu(double* a, double* b, int m, int n);
    __host__ void d_relu(double* a, double* b, int m, int n);
}