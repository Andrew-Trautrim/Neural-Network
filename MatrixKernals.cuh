#include <cuda_runtime.h>

namespace MatrixKernals
{
    __global__ void add(double* a, double* b, double* c, int m, int n);
    __global__ void add_broadcast_horizontal(double* a, double* b, double* c, int m, int n);
    __global__ void add_broadcast_vertical(double* a, double* b, double* c, int m, int n);

    __global__ void subtract(double* a, double* b, double* c, int m, int n);
    __global__ void subtract_broadcast_horizontal(double* a, double* b, double* c, int m, int n);
    __global__ void subtract_broadcast_vertical(double* a, double* b, double* c, int m, int n);

    __global__ void multiply(double* a, double* b, double* c, int m, int n);
    __global__ void divide(double* a, double* b, double* c, int m, int n);

    __global__ void dot(double* a, double* b, double* c, int a_m, int a_n, int b_m, int b_n);

    __global__ void transpose(double* a, double* b, int m, int n);
    
    __global__ void add(double* a, double num, double* b, int m, int n);
    __global__ void multiply(double* a, double num, double* b, int m, int n);

    __global__ void sigmoid(double* a, double* b, int m, int n);
    __global__ void tanh(double* a, double* b, int m, int n);
}