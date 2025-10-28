#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

namespace MatrixKernals
{
    __global__ void add(double* a, double* b, double* c, int m, int n);
    __global__ void add_broadcast_horizontal(double* a, double* b, double* c, int m, int n);
    __global__ void add_broadcast_vertical(double* a, double* b, double* c, int m, int n);

    __global__ void subtract(double* a, double* b, double* c, int m, int n);
    __global__ void subtract_broadcast_horizontal(double* a, double* b, double* c, int m, int n);
    __global__ void subtract_broadcast_vertical(double* a, double* b, double* c, int m, int n);

    __global__ void multiply(double* a, double* b, double* c, int a_m, int a_n, int b_m, int b_n);
    __global__ void hadamardProduct(double* a, double* b, double* c, int m, int n);
    __global__ void divide(double* a, double* b, double* c, int m, int n);
    
    __global__ void add(double* a, double num, double* b, int m, int n);
    __global__ void subtract(double* a, double num, double* b, int m, int n);
    __global__ void multiply(double* a, double num, double* b, int m, int n);
    __global__ void divide(double* a, double num, double* b, int m, int n);

    __global__ void transpose(double* a, double* b, int m, int n);

    __global__ void row(double* a, double* b, int row, int m, int n);
    __global__ void col(double* a, double* b, int col, int m, int n);

    __global__ void setup_random_states(curandState* state, unsigned long seed, int m, int n);
    __global__ void randomize_uniform(curandState* state, double* a, int m, int n, int min, int max);
    __global__ void randomize_normal(curandState* state, double* a, int m, int n);

    __global__ void cross_entropy(double* a, double* b, double* c, int m, int n);

    __global__ void sigmoid(double* a, double* b, int m, int n);
    __global__ void d_sigmoid(double* a, double* b, int m, int n);

    __global__ void tanh(double* a, double* b, int m, int n);
    __global__ void d_tanh(double* a, double* b, int m, int n);

    __global__ void relu(double* a, double* b, int m, int n);
    __global__ void d_relu(double* a, double* b, int m, int n);

    __global__ void log(double* a, double* b, int m, int n);

    __global__ void sum_vertical(double* a, double* b, int m, int n);
    __global__ void sum_horizontal(double* a, double* b, int m, int n);
}