#include <cuda_runtime.h>

#include "MatrixKernals.cuh"

namespace MatrixKernals
{
    __global__ void add(double* a, double* b, double* c, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = row * n + col;
        c[i] = a[i] + b[i]; 
    }
    
    __global__ void add_broadcast_horizontal(double* a, double* b, double* c, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = row * n + col;
        c[i] = a[i] + b[row]; 
    }
    
    __global__ void add_broadcast_vertical(double* a, double* b, double* c, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = row * n + col;
        c[i] = a[i] + b[col]; 
    }

    __global__ void subtract(double* a, double* b, double* c, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = row * n + col;
        c[i] = a[i] - b[i]; 
    }

    __global__ void subtract_broadcast_horizontal(double* a, double* b, double* c, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = row * n + col;
        c[i] = a[i] - b[row]; 
    }

    __global__ void subtract_broadcast_vertical(double* a, double* b, double* c, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = row * n + col;
        c[i] = a[i] - b[col]; 
    }

    __global__ void multiply(double* a, double* b, double* c, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = row * n + col;
        c[i] = a[i] * b[i]; 
    }

    __global__ void divide(double* a, double* b, double* c, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = row * n + col;
        c[i] = a[i] / b[i]; 
    }

    __global__ void dot(double* a, double* b, double* c, int a_m, int a_n, int b_m, int b_n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= a_m || col >= b_n)
        {
            return;
        }

        int idx = row * b_n + col; 
        for (int i = 0; i < a_n; ++i)
        {
            c[idx] += a[row * a_n + i] * b[i * b_n + col];
        }
    }
    
    __global__ void transpose(double* a, double* b, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        b[col * m + row] = a[row * n + col];
    }

    __global__ void add(double* a, double num, double* b, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = row * n + col;
        b[i] = a[i] + num; 
    }

    __global__ void multiply(double* a, double num, double* b, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = row * n + col;
        b[i] = a[i] * num; 
    }

    __global__ void sigmoid(double* a, double* b, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = row * n + col;
        b[i] = 1 / (1 + exp(-1 * a[i])); 
    }

    __global__ void tanh(double* a, double* b, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = row * n + col;
        b[i] = tanhf(a[i]);
    }
}