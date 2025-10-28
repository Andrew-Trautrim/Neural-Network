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
    
    __global__ void multiply(double* a, double* b, double* c, int a_m, int a_n, int b_m, int b_n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= a_m || col >= b_n)
        {
            return;
        }

        int idx = row * b_n + col;
        c[idx] = 0;
        for (int i = 0; i < a_n; ++i)
        {
            c[idx] += a[row * a_n + i] * b[i * b_n + col];
        }
    }

    __global__ void hadamardProduct(double* a, double* b, double* c, int m, int n)
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

    __global__ void subtract(double* a, double num, double* b, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = row * n + col;
        b[i] = a[i] - num; 
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

    __global__ void divide(double* a, double num, double* b, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = row * n + col;
        b[i] = a[i] / num; 
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

    __global__ void row(double* a, double* b, int row, int m, int n)
    {
        // calculate col for each thread
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (col >= n)
        {
            return;
        }

        b[col] = a[row * n + col];
    }

    __global__ void col(double* a, double* b, int col, int m, int n)
    {
        // calculate row for each thread
        int row = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m)
        {
            return;
        }

        b[row] = a[row * n + col];
    }
    
    __global__ void setup_random_states(curandState* state, unsigned long seed, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }
        
        int i = col * m + row;
        curand_init(seed, i, 0, &state[i]);
    }

    __global__ void randomize_uniform(curandState* state, double* a, int m, int n, int min, int max)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }
        
        int i = col * m + row;

        // get random number between min and max
        curandState localState = state[i];
        double r = (curand_uniform(&localState) * (max - min)) + min;
        state[i] = localState;

        a[i] = r;
    }

    __global__ void randomize_normal(curandState* state, double* a, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }
        
        int i = col * m + row;

        // get random number between min and max
        curandState localState = state[i];
        double r = curand_normal(&localState);
        state[i] = localState;

        a[i] = r;
    }

    __global__ void cross_entropy(double* a, double* b, double* c, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        // L(y_hat, y) = y * log(y_hat) + (1 - y) * log(1 - y_hat)
        int i = row * n + col;
        double a_capped = fmaxf(fminf(a[i], 1.0f - 1e-7f), 1e-7f); // make sure a[i] can't be too close to 0 or 1
        c[i] = -1 * (b[i] * logf(a_capped) + (1 - b[i]) * logf(1 - a_capped));
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

    __global__ void d_sigmoid(double* a, double* b, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = row * n + col;
        double s = 1 / (1 + exp(-1 * a[i]));
        b[i] =  s * (1 - s); // d/dx (sigmoid) = sigmoid * (1 - sigmoid) 
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

    __global__ void d_tanh(double* a, double* b, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = row * n + col;

        // d/dx (tan_h) = 1 - tan_h^2
        double tanh_x = tanhf(a[i]);
        b[i] = 1 - (tanh_x * tanh_x);
    }

    __global__ void relu(double* a, double* b, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = row * n + col;
        b[i] = a[i] > 0 ? a[i] : 0;
    }

    __global__ void d_relu(double* a, double* b, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = row * n + col;
        b[i] = a[i] > 0 ? 1 : 0;
    }

    __global__ void log(double* a, double* b, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = row * n + col;
        b[i] = logf(a[i]);
    }

    __global__ void sum_vertical(double* a, double* b, int m, int n)
    {
        // calculate col for each thread
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (col >= n)
        {
            return;
        }

        for (int i = 0; i < m; ++i)
        {
            b[col] += a[i * n + col];
        }
    }

    __global__ void sum_horizontal(double* a, double* b, int m, int n)
    {
        // calculate col for each thread
        int row = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m)
        {
            return;
        }

        for (int i = 0; i < n; ++i)
        {
            b[row] += a[row * n + i];
        }
    }
}