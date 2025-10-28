#include <cuda_runtime.h>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "Matrix.cuh"
#include "MatrixKernals.cuh"

const static int THREADS_PER_DIM = 16;

static int seed = 0;

Matrix::Matrix() : MatrixExpr(0, 0), data(nullptr) {}

Matrix::Matrix(int m, int n) : MatrixExpr(m, n)
{
    double* raw_ptr;
    cudaError_t err = cudaMallocManaged(&raw_ptr, m * n * sizeof(double));
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    auto cuda_deleter = [](double* p)
    {
        cudaFree(p);
    };

    data = std::shared_ptr<double>(raw_ptr, cuda_deleter);

    zero();
}

Matrix& Matrix::operator=(const Matrix& expr) = default;

double* Matrix::evaluate(const Matrix& result) const
{
    return data.get();
}

double* Matrix::evaluate(const Buffer& result) const
{
    return data.get();
}

bool Matrix::references(double* a) const
{
    return data.get() == a;
}

double Matrix::sum()
{
    double sum = 0;
    for (int i = 0, l = m * n; i < l; ++i)
    {
        sum += data.get()[i];
    }

    return sum;
}

int Matrix::rows()
{
    return m;
}

int Matrix::cols()
{
    return n;
}

void Matrix::set(int i, int j, double value)
{
    data.get()[i * n + j] = value;
}

double Matrix::get(int i, int j)
{
    return data.get()[i * n + j];
}

void Matrix::zero()
{
    cudaMemset(data.get(), 0, m * n * sizeof(double));
}

void Matrix::randomize()
{
    // Set kernal parameters
    int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
    int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_x, blocks_y);

    // Setup seeds
    curandState *states;
    cudaMalloc(&states, m * n * sizeof(curandState));
    MatrixKernals::setup_random_states<<<BLOCKS,THREADS>>>(states, seed++, m, n);
    cudaDeviceSynchronize();

    MatrixKernals::randomize_normal<<<BLOCKS,THREADS>>>(states, data.get(), m, n);
    cudaDeviceSynchronize();

    cudaFree(states);
}

void Matrix::randomize(int min, int max)
{
    // Set kernal parameters
    int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
    int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_x, blocks_y);

    // Setup seeds
    curandState *states;
    cudaMalloc(&states, m * n * sizeof(curandState));
    MatrixKernals::setup_random_states<<<BLOCKS,THREADS>>>(states, seed++, m, n);
    cudaDeviceSynchronize();

    // Randomize values between min and max
    MatrixKernals::randomize_uniform<<<BLOCKS,THREADS>>>(states, data.get(), m, n, min, max);
    cudaDeviceSynchronize();

    cudaFree(states);
}

void Matrix::print() const
{
    std::cout << std::fixed << std::setprecision(5);
    for (int i = 0; i < m * n; ++i)
    {
        std::cout << std::setw(10) << data.get()[i] << "  ";
        if ((i+1) % n == 0)
        {
            std::cout << std::endl;
        }
    }
}