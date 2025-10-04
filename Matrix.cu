#include <cuda_runtime.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "MatrixKernals.cuh"
#include "Matrix.cuh"

const int THREADS_PER_DIM = 16;

int seed = 0;

Matrix::Matrix(int m, int n) : m(m), n(n)
{
    cudaError_t err = cudaMallocManaged(&data, m * n * sizeof(double));
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    zero();
}

// TODO: make this better so we're not doing a deep copy for every = assignment
Matrix::Matrix(const Matrix& other) : m(other.m), n(other.n)
{
    cudaError_t err = cudaMallocManaged(&data, m * n * sizeof(double));
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    err = cudaMemcpy(data, other.data, m * n * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cudaFree(data);
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

Matrix::~Matrix()
{
    cudaFree(data);
}

int Matrix::rows() const
{
    return m;
}

int Matrix::cols() const
{
    return n;
}

Matrix Matrix::operator+(const Matrix& other) const
{
    // Set kernal parameters
    int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
    int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_x, blocks_y);
    
    Matrix result(m, n);

    if (m == other.m && n == other.n)
    {
        MatrixKernals::add<<<BLOCKS,THREADS>>>(data, other.data, result.data, m, n);
        cudaDeviceSynchronize();
    }
    else if (m == other.m && other.n == 1)
    {
        MatrixKernals::add_broadcast_horizontal<<<BLOCKS,THREADS>>>(data, other.data, result.data, m, n);
        cudaDeviceSynchronize();
    }
    else if (n == other.n && other.m == 1)
    {
        MatrixKernals::add_broadcast_vertical<<<BLOCKS,THREADS>>>(data, other.data, result.data, m, n);
        cudaDeviceSynchronize();
    }
    else 
    {
        std::ostringstream err;
        err << "Invalid dimensions: cannot add "
            << m << "x" << n
            << " matrix to "
            << other.m << "x" << other.n
            << " matrix.";
        throw std::invalid_argument(err.str()); 
    }

    return result;
}

Matrix Matrix::operator-(const Matrix& other) const
{
    // Set kernal parameters
    int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
    int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_x, blocks_y);
    
    Matrix result(m, n);

    if (m == other.m && n == other.n)
    {
        MatrixKernals::subtract<<<BLOCKS,THREADS>>>(data, other.data, result.data, m, n);
        cudaDeviceSynchronize();
    }
    else if (m == other.m && other.n == 1)
    {
        MatrixKernals::subtract_broadcast_horizontal<<<BLOCKS,THREADS>>>(data, other.data, result.data, m, n);
        cudaDeviceSynchronize();
    }
    else if (n == other.n && other.m == 1)
    {
        MatrixKernals::subtract_broadcast_vertical<<<BLOCKS,THREADS>>>(data, other.data, result.data, m, n);
        cudaDeviceSynchronize();
    }
    else 
    {
        std::ostringstream err;
        err << "Invalid dimensions: cannot subtract "
            << other.m << "x" << other.n
            << " matrix from "
            << m << "x" << n
            << " matrix.";
        throw std::invalid_argument(err.str()); 
    }

    return result;
}

Matrix Matrix::operator*(const Matrix& other) const
{
    if (m != other.m || n != other.n)
    {
        std::ostringstream err;
        err << "Invalid dimensions: cannot multiply "
            << m << "x" << n
            << " matrix with "
            << other.m << "x" << other.n
            << " matrix.";
        throw std::invalid_argument(err.str()); 
    }

    Matrix result(m, n);

    // Set kernal parameters
    int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
    int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_x, blocks_y);

    // Execute kernal
    MatrixKernals::multiply<<<BLOCKS,THREADS>>>(data, other.data, result.data, m, n);
    cudaDeviceSynchronize();

    return result;
}

Matrix Matrix::operator/(const Matrix& other) const
{
    if (m != other.m || n != other.n)
    {
        std::ostringstream err;
        err << "Invalid dimensions: cannot divide "
            << m << "x" << n
            << " matrix with "
            << other.m << "x" << other.n
            << " matrix.";
        throw std::invalid_argument(err.str()); 
    }

    Matrix result(m, n);

    // Set kernal parameters
    int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
    int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_x, blocks_y);

    // Execute kernal
    MatrixKernals::divide<<<BLOCKS,THREADS>>>(data, other.data, result.data, m, n);
    cudaDeviceSynchronize();

    return result;
}

Matrix Matrix::operator+(double num) const
{
    Matrix result(m, n);

    // Set kernal parameters
    int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
    int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_x, blocks_y);

    // Execute kernal
    MatrixKernals::add<<<BLOCKS,THREADS>>>(data, num, result.data, m, n);
    cudaDeviceSynchronize();

    return result;
}

Matrix Matrix::operator*(double num) const
{
    Matrix result(m, n);

    // Set kernal parameters
    int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
    int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_x, blocks_y);

    // Execute kernal
    MatrixKernals::multiply<<<BLOCKS,THREADS>>>(data, num, result.data, m, n);
    cudaDeviceSynchronize();

    return result;
}

Matrix Matrix::dot(const Matrix& other) const
{
    if (n != other.m)
    {
        std::ostringstream err;
        err << "Invalid dimensions: cannot dot "
            << m << "x" << n
            << " matrix with "
            << other.m << "x" << other.n
            << " matrix.";
        throw std::invalid_argument(err.str()); 
    }

    // Multpliplying an a_m x a_n matrix with an b_m x b_n matrix results in an a_m x b_n matrix.
    Matrix result(m, other.n);

    // Set kernal parameters
    int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
    int blocks_x = (other.n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_x, blocks_y);

    // Execute kernal
    MatrixKernals::dot<<<BLOCKS,THREADS>>>(data, other.data, result.data, m, n, other.m, other.n);
    cudaDeviceSynchronize();

    return result;
}

Matrix Matrix::transpose() const
{
    // The transpose of an MxN matrix is an NxM matrix
    Matrix result(n, m);

    // Set kernal parameters
    int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
    int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_x, blocks_y);

    // Execute kernal
    MatrixKernals::transpose<<<BLOCKS,THREADS>>>(data, result.data, m, n);
    cudaDeviceSynchronize();

    return result;
}

void Matrix::print()
{
    std::cout << std::fixed << std::setprecision(2);
    for (int i = 0; i < m * n; ++i)
    {
        std::cout << std::setw(5) << data[i] << "  ";
        if ((i+1) % n == 0)
        {
            std::cout << std::endl;
        }
    }
}

void Matrix::reshape(int m, int n)
{
    if (this->m * this->n != m * n)
    {
        std::ostringstream err;
        err << "Cannot reshape "
            << this->m << "x" << this->n
            << " to "
            << m << "x" << n
            << " matrix.";
        throw std::invalid_argument(err.str()); 
    }

    this->m = m;
    this->n = n;
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

    // Randomize values between 0 and 10
    MatrixKernals::randomize<<<BLOCKS,THREADS>>>(states, data, m, n, -1, 1);
    cudaDeviceSynchronize();

    cudaFree(states);
}

void Matrix::zero()
{
    cudaMemset(data, 0, m * n * sizeof(double));
}

// Static functions
Matrix Matrix::sigmoid(const Matrix& a)
{
    Matrix result(a.m, a.n);

    // Set kernal parameters
    int blocks_y = (a.m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
    int blocks_x = (a.n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_x, blocks_y);

    // Execute kernal
    MatrixKernals::sigmoid<<<BLOCKS,THREADS>>>(a.data, result.data, a.m, a.n);
    cudaDeviceSynchronize();

    return result;
}

Matrix Matrix::tanh(const Matrix& a)
{
    Matrix result(a.m, a.n);

    // Set kernal parameters
    int blocks_y = (a.m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
    int blocks_x = (a.n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_x, blocks_y);

    // Execute kernal
    MatrixKernals::tanh<<<BLOCKS,THREADS>>>(a.data, result.data, a.m, a.n);
    cudaDeviceSynchronize();

    return result;
}

double Matrix::sum(const Matrix& a)
{
    double sum = 0;
    for (int i = 0, n = a.m * a.n; i < n; ++i)
    {
        sum += a.data[i];
    }

    return sum;
}
