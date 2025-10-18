#include <cuda_runtime.h>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "Matrix.cuh"
#include "MatrixExpr.cuh"
#include "MatrixCommon.cuh"
#include "MatrixKernals.cuh"

const int THREADS_PER_DIM = 16;

int seed = 0;

Matrix::Matrix(int m, int n) : m(m), n(n)
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

Matrix::Matrix() : m(0), n(0), data(nullptr)
{
}

int Matrix::rows() const
{
    return m;
}

int Matrix::cols() const
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
        
Matrix::Matrix(const MatrixExpr& expr)
{
    m = expr.m;
    n = expr.n;

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

    expr.eval(data.get());
}

Matrix& Matrix::operator=(const MatrixExpr& expr)
{
    if (m * n != expr.m * expr.n)
    {
        double* raw_ptr;
        cudaError_t err = cudaMallocManaged(&raw_ptr, expr.m * expr.n * sizeof(double));
        if (err != cudaSuccess)
        {
            throw std::runtime_error(cudaGetErrorString(err));
        }

        auto cuda_deleter = [](double* p)
        {
            cudaFree(p);
        };

        data = std::shared_ptr<double>(raw_ptr, cuda_deleter);
    }

    m = expr.m;
    n = expr.n;

    expr.eval(data.get());

    return *this;
}

MatrixExpr Matrix::operator+(const Matrix& other) const
{
    std::function<void (double*)> expr = [this, other](double* result)
    {
        MatrixCommon::add(data.get(), other.data.get(), result, m, n, other.m, other.n);
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr Matrix::operator+(const MatrixExpr& other) const
{
    std::function<void (double*)> expr = [this, other](double* result)
    {
        double* accumulator = result;
        if (data.get() == result)
        {
            MatrixExpr::setBuffer(other, &accumulator);
        }

        other.eval(accumulator);

        MatrixCommon::add(data.get(), accumulator, result, m, n, other.m, other.n);
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr Matrix::operator-(const Matrix& other) const
{
    std::function<void (double*)> expr = [this, other](double* result)
    {
        MatrixCommon::subtract(data.get(), other.data.get(), result, m, n, other.m, other.n);
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr Matrix::operator-(const MatrixExpr& other) const
{
    std::function<void (double*)> expr = [this, other](double* result)
    {
        double* accumulator = result;
        if (data.get() == result)
        {
            MatrixExpr::setBuffer(other, &accumulator);
        }

        other.eval(accumulator);

        MatrixCommon::subtract(data.get(), accumulator, result, m, n, other.m, other.n);
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr Matrix::operator*(const Matrix& other) const
{
    std::function<void (double*)> expr = [this, other](double* result)
    {
        MatrixCommon::multiply(data.get(), other.data.get(), result, m, n, other.m, other.n);
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr Matrix::operator*(const MatrixExpr& other) const
{
    std::function<void (double*)> expr = [this, other](double* result)
    {
        double* accumulator = result;
        if (data.get() == result)
        {
            MatrixExpr::setBuffer(other, &accumulator);
        }

        other.eval(accumulator);

        MatrixCommon::multiply(data.get(), accumulator, result, m, n, other.m, other.n);
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr Matrix::operator/(const Matrix& other) const
{
    std::function<void (double*)> expr = [this, other](double* result)
    {
        MatrixCommon::divide(data.get(), other.data.get(), result, m, n, other.m, other.n);
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr Matrix::operator/(const MatrixExpr& other) const
{
    std::function<void (double*)> expr = [this, other](double* result)
    {
        double* accumulator = result;
        if (data.get() == result)
        {
            MatrixExpr::setBuffer(other, &accumulator);
        }

        other.eval(accumulator);

        MatrixCommon::divide(data.get(), accumulator, result, m, n, other.m, other.n);
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr Matrix::operator+(double num) const
{
    std::function<void (double*)> expr = [this, num](double* result)
    {
        MatrixCommon::add(data.get(), num, result, m, n);
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr Matrix::operator-(double num) const
{
    std::function<void (double*)> expr = [this, num](double* result)
    {
        MatrixCommon::subtract(data.get(), num, result, m, n);
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr Matrix::operator*(double num) const
{
    std::function<void (double*)> expr = [this, num](double* result)
    {
        MatrixCommon::multiply(data.get(), num, result, m, n);
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr Matrix::operator/(double num) const
{
    std::function<void (double*)> expr = [this, num](double* result)
    {
        MatrixCommon::divide(data.get(), num, result, m, n);
    };

    return MatrixExpr(m, n, expr);
}

// TODO
MatrixExpr Matrix::dot(const Matrix& other) const
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

    std::function<void (double*)> expr = [this, other](double* result)
    {
        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (other.n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::dot<<<BLOCKS,THREADS>>>(data.get(), other.data.get(), result, m, n, other.m, other.n);
        cudaDeviceSynchronize();
    };

    // Multpliplying an a_m x a_n matrix with an b_m x b_n matrix results in an a_m x b_n matrix.
    return MatrixExpr(m, other.n, expr);
}

MatrixExpr Matrix::transpose() const
{
    std::function<void (double*)> expr = [this](double* result)
    {
        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::transpose<<<BLOCKS,THREADS>>>(data.get(), result, m, n);
        cudaDeviceSynchronize();
    };

    // The transpose of an MxN matrix is an NxM matrix
    return MatrixExpr(n, m, expr);
}

MatrixExpr Matrix::row(int i) const
{
    if (i >= m)
    {
        std::ostringstream err;
        err << "Cannot get row " 
            << i << " of "
            << m << "x" << n
            << " matrix.";
        throw std::invalid_argument(err.str()); 
    }

    std::function<void (double*)> expr = [this, i](double* result)
    {
        // Set kernal parameters
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x);

        // Execute kernal
        MatrixKernals::row<<<BLOCKS,THREADS>>>(data.get(), result, i, m, n);
        cudaDeviceSynchronize();
    };

    return MatrixExpr(1, n, expr);
}

MatrixExpr Matrix::col(int i) const
{
    if (i >= n)
    {
        std::ostringstream err;
        err << "Cannot get column " 
            << i << " of "
            << m << "x" << n
            << " matrix.";
        throw std::invalid_argument(err.str()); 
    }

    std::function<void (double*)> expr = [this, i](double* result)
    {
        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_y);

        // Execute kernal
        MatrixKernals::col<<<BLOCKS,THREADS>>>(data.get(), result, i, m, n);
        cudaDeviceSynchronize();
    };

    return MatrixExpr(m, 1, expr);
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

void Matrix::zero()
{
    cudaMemset(data.get(), 0, m * n * sizeof(double));
}

// Static functions
MatrixExpr Matrix::cross_entropy(const Matrix& a, const Matrix& b)
{
    if (a.m != b.m || a.n != b.n)
    {
        std::ostringstream err;
        err << "Invalid dimensions: cannot compute cross entropy of "
            << a.m << "x" << a.n
            << " matrix and "
            << b.m << "x" << b.n
            << " matrix.";
        throw std::invalid_argument(err.str()); 
    }

    std::function<void (double*)> expr = [a, b](double* result)
    {
        // Set kernal parameters
        int blocks_y = (a.m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (a.n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::cross_entropy<<<BLOCKS,THREADS>>>(a.data.get(), b.data.get(), result, a.m, a.n);
        cudaDeviceSynchronize();
    };

    return MatrixExpr(a.m, a.n, expr);
}

MatrixExpr Matrix::sigmoid(const Matrix& a)
{
    std::function<void (double*)> expr = [a](double* result)
    {
        // Set kernal parameters
        int blocks_y = (a.m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (a.n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::sigmoid<<<BLOCKS,THREADS>>>(a.data.get(), result, a.m, a.n);
        cudaDeviceSynchronize();
    };

    return MatrixExpr(a.m, a.n, expr);
}

MatrixExpr Matrix::d_sigmoid(const Matrix& a)
{
    std::function<void (double*)> expr = [a](double* result)
    {
        // Set kernal parameters
        int blocks_y = (a.m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (a.n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::d_sigmoid<<<BLOCKS,THREADS>>>(a.data.get(), result, a.m, a.n);
        cudaDeviceSynchronize();
    };

    return MatrixExpr(a.m, a.n, expr);
}

MatrixExpr Matrix::tanh(const Matrix& a)
{
    std::function<void (double*)> expr = [a](double* result)
    {
        // Set kernal parameters
        int blocks_y = (a.m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (a.n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::tanh<<<BLOCKS,THREADS>>>(a.data.get(), result, a.m, a.n);
        cudaDeviceSynchronize();
    };

    return MatrixExpr(a.m, a.n, expr);
}

MatrixExpr Matrix::d_tanh(const Matrix& a)
{
    std::function<void (double*)> expr = [a](double* result)
    {
        // Set kernal parameters
        int blocks_y = (a.m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (a.n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::d_tanh<<<BLOCKS,THREADS>>>(a.data.get(), result, a.m, a.n);
        cudaDeviceSynchronize();
    };

    return MatrixExpr(a.m, a.n, expr);
}

MatrixExpr Matrix::relu(const Matrix& a)
{
    std::function<void (double*)> expr = [a](double* result)
    {
        // Set kernal parameters
        int blocks_y = (a.m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (a.n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::relu<<<BLOCKS,THREADS>>>(a.data.get(), result, a.m, a.n);
        cudaDeviceSynchronize();
    };

    return MatrixExpr(a.m, a.n, expr);
}

MatrixExpr Matrix::d_relu(const Matrix& a)
{
    std::function<void (double*)> expr = [a](double* result)
    {
        // Set kernal parameters
        int blocks_y = (a.m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (a.n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::d_relu<<<BLOCKS,THREADS>>>(a.data.get(), result, a.m, a.n);
        cudaDeviceSynchronize();
    };

    return MatrixExpr(a.m, a.n, expr);
}

MatrixExpr Matrix::log(const Matrix& a)
{
    std::function<void (double*)> expr = [a](double* result)
    {
        // Set kernal parameters
        int blocks_y = (a.m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (a.n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::log<<<BLOCKS,THREADS>>>(a.data.get(), result, a.m, a.n);
        cudaDeviceSynchronize();
    };

    return MatrixExpr(a.m, a.n, expr);
}

MatrixExpr Matrix::sum(const Matrix& a, int axis)
{
    if (axis == 0)
    {
        std::function<void (double*)> expr = [a](double* result)
        {
            // Set kernal parameters
            int blocks_x = (a.n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

            dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
            dim3 BLOCKS(blocks_x);

            // Execute kernal
            MatrixKernals::sum_vertical<<<BLOCKS,THREADS>>>(a.data.get(), result, a.m, a.n);
            cudaDeviceSynchronize();
        };

        return MatrixExpr(1, a.n, expr);
    }
    else if (axis == 1)
    {
        std::function<void (double*)> expr = [a](double* result)
        {
            // Set kernal parameters
            int blocks_y = (a.m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

            dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
            dim3 BLOCKS(blocks_y);

            // Execute kernal
            MatrixKernals::sum_horizontal<<<BLOCKS,THREADS>>>(a.data.get(), result, a.m, a.n);
            cudaDeviceSynchronize();
        };

        return MatrixExpr(a.m, 1, expr);
    }

    std::ostringstream err;
    err << "Unknown axis: " << axis << ".";
    throw std::invalid_argument(err.str()); 
}

double Matrix::sum(const Matrix& a)
{
    double sum = 0;
    for (int i = 0, n = a.m * a.n; i < n; ++i)
    {
        sum += a.data.get()[i];
    }

    return sum;
}
