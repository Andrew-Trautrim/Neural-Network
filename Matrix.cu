#include <cuda_runtime.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "Matrix.cuh"
#include "MatrixKernals.cuh"

const int THREADS_PER_DIM = 16;

int seed = 0;

void add(double* a, double* b, double* c)
{
    
}

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
    zero();

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

    zero();

    expr.eval(data.get());

    return *this;
}

MatrixExpr Matrix::operator+(const Matrix& other) const
{
    std::function<void (double*)> expr = [this, other](double* accumulator)
    {
        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        if (m == other.m && n == other.n)
        {
            MatrixKernals::add<<<BLOCKS,THREADS>>>(data.get(), other.data.get(), accumulator, m, n);
            cudaDeviceSynchronize();
        }
        else if (m == other.m && other.n == 1)
        {
            MatrixKernals::add_broadcast_horizontal<<<BLOCKS,THREADS>>>(data.get(), other.data.get(), accumulator, m, n);
            cudaDeviceSynchronize();
        }
        else if (n == other.n && other.m == 1)
        {
            MatrixKernals::add_broadcast_vertical<<<BLOCKS,THREADS>>>(data.get(), other.data.get(), accumulator, m, n);
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
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr Matrix::operator+(const MatrixExpr& other) const
{
    std::function<void (double*)> expr = [this, other](double* accumulator)
    {
        other.eval(accumulator);

        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        if (m == other.m && n == other.n)
        {
            MatrixKernals::add<<<BLOCKS,THREADS>>>(data.get(), accumulator, accumulator, m, n);
            cudaDeviceSynchronize();
        }
        else if (m == other.m && other.n == 1)
        {
            MatrixKernals::add_broadcast_horizontal<<<BLOCKS,THREADS>>>(data.get(), accumulator, accumulator, m, n);
            cudaDeviceSynchronize();
        }
        else if (n == other.n && other.m == 1)
        {
            MatrixKernals::add_broadcast_vertical<<<BLOCKS,THREADS>>>(data.get(), accumulator, accumulator, m, n);
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
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr Matrix::operator-(const Matrix& other) const
{
    std::function<void (double*)> expr = [this, other](double* accumulator)
    {
        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        if (m == other.m && n == other.n)
        {
            MatrixKernals::subtract<<<BLOCKS,THREADS>>>(data.get(), other.data.get(), accumulator, m, n);
            cudaDeviceSynchronize();
        }
        else if (m == other.m && other.n == 1)
        {
            MatrixKernals::subtract_broadcast_horizontal<<<BLOCKS,THREADS>>>(data.get(), other.data.get(), accumulator, m, n);
            cudaDeviceSynchronize();
        }
        else if (n == other.n && other.m == 1)
        {
            MatrixKernals::subtract_broadcast_vertical<<<BLOCKS,THREADS>>>(data.get(), other.data.get(), accumulator, m, n);
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
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr Matrix::operator-(const MatrixExpr& other) const
{
    std::function<void (double*)> expr = [this, other](double* accumulator)
    {
        other.eval(accumulator);

        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        if (m == other.m && n == other.n)
        {
            MatrixKernals::subtract<<<BLOCKS,THREADS>>>(data.get(), accumulator, accumulator, m, n);
            cudaDeviceSynchronize();
        }
        else if (m == other.m && other.n == 1)
        {
            MatrixKernals::subtract_broadcast_horizontal<<<BLOCKS,THREADS>>>(data.get(), accumulator, accumulator, m, n);
            cudaDeviceSynchronize();
        }
        else if (n == other.n && other.m == 1)
        {
            MatrixKernals::subtract_broadcast_vertical<<<BLOCKS,THREADS>>>(data.get(), accumulator, accumulator, m, n);
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
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr Matrix::operator*(const Matrix& other) const
{
    std::function<void (double*)> expr = [this, other](double* accumulator)
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

        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::multiply<<<BLOCKS,THREADS>>>(data.get(), other.data.get(), accumulator, m, n);
        cudaDeviceSynchronize();
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr Matrix::operator*(const MatrixExpr& other) const
{
    std::function<void (double*)> expr = [this, other](double* accumulator)
    {
        other.eval(accumulator);

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

        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::multiply<<<BLOCKS,THREADS>>>(data.get(), accumulator, accumulator, m, n);
        cudaDeviceSynchronize();
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr Matrix::operator/(const Matrix& other) const
{
    std::function<void (double*)> expr = [this, other](double* accumulator)
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

        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::divide<<<BLOCKS,THREADS>>>(data.get(), other.data.get(), accumulator, m, n);
        cudaDeviceSynchronize();
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr Matrix::operator/(const MatrixExpr& other) const
{
    std::function<void (double*)> expr = [this, other](double* accumulator)
    {
        other.eval(accumulator);

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

        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::divide<<<BLOCKS,THREADS>>>(data.get(), accumulator, accumulator, m, n);
        cudaDeviceSynchronize();
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr Matrix::operator+(double num) const
{
    std::function<void (double*)> expr = [this, num](double* accumulator)
    {
        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::add<<<BLOCKS,THREADS>>>(data.get(), num, accumulator, m, n);
        cudaDeviceSynchronize();
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr Matrix::operator*(double num) const
{
    std::function<void (double*)> expr = [this, num](double* accumulator)
    {
        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::multiply<<<BLOCKS,THREADS>>>(data.get(), num, accumulator, m, n);
        cudaDeviceSynchronize();
    };

    return MatrixExpr(m, n, expr);
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
    MatrixKernals::dot<<<BLOCKS,THREADS>>>(data.get(), other.data.get(), result.data.get(), m, n, other.m, other.n);
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
    MatrixKernals::transpose<<<BLOCKS,THREADS>>>(data.get(), result.data.get(), m, n);
    cudaDeviceSynchronize();

    return result;
}

Matrix Matrix::row(int i) const
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

    Matrix result(1, n);

    // Set kernal parameters
    int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_x);

    // Execute kernal
    MatrixKernals::row<<<BLOCKS,THREADS>>>(data.get(), result.data.get(), i, m, n);
    cudaDeviceSynchronize();

    return result;
}

Matrix Matrix::col(int i) const
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

    // The transpose of an MxN matrix is an NxM matrix
    Matrix result(m, 1);

    // Set kernal parameters
    int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_y);

    // Execute kernal
    MatrixKernals::col<<<BLOCKS,THREADS>>>(data.get(), result.data.get(), i, m, n);
    cudaDeviceSynchronize();

    return result;
}

void Matrix::print() const
{
    std::cout << std::fixed << std::setprecision(2);
    for (int i = 0; i < m * n; ++i)
    {
        std::cout << std::setw(5) << data.get()[i] << "  ";
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

    // Randomize values between 0 and 10
    MatrixKernals::randomize<<<BLOCKS,THREADS>>>(states, data.get(), m, n, min, max);
    cudaDeviceSynchronize();

    cudaFree(states);
}

void Matrix::zero()
{
    cudaMemset(data.get(), 0, m * n * sizeof(double));
}

// Static functions
Matrix Matrix::cross_entropy(const Matrix& a, const Matrix& b)
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

    Matrix result(a.m, a.n);

    // Set kernal parameters
    int blocks_y = (a.m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
    int blocks_x = (a.n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_x, blocks_y);

    // Execute kernal
    MatrixKernals::cross_entropy<<<BLOCKS,THREADS>>>(a.data.get(), b.data.get(), result.data.get(), a.m, a.n);
    cudaDeviceSynchronize();

    return result;
}

Matrix Matrix::sigmoid(const Matrix& a)
{
    Matrix result(a.m, a.n);

    // Set kernal parameters
    int blocks_y = (a.m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
    int blocks_x = (a.n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_x, blocks_y);

    // Execute kernal
    MatrixKernals::sigmoid<<<BLOCKS,THREADS>>>(a.data.get(), result.data.get(), a.m, a.n);
    cudaDeviceSynchronize();

    return result;
}

Matrix Matrix::d_sigmoid(const Matrix& a)
{
    Matrix result(a.m, a.n);

    // Set kernal parameters
    int blocks_y = (a.m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
    int blocks_x = (a.n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_x, blocks_y);

    // Execute kernal
    MatrixKernals::d_sigmoid<<<BLOCKS,THREADS>>>(a.data.get(), result.data.get(), a.m, a.n);
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
    MatrixKernals::tanh<<<BLOCKS,THREADS>>>(a.data.get(), result.data.get(), a.m, a.n);
    cudaDeviceSynchronize();

    return result;
}

Matrix Matrix::d_tanh(const Matrix& a)
{
    Matrix result(a.m, a.n);

    // Set kernal parameters
    int blocks_y = (a.m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
    int blocks_x = (a.n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_x, blocks_y);

    // Execute kernal
    MatrixKernals::d_tanh<<<BLOCKS,THREADS>>>(a.data.get(), result.data.get(), a.m, a.n);
    cudaDeviceSynchronize();

    return result;
}

Matrix Matrix::log(const Matrix& a)
{
    Matrix result(a.m, a.n);

    // Set kernal parameters
    int blocks_y = (a.m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
    int blocks_x = (a.n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_x, blocks_y);

    // Execute kernal
    MatrixKernals::log<<<BLOCKS,THREADS>>>(a.data.get(), result.data.get(), a.m, a.n);
    cudaDeviceSynchronize();

    return result;
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

Matrix Matrix::sum(const Matrix& a, int axis)
{
    if (axis == 0)
    {
        Matrix result(1, a.n);

        // Set kernal parameters
        int blocks_x = (a.n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x);

        // Execute kernal
        MatrixKernals::sum_vertical<<<BLOCKS,THREADS>>>(a.data.get(), result.data.get(), a.m, a.n);
        cudaDeviceSynchronize();

        return result;
    }
    else if (axis == 1)
    {
        Matrix result(a.m, 1);

        // Set kernal parameters
        int blocks_y = (a.m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_y);

        // Execute kernal
        MatrixKernals::sum_horizontal<<<BLOCKS,THREADS>>>(a.data.get(), result.data.get(), a.m, a.n);
        cudaDeviceSynchronize();

        return result;
    }

    std::ostringstream err;
    err << "Unknown axis: " << axis << ".";
    throw std::invalid_argument(err.str()); 
}

MatrixExpr::MatrixExpr(int m, int n, std::function<void (double*)> eval) : m(m), n(n), eval(eval)
{
}

MatrixExpr MatrixExpr::operator+(const Matrix& other) const
{
    std::function<void (double*)> expr = [this, other](double* accumulator)
    {
        eval(accumulator);

        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        if (m == other.m && n == other.n)
        {
            MatrixKernals::add<<<BLOCKS,THREADS>>>(accumulator, other.data.get(), accumulator, m, n);
            cudaDeviceSynchronize();
        }
        else if (m == other.m && other.n == 1)
        {
            MatrixKernals::add_broadcast_horizontal<<<BLOCKS,THREADS>>>(accumulator, other.data.get(), accumulator, m, n);
            cudaDeviceSynchronize();
        }
        else if (n == other.n && other.m == 1)
        {
            MatrixKernals::add_broadcast_vertical<<<BLOCKS,THREADS>>>(accumulator, other.data.get(), accumulator, m, n);
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
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr MatrixExpr::operator-(const Matrix& other) const
{
    std::function<void (double*)> expr = [this, other](double* accumulator)
    {
        eval(accumulator);

        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        if (m == other.m && n == other.n)
        {
            MatrixKernals::subtract<<<BLOCKS,THREADS>>>(accumulator, other.data.get(), accumulator, m, n);
            cudaDeviceSynchronize();
        }
        else if (m == other.m && other.n == 1)
        {
            MatrixKernals::subtract_broadcast_horizontal<<<BLOCKS,THREADS>>>(accumulator, other.data.get(), accumulator, m, n);
            cudaDeviceSynchronize();
        }
        else if (n == other.n && other.m == 1)
        {
            MatrixKernals::subtract_broadcast_vertical<<<BLOCKS,THREADS>>>(accumulator, other.data.get(), accumulator, m, n);
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
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr MatrixExpr::operator*(const Matrix& other) const
{
    std::function<void (double*)> expr = [this, other](double* accumulator)
    {
        eval(accumulator);

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

        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::multiply<<<BLOCKS,THREADS>>>(accumulator, other.data.get(), accumulator, m, n);
        cudaDeviceSynchronize();
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr MatrixExpr::operator/(const Matrix& other) const
{
    std::function<void (double*)> expr = [this, other](double* accumulator)
    {
        eval(accumulator);

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

        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::divide<<<BLOCKS,THREADS>>>(accumulator, other.data.get(), accumulator, m, n);
        cudaDeviceSynchronize();
    };

    return MatrixExpr(m, n, expr);
}

Matrix MatrixExpr::evaluate(const MatrixExpr& expr)
{
    Matrix result(expr.m, expr.n);
    expr.eval(result.data.get());

    return result;
}
