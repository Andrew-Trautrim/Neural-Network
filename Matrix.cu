#include <cuda_runtime.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "MatrixKernals.cuh"
#include "Matrix.cuh"

const int THREADS_PER_DIM = 16;

Matrix::Matrix(int rows, int cols) : _rows(rows), _cols(cols)
{
    cudaError_t err = cudaMallocManaged(&_data, _rows * _cols * sizeof(double));
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    zero();
}

Matrix::Matrix(const Matrix& other)
{
    _rows = other.rows();
    _cols = other.cols();
    cudaMemcpy(_data, other.data(), _rows * _cols * sizeof(double), cudaMemcpyHostToDevice);
}

Matrix::~Matrix()
{
    cudaFree(_data);
}

int Matrix::rows() const
{
    return _rows;
}

int Matrix::cols() const
{
    return _cols;
}

double* Matrix::data() const
{
    return _data;
}

Matrix Matrix::operator+(const Matrix& other) const
{
    // Set kernal parameters
    int blocks_y = (_rows + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
    int blocks_x = (_cols + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_x, blocks_y);
    
    Matrix result(_rows, _cols);

    if (_rows == other.rows() && _cols == other.cols())
    {
        MatrixKernals::add<<<BLOCKS,THREADS>>>(_data, other.data(), result.data(), _rows, _cols);
        cudaDeviceSynchronize();
    }
    else if (_rows == other.rows() && other.cols() == 1)
    {
        MatrixKernals::add_broadcast_horizontal<<<BLOCKS,THREADS>>>(_data, other.data(), result.data(), _rows, _cols);
        cudaDeviceSynchronize();
    }
    else if (_cols == other.cols() && other.rows() == 1)
    {
        MatrixKernals::add_broadcast_vertical<<<BLOCKS,THREADS>>>(_data, other.data(), result.data(), _rows, _cols);
        cudaDeviceSynchronize();
    }
    else 
    {
        std::ostringstream err;
        err << "Invalid dimensions: cannot add "
            << _rows << "x" << _cols
            << " matrix to "
            << other.rows() << "x" << other.cols()
            << " matrix.";
        throw std::invalid_argument(err.str()); 
    }

    return result;
}

Matrix Matrix::operator-(const Matrix& other) const
{
    // Set kernal parameters
    int blocks_y = (_rows + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
    int blocks_x = (_cols + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_x, blocks_y);
    
    Matrix result(_rows, _cols);

    if (_rows == other.rows() && _cols == other.cols())
    {
        MatrixKernals::subtract<<<BLOCKS,THREADS>>>(_data, other.data(), result.data(), _rows, _cols);
        cudaDeviceSynchronize();
    }
    else if (_rows == other.rows() && other.cols() == 1)
    {
        MatrixKernals::subtract_broadcast_horizontal<<<BLOCKS,THREADS>>>(_data, other.data(), result.data(), _rows, _cols);
        cudaDeviceSynchronize();
    }
    else if (_cols == other.cols() && other.rows() == 1)
    {
        MatrixKernals::subtract_broadcast_vertical<<<BLOCKS,THREADS>>>(_data, other.data(), result.data(), _rows, _cols);
        cudaDeviceSynchronize();
    }
    else 
    {
        std::ostringstream err;
        err << "Invalid dimensions: cannot subtract "
            << other.rows() << "x" << other.cols()
            << " matrix from "
            << _rows << "x" << _cols
            << " matrix.";
        throw std::invalid_argument(err.str()); 
    }

    return result;
}

Matrix Matrix::operator*(const Matrix& other) const
{
    if (_rows != other.rows() || _cols != other.cols())
    {
        std::ostringstream err;
        err << "Invalid dimensions: cannot multiply "
            << _rows << "x" << _cols
            << " matrix with "
            << other.rows() << "x" << other.cols()
            << " matrix.";
        throw std::invalid_argument(err.str()); 
    }

    Matrix result(_rows, _cols);

    // Set kernal parameters
    int blocks_y = (_rows + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
    int blocks_x = (_cols + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_x, blocks_y);

    // Execute kernal
    MatrixKernals::multiply<<<BLOCKS,THREADS>>>(_data, other.data(), result.data(), _rows, _cols);
    cudaDeviceSynchronize();

    return result;
}

Matrix Matrix::operator/(const Matrix& other) const
{
    if (_rows != other.rows() || _cols != other.cols())
    {
        std::ostringstream err;
        err << "Invalid dimensions: cannot divide "
            << _rows << "x" << _cols
            << " matrix with "
            << other.rows() << "x" << other.cols()
            << " matrix.";
        throw std::invalid_argument(err.str()); 
    }

    Matrix result(_rows, _cols);

    // Set kernal parameters
    int blocks_y = (_rows + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
    int blocks_x = (_cols + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_x, blocks_y);

    // Execute kernal
    MatrixKernals::divide<<<BLOCKS,THREADS>>>(_data, other.data(), result.data(), _rows, _cols);
    cudaDeviceSynchronize();

    return result;
}

Matrix Matrix::operator+(double num) const
{
    Matrix result(_rows, _cols);

    // Set kernal parameters
    int blocks_y = (_rows + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
    int blocks_x = (_cols + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_x, blocks_y);

    // Execute kernal
    MatrixKernals::add<<<BLOCKS,THREADS>>>(_data, num, result.data(), _rows, _cols);
    cudaDeviceSynchronize();

    return result;
}

Matrix Matrix::operator*(double num) const
{
    Matrix result(_rows, _cols);

    // Set kernal parameters
    int blocks_y = (_rows + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
    int blocks_x = (_cols + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_x, blocks_y);

    // Execute kernal
    MatrixKernals::multiply<<<BLOCKS,THREADS>>>(_data, num, result.data(), _rows, _cols);
    cudaDeviceSynchronize();

    return result;
}

Matrix Matrix::dot(const Matrix& other) const
{
    if (_cols != other.rows())
    {
        std::ostringstream err;
        err << "Invalid dimensions: cannot dot "
            << _rows << "x" << _cols
            << " matrix with "
            << other.rows() << "x" << other.cols()
            << " matrix.";
        throw std::invalid_argument(err.str()); 
    }

    // Multpliplying an a_m x a_n matrix with an b_m x b_n matrix results in an a_m x b_n matrix.
    Matrix result(_rows, other.cols());

    // Set kernal parameters
    int blocks_y = (_rows + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
    int blocks_x = (other.cols() + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_x, blocks_y);

    // Execute kernal
    MatrixKernals::dot<<<BLOCKS,THREADS>>>(_data, other.data(), result.data(), _rows, _cols, other.rows(), other.cols());
    cudaDeviceSynchronize();

    return result;
}

Matrix Matrix::transpose() const
{
    // The transpose of an MxN matrix is an NxM matrix
    Matrix result(_cols, _rows);

    // Set kernal parameters
    int blocks_y = (_rows + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
    int blocks_x = (_cols + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_x, blocks_y);

    // Execute kernal
    MatrixKernals::transpose<<<BLOCKS,THREADS>>>(_data, result.data(), _rows, _cols);
    cudaDeviceSynchronize();

    return result;
}

void Matrix::print()
{
    std::cout << std::fixed << std::setprecision(2);
    for (int i = 0; i < _rows * _cols; ++i)
    {
        std::cout << std::setw(5) << _data[i] << "  ";
        if ((i+1) % _cols == 0)
        {
            std::cout << std::endl;
        }
    }
}

void Matrix::reshape(int m, int n)
{
    if (_rows * _cols != m * n)
    {
        std::ostringstream err;
        err << "Cannot reshape "
            << _rows << "x" << _cols
            << " to "
            << m << "x" << n
            << " matrix.";
        throw std::invalid_argument(err.str()); 
    }

    _rows = m;
    _cols = n;
}

void Matrix::randomize()
{
    for (int i = 0; i < _rows * _cols; ++i)
    {
        _data[i] = ((double)rand() / RAND_MAX) * 10;
    }
}

void Matrix::zero()
{
    cudaMemset(_data, 0, _rows * _cols * sizeof(double));
}

// Static functions
Matrix Matrix::sigmoid(const Matrix& a)
{
    Matrix result(a.rows(), a.cols());

    // Set kernal parameters
    int blocks_y = (a.rows() + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
    int blocks_x = (a.cols() + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_x, blocks_y);

    // Execute kernal
    MatrixKernals::sigmoid<<<BLOCKS,THREADS>>>(a.data(), result.data(), a.rows(), a.cols());
    cudaDeviceSynchronize();

    return result;
}

Matrix Matrix::tanh(const Matrix& a)
{
    Matrix result(a.rows(), a.cols());

    // Set kernal parameters
    int blocks_y = (a.rows() + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
    int blocks_x = (a.cols() + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

    dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 BLOCKS(blocks_x, blocks_y);

    // Execute kernal
    MatrixKernals::tanh<<<BLOCKS,THREADS>>>(a.data(), result.data(), a.rows(), a.cols());
    cudaDeviceSynchronize();

    return result;
}

double Matrix::sum(const Matrix& a)
{
    double sum = 0;
    for (int i = 0, n = a.rows() * a.cols(); i < n; ++i)
    {
        sum += a.data()[i];
    }

    return sum;
}
