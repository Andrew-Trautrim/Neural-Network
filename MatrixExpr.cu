#include <cuda_runtime.h>
#include <functional>
#include <sstream>
#include <stdexcept>

#include "Matrix.cuh"
#include "MatrixExpr.cuh"
#include "MatrixKernals.cuh"

const int THREADS_PER_DIM = 16;

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

    std::function<void (double*)> expr = [this, other](double* accumulator)
    {
        eval(accumulator);

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

    std::function<void (double*)> expr = [this, other](double* accumulator)
    {
        eval(accumulator);

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


MatrixExpr MatrixExpr::operator+(double num) const
{
    std::function<void (double*)> expr = [this, num](double* accumulator)
    {
        eval(accumulator);

        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::add<<<BLOCKS,THREADS>>>(accumulator, num, accumulator, m, n);
        cudaDeviceSynchronize();
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr MatrixExpr::operator*(double num) const
{
    std::function<void (double*)> expr = [this, num](double* accumulator)
    {
        eval(accumulator);

        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::multiply<<<BLOCKS,THREADS>>>(accumulator, num, accumulator, m, n);
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
