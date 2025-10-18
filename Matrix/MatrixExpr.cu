#include <cuda_runtime.h>
#include <functional>
#include <sstream>
#include <stdexcept>

#include "Matrix.cuh"
#include "MatrixExpr.cuh"
#include "MatrixKernals.cuh"
#include "MatrixCommon.cuh"

Buffer MatrixExpr::buffer = { nullptr, 0 };

MatrixExpr::MatrixExpr(int m, int n, std::function<void (double*)> eval) : m(m), n(n), eval(eval)
{
}

MatrixExpr MatrixExpr::operator+(const Matrix& other) const
{
    std::function<void (double*)> expr = [this, other](double* result)
    {
        double* accumulator = result;
        if (other.data.get() == result)
        {
            MatrixExpr::setBuffer(*this, &accumulator);
        }

        eval(accumulator);

        MatrixCommon::add(accumulator, other.data.get(), result, m, n, other.m, other.n);
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr MatrixExpr::operator-(const Matrix& other) const
{
    std::function<void (double*)> expr = [this, other](double* result)
    {
        double* accumulator = result;
        if (other.data.get() == result)
        {
            MatrixExpr::setBuffer(*this, &accumulator);
        }

        eval(accumulator);
        
        MatrixCommon::subtract(accumulator, other.data.get(), result, m, n, other.m, other.n);
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr MatrixExpr::operator*(const Matrix& other) const
{
    std::function<void (double*)> expr = [this, other](double* result)
    {
        double* accumulator = result;
        if (other.data.get() == result)
        {
            MatrixExpr::setBuffer(*this, &accumulator);
        }

        eval(accumulator);

        MatrixCommon::multiply(accumulator, other.data.get(), result, m, n, other.m, other.n);
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr MatrixExpr::operator/(const Matrix& other) const
{
    std::function<void (double*)> expr = [this, other](double* result)
    {
        double* accumulator = result;
        if (other.data.get() == result)
        {
            MatrixExpr::setBuffer(*this, &accumulator);
        }

        eval(accumulator);

        MatrixCommon::divide(accumulator, other.data.get(), result, m, n, other.m, other.n);
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr MatrixExpr::operator+(double num) const
{
    std::function<void (double*)> expr = [this, num](double* result)
    {
        eval(result);

        MatrixCommon::add(result, num, result, m, n);
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr MatrixExpr::operator-(double num) const
{
    std::function<void (double*)> expr = [this, num](double* result)
    {
        eval(result);

        MatrixCommon::subtract(result, num, result, m, n);
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr MatrixExpr::operator*(double num) const
{
    std::function<void (double*)> expr = [this, num](double* result)
    {
        eval(result);

        MatrixCommon::multiply(result, num, result, m, n);
    };

    return MatrixExpr(m, n, expr);
}

MatrixExpr MatrixExpr::operator/(double num) const
{
    std::function<void (double*)> expr = [this, num](double* result)
    {
        eval(result);

        MatrixCommon::divide(result, num, result, m, n);
    };

    return MatrixExpr(m, n, expr);
}

Matrix MatrixExpr::evaluate(const MatrixExpr& expr)
{
    Matrix result(expr.m, expr.n);
    expr.eval(result.data.get());

    return result;
}

void MatrixExpr::setBuffer(const MatrixExpr& expr, double** result)
{
    if (buffer.size < expr.m * expr.n)
    {
        if (buffer.data != nullptr)
        {
            cudaFree(buffer.data);
        }

        cudaError_t err = cudaMallocManaged(&buffer.data, expr.m * expr.n * sizeof(double));
        if (err != cudaSuccess)
        {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }

    *result = buffer.data;
}
