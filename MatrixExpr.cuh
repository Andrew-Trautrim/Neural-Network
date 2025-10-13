#ifndef MATRIXEXPR_H
#define MATRIXEXPR_H

#include <cuda_runtime.h>
#include <functional>

#include "Matrix.cuh"

class Matrix;

class MatrixExpr
{
    friend class Matrix;

    public:
        __host__ MatrixExpr(int m, int n, std::function<void (double*)> eval);

        __host__ MatrixExpr operator+(const Matrix& other) const;
        __host__ MatrixExpr operator-(const Matrix& other) const;
        __host__ MatrixExpr operator*(const Matrix& other) const;
        __host__ MatrixExpr operator/(const Matrix& other) const;

        __host__ MatrixExpr operator+(double num) const;
        __host__ MatrixExpr operator*(double num) const;

        __host__ static Matrix evaluate(const MatrixExpr& expr);

    private:
        int m;
        int n;

        std::function<void (double*)> eval;
};

#endif // MATRIXEXPR_H