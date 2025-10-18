#ifndef MATRIX_H
#define MATRIX_H

#include <cuda_runtime.h>
#include <memory>

#include "MatrixExpr.cuh"

class MatrixExpr;

class Matrix
{
    friend class MatrixExpr;

    public:
        Matrix();
        Matrix(int m, int n);

        Matrix(const Matrix&) = default;
        Matrix(const MatrixExpr&);

        Matrix& operator=(const Matrix&) = default;
        Matrix& operator=(const MatrixExpr&);

        int rows() const;
        int cols() const;

        void set(int i, int j, double value);
        double get(int i, int j);

        MatrixExpr operator+(const Matrix& other) const;
        MatrixExpr operator+(const MatrixExpr& other) const;

        MatrixExpr operator-(const Matrix& other) const;
        MatrixExpr operator-(const MatrixExpr& other) const;

        MatrixExpr operator*(const Matrix& other) const;
        MatrixExpr operator*(const MatrixExpr& other) const;

        MatrixExpr operator/(const Matrix& other) const;
        MatrixExpr operator/(const MatrixExpr& other) const;

        MatrixExpr operator+(double num) const;
        MatrixExpr operator-(double num) const;
        MatrixExpr operator*(double num) const;
        MatrixExpr operator/(double num) const;

        MatrixExpr dot(const Matrix& other) const;
        MatrixExpr transpose() const;

        MatrixExpr row(int i) const;
        MatrixExpr col(int i) const;

        void print() const;
        void reshape(int m, int n);
        void randomize(int min, int max); // uniform distribution between min and max
        void randomize(); // normal distribution
        void zero();

        static MatrixExpr cross_entropy(const Matrix& a, const Matrix& b);

        static MatrixExpr sigmoid(const Matrix& a);
        static MatrixExpr d_sigmoid(const Matrix& a);

        static MatrixExpr tanh(const Matrix& a);
        static MatrixExpr d_tanh(const Matrix& a);

        static MatrixExpr relu(const Matrix& a);
        static MatrixExpr d_relu(const Matrix& a);

        static MatrixExpr log(const Matrix& a);

        static MatrixExpr sum(const Matrix& a, int axis);
        static double sum(const Matrix& a);

    private:
        int m;
        int n;

        std::shared_ptr<double> data;
};

#endif // MATRIX_H
