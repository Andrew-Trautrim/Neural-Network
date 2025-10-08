#ifndef MATRIX_H
#define MATRIX_H

#include <memory>

class Matrix
{
    public:
        Matrix();
        Matrix(int m, int n);

        Matrix(const Matrix&) = default;
        Matrix& operator=(const Matrix&) = default;

        int rows() const;
        int cols() const;

        Matrix operator+(const Matrix& other) const;
        Matrix operator-(const Matrix& other) const;
        Matrix operator*(const Matrix& other) const;
        Matrix operator/(const Matrix& other) const;

        Matrix operator+(double num) const;
        Matrix operator*(double num) const;

        Matrix dot(const Matrix& other) const;
        Matrix transpose() const;

        Matrix row(int i) const;
        Matrix col(int i) const;

        void print() const;
        void reshape(int m, int n);
        void randomize(int min, int max);
        void zero();

        static Matrix cross_entropy(const Matrix& a, const Matrix& b);

        static Matrix sigmoid(const Matrix& a);
        static Matrix d_sigmoid(const Matrix& a);

        static Matrix tanh(const Matrix& a);
        static Matrix d_tanh(const Matrix& a);

        static Matrix log(const Matrix& a);

        static Matrix sum(const Matrix& a, int axis);
        static double sum(const Matrix& a);

    private:
        int m;
        int n;

        std::shared_ptr<double> data;
};

#endif // MATRIX_H
