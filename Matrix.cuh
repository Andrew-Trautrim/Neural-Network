#ifndef MATRIX_H
#define MATRIX_H

class Matrix
{
    public:
        Matrix(int m, int n);
        Matrix(const Matrix& other);
        ~Matrix();

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

        void reshape(int m, int n);
        void randomize();
        void zero();
        void print();

        static Matrix cross_entropy(const Matrix& a, const Matrix& b);
        static Matrix sigmoid(const Matrix& a);
        static Matrix tanh(const Matrix& a);
        static Matrix d_tanh(const Matrix& a);
        static Matrix log(const Matrix& a);
        static Matrix sum(const Matrix& a, int axis);
        static double sum(const Matrix& a);

    private:
        int m;
        int n;
        double* data;
};

#endif // MATRIX_H
