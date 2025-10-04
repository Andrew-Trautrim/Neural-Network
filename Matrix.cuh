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

        static Matrix sigmoid(const Matrix& a);
        static Matrix tanh(const Matrix& a);
        static double sum(const Matrix& a);

    private:
        int m;
        int n;
        double* data;
};

#endif // MATRIX_H
