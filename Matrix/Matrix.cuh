#ifndef MATRIX_H
#define MATRIX_H

#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <iostream>
#include <unordered_map>
#include <stack>
#include <stdexcept>

#include "MatrixCommon.cuh"

class Matrix;
template<typename L, typename R> class BinaryExpr;
template<typename E> class UnaryExpr;
template<typename E> class NumExpr;

struct Buffer
{
    std::shared_ptr<double> data; 
    size_t size;
};

template<typename A>
class MatrixExpr
{
    friend class Matrix;
    template<typename E> friend class MatrixExpr;
    template<typename L, typename R> friend class BinaryExpr;
    template<typename E> friend class UnaryExpr;
    template<typename E> friend class NumExpr;

    public:
        __host__ MatrixExpr(int m, int n);
        __host__ MatrixExpr();

        // Binary expressions
        template<typename B> __host__ BinaryExpr<A, B> operator+(const MatrixExpr<B>& other) const;
        template<typename B> __host__ BinaryExpr<A, B> operator-(const MatrixExpr<B>& other) const;
        template<typename B> __host__ BinaryExpr<A, B> operator*(const MatrixExpr<B>& other) const;
        template<typename B> __host__ BinaryExpr<A, B> operator&(const MatrixExpr<B>& other) const;
        template<typename B> __host__ BinaryExpr<A, B> operator/(const MatrixExpr<B>& other) const;

        // Unary expressions
        __host__ UnaryExpr<A> transpose() const;
        __host__ UnaryExpr<A> sum(int axis) const;
        
        // Num expressions
        __host__ NumExpr<A> operator+(double num) const;
        __host__ NumExpr<A> operator*(double num) const;
        __host__ NumExpr<A> operator/(double num) const;
        __host__ NumExpr<A> operator-(double num) const;

        __host__ double* evaluate(const Matrix& result) const;
        __host__ double* evaluate(const Buffer& result) const;
        __host__ bool references(double* a) const;

        // Static binary expressions
        template<typename B> static __host__ BinaryExpr<A, B> cross_entropy(const MatrixExpr<A>& a, const MatrixExpr<B>& b);

        // Static unary expressions
        static __host__ UnaryExpr<A> sigmoid(const MatrixExpr<A>& a);
        static __host__ UnaryExpr<A> d_sigmoid(const MatrixExpr<A>& a);
        static __host__ UnaryExpr<A> tanh(const MatrixExpr<A>& a);
        static __host__ UnaryExpr<A> d_tanh(const MatrixExpr<A>& a);
        static __host__ UnaryExpr<A> relu(const MatrixExpr<A>& a);
        static __host__ UnaryExpr<A> d_relu(const MatrixExpr<A>& a);

    protected:
        int m;
        int n;

        __host__ static Buffer getBuffer(int required_size);
        __host__ static void releaseBuffer(Buffer buff);

    private:
        static std::stack<Buffer> buffers;
};

class Matrix : public MatrixExpr<Matrix>
{
    template<typename E> friend class MatrixExpr;
    template<typename E> friend class NumExpr;
    template<typename E> friend class UnaryExpr;
    template<typename L, typename R> friend class BinaryExpr;

    public:
        Matrix();
        Matrix(int m, int n);
                
        template<typename A>
        Matrix(const MatrixExpr<A>& expr);

        Matrix& operator=(const Matrix& expr);

        template<typename A>
        Matrix& operator=(const MatrixExpr<A>& expr);

        using MatrixExpr<Matrix>::operator+;
        using MatrixExpr<Matrix>::operator-;
        using MatrixExpr<Matrix>::operator*;
        using MatrixExpr<Matrix>::operator/;
        
        using MatrixExpr<Matrix>::transpose;
        using MatrixExpr<Matrix>::sum;

        int cols();
        int rows();

        void set(int i, int j, double value);
        double get(int i, int j);

        double sum();

        void randomize();
        void randomize(int min, int max);
        void zero();
        void print() const;

        double* evaluate(const Matrix& result) const;
        double* evaluate(const Buffer& result) const;
        bool references(double* a) const;
    
    private:
        std::shared_ptr<double> data; 
};

template<typename L, typename R>
class BinaryExpr : public MatrixExpr<BinaryExpr<L, R>>
{
    friend class Matrix;
    template<typename E> friend class MatrixExpr;
    template<typename E> friend class NumExpr;
    template<typename E> friend class UnaryExpr;

    public:
        BinaryExpr(const L& lhs, const R& rhs, std::function<void (double*, double*, double*, int, int, int, int)> eval, bool self_referencing = true);
        BinaryExpr(const L& lhs, const R& rhs, std::function<void (double*, double*, double*, int, int, int, int)> eval, int m, int n, bool self_referencing = true);

        __host__ double* evaluate(const Matrix& result) const;
        __host__ double* evaluate(const Buffer& result) const;
        __host__ bool references(double* a) const;

    private:
        const L& lhs;
        const R& rhs;

        bool self_referencing; // bool to determine if the expression allowed to reference the result

        std::function<void (double*, double*, double*, int, int, int, int)> eval;
};

template<typename E>
class UnaryExpr : public MatrixExpr<UnaryExpr<E>>
{
    friend class Matrix;
    template<typename A> friend class MatrixExpr;
    template<typename A> friend class NumExpr;
    template<typename L, typename R> friend class BinaryExpr;

    public:
        UnaryExpr(const E& expr, std::function<void (double*, double*, int, int)> eval);
        UnaryExpr(const E& expr, std::function<void (double*, double*, int, int)> eval, int m, int n);

        __host__ double* evaluate(const Matrix& result) const;
        __host__ double* evaluate(const Buffer& result) const;
        __host__ bool references(double* a) const;

    private:
        const E& expr;

        std::function<void (double*, double*, int, int)> eval;
};

template<typename E>
class NumExpr : public MatrixExpr<NumExpr<E>>
{
    friend class Matrix;
    template<typename A> friend class MatrixExpr;
    template<typename A> friend class UnaryExpr;
    template<typename L, typename R> friend class BinaryExpr;

    public:
        NumExpr(const E& expr, std::function<void (double*, double, double*, int, int)> eval, double num);

        __host__ double* evaluate(const Matrix& result) const;
        __host__ double* evaluate(const Buffer& result) const;
        __host__ bool references(double* a) const;

    private:
        const E& expr;
        double num;

        std::function<void (double*, double, double*, int, int)> eval;
};

/*
 * Matix class
 */
        
template<typename A>
Matrix::Matrix(const MatrixExpr<A>& expr)
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

    double* result = expr.evaluate(*this);
    if (result != data.get())
    {
        cudaMemcpy(data.get(), result, m * n * sizeof(double), cudaMemcpyHostToHost);
    }
}

template<typename A>
Matrix& Matrix::operator=(const MatrixExpr<A>& expr)
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

    double* result = expr.evaluate(*this);
    if (result != data.get())
    {
        cudaMemcpy(data.get(), result, m * n * sizeof(double), cudaMemcpyHostToHost);
    }

    return *this;
}

/*
 * MatrixExpr class
 */

// PUBLIC

template<typename A>
MatrixExpr<A>::MatrixExpr(int m, int n) : m(m), n(n) {}

template<typename A>
MatrixExpr<A>::MatrixExpr() = default;

template<typename A>
template<typename B>
BinaryExpr<A, B> MatrixExpr<A>::operator+(const MatrixExpr<B>& other) const
{
    auto eval = [](double* a, double* b, double* c, int a_m, int a_n, int b_m, int b_n) 
    { 
        MatrixCommon::add(a, b, c, a_m, a_n, b_m, b_n);
    };

    return BinaryExpr<A, B>(static_cast<const A&>(*this), static_cast<const B&>(other), eval);
}

template<typename A>
template<typename B>
BinaryExpr<A, B> MatrixExpr<A>::operator-(const MatrixExpr<B>& other) const
{
    auto eval = [](double* a, double* b, double* c, int a_m, int a_n, int b_m, int b_n) 
    { 
        MatrixCommon::subtract(a, b, c, a_m, a_n, b_m, b_n);
    };

    return BinaryExpr<A, B>(static_cast<const A&>(*this), static_cast<const B&>(other), eval);
}

template<typename A>
template<typename B>
BinaryExpr<A, B> MatrixExpr<A>::operator*(const MatrixExpr<B>& other) const
{
    auto eval = [](double* a, double* b, double* c, int a_m, int a_n, int b_m, int b_n) 
    { 
        MatrixCommon::multiply(a, b, c, a_m, a_n, b_m, b_n);
    };

    return BinaryExpr<A, B>(static_cast<const A&>(*this), static_cast<const B&>(other), eval, m, other.n, false);
}

template<typename A>
template<typename B>
BinaryExpr<A, B> MatrixExpr<A>::operator&(const MatrixExpr<B>& other) const
{
    auto eval = [](double* a, double* b, double* c, int a_m, int a_n, int b_m, int b_n) 
    { 
        MatrixCommon::hadamardProduct(a, b, c, a_m, a_n, b_m, b_n);
    };

    return BinaryExpr<A, B>(static_cast<const A&>(*this), static_cast<const B&>(other), eval);
}

template<typename A>
template<typename B>
BinaryExpr<A, B> MatrixExpr<A>::operator/(const MatrixExpr<B>& other) const
{
    auto eval = [](double* a, double* b, double* c, int a_m, int a_n, int b_m, int b_n) 
    { 
        MatrixCommon::divide(a, b, c, a_m, a_n, b_m, b_n);
    };

    return BinaryExpr<A, B>(static_cast<const A&>(*this), static_cast<const B&>(other), eval);
}

template<typename A>
template<typename B>
BinaryExpr<A, B> MatrixExpr<A>::cross_entropy(const MatrixExpr<A>& a, const MatrixExpr<B>& b)
{
    auto eval = [](double* a, double* b, double* c, int a_m, int a_n, int b_m, int b_n) 
    {
        MatrixCommon::cross_entropy(a, b, c, a_m, a_n, b_m, b_n);
    };

    return BinaryExpr<A, B>(static_cast<const A&>(a), static_cast<const B&>(b), eval);
}

template<typename E>
UnaryExpr<E> MatrixExpr<E>::transpose() const
{
    auto eval = [](double* a, double* b, int a_m, int a_n) 
    { 
        MatrixCommon::transpose(a, b, a_m, a_n);
    };

    return UnaryExpr<E>(static_cast<const E&>(*this), eval, n, m);
}

template<typename E>
UnaryExpr<E> MatrixExpr<E>::sum(int axis) const
{
    auto eval = [axis](double* a, double* b, int a_m, int a_n) 
    { 
        MatrixCommon::sum(a, b, a_m, a_n, axis);
    };

    return UnaryExpr<E>(static_cast<const E&>(*this), eval, axis == 0 ? 1 : m, axis == 1 ? 1 : n);
}

template<typename E>
UnaryExpr<E> MatrixExpr<E>::sigmoid(const MatrixExpr<E>& expr)
{
    auto eval = [](double* a, double* b, int a_m, int a_n) 
    {
        MatrixCommon::sigmoid(a, b, a_m, a_n);
    };

    return UnaryExpr<E>(static_cast<const E&>(expr), eval);
}

template<typename E>
UnaryExpr<E> MatrixExpr<E>::d_sigmoid(const MatrixExpr<E>& expr)
{
    auto eval = [](double* a, double* b, int a_m, int a_n) 
    { 
        MatrixCommon::d_sigmoid(a, b, a_m, a_n);
    };

    return UnaryExpr<E>(static_cast<const E&>(expr), eval);
}

template<typename E>
UnaryExpr<E> MatrixExpr<E>::tanh(const MatrixExpr<E>& expr)
{
    auto eval = [](double* a, double* b, int a_m, int a_n) 
    { 
        MatrixCommon::tanh(a, b, a_m, a_n);
    };

    return UnaryExpr<E>(static_cast<const E&>(expr), eval);
}

template<typename E>
UnaryExpr<E> MatrixExpr<E>::d_tanh(const MatrixExpr<E>& expr)
{
    auto eval = [](double* a, double* b, int a_m, int a_n) 
    { 
        MatrixCommon::d_tanh(a, b, a_m, a_n);
    };

    return UnaryExpr<E>(static_cast<const E&>(expr), eval);
}

template<typename E>
UnaryExpr<E> MatrixExpr<E>::relu(const MatrixExpr<E>& expr)
{
    auto eval = [](double* a, double* b, int a_m, int a_n) 
    { 
        MatrixCommon::relu(a, b, a_m, a_n);
    };

    return UnaryExpr<E>(static_cast<const E&>(expr), eval);
}

template<typename E>
UnaryExpr<E> MatrixExpr<E>::d_relu(const MatrixExpr<E>& expr)
{
    auto eval = [](double* a, double* b, int a_m, int a_n) 
    { 
        MatrixCommon::d_relu(a, b, a_m, a_n);
    };

    return UnaryExpr<E>(static_cast<const E&>(expr), eval);
}

template<typename E>
NumExpr<E> MatrixExpr<E>::operator+(double num) const
{
    auto eval = [](double* a, double num, double* b, int a_m, int a_n) 
    { 
        MatrixCommon::add(a, num, b, a_m, a_n);
    };

    return NumExpr<E>(static_cast<const E&>(*this), eval, num);
}

template<typename E>
NumExpr<E> MatrixExpr<E>::operator-(double num) const
{
    auto eval = [](double* a, double num, double* b, int a_m, int a_n) 
    { 
        MatrixCommon::subtract(a, num, b, a_m, a_n);
    };

    return NumExpr<E>(static_cast<const E&>(*this), eval, num);
}

template<typename E>
NumExpr<E> MatrixExpr<E>::operator*(double num) const
{
    auto eval = [](double* a, double num, double* b, int a_m, int a_n) 
    { 
        MatrixCommon::multiply(a, num, b, a_m, a_n);
    };

    return NumExpr<E>(static_cast<const E&>(*this), eval, num);
}

template<typename E>
NumExpr<E> MatrixExpr<E>::operator/(double num) const
{
    auto eval = [](double* a, double num, double* b, int a_m, int a_n) 
    { 
        MatrixCommon::divide(a, num, b, a_m, a_n);
    };

    return NumExpr<E>(static_cast<const E&>(*this), eval, num);
}

template<typename A>
double* MatrixExpr<A>::evaluate(const Matrix& result) const
{
    return static_cast<const A&>(*this).evaluate(result);
}

template<typename A>
double* MatrixExpr<A>::evaluate(const Buffer& result) const
{
    return static_cast<const A&>(*this).evaluate(result);
}

template<typename A>
bool MatrixExpr<A>::references(double* a) const
{
    return static_cast<const A&>(*this).references(a);
}

// PROTECTED

template<typename A>
Buffer MatrixExpr<A>::getBuffer(int required_size)
{
    Buffer buff { nullptr, 0 };
    if (!buffers.empty())
    {
        buff = buffers.top();
        buffers.pop();
    }

    if (buff.size < required_size)
    {
        double* raw_ptr;
        cudaError_t err = cudaMallocManaged(&raw_ptr, required_size * sizeof(double));
        if (err != cudaSuccess)
        {
            throw std::runtime_error(cudaGetErrorString(err));
        }

        auto cuda_deleter = [](double* p)
        {
            cudaFree(p);
        };

        buff.size = required_size;
        buff.data = std::shared_ptr<double>(raw_ptr, cuda_deleter);
    }
    
    return buff;
}

template<typename A>
void MatrixExpr<A>::releaseBuffer(Buffer buff)
{
    if (buff.data == nullptr)
    {
        return;
    }

    buffers.push(buff);
}

/*
 * BinaryExpr class
 */

template<typename A>
inline std::stack<Buffer> MatrixExpr<A>::buffers;

template<typename L, typename R>
BinaryExpr<L, R>::BinaryExpr(const L& lhs, const R& rhs, std::function<void (double*, double*, double*, int, int, int, int)> eval, bool self_referencing) 
    : MatrixExpr<BinaryExpr<L, R>>(lhs.m, lhs.n), 
        lhs(lhs), rhs(rhs), eval(eval), self_referencing(self_referencing)
{
}

template<typename L, typename R>
BinaryExpr<L, R>::BinaryExpr(const L& lhs, const R& rhs, std::function<void (double*, double*, double*, int, int, int, int)> eval, int m, int n, bool self_referencing) 
    : MatrixExpr<BinaryExpr<L, R>>(m, n), 
        lhs(lhs), rhs(rhs), eval(eval), self_referencing(self_referencing)
{
}

template<typename L, typename R>
double* BinaryExpr<L, R>::evaluate(const Matrix& result) const 
{
    // Evaluate left hand side
    Buffer lhs_buff { nullptr, 0 };
    double* lhs_result;

    // Can we use the result as a buffer?
    if (std::is_same_v<L, Matrix> || (
            result.m * result.n == lhs.m * lhs.n && 
            !rhs.references(result.data.get())))
    {
        lhs_result = lhs.evaluate(result);
    }
    else 
    {
        lhs_buff = this->getBuffer(lhs.m * lhs.n);
        lhs_result = lhs.evaluate(lhs_buff);
    }

    // evaluate right hand side
    Buffer rhs_buff { nullptr, 0 };
    double* rhs_result;
    
    // Can we use the result as a buffer AND has it not already been used?
    if (std::is_same_v<R, Matrix> || (
            lhs_result != result.data.get() && 
            result.m * result.n == rhs.m * rhs.n && 
            !lhs.references(result.data.get())))
    {
        rhs_result = rhs.evaluate(result);
    }
    else 
    {
        rhs_buff = this->getBuffer(rhs.m * rhs.n);
        rhs_result = rhs.evaluate(rhs_buff);
    }

    // if we have something like a = a * b then we must store the result in a buffer
    Buffer result_buff { nullptr, 0 };
    double* accumulator = result.data.get();
    if (!self_referencing && (lhs_result == result.data.get() || rhs_result == result.data.get()))
    {
        result_buff = this->getBuffer(result.m * result.n);
        accumulator = result_buff.data.get();
    }

    eval(lhs_result, rhs_result, accumulator, lhs.m, lhs.n, rhs.m, rhs.n);

    this->releaseBuffer(lhs_buff);
    this->releaseBuffer(rhs_buff);
    this->releaseBuffer(result_buff);

    return accumulator;
}

template<typename L, typename R>
double* BinaryExpr<L, R>::evaluate(const Buffer& result) const 
{
    // Evaluate left hand side
    Buffer lhs_buff { nullptr, 0 };
    double* lhs_result;

    // Can we use the result as a buffer?
    if (std::is_same_v<L, Matrix> || result.size == lhs.m * lhs.n)
    {
        lhs_result = lhs.evaluate(result);
    }
    else 
    {
        lhs_buff = this->getBuffer(lhs.m * lhs.n);
        lhs_result = lhs.evaluate(lhs_buff);
    }

    // evaluate right hand side
    Buffer rhs_buff { nullptr, 0 };
    double* rhs_result;
    
    // Can we use the result as a buffer AND has it not already been used?
    if (std::is_same_v<R, Matrix> || (lhs_result != result.data.get() && result.size == rhs.m * rhs.n))
    {
        rhs_result = rhs.evaluate(result);
    }
    else 
    {
        rhs_buff = this->getBuffer(rhs.m * rhs.n);
        rhs_result = rhs.evaluate(rhs_buff);
    }
    
    // if we have something like a = a * b then we must store the result in a buffer
    Buffer result_buff { nullptr, 0 };
    double* accumulator = result.data.get();
    if (!self_referencing && (lhs_result == result.data.get() || rhs_result == result.data.get()))
    {
        result_buff = this->getBuffer(result.size);
        accumulator = result_buff.data.get();
    }

    eval(lhs_result, rhs_result, accumulator, lhs.m, lhs.n, rhs.m, rhs.n);

    this->releaseBuffer(lhs_buff);
    this->releaseBuffer(rhs_buff);
    this->releaseBuffer(result_buff);

    return accumulator;
}

template<typename L, typename R>
bool BinaryExpr<L, R>::references(double* a) const
{
    return lhs.references(a) || rhs.references(a);
}

/*
 * NumExpr class
 */

template<typename E>
NumExpr<E>::NumExpr(const E& expr, std::function<void (double*, double, double*, int, int)> eval, double num) 
    : MatrixExpr<NumExpr<E>>(expr.m, expr.n), 
        expr(expr), eval(eval), num(num)
{
}

template<typename E>
double* NumExpr<E>::evaluate(const Matrix& result) const 
{
    // Evaluate sub-expression
    Buffer buff { nullptr, 0 };
    double* expr_result;

    // Can we use the result as a buffer?
    if (std::is_same_v<E, Matrix> || result.m * result.n == expr.m * expr.n)
    {
        expr_result = expr.evaluate(result);
    }
    else 
    {
        buff = this->getBuffer(expr.m * expr.n);
        expr_result = expr.evaluate(buff);
    }

    eval(expr_result, num, result.data.get(), expr.m, expr.n);

    this->releaseBuffer(buff);

    return result.data.get();
}

template<typename E>
double* NumExpr<E>::evaluate(const Buffer& result) const 
{
    // Evaluate sub-expression
    Buffer buff { nullptr, 0 };
    double* expr_result;

    // Can we use the result as a buffer?
    if (std::is_same_v<E, Matrix> || result.size == expr.m * expr.n)
    {
        expr_result = expr.evaluate(result);
    }
    else 
    {
        buff = this->getBuffer(expr.m * expr.n);
        expr_result = expr.evaluate(buff);
    }

    eval(expr_result, num, result.data.get(), expr.m, expr.n);

    this->releaseBuffer(buff);

    return result.data.get();
}

template<typename E>
bool NumExpr<E>::references(double* a) const
{
    return expr.references(a);
}

/*
 * UnaryExpr class
 */

template<typename E>
UnaryExpr<E>::UnaryExpr(const E& expr, std::function<void (double*, double*, int, int)> eval) 
    : MatrixExpr<UnaryExpr<E>>(expr.m, expr.n), 
        expr(expr), eval(eval)
{
}

template<typename E>
UnaryExpr<E>::UnaryExpr(const E& expr, std::function<void (double*, double*, int, int)> eval, int m, int n) 
    : MatrixExpr<UnaryExpr<E>>(m, n), 
        expr(expr), eval(eval)
{
}

template<typename E>
double* UnaryExpr<E>::evaluate(const Matrix& result) const 
{
    // Evaluate sub-expression
    Buffer buff { nullptr, 0 };
    double* expr_result;

    // Can we use the result as a buffer?
    if (std::is_same_v<E, Matrix> || result.m * result.n == expr.m * expr.n)
    {
        expr_result = expr.evaluate(result);
    }
    else 
    {
        buff = this->getBuffer(expr.m * expr.n);
        expr_result = expr.evaluate(buff);
    }

    eval(expr_result, result.data.get(), expr.m, expr.n);

    this->releaseBuffer(buff);

    return result.data.get();
}

template<typename E>
double* UnaryExpr<E>::evaluate(const Buffer& result) const 
{
    // Evaluate sub-expression
    Buffer buff { nullptr, 0 };
    double* expr_result;

    // Can we use the result as a buffer?
    if (std::is_same_v<E, Matrix> || result.size == expr.m * expr.n)
    {
        expr_result = expr.evaluate(result);
    }
    else 
    {
        buff = this->getBuffer(expr.m * expr.n);
        expr_result = expr.evaluate(buff);
    }

    eval(expr_result, result.data.get(), expr.m, expr.n);

    this->releaseBuffer(buff);

    return result.data.get();
}

template<typename E>
bool UnaryExpr<E>::references(double* a) const
{
    return expr.references(a);
}

#endif // MATRIX_H
