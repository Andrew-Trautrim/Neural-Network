#include <iostream>
#include <sstream>
#include <stdexcept>

#include "MatrixCommon.cuh"
#include "MatrixKernals.cuh"

const int THREADS_PER_DIM = 16;

namespace MatrixCommon
{
    void add(double* a, double* b, double* c, int a_m, int a_n, int b_m, int b_n)
    {
        // Set kernal parameters
        int blocks_y = (a_m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (a_n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        if (a_m == b_m && a_n == b_n)
        {
            MatrixKernals::add<<<BLOCKS,THREADS>>>(a, b, c, a_m, a_n);
            cudaDeviceSynchronize();
        }
        else if (a_m == b_m && b_n == 1)
        {
            MatrixKernals::add_broadcast_horizontal<<<BLOCKS,THREADS>>>(a, b, c, a_m, a_n);
            cudaDeviceSynchronize();
        }
        else if (a_n == b_n && b_m == 1)
        {
            MatrixKernals::add_broadcast_vertical<<<BLOCKS,THREADS>>>(a, b, c, a_m, a_n);
            cudaDeviceSynchronize();
        }
        else 
        {
            std::ostringstream err;
            err << "Invalid dimensions: cannot add "
                << a_m << "x" << a_n
                << " matrix and "
                << b_m << "x" << b_n
                << " matrix.";
            throw std::invalid_argument(err.str()); 
        }
    }

    void subtract(double* a, double* b, double* c, int a_m, int a_n, int b_m, int b_n)
    {
        // Set kernal parameters
        int blocks_y = (a_m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (a_n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        if (a_m == b_m && a_n == b_n)
        {
            MatrixKernals::subtract<<<BLOCKS,THREADS>>>(a, b, c, a_m, a_n);
            cudaDeviceSynchronize();
        }
        else if (a_m == b_m && b_n == 1)
        {
            MatrixKernals::subtract_broadcast_horizontal<<<BLOCKS,THREADS>>>(a, b, c, a_m, a_n);
            cudaDeviceSynchronize();
        }
        else if (a_n == b_n && b_m == 1)
        {
            MatrixKernals::subtract_broadcast_vertical<<<BLOCKS,THREADS>>>(a, b, c, a_m, a_n);
            cudaDeviceSynchronize();
        }
        else 
        {
            std::ostringstream err;
            err << "Invalid dimensions: cannot subtract "
                << b_m << "x" << b_n
                << " matrix from "
                << a_m << "x" << a_n
                << " matrix.";
            throw std::invalid_argument(err.str()); 
        }
    }

    void multiply(double* a, double* b, double* c, int a_m, int a_n, int b_m, int b_n)
    {
        if (a_n != b_m)
        {
            std::ostringstream err;
            err << "Invalid dimensions: cannot multiply "
                << a_m << "x" << a_n
                << " matrix and "
                << b_m << "x" << b_n
                << " matrix.";
            throw std::invalid_argument(err.str()); 
        }

        // Set kernal parameters
        int blocks_y = (a_m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (a_n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::multiply<<<BLOCKS,THREADS>>>(a, b, c, a_m, a_n, b_m, b_n);
        cudaDeviceSynchronize();
    }

    void hadamardProduct(double* a, double* b, double* c, int a_m, int a_n, int b_m, int b_n)
    {
        if (a_m != b_m || a_n != b_n)
        {
            std::ostringstream err;
            err << "Invalid dimensions: cannot Hadamard product "
                << a_m << "x" << a_n
                << " matrix and "
                << b_m << "x" << b_n
                << " matrix.";
            throw std::invalid_argument(err.str()); 
        }

        // Set kernal parameters
        int blocks_y = (a_m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (a_n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::hadamardProduct<<<BLOCKS,THREADS>>>(a, b, c, a_m, a_n);
        cudaDeviceSynchronize();
    }

    void divide(double* a, double* b, double* c, int a_m, int a_n, int b_m, int b_n)
    {
        if (a_m != b_m || a_n != b_n)
        {
            std::ostringstream err;
            err << "Invalid dimensions: cannot divide "
                << a_m << "x" << a_n
                << " matrix by "
                << b_m << "x" << b_n
                << " matrix.";
            throw std::invalid_argument(err.str()); 
        }

        // Set kernal parameters
        int blocks_y = (a_m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (a_n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::divide<<<BLOCKS,THREADS>>>(a, b, c, a_m, a_n);
        cudaDeviceSynchronize();
    }
    
    void add(double* a, double num, double* b, int m, int n)
    {
        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::add<<<BLOCKS,THREADS>>>(a, num, b, m, n);
        cudaDeviceSynchronize();
    }

    void subtract(double* a, double num, double* b, int m, int n)
    {
        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::subtract<<<BLOCKS,THREADS>>>(a, num, b, m, n);
        cudaDeviceSynchronize();
    }

    void multiply(double* a, double num, double* b, int m, int n)
    {
        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::multiply<<<BLOCKS,THREADS>>>(a, num, b, m, n);
        cudaDeviceSynchronize();
    }

    void divide(double* a, double num, double* b, int m, int n)
    {
        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::divide<<<BLOCKS,THREADS>>>(a, num, b, m, n);
        cudaDeviceSynchronize();
    }

    void transpose(double* a, double* b, int m, int n)
    {
        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::transpose<<<BLOCKS,THREADS>>>(a, b, m, n);
        cudaDeviceSynchronize();
    }

    void sum(double* a, double* b, int m, int n, int axis)
    {
        if (axis == 0)
        {
            // Set kernal parameters
            int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

            dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
            dim3 BLOCKS(blocks_x);

            // Execute kernal
            MatrixKernals::sum_vertical<<<BLOCKS,THREADS>>>(a, b, m, n);
            cudaDeviceSynchronize();

            return;
        }
        else if (axis == 1)
        {
            // Set kernal parameters
            int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

            dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
            dim3 BLOCKS(blocks_y);

            // Execute kernal
            MatrixKernals::sum_horizontal<<<BLOCKS,THREADS>>>(a, b, m, n);
            cudaDeviceSynchronize();

            return;
        }

        std::ostringstream err;
        err << "Unknown axis: " << axis << ".";
        throw std::invalid_argument(err.str()); 
    }

    void cross_entropy(double* a, double* b, double* c, int a_m, int a_n, int b_m, int b_n)
    {
        if (a_m != b_m || a_n != b_n)
        {
            std::ostringstream err;
            err << "Invalid dimensions: cannot compute cross entropy of "
                << a_m << "x" << a_n
                << " matrix and "
                << b_m << "x" << b_n
                << " matrix.";
            throw std::invalid_argument(err.str()); 
        }

        // Set kernal parameters
        int blocks_y = (a_m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (a_n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::cross_entropy<<<BLOCKS,THREADS>>>(a, b, c, a_m, a_n);
        cudaDeviceSynchronize();
    }

    void sigmoid(double* a, double* b, int m, int n)
    {
        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::sigmoid<<<BLOCKS,THREADS>>>(a, b, m, n);
        cudaDeviceSynchronize();
    }

    void d_sigmoid(double* a, double* b, int m, int n)
    {
        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::d_sigmoid<<<BLOCKS,THREADS>>>(a, b, m, n);
        cudaDeviceSynchronize();
    }

    void tanh(double* a, double* b, int m, int n)
    {
        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::tanh<<<BLOCKS,THREADS>>>(a, b, m, n);
        cudaDeviceSynchronize();
    }

    void d_tanh(double* a, double* b, int m, int n)
    {
        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::d_tanh<<<BLOCKS,THREADS>>>(a, b, m, n);
        cudaDeviceSynchronize();
    }

    void relu(double* a, double* b, int m, int n)
    {
        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::relu<<<BLOCKS,THREADS>>>(a, b, m, n);
        cudaDeviceSynchronize();
    }

    void d_relu(double* a, double* b, int m, int n)
    {
        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::d_relu<<<BLOCKS,THREADS>>>(a, b, m, n);
        cudaDeviceSynchronize();
    }

    void log(double* a, double* b, int m, int n)
    {
        // Set kernal parameters
        int blocks_y = (m + THREADS_PER_DIM - 1) / THREADS_PER_DIM;
        int blocks_x = (n + THREADS_PER_DIM - 1) / THREADS_PER_DIM;

        dim3 THREADS(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 BLOCKS(blocks_x, blocks_y);

        // Execute kernal
        MatrixKernals::log<<<BLOCKS,THREADS>>>(a, b, m, n);
        cudaDeviceSynchronize();
    }
}