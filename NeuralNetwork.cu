#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "Matrix.cuh"

class NeuralNetwork
{
    public:
        NeuralNetwork(
            Matrix training_set_X, 
            Matrix training_set_Y,
            double learning_rate,
            int num_iterations,
            std::vector<int> layer_sizes) 
            : training_set_X(training_set_X), 
              training_set_Y(training_set_Y), 
              learning_rate(learning_rate), 
              num_iterations(num_iterations),
              num_layers(layer_sizes.size())
        {
            complete = false;

            initialize_parameters(layer_sizes);
        }

        ~NeuralNetwork()
        {
            free(W);
            free(b);
        }

        void test(int i)
        {
            Matrix x = training_set_X.col(i);
            Matrix y = training_set_Y.col(i);

            std::cout << "x = " << std::endl;
            x.print();

            std::cout << "y = " << std::endl;
            y.print();

            Matrix output = predict(x);
            std::cout << "output = " << std::endl;
            output.print();
        }

        void model()
        {
            if (complete)
            {
                throw std::logic_error("Neural network already completed learning."); 
            }

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            for (int i = 0; i < num_iterations; ++i)
            {
                // start recording
                cudaEventRecord(start);

                forward_propagation();
                backward_propagation();

                // stop recording
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                if ((i + 1) % 10 == 0)
                {
                    float diff = 0;
                    cudaEventElapsedTime(&diff, start, stop);
                    std::cout << i + 1 << ": C = " << cost(A[num_layers - 1]) << ", t = " << diff << "ms" << std::endl;
                }
            }

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            complete = true;
        }

        Matrix predict(Matrix x)
        {
            Matrix A = x;
            for (int i = 0; i < num_layers; ++i)
            {
                Matrix Z = W[i].dot(A) + b[i];
                A = i == num_layers - 1
                    ? Matrix::sigmoid(Z)
                    : Matrix::tanh(Z);
            }

            return A;
        }

        void print()
        {
            for (int i = 0; i < num_layers; ++i)
            {
                std::cout << "W_" << i << " = " << std::endl;
                W[i].print();
                std::cout << "b_" << i << " = " << std::endl;
                b[i].print();
                std::cout << std::endl;
            }
        }

    private:
        bool complete;

        // hyperparameters
        Matrix training_set_X; 
        Matrix training_set_Y;

        double learning_rate;
        int num_iterations;
        int num_layers;

        // parameters
        Matrix* W;
        Matrix* b;
        
        // cached values
        Matrix* Z;
        Matrix* A;

        void initialize_parameters(std::vector<int> layer_sizes)
        {
            W = (Matrix*)malloc(num_layers * sizeof(Matrix));
            b = (Matrix*)malloc(num_layers * sizeof(Matrix));

            Z = (Matrix*)malloc(num_layers * sizeof(Matrix));
            A = (Matrix*)malloc(num_layers * sizeof(Matrix));

            int prev_layer_size = training_set_X.rows();
            for (int i = 0; i < num_layers; ++i)
            {
                W[i] = Matrix(layer_sizes[i], prev_layer_size);
                b[i] = Matrix(layer_sizes[i], 1);

                W[i].randomize(-1, 1);
                b[i].randomize(-1, 1);

                prev_layer_size = layer_sizes[i];
            }
        }

        double cost(Matrix A)
        {
            Matrix loss = Matrix::cross_entropy(A, training_set_Y);
            return (1 / (double)training_set_X.cols()) * Matrix::sum(loss);
        }

        void forward_propagation()
        {
            Matrix prev_A = training_set_X;
            for (int i = 0; i < num_layers; ++i)
            {
                Z[i] = W[i].dot(prev_A) + b[i];
                A[i] = i == num_layers - 1
                    ? Matrix::sigmoid(Z[i])
                    : Matrix::tanh(Z[i]);

                prev_A = A[i];
            }
        }

        void backward_propagation()
        {
            Matrix dA = (((training_set_Y * -1) + 1) / ((A[num_layers - 1] * -1) + 1)) - (training_set_Y / A[num_layers - 1]);

            for (int i = num_layers - 1; i >= 0; --i)
            {
                // dZ for this layer
                Matrix dZ = dA * (i == num_layers - 1
                    ? Matrix::d_sigmoid(Z[i])
                    : Matrix::d_tanh(Z[i]));

                // update dA for next (i - 1) layer
                dA = W[i].transpose().dot(dZ);

                // calculate gradients
                Matrix layer_input = i == 0
                    ? training_set_X
                    : A[i-1];
                Matrix dW = dZ.dot(layer_input.transpose());
                Matrix db = Matrix::sum(dZ, 1);

                // update parameters
                W[i] = W[i] - (dW * learning_rate);
                b[i] = b[i] - (db * learning_rate);
            }
        }
};
