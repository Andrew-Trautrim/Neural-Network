#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "Matrix.cuh"

struct propagate_cache
{
    Matrix* Z;
    Matrix* A;
};

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

            initializeParameters(layer_sizes);
        }

        ~NeuralNetwork()
        {
            free(W);
            free(b);
        }

        void model()
        {
            if (complete)
            {
                throw std::logic_error("Neural network already completed learning."); 
            }

            for (int i = 0; i < num_iterations; ++i)
            {
                propagate_cache cache = forward_propagation();
                backward_propagation(cache);

                free(cache.Z);
                free(cache.A);
            }

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

        void initializeParameters(std::vector<int> layer_sizes)
        {
            W = (Matrix*)malloc(num_layers * sizeof(Matrix));
            b = (Matrix*)malloc(num_layers * sizeof(Matrix));

            int prev_layer_size = training_set_X.rows();
            for (int i = 0; i < num_layers; ++i)
            {
                W[i] = Matrix(layer_sizes[i], prev_layer_size);
                b[i] = Matrix(layer_sizes[i], 1);

                W[i].randomize();
                b[i].randomize();

                prev_layer_size = layer_sizes[i];
            }
        }

        double cost(Matrix A)
        {
            Matrix loss = Matrix::cross_entropy(training_set_Y, A);
            return (-1 / (double)training_set_X.cols()) * Matrix::sum(loss);
        }

        propagate_cache forward_propagation()
        {
            // We need to cache Z and A values for back propagation
            propagate_cache cache;
            cache.Z = (Matrix*)malloc(num_layers * sizeof(Matrix));
            cache.A = (Matrix*)malloc(num_layers * sizeof(Matrix));

            Matrix prev_A = training_set_X;
            for (int i = 0; i < num_layers; ++i)
            {
                cache.Z[i] = W[i].dot(prev_A) + b[i];
                cache.A[i] = i == num_layers - 1
                    ? Matrix::sigmoid(cache.Z[i])
                    : Matrix::tanh(cache.Z[i]);

                prev_A = cache.A[i];
            }

            return cache;
        }

        void backward_propagation(propagate_cache cache)
        {
            // Matrix dA;
            // for (int i = num_layers - 1; i >= 0; --i)
            // {
            //     Matrix dZ = dA * Matrix::d_tanh(cache.Z[i]);

            //     Matrix dW = dZ.dot(cache.A[i-1].transpose()) * (1 / (double)m);
            //     Matrix db = dZ.sum();
                
            //     dA = W[i].transpose().dot(dZ);
            // }
        }
};
