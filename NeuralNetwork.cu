#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "Matrix.cuh"

struct propagate_cache
{
    std::vector<Matrix> Z;
    std::vector<Matrix> A;
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

        void Model()
        {
            if (!complete)
            {
                throw std::logic_error("Neural network already completed learning."); 
            }

            for (int i = 0; i < num_iterations; ++i)
            {
                propagate_cache cache = forward_propagation();
                backward_propagation(cache);
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
        std::vector<Matrix> W;
        std::vector<Matrix> b;

        void initializeParameters(std::vector<int> layer_sizes)
        {
            int prev_layer_size = training_set_X.rows();
            for (int i = 0; i < layer_sizes.size(); ++i)
            {
                W.push_back(Matrix(layer_sizes[i], prev_layer_size));
                b.push_back(Matrix(layer_sizes[i], 1));
                prev_layer_size = layer_sizes[i];

                W[i].randomize();
                b[i].randomize();
            }
        }

        propagate_cache forward_propagation()
        {
            // We need to cache Z and A values for back propagation
            propagate_cache cache;

            Matrix A = training_set_X;
            for (int i = 0; i < num_layers; ++i)
            {
                cache.Z.push_back(W[i].dot(A) + b[i]);
                A = i == num_layers - 1
                    ? Matrix::sigmoid(cache.Z[i])
                    : Matrix::tanh(cache.Z[i]);
                cache.A.push_back(A);
            }

            return cache;
        }

        void backward_propagation(propagate_cache cache)
        {
            // TODO
        }
};
