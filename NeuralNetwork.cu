#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "NeuralNetwork.cuh"

NeuralNetwork::NeuralNetwork()
{
}

NeuralNetwork::NeuralNetwork(
    Matrix training_set_X, 
    Matrix training_set_Y,
    Matrix test_set_X,
    Matrix test_set_Y,
    double learning_rate,
    int num_iterations,
    std::vector<int> layer_sizes) 
    : training_set_X(training_set_X), 
        training_set_Y(training_set_Y), 
        test_set_X(test_set_X),
        test_set_Y(test_set_Y),
        learning_rate(learning_rate), 
        num_iterations(num_iterations),
        num_layers(layer_sizes.size()),
        W(num_layers),
        dW(num_layers),
        b(num_layers),
        db(num_layers),
        Z(num_layers),
        dZ(num_layers),
        A(num_layers),
        dA(num_layers)
{
    complete = false;

    initialize_parameters(layer_sizes);
}

Matrix NeuralNetwork::predict(Matrix x)
{
    Matrix A = x;
    for (int i = 0; i < num_layers; ++i)
    {
        Matrix Z = (W[i] * A) + b[i];
        A = i == num_layers - 1
            ? Matrix::sigmoid(Z)
            : Matrix::relu(Z);
    }

    return A;
}

void NeuralNetwork::train()
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

        if ((i + 1) % 1 == 0)
        {
            float diff = 0;
            cudaEventElapsedTime(&diff, start, stop);
            double c = cost(A[num_layers - 1], training_set_Y);
            std::cout << "Epoch " << i + 1 << ": C = " << c << ", t = " << diff << "ms" << std::endl;
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    complete = true;
}

// double accuracy(Matrix A, Matrix Y)
// {
//     int num_correct = 0;
//     for (int i = 0; i < A.cols(); ++i)
//     {
//         int prediction = 0;
//         int actual = 0;
//         for (int j = 0; j < A.rows(); ++j)
//         {
//             if (A.get(j, i) > A.get(prediction, i))
//             {
//                 prediction = j;
//             }
            
//             if (Y.get(j, i) > Y.get(actual, i))
//             {
//                 actual = j;
//             }
//         }

//         if (prediction == actual)
//         {
//             num_correct++;
//         }
//     }

//     return (double)num_correct / A.cols();
// }

// void NeuralNetwork::test()
// {
//     std::cout << "Training set: " << std::endl;

//     Matrix train_result = predict(training_set_X);

//     double cost_train = cost(train_result, training_set_Y);
//     std::cout << "\tCost = " << cost_train << std::endl;
//     double accuracy_train = accuracy(train_result, training_set_Y);
//     std::cout << "\tAccuracy = " << accuracy_train << std::endl << std::endl;
    
//     std::cout << "Testing set: " << std::endl;

//     Matrix test_result = predict(test_set_X);

//     double cost_test = cost(test_result, test_set_Y);
//     std::cout << "\tCost = " << cost_test << std::endl;
//     double accuracy_test = accuracy(test_result, test_set_Y);
//     std::cout << "\tAccuracy = " << accuracy_test << std::endl << std::endl;
// }

void NeuralNetwork::print()
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

// Private methods

void NeuralNetwork::initialize_parameters(std::vector<int> layer_sizes)
{
    int prev_layer_size = training_set_X.rows();
    for (int i = 0; i < num_layers; ++i)
    {
        W[i] = Matrix(layer_sizes[i], prev_layer_size);

        // He initialization
        W[i].randomize();
        W[i] = W[i] * sqrt(2 / (double)prev_layer_size);

        b[i] = Matrix(layer_sizes[i], 1); // initialize b to 0

        prev_layer_size = layer_sizes[i];
    }
}

void NeuralNetwork::forward_propagation()
{
    Matrix prev_A = training_set_X;
    for (int i = 0; i < num_layers; ++i)
    {
        Z[i] = (W[i] * prev_A) + b[i];
        A[i] = i == num_layers - 1
            ? Matrix::sigmoid(Z[i])
            : Matrix::relu(Z[i]);

        prev_A = A[i];
    }
}

void NeuralNetwork::backward_propagation()
{
    // Output layer
    dZ[num_layers - 1] = A[num_layers - 1] - training_set_Y;

    for (int i = num_layers - 1; i >= 0; --i)
    {
        // calculate gradients
        Matrix layer_input = i == 0 
            ? training_set_X 
            : A[i - 1];
        dW[i] = (dZ[i] * layer_input.transpose()) / (double)training_set_X.cols();
        db[i] = dZ[i].sum(1) / (double)training_set_X.cols();
        
        // update parameters
        W[i] = W[i] - (dW[i] * learning_rate);
        b[i] = b[i] - (db[i] * learning_rate);

        if (i != 0)
        {
            dZ[i - 1] = (W[i].transpose() * dZ[i]) & Matrix::d_relu(Z[i - 1]);
        }
    }
}

double NeuralNetwork::cost(Matrix A, Matrix Y)
{
    Matrix loss = Matrix::cross_entropy(A, Y);
    return  loss.sum() / (double)training_set_X.cols();
}
