#include <iostream>

#include "Matrix.cuh"
#include "NeuralNetwork.cu"

int main() 
{
    int input_size = 5;
    int output_size = 10;

    Matrix training_set_X(input_size, 5);
    Matrix training_set_Y(output_size, 5);

    training_set_X.randomize();
    training_set_Y.randomize();

    std::vector<int> layer_sizes = {10, 5, 5, 6, 20, output_size};
    NeuralNetwork nn(training_set_X, training_set_Y, 0.01, 1, layer_sizes);

    Matrix x(input_size, 1);
    nn.model();
    
    std::cout << "complete :)" << std::endl;
}