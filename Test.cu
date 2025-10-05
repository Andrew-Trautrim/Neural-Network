#include <iostream>

#include "Matrix.cuh"
#include "NeuralNetwork.cu"

int main() 
{
    int input_size = 5;
    int output_size = 1;

    Matrix training_set_X(input_size, 400);
    Matrix training_set_Y(output_size, 400);

    training_set_X.randomize(0, 10);
    training_set_Y.randomize(0, 1);

    std::vector<int> layer_sizes = {5, output_size};
    NeuralNetwork nn(training_set_X, training_set_Y, 0.01, 1000, layer_sizes);

    std::cout << "before:" << std::endl;
    nn.test(0);

    nn.model();
    
    std::cout << "after:" << std::endl;
    nn.test(0);
}