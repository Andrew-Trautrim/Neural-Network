#include <iostream>

#include "Matrix.cuh"
#include "NeuralNetwork.cu"

int main() 
{
    int input_size = 5;
    int output_size = 2;
    Matrix training_set_X(input_size, 5);
    Matrix training_set_Y(output_size, 5);

    training_set_X.randomize();
    training_set_Y.randomize();

    std::cout << "test1" << std::endl;

    std::vector<int> layer_sizes = {10, output_size};
    NeuralNetwork nn(training_set_X, training_set_Y, 0.01, 100, layer_sizes);

    std::cout << "test2" << std::endl;

    Matrix x(input_size, 1);
    nn.predict(x);
    
    std::cout << "complete :)" << std::endl;
}