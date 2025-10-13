#include <iostream>

#include "Matrix.cuh"

int main() 
{
    Matrix a(1, 1);
    a.randomize(1, 10);
    std::cout << "a = " << std::endl;
    a.print();

    Matrix b(1, 1);
    b.randomize(1, 10);
    std::cout << "b = " << std::endl;
    b.print();

    Matrix c(1, 1);
    c.randomize(1, 10);
    std::cout << "c = " << std::endl;
    c.print();

    Matrix d = a.dot(b) + c;
    std::cout << "a*b + c = " << std::endl;
    d.print();

    // int input_size = 5;
    // int output_size = 5;
    // int training_set_size = 1000;

    // Matrix training_set_X(input_size, training_set_size);
    // Matrix training_set_Y(output_size, training_set_size);

    // training_set_X.randomize(0, 10);
    // training_set_Y.randomize(0, 1);

    // std::vector<int> layer_sizes = {10, 10, output_size};
    // NeuralNetwork nn(training_set_X, training_set_Y, 0.01, 10000, layer_sizes);

    // std::cout << "before:" << std::endl;
    // nn.test(50);

    // nn.model();
    
    // std::cout << "after:" << std::endl;
    // nn.test(50);
}