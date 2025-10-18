#include <vector>

#include "Matrix/Matrix.cuh"

class NeuralNetwork
{
    public:
        NeuralNetwork();
        NeuralNetwork(Matrix training_set_X, Matrix training_set_Y, Matrix test_set_X, Matrix test_set_Y, double learning_rate, int num_iterations, std::vector<int> layer_sizes);

        Matrix predict(Matrix x);

        void train();
        void test();
        void print();

    private:
        bool complete;

        // hyperparameters
        Matrix training_set_X; 
        Matrix training_set_Y;
        
        Matrix test_set_X; 
        Matrix test_set_Y;

        double learning_rate;
        int num_iterations;
        int num_layers;

        // parameters
        std::vector<Matrix> W;
        std::vector<Matrix> b;
        
        // cached values
        std::vector<Matrix> Z;
        std::vector<Matrix> A;
        
        std::vector<Matrix> dW;
        std::vector<Matrix> db;
        
        std::vector<Matrix> dZ;
        std::vector<Matrix> dA;

        void initialize_parameters(std::vector<int> layer_sizes);
        void forward_propagation();
        void backward_propagation();

        double cost(Matrix A, Matrix Y);
};
