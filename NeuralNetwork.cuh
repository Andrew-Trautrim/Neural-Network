#include <vector>

#include "Matrix.cuh"

class NeuralNetwork
{
    public:
        NeuralNetwork(Matrix training_set_X, Matrix training_set_Y, double learning_rate, int num_iterations, std::vector<int> layer_sizes);

        Matrix predict(Matrix x);

        void model();
        void print();
        void test(int i);

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

        double cost(Matrix A);
};
