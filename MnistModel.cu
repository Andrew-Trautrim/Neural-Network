#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "NeuralNetwork.cuh"

class MnistModel
{
    public:
        MnistModel()
        {
            load_set("./MNIST/mnist_train.csv", training_set_X, training_set_Y);
            load_set("./MNIST/mnist_test.csv", test_set_X, test_set_Y);
            
            std::vector<int> layer_sizes = { 5, training_set_Y.rows() };
            network = NeuralNetwork(
                training_set_X, 
                training_set_Y, 
                test_set_X,
                test_set_Y,
                0.001, 
                2400, 
                layer_sizes);
        }

        void train()
        {
            // network.test();

            std::cout << "Starting training..." << std::endl;
            network.train();
            std::cout << std::endl;
            
            // network.test();
        }

    private:
        Matrix training_set_X;
        Matrix training_set_Y;
        Matrix test_set_X;
        Matrix test_set_Y;

        NeuralNetwork network;

        void load_set(std::string training_data_file, Matrix& X, Matrix& Y)
        {
            // setup matrices
            auto [lines, values_per_line] = get_csv_dimentions(training_data_file);
            X = Matrix(values_per_line - 1, lines); // the first entry is the classification
            Y = Matrix(10, lines);

            // fill matrices
            std::ifstream file(training_data_file); 

            if (!file.is_open()) { 
                std::ostringstream err;
                err << "Unable to open training data file: " << training_data_file << ".";
                throw std::invalid_argument(err.str()); 
            }

            int i = 0;
            std::string line;
            while (std::getline(file, line)) 
            {
                std::stringstream ss(line);
                std::string cell;

                // get first value for output
                std::getline(ss, cell, ',');
                Y.set(std::stoi(cell), i, 1);

                // Parse the line by comma delimiter
                int j = 0;
                while (std::getline(ss, cell, ',')) 
                { 
                    X.set(j++, i, std::stoi(cell));
                }

                i++;
            }

            file.close();
        }

        std::tuple<int, int> get_csv_dimentions(std::string filename)
        {
            std::ifstream file(filename); 

            if (!file.is_open()) { 
                std::ostringstream err;
                err << "Unable to open file: " << filename << ".";
                throw std::invalid_argument(err.str()); 
            }

            std::string line;

            // get first line
            std::getline(file, line);
            std::stringstream ss(line);
            std::string cell;
            
            // count number of values in first line
            int values_per_line = 0;
            while (std::getline(ss, cell, ',')) 
            { 
                values_per_line++;
            }

            // count number of lines in file
            // start at 1 since we already got the first line
            int lines = 1;
            while (std::getline(file, line)) 
            {
                lines++;
            }

            file.close();

            return { lines, values_per_line };
        }
};
