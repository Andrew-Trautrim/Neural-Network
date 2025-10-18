#include <iostream>

#include "Matrix/Matrix.cuh"

int main() 
{
    Matrix a(2, 2);
    a.randomize(1, 10);
    std::cout << "a = " << std::endl;
    a.print();

    Matrix b(2, 2);
    b.randomize(1, 10);
    std::cout << "b = " << std::endl;
    b.print();

    Matrix c(2, 2);
    c.randomize(1, 10);
    std::cout << "c = " << std::endl;
    c.print();

    a = (c * c) + (b * c);
    std::cout << "(c * c) + (b * c) = " << std::endl;
    a.print();
}