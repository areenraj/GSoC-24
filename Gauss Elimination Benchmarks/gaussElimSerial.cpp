#include <cstdlib> 
#include <iostream> 
#include <fstream>
#include <chrono>
#include <unistd.h>
#include <iomanip>
#include <random>

#define IDX2C(i,j,ld) (((i)*(ld))+(j))

void setMatrix(float* a, float* b, int matrixSize)
{
    /*Random Number Generator*/
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

    for(int i=0;i<matrixSize;i++)    
    {    
        for(int j=0;j<matrixSize;j++)    
        {    
            a[IDX2C(i,j,matrixSize)] = rng() % 1000;
        }

        b[i] = rng() % 1000;
    }
}

void putMatrix(std::ofstream& output, float* a, float* b, int matrixSize)
{
    for(int i=0; i<matrixSize; i++) 
    {
        for(int j=0;j<matrixSize;j++)
        {
            output << std::setw(4) << a[IDX2C(i, j, matrixSize)];
        }
        output << "\t|\t" << b[i] << "\n";
    }
}

void Gauss_Elimination(float* matrix, float* vec, int matrixSize) 
{
    #define A(I, J) matrix[(I)*matrixSize + (J)]

        /*--- Transform system in Upper Matrix ---*/
        for (auto iVar = 1; iVar < matrixSize; iVar++) {
        for (auto jVar = 0; jVar < iVar; jVar++) {
            float weight = A(iVar, jVar) / A(jVar, jVar);
            for (auto kVar = jVar; kVar < matrixSize; kVar++) A(iVar, kVar) -= weight * A(jVar, kVar);
            vec[iVar] -= weight * vec[jVar];
        }
        }

        /*--- Backwards substitution ---*/
        for (auto iVar = matrixSize; iVar > 0;) {
        iVar--;  // unsigned type
        for (auto jVar = iVar + 1; jVar < matrixSize; jVar++) vec[iVar] -= A(iVar, jVar) * vec[jVar];
        vec[iVar] /= A(iVar, iVar);
        }
    #undef A
    }

int main()
{

    int matrixSize = 0.0;

    std::cout << "Enter Matrix Size (Max 32): ";
    std::cin >> matrixSize;

    double time = 0.0;

    /*Allocate Memory to input square matrices of dimensions MxM (a and b) and output matrix mul*/
    float *a = (float*) malloc(sizeof(float)*matrixSize*matrixSize);
    float *b = (float*) malloc(sizeof(float)*matrixSize);

    /*Initialize the input matrix and vector to random values between 0-999*/
    setMatrix(a,b,matrixSize);   

    /*Create Output File and Print Matrix to File*/
    std::ofstream serial("serial.txt");
    serial<< std::right;
    serial << "Input Augmented Matrix with Vector" << "\n\n";
    putMatrix(serial, a, b, matrixSize);
    serial.close();

    /*Run and Time the function*/
    auto start = std::chrono::high_resolution_clock::now();
    Gauss_Elimination(a, b, matrixSize);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

    time = duration.count();

    /*Write dimension and execution time to file*/
    serial.open("serial.txt", std::ios::app);
    serial << "\n\n" << "Time taken to execute in Nanoseconds - " << time << " ns" << "\n";
    serial << "Matrix Size - " << matrixSize << "\n";
    serial.close();

    free(a);
    free(b);
    
    return 0;

}
