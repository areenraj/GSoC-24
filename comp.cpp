#include <cstdlib> 
#include <iostream> 
#include <fstream>
#include <chrono>
#include "cblas.h"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

int main()
{

    /*Define Initial Dimension (M), Final Dimenions (limit), and incremental dimension every iteration (inc)*/
    int limit = 5000;
    int inc = 5;
    int M = 5;

    /*Create output file*/
    std::ofstream serial("serial.txt");

    serial.open("serial.txt", std::ios::app);
    serial << "Dimension" << "\t\t\t" << "Time(ms)" << "\n";
    serial.close();

    /*While Loop that increases the dimension of the square matrices each iteration*/
    while(M<limit)
    {

        /*Allocate Memory to input square matrices of dimensions MxM (a and b) and output matrix mul*/
        float *a = (float*) malloc(sizeof(double)*M*M);
        float *b = (float*) malloc(sizeof(double)*M*M);
        float *mul = (float*) malloc(sizeof(double)*M*M);

        /*initialize the input matrices to random values and the output matrix to zero*/
        for(int i=0;i<M;i++)    
        {    
            for(int j=0;j<M;j++)    
            {    
                a[IDX2C(i,j,M)] = rand() % 2000;
                b[IDX2C(i,j,M)] = rand() % 2000;
                mul[IDX2C(i,j,M)] = 0.0;
            }    
        }    
        
        /*OpenBLAS computations*/
        auto start = std::chrono::high_resolution_clock::now();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, M, M, 1.0, a, M, b, M, 0.0, mul, M);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        /*Write duration to file along with dimension number*/
        serial.open("serial.txt", std::ios::app);
        serial << M << "\t\t\t" << duration.count() << "\n";
        serial.close();

        M += inc;

    } 

    return 0;

}
