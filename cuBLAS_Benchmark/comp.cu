#include <cstdlib> 
#include <iostream> 
#include <fstream>
#include <chrono>
#include "cblas.h"
#include <time.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

int main()
{

    /*Define Initial Dimension (M), Final Dimenions (limit), and incremental dimension every iteration (inc)*/
    int limit = 10000;
    int inc = 500;
    int M = 5;
    
    /*Create output file*/
    std::ofstream parallel("parallel.txt");

    parallel.open("parallel.txt", std::ios::app);
    parallel << "Dimension" << "\t\t\t" << "Time(ms)" << "\n";
    parallel.close();
   
    float alpha = 1.0;
    float beta = 0.0;
    /*While Loop that increases the dimension of the square matrices each iteration*/

    while(M<limit)
    {

        cublasHandle_t handle; 

        double time = 0.0;

        /*Allocate Memory to input square matrices of dimensions MxM (a and b) and output matrix mul*/
        float *a = (float*) malloc(sizeof(float)*M*M);
        float *b = (float*) malloc(sizeof(float)*M*M);
        float *mul = (float*) malloc(sizeof(float)*M*M);

        float  *devPtra, *devPtrb, *devPtrmul;
    
        cudaMalloc ((void**)&devPtra, M*M*sizeof(float));
        cudaMalloc ((void**)&devPtrb, M*M*sizeof(float));
        cudaMalloc ((void**)&devPtrmul, M*M*sizeof(float));

        /*initialize the input matrices to random values and the output matrix to zero*/
        for(int i=0;i<M;i++)    
        {    
            for(int j=0;j<M;j++)    
            {    
                a[IDX2C(i,j,M)] = 2500;
                b[IDX2C(i,j,M)] = 2500;
                mul[IDX2C(i,j,M)] = 0.0;
            }    
        }    
        
        /*OpenBLAS computations*/
        
        cublasCreate(&handle);
        cublasSetMatrix (M, M, sizeof(*a), a, M, devPtra, M);
        cublasSetMatrix (M, M, sizeof(*b), b, M, devPtrb, M);
        cublasSetMatrix (M, M, sizeof(*mul), b, M, devPtrmul, M);

        for(int i=0; i<200; i++)
        {

            auto start = std::chrono::high_resolution_clock::now();
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, M, M, &alpha , devPtra, M, devPtrb, M, &beta, devPtrmul, M);
            cudaDeviceSynchronize();
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);    

            time += duration.count(); 

        
        }
 
        time = time/200.0;

        /*Write duration to file along with dimension number*/
        parallel.open("parallel.txt", std::ios::app);
        parallel << M << "\t\t\t" << time << "\n";
        parallel.close();

        free(a);
        free(b);
        free(mul);
        cudaFree(devPtra);
        cudaFree(devPtrb);
        cudaFree(devPtrmul);

        cublasDestroy(handle);

        M += inc;

    } 


    return 0;

}
