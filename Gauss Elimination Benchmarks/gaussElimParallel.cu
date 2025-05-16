#include <cstdlib> 
#include <iostream> 
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include <unistd.h>
#include <random>

/*The method for row swap seems a bit messy but is a pretty inexpensive solution for what would require memory transfer through additional functions.*/

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

__global__ void Gauss_Elimination(float* a, float* b, int matrixSize)
{
    int x = threadIdx.x/matrixSize;
    int y = threadIdx.x%matrixSize;

    float weight = 0.0;     // Weight for each row
    float pivot = 0.0;        // Matrix value at current pivot position
    int offset = 0;             // Row swapping 

    for(int currentRow=0; currentRow<matrixSize; currentRow++)
    {
        /*If the next pivot value is zero and the current row is not the last row then use offset variable to swap rows*/
        if(a[IDX2C(currentRow+offset, currentRow, matrixSize)] == 0  && currentRow!=(matrixSize-1)) 
        {
            offset = 1;     // Switch to the next row with non-zero pivot
        }
        else if(offset==1)
        {
            offset = -1;    // Switch back to the previous row which now has non-zero pivot
        }
        else 
        {
            offset = 0;     // Reset to the original algorithm
        }

        pivot = a[IDX2C(currentRow+offset, currentRow, matrixSize)];        // Get value at pivot

        __syncthreads();

        a[IDX2C(currentRow+offset, y, matrixSize)] /= pivot;                          // Divide vector and matrix pivot row by pivot value
        if(y==0) b[currentRow+offset] /= pivot;
        
        weight = a[IDX2C(x, currentRow, matrixSize)];                                      // Assign weight

        if(x!=currentRow+offset) 
        {
            a[IDX2C(x, y, matrixSize)] -= weight * a[IDX2C(currentRow+offset, y, matrixSize)];      // Row operations
            if(y==0) b[x] -= weight * b[currentRow+offset];
        }

        __syncthreads();

    }
}

int main()
{

    int matrixSize=0.0;   

    std::cout << "Enter Matrix Size (Max 32): ";
    std::cin >> matrixSize;
   
    double time = 0.0;

    /*Host Pointers - a is the matrix and b is the vector*/
    float *a = (float*) malloc(sizeof(float)*matrixSize*matrixSize);
    float *b = (float*) malloc(sizeof(float)*matrixSize);

    float  *devPtra, *devPtrb;

    /*Device Pointers*/
    cudaMalloc ((void**)&devPtra, matrixSize*matrixSize*sizeof(float));
    cudaMalloc ((void**)&devPtrb, matrixSize*sizeof(float));

    /*Initialize the input matrix and vector to random values between 0-999*/
    setMatrix(a,b,matrixSize);    

    /*Copying the values to GPU*/
    cudaMemcpy((void*)devPtra, (void*)a, matrixSize*matrixSize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)devPtrb, (void*)b, matrixSize*sizeof(float), cudaMemcpyHostToDevice);
    
    /*Create Output File and Print Matrix to File*/
    std::ofstream parallel("parallel.txt");
    parallel << std::right;
    parallel << "Input Augmented Matrix with Vector" << "\n\n";
    putMatrix(parallel, a, b, matrixSize);
    parallel.close();

    /*Run and Time the Kernel*/
    dim3 blockSize(matrixSize*matrixSize, 1, 1);
    dim3 gridSize(1, 1, 1);

    auto start = std::chrono::high_resolution_clock::now();
    Gauss_Elimination<<<gridSize, blockSize>>>(devPtra, devPtrb, matrixSize);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);    

    time = duration.count(); 

    cudaMemcpy((void*)b, (void*)devPtrb, matrixSize*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)a, (void*)devPtra, matrixSize*matrixSize*sizeof(float), cudaMemcpyDeviceToHost);

    /*Write output to file*/
    parallel.open("parallel.txt", std::ios::app);
    parallel << std::right;
    parallel<< "\n\n" << "Output Reduced Augmented matrix with Solution Vector" << "\n\n";
    putMatrix(parallel, a, b, matrixSize);
    parallel.close();

    /*Write duration to file along with dimension number*/
    parallel.open("parallel.txt", std::ios::app);
    parallel << "\n\n" << "Time taken to execute in Nanoseconds - " << time << " ns" << "\n";
    parallel << "Matrix Size - " << matrixSize << "\n";
    parallel.close();

    /*Free Memory*/
    free(b);
    free(a);
    cudaFree(devPtra);
    cudaFree(devPtrb);

    return 0;
}