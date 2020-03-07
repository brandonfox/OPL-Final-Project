#include<iostream>
#include<stdlib.h>
#include<math.h>

#define MAX_THREADS 512

typedef struct _matrix {
    int xDim;
    int yDim;
    int *vals;
} matrix;

__global__
void doMultiplications(_matrix *a, _matrix *b, int* resultMatrix,int row, int col){ //Result matrix must be a.xDim * b.xDim * a.yDim length array
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int startIndex = (row * a->xDim * b->xDim + col);
    // std::cout << "Doing request, Start index for result array:" << startIndex << ". Doing row:" << row << ", " << col << std::endl; 
    for(int i = index; i < b->xDim; i+= stride){
        resultMatrix[startIndex + i*a->xDim] = a->vals[row * a->xDim + col] * b->vals[col * b->xDim + i];
    }
}
__global__
void doAdds(_matrix *a, _matrix *b, int* multVals, _matrix *c){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int numToDo = a->xDim;
    int matrixDimensions = c->xDim * c->yDim;
    for(int a = index; a < matrixDimensions; a += stride){
        for(int i = 0; i < numToDo; i++){
            c->vals[a] += multVals[a * numToDo + i];
        }
    }
}

void printArray(int *vals, int size){
    std::cout << "[";
    for(int i = 0; i < size; i++){
        std::cout << vals[i] << ", ";
    }
    std::cout << "]" << std::endl;
}

void printMatrix(_matrix *m){
    for(int i = 0; i < m->yDim; i++){
        std::cout << "[ ";
        for(int j = 0; j < m->xDim; j++){
            std::cout << m->vals[i*m->xDim + j] << " ";
        }
        std:: cout << "]" << std::endl;
    }
    std::cout << std::endl;
}

_matrix *multiplyMatrices(_matrix *a, _matrix *b){
    int *multiplications;
    int multiplicationSize = a->xDim * b->xDim * a->yDim;
    cudaMallocManaged(&multiplications,multiplicationSize * sizeof(int));
    int threads = (b->xDim / 32 + 1) * 32;
    int blocks = 1;
    if(threads > MAX_THREADS){
        blocks = threads/MAX_THREADS + 1;
        threads = MAX_THREADS;
    }
    std::cout << "Determined that the matrices needs " << blocks << " blocks and " << threads << " threads." << std::endl;
    std::cout << "Sending multiplication requests..." << std::endl;
    int rows = a->yDim;
    int cols = a->xDim;
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            doMultiplications<<<blocks,threads>>>(a,b,multiplications,i,j);
        }
    }
    std::cout << "Sent multiplication requests." << std::endl;
    matrix *c;
    cudaMallocManaged(&c,sizeof(matrix));
    cudaDeviceSynchronize();
    std::cout << "Finished multiplication operations." << std::endl;
    c->yDim = a->yDim;
    c->xDim = b->xDim;
    int matrixSize = c->yDim * c->xDim;
    cudaMallocManaged(&c->vals,sizeof(int) * matrixSize);
    threads = (matrixSize/32 +1)* 32;
    blocks = 1;
    if(threads > MAX_THREADS){
        blocks = threads/MAX_THREADS + 1;
        threads = MAX_THREADS;
    }
    std::cout << "Determined that the matrix needs: " << blocks << " blocks and " << threads << " threads to add. Needs a total of " << matrixSize << " add operations." << std::endl;
    doAdds<<<blocks,threads>>>(a,b,multiplications,c);
    cudaDeviceSynchronize();
    std::cout << "Finished all adds." << std::endl;
    cudaFree(multiplications);
    return c;
}

_matrix *generateRandomMatrix(int x, int y){
    int *vals;
    cudaMallocManaged(&vals,sizeof(int) * x * y);
    for(int i = 0; i < y; i++){
        for(int j = 0; j < x; j++){
            vals[i*x + j] = rand()%5;
        }
    }
    matrix *m;
    cudaMallocManaged(&m,sizeof(matrix));
    m->xDim = x;
    m->yDim = y;
    m->vals = vals;
    return m;
}

void deleteMatrix(_matrix *m){
    cudaFree(m->vals);
    cudaFree(m);
}

int main(void){
    matrix *a = generateRandomMatrix(450,450);
    matrix *b = generateRandomMatrix(450,450);

    // printMatrix(a);
    // printMatrix(b);

    matrix *c = multiplyMatrices(a,b);

    // printMatrix(c);

    deleteMatrix(a);
    deleteMatrix(b);
    deleteMatrix(c);

    return 0;
}