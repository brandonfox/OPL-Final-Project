#include<iostream>

typedef struct _matrix {
    int xDim;
    int yDim;
    int *vals;
};

__global__
void doMultiplications(_matrix *a, _matrix *b, int* resultMatrix){ //Result matrix must be a.yDim * b.xDim * a.xDim length array
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    

_matrix addMatrices(_matrix *a, _matrix *b){
    int *multiplications;
    int multiplicationSize = a->yDim * b->xDim * a->xDim * sizeof(int);
    cudaMallocManaged(&multiplications,matrixSize);
    doMultiplications<<<1, 256>>>(a,b,multiplications);

}

void main(void){

}