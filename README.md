# OPL-Final-Project
# Matrix Multiplication in CUDA

## Explanation
Matrix multiplication is a very tedious job as it involves a lot of repeated addition and multiplication operations.
On a normal CPU this can take a very long time depending on the size of the matrix. The time complexity for matrix multiplication is O(n^3) which could take a very long time to compute. This problem can be solved through multithreading and in my case more specifically GPU processing. For my final project I am implementing a simple CUDA matrix multiplication program.

## What I have done
I have wrote a CUDA program that multiplies two matrices together following CUDA guidelines on multithreaded processes. I have distributed the multiplication and addition jobs to different GPU threads in the aims of keeping processing depth as small as possible.

## Breakdown of files
add, add_block, add_cuda and 64block_add are CUDA compiled files that I used for learning the basics of CUDA programs.
add.cu is the file I wrote to learn how to use CUDA.
matrix is the CUDA compiled code for matrixMult.cu.
matrixMult.cu is the file in which I implement matrix multiplication.

## Dependencies
CUDA

## Methodology
I have used the nvprof command that comes with the CUDA package to profile my application. I thought it was sufficient as it does a very good job at profiling performance.

## Performance
 Running matrix multiplication on two 450x450 matrices finish arithmetic operations in 720ms. The program took longer to do API calls than to process the matrix taking almost 1 second on API calls. This is however far better than processing matrix multiplication on a CPU

 ## Conclusion
 GPU processing speeds up what would be considered tedious tasks considerably and programs that can make use of both the CPU and GPU will in turn perform significantly better than just on either.