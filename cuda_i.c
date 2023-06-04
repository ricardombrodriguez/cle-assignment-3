#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

#define MATRIX_SIZE 1024
#define MAX_ITERATIONS 10

/* CUDA kernel to sort the integer sequence with a specific size */
__global__ void sort_subsequence(int* sequence, int size) {

    /* CUDA Thread index */
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    /* Sort subsequence */

}

/* CUDA kernel responsible for merging two sorted subsequences from previous iterations */
__global__ void merge_subsequences(int* sequence, int size) {
    
    /* CUDA Thread index */
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    /* Merge the two sorted subsequences */

}

int main() {

    /* Load file data into host memory */
    ifstream input_file("datSeq1M.bin", ios::binary);

    /* Matrix */
    vector<int> matrix(MATRIX_SIZE * MATRIX_SIZE);


    input_file.read(reinterpret_cast<char*>(matrix.data()), MATRIX_SIZE * MATRIX_SIZE * sizeof(int));

    /* Allocate device memory for the sequence */
    int* device_sequence;
    cudaMalloc((void**)&device_sequence, MATRIX_SIZE * MATRIX_SIZE * sizeof(int));

    // Copy the host data to device memory
    cudaMemcpy(device_sequence, matrix.data(), MATRIX_SIZE * MATRIX_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    /* Perform iterations for a number of MAX_ITERATIONS. Each iteration will create the kernels, to sort and merge the subsequences, according to the estimated size */
    for (int iteration = 1; iteration <= MAX_ITERATIONS; iteration++) {

        int subsequence_size = MATRIX_SIZE / (1 << (iteration - 1));
        int num_subsequences = 1 << (iteration - 1);
        int block_size = 1024;  // Adjust the block size as needed
        int grid_size = (subsequence_size * num_subsequences + block_size - 1) / block_size;

        /* Launch sorting kernel, for each subsequence */
        sortSubsequence<<<grid_size, block_size>>>(device_sequence, subsequence_size);

        /* Syncronize all sorting kernels to proceed to the merging phase */
        cudaDeviceSynchronize();

        /* Launch merging kernel, for each pair of previously sorted subsequences */
        for (int i = 0; i < num_subsequences / 2; i++) {
            mergeSubsequences<<<grid_size, block_size>>>(device_sequence + i * subsequence_size * 2, subsequence_size * 2);
        }

        /* Syncronize all sorting kernels to mark the end of the merging phase */
        cudaDeviceSynchronize();

    }

    /* Copy the sorted sequence from device memory to host memory */
    vector<int> sorted_sequence(MATRIX_SIZE * MATRIX_SIZE);
    cudaMemcpy(sorted_sequence.data(), device_sequence, MATRIX_SIZE * MATRIX_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    /* Clean up device memory */
    cudaFree(device_sequence);

    /* Perform validation of the sorted full sequence */

}