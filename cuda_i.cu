#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

#include "utils.h"

#define MATRIX_SIZE 1024 * 1024         /* Total size of the matrix */
#define MATRIX_SUBSIZE 1024             /* One dimension size of the matrix */
#define MAX_ITERATIONS 10               /* Number of iterations to merge the sequence file */

int main() {

    /* Load file data into host memory */
    ifstream input_file("datSeq1M.bin", ios::binary);

    /* Matrix to store data (integers) *from the input file */
    vector<int> matrix(MATRIX_SIZE);

    /* Allocate device memory for the sequence */
    int* device_sequence;
    CHECK (cudaMalloc((void**) &device_sequence, MATRIX_SIZE * sizeof(int)));

    (void) get_delta_time ();

    /* Read file data (MATRIX_SIZEintegers) and store it in the matrix vector */
    input_file.read(reinterpret_cast<char*>(matrix.data()), MATRIX_SIZE * sizeof(int));
    print("Data was read and stored in the host.");

    /* Copy the host data to device memory */
    CHECK (cudaMemcpy(device_sequence, matrix.data(), MATRIX_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    printf("Data was copied from the host memory to the device memory");

    /* Perform iterations for a number of MAX_ITERATIONS. Each iteration will create the kernels, to sort and merge the subsequences, according to the estimated size */
    for (int iteration = 1; iteration <= MAX_ITERATIONS; iteration++) {

        /* Get the subsequence current size by multiplying the MATRIX_SUBSIZE with power of 2 values, using the left shift operator */
        int subsequence_size = MATRIX_SUBSIZE * (1 << (iteration - 1));    

        /* Get the number of subsequences by dividing the MATRIX_SUBSIZE with power of 2 values, using the left shift operator */
        int num_subsequences = MATRIX_SUBSIZE / (1 << (iteration - 1));

        /* Launch sorting kernel, for each subsequence. With num_subsequence grids, each one having subsequence_size threads */
        sortSubsequence<<<num_subsequences, subsequence_size>>>(device_sequence, subsequence_size);

        /* Syncronize all sorting kernels to proceed to the merging phase */
        CHECK (cudaDeviceSynchronize());
        CHECK (cudaGetLastError());

        /* Launch merging kernel, for each pair of previously sorted subsequences */
        for (int i = 0; i < num_subsequences / 2; i++) {
            mergeSubsequences<<<num_subsequences, subsequence_size>>>(device_sequence + i * subsequence_size * 2, subsequence_size * 2);
        }

        /* Syncronize all sorting kernels to mark the end of the merging phase */
        CHECK (cudaDeviceSynchronize());
        CHECK (cudaGetLastError());

    }

    /* Copy the sorted sequence from device memory to host memory */
    vector<int> sorted_sequence(MATRIX_SIZE);
    CHECK (cudaMemcpy(sorted_sequence.data(), device_sequence, MATRIX_SIZE * sizeof(int), cudaMemcpyDeviceToHost));
    printf("Sorted sequence was copied from the device memory to the host memory");

    /* Perform validation of the sorted full sequence */
    if (validation(sorted_sequence, MATRIX_SIZE) == EXIT_FAILURE) {
        printf("[ERROR] Validation of the sequence failed, the sequence is not properly sorted.");
        exit(EXIT_FAILURE);
    };

    printf("The validation of the final sequence was a success. The sequence is properly sorted.");

    /* Clean up device memory */
    CHECK (cudaFree(device_sequence));

    printf("End.");

}

/* CUDA kernel to sort the integer sequence with a specific size */
__global__ void sort_subsequence(int* sequence, int size) {

    /* CUDA Thread index */
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    /* Ver subarray da thread consoante a thread_idx */

    /* Sort subsequence */
    bitonic_sort(sequence, size);

}

/* CUDA kernel responsible for merging two sorted subsequences from previous iterations */
__global__ void merge_subsequences(int *sequence, int size) {
    
    /* CUDA Thread index */
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    /* Ver subarrays que vão formar um par */

    /* Merge the two sorted subsequences */
    merge_sorted_arrays();

}

static double get_delta_time(void)
{
  static struct timespec t0,t1;

  t0 = t1;
  if (clock_gettime(CLOCK_MONOTONIC,&t1) != 0)
  {
    perror("clock_gettime");
    exit(1);
  }
  return (double)(t1.tv_sec - t0.tv_sec) + 1.0e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
}