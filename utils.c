/**
 *  @file utils.c 
 *
 *  @brief Important methods for bitonic sorting and validation
 *
 *  @author Pedro Sobral & Ricardo Rodriguez
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "utils.h"

/**
 * @brief Recursively sorts a bitonic sequence, sorting two smaller subsequences and merging them.
 * 
 * @param sequence  Sequence containing the elements to be sorted.
 * @param low Starting index of the sub-array to be sorted.
 * @param count Number of elements in the sub-array to be sorted.
 * @param direction The direction of sorting (1 for ascending, 0 for descending).
 */
void bitonic_sort_recursive(int *sequence, int low, int count, int direction) {
    if (count > 1) {
        int k = count / 2;
        bitonic_sort_recursive(sequence, low, k, 1);
        bitonic_sort_recursive(sequence, low + k, k, 0);
        bitonic_merge(sequence, low, count, direction);
    }
}

/**
 * @brief Sort sequence using bitonic sort, according to the sequence size.
 * 
 * @param sequence Sequence containing the elements to be sorted.
 * @param sequence_size Number of elements in the sequence.
 */
void bitonic_sort(int *sequence, int sequence_size) {
    int count = 2;
    while (count <= sequence_size) {
        for (int i = 0; i < sequence_size; i += count) {
            bitonic_sort_recursive(sequence, i, count, 1);
        }
        count *= 2;
    }
}

void bitonic_merge(int arr[], int low, int cnt, int dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++) {
            if (1 == (arr[i] > arr[i + k])) {
                int temp = arr[i];
                arr[i] = arr[i + k];
                arr[i + k] = temp;
            }
        }
        bitonic_merge(arr, low, k, dir);
        bitonic_merge(arr, low + k, k, dir);
    }
}

/**
 * @brief Validates if the sequence is sorted in ascending order.
 * 
 * @param sequence Sequence to be validated.
 * @param numValues Number of elements in the sequence.
 * @return 1 if the sequence is sorted, 0 otherwise.
 */
int validation(int *sequence, int numValues)
{

    for (int i = 1; i < numValues; i++)
    {
        if (sequence[i - 1] > sequence[i])
        {
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;

}