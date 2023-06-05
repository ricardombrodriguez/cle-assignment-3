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

void bitonic_sort(int *sequence, int sequence_size) {
    int count = 2;
    while (count <= sequence_size) {
        for (int i = 0; i < sequence_size; i += count) {
            bitonic_sort_recursive(sequence, i, count, 1);
        }
        count *= 2;
    }
}

void bitonic_sort_recursive(int *sequence, int low, int count, int direction) {
    if (count > 1) {
        int k = count / 2;
        bitonic_sort_recursive(sequence, low, k, 1);
        bitonic_sort_recursive(sequence, low + k, k, 0);
        bitonic_merge(sequence, low, count, direction);
    }
}

/**
*
* @brief Recursively sorts a bitonic sequence.
*
* @param array Array containing the elements to be sorted.
* @param low Starting index of the sub-array to be sorted.
* @param count Number of elements in the sub-array to be sorted.
* @param direction The direction of sorting (1 for ascending, 0 for descending).
*/
void bitonicMergeSort(int *array, int low, int count, int direction)
{
    if (count > 1)
    {
        int k = count / 2;
        bitonicMergeSort(array, low, k, 1);
        bitonicMergeSort(array, low + k, k, 0);
        bitonicMerge(array, low, count, direction);
    }
}

void merge_sorted_arrays(int* arr1, int n1, int* arr2, int n2, int* result) {

    /* Temporary sequence to hold sorted integers */
    int* temp = (int*) malloc((n1 + n2) * sizeof(int));

    /* Merging the two subsequences */
    int i = 0;
    int j = 0;
    int k = 0;

    while (i < n1 && j < n2) {
        if (arr1[i] <= arr2[j]) {
            temp[k++] = arr1[i++];
        } else {
            temp[k++] = arr2[j++];
        }
    }

    while (i < n1) {
        temp[k++] = arr1[i++];
    }

    while (j < n2) {
        temp[k++] = arr2[j++];
    }

    /* Copying results from temporary array variable to the result one */
    for (int idx = 0; idx < n1 + n2; idx++) {
        result[idx] = temp[idx];
    }

    /* Free temp from memory */
    free(temp);
}

/**
 * @brief Validates if the array is sorted in ascending order.
 * 
 * @param array Array to be validated.
 * @param numValues Number of elements in the array.
 * @return 1 if the array is sorted, 0 otherwise.
 */
int validation(int *array, int numValues)
{
    for (int i = 1; i < numValues; i++)
    {
        if (array[i - 1] > array[i])
        {
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}