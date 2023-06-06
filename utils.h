/**
 *  @file utils.h (interface file)
 *
 *  @brief Interface for important program methods
 *
 *  @author Pedro Sobral & Ricardo Rodriguez
 */

#ifndef UTILS_H
# define UTILS_H

#include <stdlib.h>
#include <string.h>

/**
 * @brief Recursively sorts a bitonic sequence, sorting two smaller subsequences and merging them.
 * 
 * @param sequence  Sequence containing the elements to be sorted.
 * @param low Starting index of the sub-array to be sorted.
 * @param count Number of elements in the sub-array to be sorted.
 * @param direction The direction of sorting (1 for ascending, 0 for descending).
 */
void bitonic_sort_recursive(int *sequence, int low, int count, int direction);

/**
 * @brief Sort sequence using bitonic sort, according to the sequence size.
 * 
 * @param sequence Sequence containing the elements to be sorted.
 * @param sequence_size Number of elements in the sequence.
 */
void bitonic_sort(int *sequence, int sequence_size);

/**
 * @brief Validates if the sequence is sorted in ascending order.
 * 
 * @param sequence Sequence to be validated.
 * @param numValues Number of elements in the sequence.
 * @return 1 if the sequence is sorted, 0 otherwise.
 */
int validation(int *sequence, int numValues);

#endif /* UTILS_H */