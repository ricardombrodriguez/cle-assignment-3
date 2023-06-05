#include <stdlib.h>
#include <string.h>

/**
 * @brief Merges sub-sequences in a bitonic sequence.
 * 
 * @param array Sequence containing the elements to be merged.
 * @param low Starting index of the sub-sequence to be merged.
 * @param sequence_size Number of elements in the sub-sequence to be merged.
 * @param direction The direction of sorting (1 for ascending, 0 for descending).
 */
extern void bitonicMerge(int *sequence, int low, int sequence_size, int direction);

/**
 * @brief Recursively sorts a bitonic sequence.
 * 
 * @param sequence Sequence containing the elements to be sorted.
 * @param low Starting index of the sub-sequence to be sorted.
 * @param sequence_size Number of elements in the sub-sequence to be sorted.
 * @param direction The direction of sorting (1 for ascending, 0 for descending).
 */
extern void bitonicMergeSort(int *sequence, int low, int sequence_size, int direction);

/**
 * @brief Validates if the sequence is sorted in ascending order.
 * 
 * @param sequence Sequence to be validated.
 * @param sequence_size Number of elements in the sequence.
 * @return EXIT_SUCCESS if the sequence is sorted, EXIT_FAILURE otherwise.
 */
extern int validation(int *sequence, int sequence_size);


#endif /* UTILS_H */