#ifndef BYTEA_FILTER
#define BYTEA_FILTER

#include "postgres.h"
#include "fmgr.h"

int get_bit_bloom(char *bloom_array, size_t length, int index);
void set_bit_bloom(char *bloom_array, size_t length, int index);
bytea *to_bytea(char *bloom_array, size_t length_bits);
void get_blmfl_result_attributes(HeapTupleHeader tuple, char **filt, int32 *m, int32 *n, int32 *k, int32 *t);
void get_input_data(bytea *input_data, size_t *data_raw_length, char **raw_data);

#endif  // BYTEA_FILTER