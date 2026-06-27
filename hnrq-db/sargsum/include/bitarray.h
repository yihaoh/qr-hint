#include "bitarray.c"

BitArray *alloc_bit_array(size_t size);
void set_bit(BitArray *bit_array, size_t index);
unsigned char reverse(unsigned char b);
BitArray *to_bit_array(const bytea *bytea_data);
char *to_string(const BitArray *bit_array);
void free_bit_array(BitArray *bit_array);