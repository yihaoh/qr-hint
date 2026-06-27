#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "postgres.h"
#include "fmgr.h"
#include "utils/bytea.h"

typedef struct
{
    unsigned char *bits;
    size_t size;
} BitArray;

BitArray *alloc_bit_array(size_t size)
{
    BitArray *bit_array = (BitArray *)palloc(sizeof(BitArray));
    if (!bit_array)
        return NULL;
    size_t num_bytes = (size + 7) / 8;
    bit_array->bits = (unsigned char *)palloc(num_bytes);
    if (!bit_array->bits)
    {
        pfree(bit_array);
        return NULL;
    }
    for (size_t i = 0; i < num_bytes; i++)
        bit_array->bits[i] = 0;
    bit_array->size = size;
    return bit_array;
}

void set_bit(BitArray *bit_array, size_t index)
{
    if (index < bit_array->size)
        bit_array->bits[index / 8] |= (1 << (index % 8));
}

void reset_bit(BitArray *bit_array, size_t index)
{
    if (index < bit_array->size)
        bit_array->bits[index / 8] &= ~(1 << (index % 8));
}

void flip_bit(BitArray *bit_array, size_t index)
{
    if (index < bit_array->size)
        bit_array->bits[index / 8] ^= (1 << (index % 8));
}

int get_bit(const BitArray *bit_array, size_t index)
{
    if (index >= bit_array->size)
        return -1;
    return (bit_array->bits[index / 8] >> (index % 8)) & 1;
}

unsigned char reverse(unsigned char b)
{
    b = (b & 0xF0) >> 4 | (b & 0x0F) << 4;
    b = (b & 0xCC) >> 2 | (b & 0x33) << 2;
    b = (b & 0xAA) >> 1 | (b & 0x55) << 1;
    return b;
}

char *to_string(const BitArray *bit_array)
{
    size_t bytes = (bit_array->size + 7) / 8;
    char *string = (char *)palloc(bytes * 2 + 1);
    if (!string)
        return NULL;

    for (size_t i = 0; i < bytes; i++)
    {
        sprintf(string + i * 2, "%02X", reverse(bit_array->bits[i]));
        for (int k = 0; k < 8; k++)
        {
            printf("%d", get_bit(bit_array, i * 8 + k));
        }
        printf("\n");
    }

    string[bytes * 2] = '\0';
    return string;
}

bytea *to_bytea(const BitArray *bit_array)
{
    size_t num_bytes = (bit_array->size + 7) / 8;

    bytea *result = (bytea *)palloc(VARHDRSZ + num_bytes);
    SET_VARSIZE(result, VARHDRSZ + num_bytes);

    // Copy bit_array bits into BYTEA
    memcpy(VARDATA(result), bit_array->bits, num_bytes);

    // Return the BYTEA data
    return result;
}

void free_bit_array(BitArray *bit_array)
{
    if (bit_array)
    {
        pfree(bit_array->bits);
        pfree(bit_array);
    }
}

BitArray *to_bit_array(const bytea *bytea_data)
{
    // Get the size of the bytea data
    size_t num_bytes = VARSIZE(bytea_data) - VARHDRSZ;
    if (num_bytes == 0)
        return NULL; // Handle empty bytea case

    // Create a new BitArray structure
    BitArray *bit_array = (BitArray *)palloc(sizeof(BitArray));
    if (!bit_array)
        return NULL;

    bit_array->size = num_bytes * 8; // Number of bits
    bit_array->bits = (unsigned char*) palloc(num_bytes);
    if (!bit_array->bits)
    {
        pfree(bit_array);
        return NULL;
    }

    // Copy data from bytea to BitArray
    memcpy(bit_array->bits, VARDATA(bytea_data), num_bytes);

    return bit_array;
}