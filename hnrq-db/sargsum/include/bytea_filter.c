// Support for the bloomfilter
#include "bytea_filter.h"

#include "postgres.h"
#include "fmgr.h"

// Return the bit at given `index` in the bloomfilter
int get_bit_bloom(char *bloom_array, size_t length, int index)
{
    if (index < (length * 8))
    {
        return (bloom_array[index / 8] >> (index % 8)) & 1;
    }
}

// Set the bit at given `index` in the bloomfilter
void set_bit_bloom(char *bloom_array, size_t length, int index)
{
    if (index < (length * 8))
    {
        bloom_array[index / 8] |= (1 << (index % 8));
    }
    return;
}

// Convert the char* bloomfilter to a bytea*
bytea *to_bytea(char *bloom_array, size_t length_bits)
{
    size_t num_bytes = (length_bits + 7) / 8;
    bytea *result = (bytea *)palloc(VARHDRSZ + num_bytes);
    SET_VARSIZE(result, VARHDRSZ + num_bytes);
    memcpy(VARDATA(result), bloom_array, num_bytes);
    return result;
}

// Given a HeapTupleHeader, extract the values for filt, m, n,
void get_blmfl_result_attributes(HeapTupleHeader tuple, char **filt, int32 *m, int32 *n, int32 *k, int32 *t)
{
    bool isnull;

    *m = GetAttributeByName(tuple, "m", &isnull);
    *n = GetAttributeByName(tuple, "n", &isnull);
    *k = GetAttributeByName(tuple, "k", &isnull);
    *t = GetAttributeByName(tuple, "t", &isnull);
    bytea *bloom = DatumGetByteaP(GetAttributeByName(tuple, "filt", &isnull));

    if (isnull)
    {
        ereport(ERROR,
                errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
                errmsg("Attributes cannot be null."));
    }

    size_t bloom_raw_length = VARSIZE_ANY(bloom) - VARHDRSZ;
    *filt = VARDATA_ANY(bloom);

    if (*m != (bloom_raw_length * 8))
    {
        ereport(ERROR,
                errcode(ERRCODE_DATA_EXCEPTION),
                errmsg("Lengths do not match"));
    }
}

// Get data to be inserted into/checked in bloomfilter
void get_input_data(bytea *input_data, size_t *data_raw_length, char **raw_data)
{
    *data_raw_length = VARSIZE_ANY(input_data) - VARHDRSZ;
    *raw_data = VARDATA_ANY(input_data);
}
