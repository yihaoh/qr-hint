#include <stdint.h>
#include <math.h>
#include <string.h>

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/memutils.h"
#include "catalog/pg_type.h"
#include "utils/lsyscache.h"
#include "utils/errcodes.h"
#include "utils/array.h"
#include "utils/numeric.h"
#include "utils/builtins.h"
#include "utils/typcache.h"
#include "utils/elog.h"
#include "utils/geo_decls.h"
#include "utils/guc.h"
#include "executor/executor.h"

#include "include/bytea_filter.c"
#include "include/converters.c"
#include "include/murmur3.c"

PG_MODULE_MAGIC;

typedef struct
{
    char *filt;
    int32 m; // size of filter (in bits)
    int32 n; // expected capacity
    int32 k; // number of hash functions
    int32 t; // total inserted count
} blmfl_state_t;

typedef struct
{
    bytea filt;   // Bloomfilter
    int32 m;      // Size of filter (in bits)
    int32 n;      // Expected capacity (i.e. number of elements expected to be inserted into bloomfilter)
    int32 k;      // Number of hash functions
    int32 t;      // Total number of elements inserted
} BlmflResult;

static int BLMFL_M;
static int BLMFL_N;
static int BLMFL_K;

void _PG_init(void) {
    // Define custom GUC parameters
    // These values are set in SQL; see sql/blmfl_test.sql for examples. 

    DefineCustomIntVariable(
        "blmfl.bloomfilter_bitsize",
        "Size of the bloomfilter in bits",
        NULL, &BLMFL_M, 0, 0, INT_MAX, PGC_USERSET, 0, 
        NULL, NULL, NULL);   

    DefineCustomIntVariable(
        "blmfl.estimated_count",
        "Estimated ount of elements in bloomfilter",
        NULL, &BLMFL_N, 0, 0, INT_MAX, PGC_USERSET, 0, 
        NULL, NULL, NULL);    

    DefineCustomIntVariable(
        "blmfl.num_hashes",
        "Number of times the input is hashed",
        NULL, &BLMFL_K, 0, 0, INT_MAX, PGC_USERSET, 0, 
        NULL, NULL, NULL);
}

/********************************************************************************
  bloom filter utils
 ********************************************************************************/

unsigned int optimal_k(int m, int n)
{
    return round(log(2) * m / n);
}

double false_positive_rate(int m, int n, int k)
{
    return pow((1 - exp(-1.0 * k * n / m)), (double)k);
}

unsigned int murmur3_hash(char *data, u_int32_t data_length, int i, int m)
{
    uint32_t hashed_data;
    MurmurHash3_x86_32(data, data_length, i, &hashed_data);
    int result = hashed_data % m;
    return result;
}

unsigned int murmur3_hash_any(char **data, int* data_length, int nargs, int i, int m)
{
    int result = 0;
    for (int index = 1; index < nargs; index++) {
        uint32_t hashed_data;
        MurmurHash3_x86_32(data[index], data_length[index], 0, &hashed_data);
        result = hashed_data;
    }
    return (result % m + m) % m;
}

unsigned int unique_count_estimate(int m, int k, char *filt)
{
    int num_set_bits = 0;
    int count = 0;

    for (int i = 0; i < m; i++) {
        if (get_bit_bloom(filt, m, i) == 1){
            num_set_bits++;
        }
    }

    // Calculate the logarithm (cast to double for correct division)
    double ratio = (double)num_set_bits / m;
    double log_value = log(ratio);

    // Calculate count
    count = -(unsigned int)((m * log_value) / k);    
    return count;
}

// Create the return tuple in required format
HeapTuple create_return_tuple(MemoryContext aggr_context, blmfl_state_t *sp)
{
    const int tup_len = 5;
    TupleDesc tupdesc;
    Datum values[5];
    bool nulls[5];

    MemSet(values, 0, sizeof(values));
    MemSet(nulls, 0, sizeof(nulls));

    tupdesc = CreateTemplateTupleDesc(tup_len);
    TupleDescInitEntry(tupdesc, (AttrNumber)1, "filt", BYTEAOID, -1, 0);
    TupleDescInitEntry(tupdesc, (AttrNumber)2, "m", INT4OID, -1, 0);
    TupleDescInitEntry(tupdesc, (AttrNumber)3, "n", INT4OID, -1, 0);
    TupleDescInitEntry(tupdesc, (AttrNumber)4, "k", INT4OID, -1, 0);
    TupleDescInitEntry(tupdesc, (AttrNumber)5, "t", INT4OID, -1, 0);

    tupdesc = BlessTupleDesc(tupdesc);

    values[0] = PointerGetDatum(to_bytea(sp->filt, sp->m));
    values[1] = Int32GetDatum(sp->m);
    values[2] = Int32GetDatum(sp->n);
    values[3] = Int32GetDatum(sp->k);
    values[4] = Int32GetDatum(sp->t);

    HeapTuple rettuple = heap_form_tuple(tupdesc, values, nulls);
    return rettuple;
}

/********************************************************************************
  function for testing membership of a particular element
 ********************************************************************************/

PG_FUNCTION_INFO_V1(blmfl_test);
Datum blmfl_test(PG_FUNCTION_ARGS)
{
    HeapTupleHeader tuple = PG_GETARG_HEAPTUPLEHEADER(0); // Header is of type BLMFL_RESULT
    int32 m, n, k, t;
    char *filt;
    
    // Read the BLMFL_RESULT fields
    get_blmfl_result_attributes(tuple, &filt, &m, &n, &k, &t);

    // Get the data to hash and check presence of in bloomfilter
    u_int32_t raw_length;
    char *raw_data;
    get_input_data(PG_GETARG_BYTEA_P(1), &raw_length, &raw_data);

    // Check if data is present
    bool is_present = true;
    for (int i = 0; i < k; i++) {
        int hash_result = murmur3_hash(raw_data, raw_length, i, m);
        if (get_bit_bloom(filt, m, hash_result) != 1){
            is_present = false;
        }
    }
    
    // Return
    Datum result = BoolGetDatum(is_present);
    PG_RETURN_DATUM(result);
}

PG_FUNCTION_INFO_V1(blmfl_test_any);
Datum blmfl_test_any(PG_FUNCTION_ARGS)
{
    HeapTupleHeader tuple = PG_GETARG_HEAPTUPLEHEADER(0); // Header is of type BLMFL_RESULT
    int32 m, n, k, t;
    char *filt;
    
    // Read the BLMFL_RESULT fields
    get_blmfl_result_attributes(tuple, &filt, &m, &n, &k, &t);

    int nargs = PG_NARGS(); // Get the number of arguments
    char **argument_array = palloc((nargs) * sizeof(char*)); // Initialize a buffer to store concatenated argument data
    int *argument_sizes = palloc((nargs) * sizeof(int));

    convert_input_data(fcinfo, &argument_array, &argument_sizes, nargs);

    // Check if data is present
    bool is_present = true;
    for (int i = 0; i < k; i++) {
        int hash_result = murmur3_hash_any(argument_array, argument_sizes, nargs, i, m);
        if (get_bit_bloom(filt, m, hash_result) != 1){
            is_present = false;
        }
    }

    pfree(argument_array);
    pfree(argument_sizes);
    
    Datum result = BoolGetDatum(is_present);
    PG_RETURN_DATUM(result);
}

/********************************************************************************
  functions for creating and updating the bloomfilter
 ********************************************************************************/

static blmfl_state_t *blmfl_state_new_n(MemoryContext aggr_context)
{
    MemoryContext tmp_context = AllocSetContextCreate(aggr_context,
                                                      "blmfl_state",
                                                      ALLOCSET_DEFAULT_MINSIZE,
                                                      ALLOCSET_DEFAULT_INITSIZE,
                                                      ALLOCSET_DEFAULT_MAXSIZE);
    MemoryContext old_context = MemoryContextSwitchTo(tmp_context);
    blmfl_state_t *sp = (blmfl_state_t *)palloc(sizeof(blmfl_state_t));
    sp->m = BLMFL_M;
    sp->n = BLMFL_N;
    sp->k = BLMFL_K;
    sp->t = 0;
    sp->filt = (char*) palloc(sp->m);
    
    MemSet(sp->filt, 0, sp->m);
    MemoryContextSwitchTo(old_context);
    return sp;
}

static void blmfl_state_add(blmfl_state_t *sp, char *data, u_int32_t data_length)
{
    for (int i = 0; i < sp->k; i++) {
        int hash_result = murmur3_hash(data, data_length, i, sp->m);
        set_bit_bloom(sp->filt, sp->m, hash_result);
    }
    sp->t = sp->t + 1;
    return;
}

// State Function
PG_FUNCTION_INFO_V1(blmfl_sfunc);
Datum blmfl_sfunc(PG_FUNCTION_ARGS)
{
    MemoryContext aggr_context;
    if (!AggCheckCallContext(fcinfo, &aggr_context)){
        ereport(ERROR,
                (errcode(ERRCODE_DATA_EXCEPTION),
                 errmsg("blmfl_sfunc outside transition context")));
    }

    if (PG_NARGS() <= 1){
        ereport(ERROR,
                (errcode(ERRCODE_DATA_EXCEPTION),
                 errmsg("The bloomfilter expects at least 1 data element")));
    }

    blmfl_state_t *sp;
    // If state does not exist (i.e. a first call), create it. 
    if (PG_ARGISNULL(0)) {
        sp = blmfl_state_new_n(aggr_context);
    // Else, assign the existing one. 
    } else {
        sp = (blmfl_state_t *)(PG_GETARG_POINTER(0));
    }

    // Get the data to hash and check presence of in bloomfilter
    u_int32_t raw_length;
    char *raw_data;
    get_input_data(PG_GETARG_BYTEA_P(1), &raw_length, &raw_data);

    // Insert into bloomfilter
    blmfl_state_add(sp, raw_data, raw_length);
    
    PG_RETURN_POINTER(sp);
}

// Final Function
PG_FUNCTION_INFO_V1(blmfl_ffunc);
Datum blmfl_ffunc(PG_FUNCTION_ARGS) {
    MemoryContext aggr_context;
    if (!AggCheckCallContext(fcinfo, &aggr_context)) {
        ereport(ERROR,
                (errcode(ERRCODE_DATA_EXCEPTION),
                 errmsg("blmfl_ffunc outside transition context")));
    }

    blmfl_state_t *sp;
    if (PG_ARGISNULL(0)) {
            ereport(ERROR,
                (errcode(ERRCODE_DATA_EXCEPTION),
                 errmsg("Bloomfilter state is null.")));    }

    sp = (blmfl_state_t *)(PG_GETARG_POINTER(0));
    if (sp->filt == 0)
    {
        ereport(ERROR,
                (errcode(ERRCODE_DATA_EXCEPTION),
                 errmsg("Bloomfilter is null.")));
    }

    HeapTuple rettuple = create_return_tuple(aggr_context, sp);
    PG_RETURN_DATUM(HeapTupleGetDatum(rettuple));
}

/********************************************************************************
  functions for getting more information about the bloomfilter
 ********************************************************************************/
PG_FUNCTION_INFO_V1(blmfl_approx_unique_count);
Datum blmfl_approx_unique_count(PG_FUNCTION_ARGS)
{
    HeapTupleHeader tuple = PG_GETARG_HEAPTUPLEHEADER(0); // Header is of type BLMFL_RESULT
    int32 m, n, k, t;
    char *filt;
    
    // Read the BLMFL_RESULT fields
    get_blmfl_result_attributes(tuple, &filt, &m, &n, &k, &t);

    // Get estimate
    int count = unique_count_estimate(m, k, filt);
    PG_RETURN_DATUM(Int32GetDatum(count));
}

PG_FUNCTION_INFO_V1(blmfl_optimal_k);
Datum blmfl_optimal_k(PG_FUNCTION_ARGS)
{
    int m = PG_GETARG_INT32(0);
    int n = PG_GETARG_INT32(1);
    
    int k = optimal_k(m, n);
    PG_RETURN_DATUM(Int32GetDatum(k));
}

PG_FUNCTION_INFO_V1(blmfl_fpr);
Datum blmfl_fpr(PG_FUNCTION_ARGS)
{
    bool isnull;
    HeapTupleHeader t = PG_GETARG_HEAPTUPLEHEADER(0);
    int32 m = GetAttributeByName(t, "m", &isnull);
    int32 n = GetAttributeByName(t, "n", &isnull);
    int32 k = GetAttributeByName(t, "k", &isnull);
    
    float fpr = false_positive_rate(m, n, k);
    PG_RETURN_DATUM(Float8GetDatum(fpr));
}

PG_FUNCTION_INFO_V1(blmfl_merge);
Datum blmfl_merge(PG_FUNCTION_ARGS)
{
    // Read the BLMFL_RESULT fields
    HeapTupleHeader r1 = PG_GETARG_HEAPTUPLEHEADER(0); // Header is of type BLMFL_RESULT
    int32 m1, n1, k1, t1;
    char *filt1;
    get_blmfl_result_attributes(r1, &filt1, &m1, &n1, &k1, &t1);

    HeapTupleHeader r2 = PG_GETARG_HEAPTUPLEHEADER(1); // Header is of type BLMFL_RESULT
    int32 m2, n2, k2, t2;
    char *filt2;
    get_blmfl_result_attributes(r2, &filt2, &m2, &n2, &k2, &t2);

    if (m1 != m2) {
        ereport(ERROR,
            (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
             errmsg("Bloomfilter lengths do not match.")));
    }

    char *bloom_merged = palloc(m1);
    for (int i = 0; i < m1; i++)
        if (get_bit_bloom(filt1, m1, i) == 1 || get_bit_bloom(filt2, m2, i) == 1)
            set_bit_bloom(bloom_merged, m1, i);

    const int tup_len = 5;
    TupleDesc tupdesc;
    Datum values[5];
    bool nulls[5];

    MemSet(values, 0, sizeof(values));
    MemSet(nulls, 0, sizeof(nulls));

    tupdesc = CreateTemplateTupleDesc(tup_len);
    TupleDescInitEntry(tupdesc, (AttrNumber)1, "filt", BYTEAOID, -1, 0);
    TupleDescInitEntry(tupdesc, (AttrNumber)2, "m", INT4OID, -1, 0);
    TupleDescInitEntry(tupdesc, (AttrNumber)3, "n", INT4OID, -1, 0);
    TupleDescInitEntry(tupdesc, (AttrNumber)4, "k", INT4OID, -1, 0);
    TupleDescInitEntry(tupdesc, (AttrNumber)5, "t", INT4OID, -1, 0);

    tupdesc = BlessTupleDesc(tupdesc);

    values[0] = PointerGetDatum(to_bytea(bloom_merged, m1));
    values[1] = Int32GetDatum(m1);
    values[2] = Int32GetDatum(n1+n2);
    values[3] = Int32GetDatum(k1);
    values[4] = Int32GetDatum(t1+t2);

    HeapTuple rettuple = heap_form_tuple(tupdesc, values, nulls);
    PG_RETURN_DATUM(HeapTupleGetDatum(rettuple));
}

/********************************************************************************
  [VARIADIC "ANY"] --> functions for creating and updating the bloomfilter
 ********************************************************************************/
static blmfl_state_t *blmfl_state_new_n_any(MemoryContext aggr_context)
{
    MemoryContext tmp_context = AllocSetContextCreate(aggr_context,
                                                      "blmfl_state",
                                                      ALLOCSET_DEFAULT_MINSIZE,
                                                      ALLOCSET_DEFAULT_INITSIZE,
                                                      ALLOCSET_DEFAULT_MAXSIZE);
    MemoryContext old_context = MemoryContextSwitchTo(tmp_context);
    blmfl_state_t *sp = (blmfl_state_t *)palloc(sizeof(blmfl_state_t));
    sp->m = BLMFL_M;
    sp->n = BLMFL_N;
    sp->k = BLMFL_K;
    sp->t = 0;
    sp->filt = (char*) palloc(sp->m);

    MemSet(sp->filt, 0, sp->m);
    MemoryContextSwitchTo(old_context);
    return sp;
}

static void blmfl_state_add_any(blmfl_state_t *sp, char **data, int *data_length, int nargs)
{
    for (int i = 0; i < sp->k; i++) {
        uint hash_result = murmur3_hash_any(data, data_length, nargs, i, sp->m);
        set_bit_bloom(sp->filt, sp->m, hash_result);
    }

    sp->t = sp->t + 1;
    return;
}

// State Function
PG_FUNCTION_INFO_V1(blmfl_sfunc_any);
Datum blmfl_sfunc_any(PG_FUNCTION_ARGS)
{
    MemoryContext aggr_context;
    if (!AggCheckCallContext(fcinfo, &aggr_context)){
        ereport(ERROR,
                (errcode(ERRCODE_DATA_EXCEPTION),
                 errmsg("blmfl_sfunc outside transition context")));
    }

    if (PG_NARGS() <= 1){
        ereport(ERROR,
                (errcode(ERRCODE_DATA_EXCEPTION),
                 errmsg("The bloomfilter expects at least 1 data element")));
    }

    blmfl_state_t *sp;
    // If state does not exist (i.e. a first call), create it. 
    if (PG_ARGISNULL(0)) {
        sp = blmfl_state_new_n_any(aggr_context);
    // Else, assign the existing one. 
    } else {
        sp = (blmfl_state_t *)(PG_GETARG_POINTER(0));
    }

    int nargs = PG_NARGS(); // Get the number of arguments
    char **argument_array = palloc((nargs) * sizeof(char*)); // Initialize a buffer to store concatenated argument data
    int *argument_sizes = palloc((nargs) * sizeof(int));

    convert_input_data(fcinfo, &argument_array, &argument_sizes, nargs);
    blmfl_state_add_any(sp, argument_array, argument_sizes, nargs);

    pfree(argument_array);
    pfree(argument_sizes);

    PG_RETURN_POINTER(sp);
}

// Final Function
PG_FUNCTION_INFO_V1(blmfl_ffunc_any);
Datum blmfl_ffunc_any(PG_FUNCTION_ARGS) {
    MemoryContext aggr_context;
    HeapTuple rettuple;
    if (!AggCheckCallContext(fcinfo, &aggr_context)) {
        ereport(ERROR,
                (errcode(ERRCODE_DATA_EXCEPTION),
                 errmsg("blmfl_ffunc outside transition context")));
    }
    
    blmfl_state_t *sp;
    if (PG_ARGISNULL(0)) {
            ereport(ERROR,
                (errcode(ERRCODE_DATA_EXCEPTION),
                 errmsg("Bloomfilter state is null.")));    }

    sp = (blmfl_state_t *)(PG_GETARG_POINTER(0));
    if (sp->filt == 0)
    {
        ereport(ERROR,
                (errcode(ERRCODE_DATA_EXCEPTION),
                 errmsg("Bloomfilter is null.")));
    }

    rettuple = create_return_tuple(aggr_context, sp);
    PG_RETURN_DATUM(HeapTupleGetDatum(rettuple));
}