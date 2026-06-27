#include "postgres.h"
#include "fmgr.h"

void itoa_32(int32_t value, char *str)
{
    // Function to convert an integer to a binary string representation
    int i = 0;

    // Handle negative numbers by converting to unsigned
    uint32_t unsigned_value = (value < 0) ? -value : value;

    // Convert integer to string in reverse order
    while (unsigned_value > 0)
    {
        str[i++] = (unsigned_value % 2) ? '1' : '0';
        unsigned_value /= 2;
    }

    i = 32;

    // Null terminate the string
    str[i] = '\0';

    // Reverse the string to get the correct order
    for (int j = 0; j < i / 2; j++)
    {
        char temp = str[j];
        str[j] = str[i - j - 1];
        str[i - j - 1] = temp;
    }
}

void itoa_64(int64_t value, char *str)
{
    // Function to convert a 64-bit integer to a binary string representation
    int i = 0;

    // Handle negative numbers by converting to unsigned
    uint64_t unsigned_value = (value < 0) ? -value : value;

    // Convert integer to string in reverse order
    while (unsigned_value > 0)
    {
        str[i++] = (unsigned_value % 2) ? '1' : '0';
        unsigned_value /= 2;
    }

    i = 64;

    // Null terminate the string
    str[i] = '\0';

    // Reverse the string to get the correct order
    for (int j = 0; j < i / 2; j++)
    {
        char temp = str[j];
        str[j] = str[i - j - 1];
        str[i - j - 1] = temp;
    }
}

void convert_input_data(FunctionCallInfo fcinfo, char ***argument_array, int **argument_sizes, int nargs){
    for (int i = 1; i < nargs; i++)
    {
        if (PG_ARGISNULL(i))
        {
            continue; // Skip null arguments
        } 

        Oid argtype = get_fn_expr_argtype(fcinfo->flinfo, i);
        switch (argtype)
        {
        case INT4OID:
        {
            int32 value = PG_GETARG_INT32(i);

            char buffer[(sizeof(int32) * 8)];
            MemSet(buffer, '0', (sizeof(int32) * 8));
            itoa_32(value, buffer);

            (*argument_array)[i] = pstrdup(buffer);
            (*argument_sizes)[i] = (sizeof(int32) * 8);
            break;
        }
        case INT8OID:
        {
            int64 value = PG_GETARG_INT64(i);

            char buffer[(sizeof(int64) * 8)];
            MemSet(buffer, '0', (sizeof(int64) * 8));
            itoa_64(value, buffer);

            (*argument_array)[i] = pstrdup(buffer);
            (*argument_sizes)[i] = (sizeof(int64) * 8);
            break;
        }
        case VARCHAROID:
        {
            VarChar *value = PG_GETARG_VARCHAR_P(i);
            char *buffer = (char *)VARDATA(value);

            (*argument_array)[i] = pstrdup(buffer);
            (*argument_sizes)[i] = VARSIZE(value) - VARHDRSZ;
            break;
        }
        case BPCHAROID:
        {
            BpChar *value = PG_GETARG_BPCHAR_P(i);
            char *buffer = (char *)VARDATA(value);

            (*argument_array)[i] = pstrdup(buffer);
            (*argument_sizes)[i] = VARSIZE(value) - VARHDRSZ;
            break;
        }
        default:
            ereport(ERROR,
                    (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                     errmsg("Unsupported argument type: %u", argtype)));
        }
    }
}