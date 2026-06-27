#include "converters.c"

void itoa_32(int32_t value, char *str);
void itoa_64(int64_t value, char *str);
void convert_input_data(FunctionCallInfo fcinfo, char ***argument_array, int **argument_sizes, int nargs);