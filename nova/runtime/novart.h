/* Nova Native Runtime Header (libnovart v0.1.0) */
#ifndef NOVA_RUNTIME_H
#define NOVA_RUNTIME_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

typedef enum {
  NV_INT = 0,
  NV_FLOAT = 1,
  NV_BOOL = 2,
  NV_CHAR = 3,
  NV_STR = 4,
  NV_STRUCT = 5,
  NV_ARRAY = 6,
  NV_MAP = 7,
  NV_RESULT = 8,
  NV_NONE = 9,
  NV_TENSOR = 10,
  NV_GRADVAR = 11,
  NV_MODEL = 12,
  NV_OPTIMIZER = 13
} NovaTag;

typedef struct {
  int32_t tag;
  int64_t i;
  double f;
  bool b;
  void* ptr;
} NVal;

/* Value constructors */
NVal nv_int(int64_t v);
NVal nv_float(double v);
NVal nv_bool(bool v);
NVal nv_char(const char* ch);
NVal nv_str(const char* s);
NVal nv_none(void);

/* Core conversions & formatting */
double nv_num(NVal v);
bool nv_truthy(NVal v);
const char* nv_to_str(NVal v);

/* Arithmetic, Matrix Multiplication & Logic */
NVal nv_add(NVal a, NVal b);
NVal nv_sub(NVal a, NVal b);
NVal nv_mul(NVal a, NVal b);
NVal nv_div(NVal a, NVal b);
NVal nv_mod(NVal a, NVal b);
NVal nv_matmul(NVal a, NVal b);
NVal nv_neg(NVal a);
NVal nv_not(NVal a);

/* Comparisons */
NVal nv_cmp_eq(NVal a, NVal b);
NVal nv_cmp_neq(NVal a, NVal b);
NVal nv_cmp_lt(NVal a, NVal b);
NVal nv_cmp_lte(NVal a, NVal b);
NVal nv_cmp_gt(NVal a, NVal b);
NVal nv_cmp_gte(NVal a, NVal b);

/* Heap containers: Structs, Arrays, Maps, Result<T, E> */
NVal nv_alloc_struct(const char* type_name, int32_t count, const char** field_names, const NVal* field_vals);
NVal nv_load_field(NVal obj, const char* field);
void nv_store_field(NVal obj, const char* field, NVal val);

NVal nv_alloc_array(int32_t count, const NVal* elements);
NVal nv_alloc_map(int32_t count, const NVal* keys, const NVal* vals);
NVal nv_index_get(NVal obj, NVal idx);
void nv_index_set(NVal obj, NVal idx, NVal val);

NVal nv_make_ok(NVal val);
NVal nv_make_error(NVal val);

/* Builtins & Methods (including Nova AI Engine) */
NVal nv_call_builtin(const char* name, int32_t argc, const NVal* args);
NVal nv_method_call(NVal obj, const char* method, int32_t argc, const NVal* args);

#endif /* NOVA_RUNTIME_H */
