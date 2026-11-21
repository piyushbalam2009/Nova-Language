/* Nova Native Runtime Implementation (libnovart v0.1.0 + Nova AI Engine) */
#include "novart.h"
#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
  char* key;
  NVal val;
} NVEntry;

typedef struct {
  char* type_name;
  int32_t len;
  int32_t cap;
  NVEntry* fields;
} NVStruct;

typedef struct {
  int32_t len;
  int32_t cap;
  NVal* elements;
} NVArray;

typedef struct {
  int32_t len;
  int32_t cap;
  NVEntry* entries;
} NVMap;

typedef struct {
  bool is_ok;
  NVal value;
} NVResult;

/* ============================================================================
 * Nova AI Engine Native Types: Tensor, GradVar, Sequential Model, Optimizers
 * ============================================================================ */

#define NV_MAX_DIMS 8

typedef struct {
  int32_t ndim;
  int32_t shape[NV_MAX_DIMS];
  int32_t len;
  double* data;
} NVTensor;

typedef enum {
  NV_GOP_LEAF = 0,
  NV_GOP_ADD = 1,
  NV_GOP_SUB = 2,
  NV_GOP_MUL = 3,
  NV_GOP_MATMUL = 4,
  NV_GOP_RELU = 5,
  NV_GOP_SIGMOID = 6,
  NV_GOP_TANH = 7,
  NV_GOP_MSE = 8,
  NV_GOP_BCE = 9
} NVGradOp;

typedef struct NVGradVar {
  NVTensor* data;
  NVTensor* grad;
  NVGradOp op;
  struct NVGradVar* parent_a;
  struct NVGradVar* parent_b;
  NVTensor* target;
  int32_t visit_mark;
} NVGradVar;

typedef enum {
  NV_ACT_RELU = 0,
  NV_ACT_SIGMOID = 1,
  NV_ACT_TANH = 2
} NVActivationKind;

typedef struct {
  bool is_linear;
  int32_t in_features;
  int32_t out_features;
  NVGradVar* weight;
  NVGradVar* bias;
  NVActivationKind activation;
} NVLayer;

typedef struct {
  int32_t len;
  int32_t cap;
  NVLayer* layers;
} NVModel;

typedef struct {
  bool is_adam;
  double lr;
  double momentum;
  double weight_decay;
  double beta1;
  double beta2;
  double eps;
  int64_t timestep;
  int32_t state_count;
  int32_t* state_lens;
  double** m_bufs;
  double** v_bufs;
} NVOptimizer;

static void nv_panic(const char* msg) {
  fprintf(stderr, "Runtime error: %s\n", msg);
  exit(1);
}

static char* nv_strdup(const char* s) {
  if (!s) s = "";
  size_t n = strlen(s);
  char* copy = (char*)malloc(n + 1);
  if (!copy) nv_panic("out of memory");
  memcpy(copy, s, n + 1);
  return copy;
}

NVal nv_int(int64_t v) { NVal n = { NV_INT, v, 0.0, false, NULL }; return n; }
NVal nv_float(double v) { NVal n = { NV_FLOAT, 0, v, false, NULL }; return n; }
NVal nv_bool(bool v) { NVal n = { NV_BOOL, 0, 0.0, v, NULL }; return n; }
NVal nv_char(const char* ch) { NVal n = { NV_CHAR, 0, 0.0, false, (void*)nv_strdup(ch ? ch : "") }; return n; }
NVal nv_str(const char* s) { NVal n = { NV_STR, 0, 0.0, false, (void*)nv_strdup(s ? s : "") }; return n; }
NVal nv_none(void) { NVal n = { NV_NONE, 0, 0.0, false, NULL }; return n; }

static NVal nv_wrap_tensor(NVTensor* t) { NVal n = { NV_TENSOR, 0, 0.0, false, (void*)t }; return n; }
static NVal nv_wrap_grad(NVGradVar* g) { NVal n = { NV_GRADVAR, 0, 0.0, false, (void*)g }; return n; }
static NVal nv_wrap_model(NVModel* m) { NVal n = { NV_MODEL, 0, 0.0, false, (void*)m }; return n; }
static NVal nv_wrap_opt(NVOptimizer* o) { NVal n = { NV_OPTIMIZER, 0, 0.0, false, (void*)o }; return n; }

double nv_num(NVal v) {
  if (v.tag == NV_INT) return (double)v.i;
  if (v.tag == NV_FLOAT) return v.f;
  nv_panic("Expected numeric value");
  return 0.0;
}

bool nv_truthy(NVal v) {
  if (v.tag == NV_BOOL) return v.b;
  if (v.tag == NV_INT) return v.i != 0;
  if (v.tag == NV_NONE) return false;
  return true;
}

typedef struct {
  char* data;
  size_t len;
  size_t cap;
} StrBuf;

static void sb_init(StrBuf* sb) {
  sb->cap = 64;
  sb->len = 0;
  sb->data = (char*)malloc(sb->cap);
  sb->data[0] = '\0';
}

static void sb_append(StrBuf* sb, const char* s) {
  size_t add = strlen(s);
  if (sb->len + add + 1 > sb->cap) {
    while (sb->len + add + 1 > sb->cap) sb->cap *= 2;
    sb->data = (char*)realloc(sb->data, sb->cap);
  }
  memcpy(sb->data + sb->len, s, add + 1);
  sb->len += add;
}

/* Match JS Number(val.toFixed(decimals)) -> Int if integer-valued else Float */
static double nv_round_dec(double v, int decimals) {
  double scale = pow(10.0, (double)decimals);
  double r = floor(v * scale + 0.5) / scale;
  if (v < 0.0) r = ceil(v * scale - 0.5) / scale;
  if (r == 0.0) r = 0.0; /* normalize -0.0 */
  return r;
}

static NVal nv_from_rounded_js(double v, int decimals) {
  double r = nv_round_dec(v, decimals);
  if (isfinite(r) && fabs(r - floor(r)) < 1e-12 && fabs(r) < 9e15) {
    return nv_int((int64_t)llround(r));
  }
  return nv_float(r);
}

static void sb_append_tensor_num(StrBuf* sb, double raw) {
  double r = nv_round_dec(raw, 4);
  if (r == 0.0) r = 0.0;
  char buf[64];
  if (isfinite(r) && fabs(r - floor(r)) < 1e-9 && fabs(r) < 9e15) {
    snprintf(buf, sizeof(buf), "%lld", (long long)llround(r));
  } else {
    snprintf(buf, sizeof(buf), "%.4f", r);
    size_t n = strlen(buf);
    while (n > 0 && buf[n - 1] == '0') buf[--n] = '\0';
    if (n > 0 && buf[n - 1] == '.') buf[--n] = '\0';
  }
  sb_append(sb, buf);
}

static int32_t sb_append_tensor_dim(StrBuf* sb, const NVTensor* t, int32_t dim, int32_t offset) {
  sb_append(sb, "[");
  int32_t count = t->shape[dim];
  if (dim == t->ndim - 1) {
    for (int32_t i = 0; i < count; i++) {
      if (i > 0) sb_append(sb, ", ");
      sb_append_tensor_num(sb, t->data[offset + i]);
    }
    sb_append(sb, "]");
    return offset + count;
  }
  int32_t cur = offset;
  for (int32_t i = 0; i < count; i++) {
    if (i > 0) sb_append(sb, ", ");
    cur = sb_append_tensor_dim(sb, t, dim + 1, cur);
  }
  sb_append(sb, "]");
  return cur;
}

static const char* nv_tensor_to_str(const NVTensor* t) {
  StrBuf sb;
  sb_init(&sb);
  sb_append(&sb, "Tensor(");
  sb_append_tensor_dim(&sb, t, 0, 0);
  sb_append(&sb, ", shape=[");
  char num[32];
  for (int32_t i = 0; i < t->ndim; i++) {
    if (i > 0) sb_append(&sb, ", ");
    snprintf(num, sizeof(num), "%d", t->shape[i]);
    sb_append(&sb, num);
  }
  sb_append(&sb, "])");
  return sb.data;
}

static int32_t nv_model_param_count_internal(const NVModel* m) {
  int32_t total = 0;
  for (int32_t i = 0; i < m->len; i++) {
    if (m->layers[i].is_linear) {
      total += m->layers[i].weight->data->len;
      total += m->layers[i].bias->data->len;
    }
  }
  return total;
}

const char* nv_to_str(NVal v) {
  switch (v.tag) {
    case NV_STR:
    case NV_CHAR:
      return v.ptr ? (const char*)v.ptr : "";
    case NV_INT: {
      char* buf = (char*)malloc(64);
      snprintf(buf, 64, "%lld", (long long)v.i);
      return buf;
    }
    case NV_FLOAT: {
      char* buf = (char*)malloc(64);
      double f = v.f == 0.0 ? 0.0 : v.f;
      double af = fabs(f);
      if (f == floor(f) && isfinite(f)) {
        snprintf(buf, 64, "%.1f", f);
      } else if (af >= 1e-6 && af < 1e-4) {
        snprintf(buf, 64, "%.12f", f);
        size_t n = strlen(buf);
        while (n > 0 && buf[n - 1] == '0') buf[--n] = '\0';
        if (n > 0 && buf[n - 1] == '.') buf[--n] = '\0';
      } else {
        snprintf(buf, 64, "%.15g", f);
      }
      return buf;
    }
    case NV_BOOL:
      return v.b ? "true" : "false";
    case NV_NONE:
      return "None";
    case NV_STRUCT: {
      NVStruct* st = (NVStruct*)v.ptr;
      StrBuf sb;
      sb_init(&sb);
      sb_append(&sb, st->type_name);
      sb_append(&sb, " { ");
      for (int32_t i = 0; i < st->len; i++) {
        if (i > 0) sb_append(&sb, ", ");
        sb_append(&sb, st->fields[i].key);
        sb_append(&sb, ": ");
        sb_append(&sb, nv_to_str(st->fields[i].val));
      }
      sb_append(&sb, " }");
      return sb.data;
    }
    case NV_ARRAY: {
      NVArray* arr = (NVArray*)v.ptr;
      StrBuf sb;
      sb_init(&sb);
      sb_append(&sb, "[");
      for (int32_t i = 0; i < arr->len; i++) {
        if (i > 0) sb_append(&sb, ", ");
        sb_append(&sb, nv_to_str(arr->elements[i]));
      }
      sb_append(&sb, "]");
      return sb.data;
    }
    case NV_MAP: {
      NVMap* m = (NVMap*)v.ptr;
      StrBuf sb;
      sb_init(&sb);
      sb_append(&sb, "{");
      for (int32_t i = 0; i < m->len; i++) {
        if (i > 0) sb_append(&sb, ", ");
        sb_append(&sb, "\"");
        sb_append(&sb, m->entries[i].key);
        sb_append(&sb, "\": ");
        sb_append(&sb, nv_to_str(m->entries[i].val));
      }
      sb_append(&sb, "}");
      return sb.data;
    }
    case NV_RESULT: {
      NVResult* r = (NVResult*)v.ptr;
      StrBuf sb;
      sb_init(&sb);
      sb_append(&sb, r->is_ok ? "Ok(" : "Error(");
      sb_append(&sb, nv_to_str(r->value));
      sb_append(&sb, ")");
      return sb.data;
    }
    case NV_TENSOR:
      return nv_tensor_to_str((const NVTensor*)v.ptr);
    case NV_GRADVAR: {
      const NVGradVar* g = (const NVGradVar*)v.ptr;
      StrBuf sb;
      sb_init(&sb);
      sb_append(&sb, "GradVar(data=");
      sb_append(&sb, nv_tensor_to_str(g->data));
      sb_append(&sb, ", grad=");
      sb_append(&sb, nv_tensor_to_str(g->grad));
      sb_append(&sb, ")");
      return sb.data;
    }
    case NV_MODEL: {
      const NVModel* m = (const NVModel*)v.ptr;
      char* buf = (char*)malloc(128);
      snprintf(buf, 128, "Sequential(layers=%d, params=%d)", m->len, nv_model_param_count_internal(m));
      return buf;
    }
    case NV_OPTIMIZER: {
      const NVOptimizer* o = (const NVOptimizer*)v.ptr;
      return o->is_adam ? "Optimizer(adam)" : "Optimizer(sgd)";
    }
    default:
      return "None";
  }
}

/* ============================================================================
 * Deterministic Mulberry32 PRNG (Exact parity with Rust nova-ai & TS ai.ts)
 * ============================================================================ */

typedef struct {
  uint32_t state;
} NVPrng;

static uint32_t nv_prng_next_u32(NVPrng* rng) {
  rng->state = (uint32_t)(rng->state + 0x6D2B79F5u);
  uint32_t t = (rng->state ^ (rng->state >> 15)) * (1u | rng->state);
  t = (t + ((t ^ (t >> 7)) * (61u | t))) ^ t;
  return t ^ (t >> 14);
}

static double nv_prng_next_signed(NVPrng* rng) {
  return ((double)nv_prng_next_u32(rng) / 4294967296.0) * 2.0 - 1.0;
}

/* ============================================================================
 * Stage 1: Native CPU Tensor Operations
 * ============================================================================ */

static NVTensor* nvt_alloc(int32_t ndim, const int32_t* shape) {
  if (ndim <= 0 || ndim > NV_MAX_DIMS) nv_panic("Invalid tensor rank");
  NVTensor* t = (NVTensor*)malloc(sizeof(NVTensor));
  t->ndim = ndim;
  int32_t len = 1;
  for (int32_t i = 0; i < ndim; i++) {
    t->shape[i] = shape[i];
    len *= shape[i];
  }
  t->len = len;
  t->data = (double*)calloc((size_t)(len > 0 ? len : 1), sizeof(double));
  return t;
}

static NVTensor* nvt_clone(const NVTensor* src) {
  NVTensor* t = nvt_alloc(src->ndim, src->shape);
  memcpy(t->data, src->data, sizeof(double) * (size_t)src->len);
  return t;
}

static NVTensor* nvt_scalar(double v) {
  int32_t sh[1] = { 1 };
  NVTensor* t = nvt_alloc(1, sh);
  t->data[0] = v;
  return t;
}

static NVTensor* nvt_full(int32_t ndim, const int32_t* shape, double val) {
  NVTensor* t = nvt_alloc(ndim, shape);
  for (int32_t i = 0; i < t->len; i++) t->data[i] = val;
  return t;
}

static NVTensor* nvt_rand_uniform(int32_t ndim, const int32_t* shape, uint32_t seed, double scale) {
  NVTensor* t = nvt_alloc(ndim, shape);
  NVPrng rng = { seed };
  for (int32_t i = 0; i < t->len; i++) {
    t->data[i] = nv_prng_next_signed(&rng) * scale;
  }
  return t;
}

static bool nvt_same_shape(const NVTensor* a, const NVTensor* b) {
  if (a->ndim != b->ndim) return false;
  for (int32_t i = 0; i < a->ndim; i++) {
    if (a->shape[i] != b->shape[i]) return false;
  }
  return true;
}

typedef enum { NV_BOP_ADD, NV_BOP_SUB, NV_BOP_MUL, NV_BOP_DIV } NVBinOp;

static double nvt_apply_bin(NVBinOp op, double x, double y) {
  switch (op) {
    case NV_BOP_ADD: return x + y;
    case NV_BOP_SUB: return x - y;
    case NV_BOP_MUL: return x * y;
    case NV_BOP_DIV:
      if (y == 0.0) nv_panic("Tensor division by zero");
      return x / y;
  }
  return 0.0;
}

static NVTensor* nvt_broadcast_bin(const NVTensor* a, const NVTensor* b, NVBinOp op) {
  if (nvt_same_shape(a, b)) {
    NVTensor* out = nvt_alloc(a->ndim, a->shape);
    for (int32_t i = 0; i < a->len; i++) out->data[i] = nvt_apply_bin(op, a->data[i], b->data[i]);
    return out;
  }
  if (b->len == 1) {
    NVTensor* out = nvt_alloc(a->ndim, a->shape);
    double bv = b->data[0];
    for (int32_t i = 0; i < a->len; i++) out->data[i] = nvt_apply_bin(op, a->data[i], bv);
    return out;
  }
  if (a->len == 1) {
    NVTensor* out = nvt_alloc(b->ndim, b->shape);
    double av = a->data[0];
    for (int32_t i = 0; i < b->len; i++) out->data[i] = nvt_apply_bin(op, av, b->data[i]);
    return out;
  }
  if (a->ndim == 2 &&
      ((b->ndim == 1 && b->shape[0] == a->shape[1]) ||
       (b->ndim == 2 && b->shape[0] == 1 && b->shape[1] == a->shape[1]))) {
    int32_t rows = a->shape[0];
    int32_t cols = a->shape[1];
    NVTensor* out = nvt_alloc(a->ndim, a->shape);
    for (int32_t r = 0; r < rows; r++) {
      for (int32_t c = 0; c < cols; c++) {
        out->data[r * cols + c] = nvt_apply_bin(op, a->data[r * cols + c], b->data[c]);
      }
    }
    return out;
  }
  nv_panic("Incompatible tensor shapes for broadcast");
  return NULL;
}

static NVTensor* nvt_matmul(const NVTensor* a, const NVTensor* b) {
  if (a->ndim != 2 || b->ndim != 2) nv_panic("matmul '@' requires 2D tensors");
  int32_t m = a->shape[0];
  int32_t k1 = a->shape[1];
  int32_t k2 = b->shape[0];
  int32_t n = b->shape[1];
  if (k1 != k2) nv_panic("matmul inner dimension mismatch");
  int32_t sh[2] = { m, n };
  NVTensor* out = nvt_alloc(2, sh);
  for (int32_t i = 0; i < m; i++) {
    for (int32_t k = 0; k < k1; k++) {
      double aik = a->data[i * k1 + k];
      for (int32_t j = 0; j < n; j++) {
        out->data[i * n + j] += aik * b->data[k * n + j];
      }
    }
  }
  return out;
}

static double nvt_dot(const NVTensor* a, const NVTensor* b) {
  if (a->len != b->len) nv_panic("dot() length mismatch");
  double acc = 0.0;
  for (int32_t i = 0; i < a->len; i++) acc += a->data[i] * b->data[i];
  return acc;
}

static NVTensor* nvt_transpose(const NVTensor* a) {
  if (a->ndim != 2) nv_panic("transpose requires a 2D tensor");
  int32_t rows = a->shape[0];
  int32_t cols = a->shape[1];
  int32_t sh[2] = { cols, rows };
  NVTensor* out = nvt_alloc(2, sh);
  for (int32_t r = 0; r < rows; r++) {
    for (int32_t c = 0; c < cols; c++) {
      out->data[c * rows + r] = a->data[r * cols + c];
    }
  }
  return out;
}

static NVTensor* nvt_reshape(const NVTensor* a, int32_t ndim, const int32_t* new_shape) {
  int32_t expected = 1;
  for (int32_t i = 0; i < ndim; i++) expected *= new_shape[i];
  if (expected != a->len) nv_panic("reshape element count mismatch");
  NVTensor* out = nvt_alloc(ndim, new_shape);
  memcpy(out->data, a->data, sizeof(double) * (size_t)a->len);
  return out;
}

static NVTensor* nvt_relu(const NVTensor* a) {
  NVTensor* out = nvt_alloc(a->ndim, a->shape);
  for (int32_t i = 0; i < a->len; i++) out->data[i] = a->data[i] > 0.0 ? a->data[i] : 0.0;
  return out;
}

static NVTensor* nvt_sigmoid(const NVTensor* a) {
  NVTensor* out = nvt_alloc(a->ndim, a->shape);
  for (int32_t i = 0; i < a->len; i++) out->data[i] = 1.0 / (1.0 + exp(-a->data[i]));
  return out;
}

static NVTensor* nvt_tanh(const NVTensor* a) {
  NVTensor* out = nvt_alloc(a->ndim, a->shape);
  for (int32_t i = 0; i < a->len; i++) out->data[i] = tanh(a->data[i]);
  return out;
}

static NVTensor* nvt_softmax(const NVTensor* a) {
  if (a->ndim == 1) {
    NVTensor* out = nvt_alloc(a->ndim, a->shape);
    double mx = -DBL_MAX;
    for (int32_t i = 0; i < a->len; i++) if (a->data[i] > mx) mx = a->data[i];
    double s = 0.0;
    for (int32_t i = 0; i < a->len; i++) {
      out->data[i] = exp(a->data[i] - mx);
      s += out->data[i];
    }
    for (int32_t i = 0; i < a->len; i++) out->data[i] /= s;
    return out;
  }
  if (a->ndim == 2) {
    int32_t rows = a->shape[0];
    int32_t cols = a->shape[1];
    NVTensor* out = nvt_alloc(a->ndim, a->shape);
    for (int32_t r = 0; r < rows; r++) {
      double mx = -DBL_MAX;
      for (int32_t c = 0; c < cols; c++) {
        double v = a->data[r * cols + c];
        if (v > mx) mx = v;
      }
      double s = 0.0;
      for (int32_t c = 0; c < cols; c++) {
        double e = exp(a->data[r * cols + c] - mx);
        out->data[r * cols + c] = e;
        s += e;
      }
      for (int32_t c = 0; c < cols; c++) out->data[r * cols + c] /= s;
    }
    return out;
  }
  nv_panic("softmax supports 1D and 2D tensors");
  return NULL;
}

static double nvt_sum(const NVTensor* a) {
  double s = 0.0;
  for (int32_t i = 0; i < a->len; i++) s += a->data[i];
  return s;
}

static double nvt_mean(const NVTensor* a) {
  return nvt_sum(a) / (double)(a->len > 0 ? a->len : 1);
}

static double nvt_max(const NVTensor* a) {
  double m = -DBL_MAX;
  for (int32_t i = 0; i < a->len; i++) if (a->data[i] > m) m = a->data[i];
  return m;
}

static double nvt_min(const NVTensor* a) {
  double m = DBL_MAX;
  for (int32_t i = 0; i < a->len; i++) if (a->data[i] < m) m = a->data[i];
  return m;
}

static int64_t nvt_argmax(const NVTensor* a) {
  int64_t best = 0;
  double best_val = -DBL_MAX;
  for (int32_t i = 0; i < a->len; i++) {
    if (a->data[i] > best_val) {
      best_val = a->data[i];
      best = i;
    }
  }
  return best;
}

static double nvt_mse_loss(const NVTensor* pred, const NVTensor* target) {
  NVTensor* diff = nvt_broadcast_bin(pred, target, NV_BOP_SUB);
  double s = 0.0;
  for (int32_t i = 0; i < diff->len; i++) s += diff->data[i] * diff->data[i];
  int32_t n = diff->len > 0 ? diff->len : 1;
  free(diff->data);
  free(diff);
  return s / (double)n;
}

static double nvt_bce_loss(const NVTensor* pred, const NVTensor* target) {
  if (!nvt_same_shape(pred, target)) nv_panic("bce_loss shape mismatch");
  const double eps = 1e-7;
  double s = 0.0;
  for (int32_t i = 0; i < pred->len; i++) {
    double p = pred->data[i];
    if (p < eps) p = eps;
    if (p > 1.0 - eps) p = 1.0 - eps;
    double y = target->data[i];
    s += -(y * log(p) + (1.0 - y) * log(1.0 - p));
  }
  return s / (double)(pred->len > 0 ? pred->len : 1);
}

static NVTensor* nvt_unbroadcast_to(const NVTensor* g, int32_t target_ndim, const int32_t* target_shape) {
  if (g->ndim == target_ndim) {
    bool same = true;
    for (int32_t i = 0; i < target_ndim; i++) {
      if (g->shape[i] != target_shape[i]) { same = false; break; }
    }
    if (same) return nvt_clone(g);
  }
  int32_t target_len = 1;
  for (int32_t i = 0; i < target_ndim; i++) target_len *= target_shape[i];
  if (target_len == 1) {
    NVTensor* out = nvt_alloc(target_ndim, target_shape);
    out->data[0] = nvt_sum(g);
    return out;
  }
  if (g->ndim == 2 &&
      ((target_ndim == 1 && target_shape[0] == g->shape[1]) ||
       (target_ndim == 2 && target_shape[0] == 1 && target_shape[1] == g->shape[1]))) {
    int32_t rows = g->shape[0];
    int32_t cols = g->shape[1];
    NVTensor* out = nvt_alloc(target_ndim, target_shape);
    for (int32_t r = 0; r < rows; r++) {
      for (int32_t c = 0; c < cols; c++) {
        out->data[c] += g->data[r * cols + c];
      }
    }
    return out;
  }
  nv_panic("Cannot unbroadcast gradient tensor");
  return NULL;
}

/* Parse nested NVArray -> NVTensor */
static void nvt_infer_shape(NVal node, int32_t depth, int32_t* ndim, int32_t* shape, double** buf, int32_t* len, int32_t* cap) {
  if (node.tag == NV_ARRAY) {
    NVArray* arr = (NVArray*)node.ptr;
    if (*ndim == depth) {
      if (depth >= NV_MAX_DIMS) nv_panic("Tensor rank exceeds maximum supported dimensions");
      shape[depth] = arr->len;
      *ndim = depth + 1;
    } else if (shape[depth] != arr->len) {
      nv_panic("Ragged tensor shape");
    }
    for (int32_t i = 0; i < arr->len; i++) {
      nvt_infer_shape(arr->elements[i], depth + 1, ndim, shape, buf, len, cap);
    }
  } else {
    double v = nv_num(node);
    if (*len >= *cap) {
      *cap = *cap > 0 ? *cap * 2 : 16;
      *buf = (double*)realloc(*buf, sizeof(double) * (size_t)(*cap));
    }
    (*buf)[(*len)++] = v;
  }
}

static NVTensor* nvt_from_nval(NVal v) {
  if (v.tag == NV_TENSOR) return nvt_clone((const NVTensor*)v.ptr);
  if (v.tag == NV_INT || v.tag == NV_FLOAT) return nvt_scalar(nv_num(v));
  if (v.tag != NV_ARRAY) nv_panic("Expected Tensor or Array");
  int32_t ndim = 0;
  int32_t shape[NV_MAX_DIMS];
  double* buf = NULL;
  int32_t len = 0;
  int32_t cap = 0;
  nvt_infer_shape(v, 0, &ndim, shape, &buf, &len, &cap);
  if (ndim == 0) {
    ndim = 1;
    shape[0] = 1;
  }
  NVTensor* t = nvt_alloc(ndim, shape);
  if (len != t->len) nv_panic("Tensor data length does not match inferred shape");
  memcpy(t->data, buf, sizeof(double) * (size_t)len);
  free(buf);
  return t;
}

static int32_t nv_read_shape_array(NVal v, int32_t* out_shape) {
  if (v.tag != NV_ARRAY) nv_panic("Expected shape Array");
  NVArray* arr = (NVArray*)v.ptr;
  if (arr->len <= 0 || arr->len > NV_MAX_DIMS) nv_panic("Invalid shape dimensions");
  for (int32_t i = 0; i < arr->len; i++) {
    out_shape[i] = (int32_t)nv_num(arr->elements[i]);
  }
  return arr->len;
}

/* ============================================================================
 * Stage 2: Native Reverse-Mode Tensor Autograd (NVGradVar)
 * ============================================================================ */

static int32_t g_autograd_visit_epoch = 1;

static NVGradVar* nvg_new(NVTensor* data, NVGradOp op, NVGradVar* pa, NVGradVar* pb, NVTensor* target) {
  NVGradVar* g = (NVGradVar*)malloc(sizeof(NVGradVar));
  g->data = data;
  g->grad = nvt_alloc(data->ndim, data->shape);
  g->op = op;
  g->parent_a = pa;
  g->parent_b = pb;
  g->target = target;
  g->visit_mark = 0;
  return g;
}

static NVGradVar* nvg_leaf(const NVTensor* data) {
  return nvg_new(nvt_clone(data), NV_GOP_LEAF, NULL, NULL, NULL);
}

static void nvg_zero_grad(NVGradVar* g) {
  memset(g->grad->data, 0, sizeof(double) * (size_t)g->grad->len);
}

static void nvg_accum_grad(NVGradVar* g, const NVTensor* delta) {
  NVTensor* reduced = nvt_unbroadcast_to(delta, g->data->ndim, g->data->shape);
  for (int32_t i = 0; i < g->grad->len; i++) g->grad->data[i] += reduced->data[i];
  free(reduced->data);
  free(reduced);
}

static NVGradVar* nvg_add(NVGradVar* a, NVGradVar* b) {
  return nvg_new(nvt_broadcast_bin(a->data, b->data, NV_BOP_ADD), NV_GOP_ADD, a, b, NULL);
}

static NVGradVar* nvg_sub(NVGradVar* a, NVGradVar* b) {
  return nvg_new(nvt_broadcast_bin(a->data, b->data, NV_BOP_SUB), NV_GOP_SUB, a, b, NULL);
}

static NVGradVar* nvg_mul(NVGradVar* a, NVGradVar* b) {
  return nvg_new(nvt_broadcast_bin(a->data, b->data, NV_BOP_MUL), NV_GOP_MUL, a, b, NULL);
}

static NVGradVar* nvg_matmul(NVGradVar* a, NVGradVar* b) {
  return nvg_new(nvt_matmul(a->data, b->data), NV_GOP_MATMUL, a, b, NULL);
}

static NVGradVar* nvg_relu(NVGradVar* a) {
  return nvg_new(nvt_relu(a->data), NV_GOP_RELU, a, NULL, NULL);
}

static NVGradVar* nvg_sigmoid(NVGradVar* a) {
  return nvg_new(nvt_sigmoid(a->data), NV_GOP_SIGMOID, a, NULL, NULL);
}

static NVGradVar* nvg_tanh(NVGradVar* a) {
  return nvg_new(nvt_tanh(a->data), NV_GOP_TANH, a, NULL, NULL);
}

static NVGradVar* nvg_mse_loss(NVGradVar* pred, const NVTensor* target) {
  double lv = nvt_mse_loss(pred->data, target);
  return nvg_new(nvt_scalar(lv), NV_GOP_MSE, pred, NULL, nvt_clone(target));
}

static NVGradVar* nvg_bce_loss(NVGradVar* pred, const NVTensor* target) {
  double lv = nvt_bce_loss(pred->data, target);
  return nvg_new(nvt_scalar(lv), NV_GOP_BCE, pred, NULL, nvt_clone(target));
}

typedef struct {
  NVGradVar** items;
  int32_t len;
  int32_t cap;
} NVGradList;

static void nvg_topo_dfs(NVGradVar* node, int32_t epoch, NVGradList* list) {
  if (!node || node->visit_mark == epoch) return;
  node->visit_mark = epoch;
  if (node->parent_a) nvg_topo_dfs(node->parent_a, epoch, list);
  if (node->parent_b) nvg_topo_dfs(node->parent_b, epoch, list);
  if (list->len >= list->cap) {
    list->cap = list->cap > 0 ? list->cap * 2 : 32;
    list->items = (NVGradVar**)realloc(list->items, sizeof(NVGradVar*) * (size_t)list->cap);
  }
  list->items[list->len++] = node;
}

static void nvg_backward(NVGradVar* root) {
  NVGradList topo = { NULL, 0, 0 };
  int32_t epoch = ++g_autograd_visit_epoch;
  nvg_topo_dfs(root, epoch, &topo);
  for (int32_t i = 0; i < root->grad->len; i++) root->grad->data[i] = 1.0;

  for (int32_t idx = topo.len - 1; idx >= 0; idx--) {
    NVGradVar* n = topo.items[idx];
    switch (n->op) {
      case NV_GOP_LEAF:
        break;
      case NV_GOP_ADD:
        nvg_accum_grad(n->parent_a, n->grad);
        nvg_accum_grad(n->parent_b, n->grad);
        break;
      case NV_GOP_SUB: {
        nvg_accum_grad(n->parent_a, n->grad);
        NVTensor* neg_one = nvt_scalar(-1.0);
        NVTensor* neg_g = nvt_broadcast_bin(n->grad, neg_one, NV_BOP_MUL);
        nvg_accum_grad(n->parent_b, neg_g);
        free(neg_one->data); free(neg_one);
        free(neg_g->data); free(neg_g);
        break;
      }
      case NV_GOP_MUL: {
        NVTensor* ga = nvt_broadcast_bin(n->grad, n->parent_b->data, NV_BOP_MUL);
        NVTensor* gb = nvt_broadcast_bin(n->grad, n->parent_a->data, NV_BOP_MUL);
        nvg_accum_grad(n->parent_a, ga);
        nvg_accum_grad(n->parent_b, gb);
        free(ga->data); free(ga);
        free(gb->data); free(gb);
        break;
      }
      case NV_GOP_MATMUL: {
        NVTensor* bt = nvt_transpose(n->parent_b->data);
        NVTensor* at = nvt_transpose(n->parent_a->data);
        NVTensor* ga = nvt_matmul(n->grad, bt);
        NVTensor* gb = nvt_matmul(at, n->grad);
        nvg_accum_grad(n->parent_a, ga);
        nvg_accum_grad(n->parent_b, gb);
        free(bt->data); free(bt);
        free(at->data); free(at);
        free(ga->data); free(ga);
        free(gb->data); free(gb);
        break;
      }
      case NV_GOP_RELU: {
        NVTensor* pa_data = n->parent_a->data;
        NVTensor* ga = nvt_alloc(pa_data->ndim, pa_data->shape);
        for (int32_t i = 0; i < pa_data->len; i++) {
          ga->data[i] = pa_data->data[i] > 0.0 ? n->grad->data[i] : 0.0;
        }
        nvg_accum_grad(n->parent_a, ga);
        free(ga->data); free(ga);
        break;
      }
      case NV_GOP_SIGMOID: {
        NVTensor* y = n->data;
        NVTensor* ga = nvt_alloc(y->ndim, y->shape);
        for (int32_t i = 0; i < y->len; i++) {
          double yi = y->data[i];
          ga->data[i] = n->grad->data[i] * yi * (1.0 - yi);
        }
        nvg_accum_grad(n->parent_a, ga);
        free(ga->data); free(ga);
        break;
      }
      case NV_GOP_TANH: {
        NVTensor* y = n->data;
        NVTensor* ga = nvt_alloc(y->ndim, y->shape);
        for (int32_t i = 0; i < y->len; i++) {
          double yi = y->data[i];
          ga->data[i] = n->grad->data[i] * (1.0 - yi * yi);
        }
        nvg_accum_grad(n->parent_a, ga);
        free(ga->data); free(ga);
        break;
      }
      case NV_GOP_MSE: {
        NVTensor* pred = n->parent_a->data;
        int32_t cnt = pred->len > 0 ? pred->len : 1;
        double scale = (2.0 * n->grad->data[0]) / (double)cnt;
        NVTensor* ga = nvt_alloc(pred->ndim, pred->shape);
        for (int32_t i = 0; i < pred->len; i++) {
          ga->data[i] = scale * (pred->data[i] - n->target->data[i]);
        }
        nvg_accum_grad(n->parent_a, ga);
        free(ga->data); free(ga);
        break;
      }
      case NV_GOP_BCE: {
        const double eps = 1e-7;
        NVTensor* pred = n->parent_a->data;
        int32_t cnt = pred->len > 0 ? pred->len : 1;
        double scale = n->grad->data[0] / (double)cnt;
        NVTensor* ga = nvt_alloc(pred->ndim, pred->shape);
        for (int32_t i = 0; i < pred->len; i++) {
          double p = pred->data[i];
          if (p < eps) p = eps;
          if (p > 1.0 - eps) p = 1.0 - eps;
          double y = n->target->data[i];
          ga->data[i] = scale * ((p - y) / (p * (1.0 - p)));
        }
        nvg_accum_grad(n->parent_a, ga);
        free(ga->data); free(ga);
        break;
      }
    }
  }
  free(topo.items);
}

/* ============================================================================
 * Stage 3: Neural Network Layers (Sequential/Linear), SGD, Adam & Training
 * ============================================================================ */

static NVModel* nvm_new(void) {
  NVModel* m = (NVModel*)malloc(sizeof(NVModel));
  m->len = 0;
  m->cap = 4;
  m->layers = (NVLayer*)malloc(sizeof(NVLayer) * (size_t)m->cap);
  return m;
}

static void nvm_ensure_cap(NVModel* m) {
  if (m->len >= m->cap) {
    m->cap *= 2;
    m->layers = (NVLayer*)realloc(m->layers, sizeof(NVLayer) * (size_t)m->cap);
  }
}

static void nvm_add_linear(NVModel* m, int32_t in_f, int32_t out_f, uint32_t seed) {
  nvm_ensure_cap(m);
  double limit = sqrt(6.0 / (double)(in_f + out_f));
  int32_t w_shape[2] = { in_f, out_f };
  int32_t b_shape[2] = { 1, out_f };
  NVLayer* l = &m->layers[m->len++];
  l->is_linear = true;
  l->in_features = in_f;
  l->out_features = out_f;
  l->weight = nvg_new(nvt_rand_uniform(2, w_shape, seed, limit), NV_GOP_LEAF, NULL, NULL, NULL);
  l->bias = nvg_new(nvt_alloc(2, b_shape), NV_GOP_LEAF, NULL, NULL, NULL);
  l->activation = NV_ACT_RELU;
}

static void nvm_add_activation(NVModel* m, NVActivationKind act) {
  nvm_ensure_cap(m);
  NVLayer* l = &m->layers[m->len++];
  l->is_linear = false;
  l->in_features = 0;
  l->out_features = 0;
  l->weight = NULL;
  l->bias = NULL;
  l->activation = act;
}

static NVGradVar* nvm_forward(NVModel* m, NVGradVar* x) {
  NVGradVar* cur = x;
  for (int32_t i = 0; i < m->len; i++) {
    NVLayer* l = &m->layers[i];
    if (l->is_linear) {
      cur = nvg_add(nvg_matmul(cur, l->weight), l->bias);
    } else if (l->activation == NV_ACT_RELU) {
      cur = nvg_relu(cur);
    } else if (l->activation == NV_ACT_SIGMOID) {
      cur = nvg_sigmoid(cur);
    } else {
      cur = nvg_tanh(cur);
    }
  }
  return cur;
}

static int32_t nvm_collect_params(NVModel* m, NVGradVar*** out_params) {
  int32_t count = 0;
  for (int32_t i = 0; i < m->len; i++) {
    if (m->layers[i].is_linear) count += 2;
  }
  NVGradVar** arr = (NVGradVar**)malloc(sizeof(NVGradVar*) * (size_t)(count > 0 ? count : 1));
  int32_t k = 0;
  for (int32_t i = 0; i < m->len; i++) {
    if (m->layers[i].is_linear) {
      arr[k++] = m->layers[i].weight;
      arr[k++] = m->layers[i].bias;
    }
  }
  *out_params = arr;
  return count;
}

static void nvm_zero_grad(NVModel* m) {
  for (int32_t i = 0; i < m->len; i++) {
    if (m->layers[i].is_linear) {
      nvg_zero_grad(m->layers[i].weight);
      nvg_zero_grad(m->layers[i].bias);
    }
  }
}

static NVOptimizer* nvo_new_sgd(double lr, double momentum) {
  NVOptimizer* o = (NVOptimizer*)calloc(1, sizeof(NVOptimizer));
  o->is_adam = false;
  o->lr = lr;
  o->momentum = momentum;
  return o;
}

static NVOptimizer* nvo_new_adam(double lr) {
  NVOptimizer* o = (NVOptimizer*)calloc(1, sizeof(NVOptimizer));
  o->is_adam = true;
  o->lr = lr;
  o->beta1 = 0.9;
  o->beta2 = 0.999;
  o->eps = 1e-8;
  return o;
}

static void nvo_step(NVOptimizer* opt, int32_t param_count, NVGradVar** params) {
  if (opt->state_count != param_count) {
    opt->state_count = param_count;
    opt->state_lens = (int32_t*)malloc(sizeof(int32_t) * (size_t)(param_count > 0 ? param_count : 1));
    opt->m_bufs = (double**)malloc(sizeof(double*) * (size_t)(param_count > 0 ? param_count : 1));
    opt->v_bufs = (double**)malloc(sizeof(double*) * (size_t)(param_count > 0 ? param_count : 1));
    for (int32_t i = 0; i < param_count; i++) {
      int32_t n = params[i]->data->len;
      opt->state_lens[i] = n;
      opt->m_bufs[i] = (double*)calloc((size_t)(n > 0 ? n : 1), sizeof(double));
      opt->v_bufs[i] = (double*)calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    }
  }

  if (!opt->is_adam) {
    for (int32_t idx = 0; idx < param_count; idx++) {
      NVGradVar* p = params[idx];
      double* vel = opt->m_bufs[idx];
      for (int32_t i = 0; i < p->data->len; i++) {
        double g = p->grad->data[i] + opt->weight_decay * p->data->data[i];
        vel[i] = opt->momentum * vel[i] + g;
        p->data->data[i] -= opt->lr * vel[i];
      }
    }
  } else {
    opt->timestep++;
    double bc1 = 1.0 - pow(opt->beta1, (double)opt->timestep);
    double bc2 = 1.0 - pow(opt->beta2, (double)opt->timestep);
    for (int32_t idx = 0; idx < param_count; idx++) {
      NVGradVar* p = params[idx];
      double* m = opt->m_bufs[idx];
      double* v = opt->v_bufs[idx];
      for (int32_t i = 0; i < p->data->len; i++) {
        double g = p->grad->data[i];
        m[i] = opt->beta1 * m[i] + (1.0 - opt->beta1) * g;
        v[i] = opt->beta2 * v[i] + (1.0 - opt->beta2) * g * g;
        double m_hat = m[i] / bc1;
        double v_hat = v[i] / bc2;
        p->data->data[i] -= (opt->lr * m_hat) / (sqrt(v_hat) + opt->eps);
      }
    }
  }
}

static double nv_train_step_internal(NVModel* m, NVOptimizer* opt, const NVTensor* x, const NVTensor* y, bool use_bce) {
  nvm_zero_grad(m);
  NVGradVar* pred = nvm_forward(m, nvg_leaf(x));
  NVGradVar* loss = use_bce ? nvg_bce_loss(pred, y) : nvg_mse_loss(pred, y);
  double val = loss->data->data[0];
  nvg_backward(loss);
  NVGradVar** params = NULL;
  int32_t pc = nvm_collect_params(m, &params);
  nvo_step(opt, pc, params);
  free(params);
  return val;
}

static NVal nv_train_model_internal(NVModel* m, NVOptimizer* opt, const NVTensor* x, const NVTensor* y, int32_t epochs, bool use_bce, bool include_meta) {
  if (epochs < 1) epochs = 1;
  if (epochs > 5000) epochs = 5000;
  double initial_loss = 0.0;
  double final_loss = 0.0;
  for (int32_t ep = 0; ep < epochs; ep++) {
    double lv = nv_train_step_internal(m, opt, x, y, use_bce);
    if (ep == 0) initial_loss = lv;
    final_loss = lv;
  }
  NVGradVar* out = nvm_forward(m, nvg_leaf(x));
  NVal* pred_vals = (NVal*)malloc(sizeof(NVal) * (size_t)(out->data->len > 0 ? out->data->len : 1));
  for (int32_t i = 0; i < out->data->len; i++) {
    pred_vals[i] = nv_from_rounded_js(out->data->data[i], 4);
  }
  NVal preds_arr = nv_alloc_array(out->data->len, pred_vals);
  free(pred_vals);

  if (include_meta) {
    NVal keys[6] = {
      nv_str("epochs"), nv_str("initialLoss"), nv_str("finalLoss"),
      nv_str("optimizer"), nv_str("paramCount"), nv_str("predictions")
    };
    NVal vals[6] = {
      nv_int(epochs),
      nv_from_rounded_js(initial_loss, 6),
      nv_from_rounded_js(final_loss, 6),
      nv_str(opt->is_adam ? "adam" : "sgd"),
      nv_int(nv_model_param_count_internal(m)),
      preds_arr
    };
    return nv_alloc_map(6, keys, vals);
  } else {
    NVal keys[4] = {
      nv_str("epochs"), nv_str("initialLoss"), nv_str("finalLoss"), nv_str("predictions")
    };
    NVal vals[4] = {
      nv_int(epochs),
      nv_from_rounded_js(initial_loss, 6),
      nv_from_rounded_js(final_loss, 6),
      preds_arr
    };
    return nv_alloc_map(4, keys, vals);
  }
}

/* ============================================================================
 * Core Operators (with Tensor Broadcasting & Matrix Multiplication)
 * ============================================================================ */

static NVal nv_tensor_binary_nval(NVal a, NVal b, NVBinOp op) {
  NVTensor* ta = (a.tag == NV_TENSOR) ? (NVTensor*)a.ptr : nvt_scalar(nv_num(a));
  NVTensor* tb = (b.tag == NV_TENSOR) ? (NVTensor*)b.ptr : nvt_scalar(nv_num(b));
  NVTensor* out = nvt_broadcast_bin(ta, tb, op);
  if (a.tag != NV_TENSOR) { free(ta->data); free(ta); }
  if (b.tag != NV_TENSOR) { free(tb->data); free(tb); }
  return nv_wrap_tensor(out);
}

NVal nv_add(NVal a, NVal b) {
  if (a.tag == NV_STR || b.tag == NV_STR) {
    const char* sa = nv_to_str(a);
    const char* sb = nv_to_str(b);
    size_t la = strlen(sa);
    size_t lb = strlen(sb);
    char* out = (char*)malloc(la + lb + 1);
    memcpy(out, sa, la);
    memcpy(out + la, sb, lb + 1);
    NVal res = { NV_STR, 0, 0.0, false, (void*)out };
    return res;
  }
  if (a.tag == NV_TENSOR || b.tag == NV_TENSOR) return nv_tensor_binary_nval(a, b, NV_BOP_ADD);
  if (a.tag == NV_INT && b.tag == NV_INT) return nv_int(a.i + b.i);
  return nv_float(nv_num(a) + nv_num(b));
}

NVal nv_sub(NVal a, NVal b) {
  if (a.tag == NV_TENSOR || b.tag == NV_TENSOR) return nv_tensor_binary_nval(a, b, NV_BOP_SUB);
  if (a.tag == NV_INT && b.tag == NV_INT) return nv_int(a.i - b.i);
  return nv_float(nv_num(a) - nv_num(b));
}

NVal nv_mul(NVal a, NVal b) {
  if (a.tag == NV_TENSOR || b.tag == NV_TENSOR) return nv_tensor_binary_nval(a, b, NV_BOP_MUL);
  if (a.tag == NV_INT && b.tag == NV_INT) return nv_int(a.i * b.i);
  return nv_float(nv_num(a) * nv_num(b));
}

NVal nv_div(NVal a, NVal b) {
  if (a.tag == NV_TENSOR || b.tag == NV_TENSOR) return nv_tensor_binary_nval(a, b, NV_BOP_DIV);
  double rv = nv_num(b);
  if (rv == 0.0) nv_panic("division by zero");
  if (a.tag == NV_INT && b.tag == NV_INT) return nv_int(a.i / b.i);
  return nv_float(nv_num(a) / rv);
}

NVal nv_mod(NVal a, NVal b) {
  double rv = nv_num(b);
  if (rv == 0.0) nv_panic("modulo by zero");
  if (a.tag == NV_INT && b.tag == NV_INT) return nv_int(a.i % b.i);
  return nv_float(fmod(nv_num(a), rv));
}

NVal nv_matmul(NVal a, NVal b) {
  if (a.tag != NV_TENSOR || b.tag != NV_TENSOR) nv_panic("Matrix multiplication '@' requires Tensor operands");
  return nv_wrap_tensor(nvt_matmul((const NVTensor*)a.ptr, (const NVTensor*)b.ptr));
}

NVal nv_neg(NVal a) {
  if (a.tag == NV_TENSOR) {
    NVTensor* neg1 = nvt_scalar(-1.0);
    NVTensor* out = nvt_broadcast_bin((const NVTensor*)a.ptr, neg1, NV_BOP_MUL);
    free(neg1->data); free(neg1);
    return nv_wrap_tensor(out);
  }
  if (a.tag == NV_INT) return nv_int(-a.i);
  return nv_float(-nv_num(a));
}

NVal nv_not(NVal a) {
  return nv_bool(!nv_truthy(a));
}

static bool nv_values_eq(NVal a, NVal b) {
  if (a.tag == NV_NONE && b.tag == NV_NONE) return true;
  if ((a.tag == NV_INT || a.tag == NV_FLOAT) && (b.tag == NV_INT || b.tag == NV_FLOAT)) {
    return nv_num(a) == nv_num(b);
  }
  if (a.tag == NV_BOOL && b.tag == NV_BOOL) return a.b == b.b;
  if ((a.tag == NV_STR && b.tag == NV_STR) || (a.tag == NV_CHAR && b.tag == NV_CHAR)) {
    return strcmp(nv_to_str(a), nv_to_str(b)) == 0;
  }
  if (a.tag == b.tag) return a.ptr == b.ptr;
  return false;
}

NVal nv_cmp_eq(NVal a, NVal b) { return nv_bool(nv_values_eq(a, b)); }
NVal nv_cmp_neq(NVal a, NVal b) { return nv_bool(!nv_values_eq(a, b)); }

NVal nv_cmp_lt(NVal a, NVal b) {
  if (a.tag == NV_STR && b.tag == NV_STR) return nv_bool(strcmp(nv_to_str(a), nv_to_str(b)) < 0);
  return nv_bool(nv_num(a) < nv_num(b));
}

NVal nv_cmp_lte(NVal a, NVal b) {
  if (a.tag == NV_STR && b.tag == NV_STR) return nv_bool(strcmp(nv_to_str(a), nv_to_str(b)) <= 0);
  return nv_bool(nv_num(a) <= nv_num(b));
}

NVal nv_cmp_gt(NVal a, NVal b) {
  if (a.tag == NV_STR && b.tag == NV_STR) return nv_bool(strcmp(nv_to_str(a), nv_to_str(b)) > 0);
  return nv_bool(nv_num(a) > nv_num(b));
}

NVal nv_cmp_gte(NVal a, NVal b) {
  if (a.tag == NV_STR && b.tag == NV_STR) return nv_bool(strcmp(nv_to_str(a), nv_to_str(b)) >= 0);
  return nv_bool(nv_num(a) >= nv_num(b));
}

NVal nv_alloc_struct(const char* type_name, int32_t count, const char** field_names, const NVal* field_vals) {
  NVStruct* st = (NVStruct*)malloc(sizeof(NVStruct));
  st->type_name = nv_strdup(type_name);
  st->len = count;
  st->cap = count > 0 ? count : 4;
  st->fields = (NVEntry*)malloc(sizeof(NVEntry) * (size_t)st->cap);
  for (int32_t i = 0; i < count; i++) {
    st->fields[i].key = nv_strdup(field_names[i]);
    st->fields[i].val = field_vals[i];
  }
  NVal v = { NV_STRUCT, 0, 0.0, false, (void*)st };
  return v;
}

NVal nv_load_field(NVal obj, const char* field) {
  if (obj.tag == NV_STRUCT) {
    NVStruct* st = (NVStruct*)obj.ptr;
    for (int32_t i = 0; i < st->len; i++) {
      if (strcmp(st->fields[i].key, field) == 0) return st->fields[i].val;
    }
    nv_panic("Struct field not found");
  }
  if (obj.tag == NV_MAP) {
    NVMap* m = (NVMap*)obj.ptr;
    for (int32_t i = 0; i < m->len; i++) {
      if (strcmp(m->entries[i].key, field) == 0) return m->entries[i].val;
    }
    return nv_none();
  }
  nv_panic("Cannot access field on non-struct/map value");
  return nv_none();
}

void nv_store_field(NVal obj, const char* field, NVal val) {
  if (obj.tag == NV_STRUCT) {
    NVStruct* st = (NVStruct*)obj.ptr;
    for (int32_t i = 0; i < st->len; i++) {
      if (strcmp(st->fields[i].key, field) == 0) {
        st->fields[i].val = val;
        return;
      }
    }
    if (st->len >= st->cap) {
      st->cap *= 2;
      st->fields = (NVEntry*)realloc(st->fields, sizeof(NVEntry) * (size_t)st->cap);
    }
    st->fields[st->len].key = nv_strdup(field);
    st->fields[st->len].val = val;
    st->len++;
    return;
  }
  if (obj.tag == NV_MAP) {
    NVMap* m = (NVMap*)obj.ptr;
    for (int32_t i = 0; i < m->len; i++) {
      if (strcmp(m->entries[i].key, field) == 0) {
        m->entries[i].val = val;
        return;
      }
    }
    if (m->len >= m->cap) {
      m->cap = m->cap > 0 ? m->cap * 2 : 4;
      m->entries = (NVEntry*)realloc(m->entries, sizeof(NVEntry) * (size_t)m->cap);
    }
    m->entries[m->len].key = nv_strdup(field);
    m->entries[m->len].val = val;
    m->len++;
    return;
  }
  nv_panic("Cannot assign field on non-struct/map value");
}

NVal nv_alloc_array(int32_t count, const NVal* elements) {
  NVArray* arr = (NVArray*)malloc(sizeof(NVArray));
  arr->len = count;
  arr->cap = count > 0 ? count : 4;
  arr->elements = (NVal*)malloc(sizeof(NVal) * (size_t)arr->cap);
  for (int32_t i = 0; i < count; i++) arr->elements[i] = elements[i];
  NVal v = { NV_ARRAY, 0, 0.0, false, (void*)arr };
  return v;
}

NVal nv_alloc_map(int32_t count, const NVal* keys, const NVal* vals) {
  NVMap* m = (NVMap*)malloc(sizeof(NVMap));
  m->len = 0;
  m->cap = count > 0 ? count : 4;
  m->entries = (NVEntry*)malloc(sizeof(NVEntry) * (size_t)m->cap);
  NVal v = { NV_MAP, 0, 0.0, false, (void*)m };
  for (int32_t i = 0; i < count; i++) {
    nv_store_field(v, nv_to_str(keys[i]), vals[i]);
  }
  return v;
}

NVal nv_index_get(NVal obj, NVal idx) {
  if (obj.tag == NV_ARRAY) {
    NVArray* arr = (NVArray*)obj.ptr;
    int64_t i = (int64_t)nv_num(idx);
    if (i < 0 || i >= arr->len) nv_panic("Array index out of bounds");
    return arr->elements[i];
  }
  if (obj.tag == NV_MAP) {
    return nv_load_field(obj, nv_to_str(idx));
  }
  if (obj.tag == NV_STR) {
    const char* s = nv_to_str(obj);
    int64_t i = (int64_t)nv_num(idx);
    size_t len = strlen(s);
    if (i < 0 || (size_t)i >= len) nv_panic("String index out of bounds");
    char ch[2] = { s[i], '\0' };
    return nv_char(ch);
  }
  if (obj.tag == NV_TENSOR) {
    const NVTensor* t = (const NVTensor*)obj.ptr;
    int64_t i = (int64_t)nv_num(idx);
    if (i < 0 || i >= t->len) nv_panic("Tensor flat index out of bounds");
    return nv_float(t->data[i]);
  }
  nv_panic("Cannot index into value");
  return nv_none();
}

void nv_index_set(NVal obj, NVal idx, NVal val) {
  if (obj.tag == NV_ARRAY) {
    NVArray* arr = (NVArray*)obj.ptr;
    int64_t i = (int64_t)nv_num(idx);
    if (i < 0 || i >= arr->len) nv_panic("Array index out of bounds");
    arr->elements[i] = val;
    return;
  }
  if (obj.tag == NV_MAP) {
    nv_store_field(obj, nv_to_str(idx), val);
    return;
  }
  nv_panic("Cannot assign index on value");
}

NVal nv_make_ok(NVal val) {
  NVResult* r = (NVResult*)malloc(sizeof(NVResult));
  r->is_ok = true;
  r->value = val;
  NVal v = { NV_RESULT, 0, 0.0, false, (void*)r };
  return v;
}

NVal nv_make_error(NVal val) {
  NVResult* r = (NVResult*)malloc(sizeof(NVResult));
  r->is_ok = false;
  r->value = val;
  NVal v = { NV_RESULT, 0, 0.0, false, (void*)r };
  return v;
}

static const char* strip_module(const char* name) {
  const char* dot = strrchr(name, '.');
  return dot ? dot + 1 : name;
}

static NVal str_upper(const char* s) {
  size_t n = strlen(s);
  char* out = (char*)malloc(n + 1);
  for (size_t i = 0; i < n; i++) out[i] = (char)toupper((unsigned char)s[i]);
  out[n] = '\0';
  NVal v = { NV_STR, 0, 0.0, false, (void*)out };
  return v;
}

static NVal str_lower(const char* s) {
  size_t n = strlen(s);
  char* out = (char*)malloc(n + 1);
  for (size_t i = 0; i < n; i++) out[i] = (char)tolower((unsigned char)s[i]);
  out[n] = '\0';
  NVal v = { NV_STR, 0, 0.0, false, (void*)out };
  return v;
}

static NVal str_trim(const char* s) {
  while (*s && isspace((unsigned char)*s)) s++;
  size_t n = strlen(s);
  while (n > 0 && isspace((unsigned char)s[n - 1])) n--;
  char* out = (char*)malloc(n + 1);
  memcpy(out, s, n);
  out[n] = '\0';
  NVal v = { NV_STR, 0, 0.0, false, (void*)out };
  return v;
}

NVal nv_method_call(NVal obj, const char* method, int32_t argc, const NVal* args) {
  if (obj.tag == NV_TENSOR) {
    const NVTensor* t = (const NVTensor*)obj.ptr;
    if (strcmp(method, "shape") == 0) {
      NVal dims[NV_MAX_DIMS];
      for (int32_t i = 0; i < t->ndim; i++) dims[i] = nv_int(t->shape[i]);
      return nv_alloc_array(t->ndim, dims);
    }
    if (strcmp(method, "sum") == 0) return nv_float(nvt_sum(t));
    if (strcmp(method, "mean") == 0) return nv_float(nvt_mean(t));
    if (strcmp(method, "max") == 0) return nv_float(nvt_max(t));
    if (strcmp(method, "min") == 0) return nv_float(nvt_min(t));
    if (strcmp(method, "argmax") == 0) return nv_int(nvt_argmax(t));
    if (strcmp(method, "transpose") == 0) return nv_wrap_tensor(nvt_transpose(t));
    if (strcmp(method, "reshape") == 0 && argc > 0) {
      int32_t sh[NV_MAX_DIMS];
      int32_t nd = nv_read_shape_array(args[0], sh);
      return nv_wrap_tensor(nvt_reshape(t, nd, sh));
    }
    if (strcmp(method, "relu") == 0) return nv_wrap_tensor(nvt_relu(t));
    if (strcmp(method, "sigmoid") == 0) return nv_wrap_tensor(nvt_sigmoid(t));
    if (strcmp(method, "tanh") == 0) return nv_wrap_tensor(nvt_tanh(t));
    if (strcmp(method, "softmax") == 0) return nv_wrap_tensor(nvt_softmax(t));
  }
  if (obj.tag == NV_ARRAY) {
    NVArray* arr = (NVArray*)obj.ptr;
    if (strcmp(method, "len") == 0) return nv_int(arr->len);
    if (strcmp(method, "push") == 0) {
      if (arr->len >= arr->cap) {
        arr->cap = arr->cap > 0 ? arr->cap * 2 : 4;
        arr->elements = (NVal*)realloc(arr->elements, sizeof(NVal) * (size_t)arr->cap);
      }
      arr->elements[arr->len++] = argc > 0 ? args[0] : nv_none();
      return nv_none();
    }
  }
  if (obj.tag == NV_STR) {
    const char* s = nv_to_str(obj);
    if (strcmp(method, "len") == 0) return nv_int((int64_t)strlen(s));
    if (strcmp(method, "upper") == 0) return str_upper(s);
    if (strcmp(method, "lower") == 0) return str_lower(s);
    if (strcmp(method, "trim") == 0) return str_trim(s);
  }
  nv_panic("Unknown method call");
  return nv_none();
}

static NVGradVar* nv_expect_grad(NVal v) {
  if (v.tag != NV_GRADVAR) nv_panic("Expected GradVar");
  return (NVGradVar*)v.ptr;
}

static NVModel* nv_expect_model(NVal v) {
  if (v.tag != NV_MODEL) nv_panic("Expected Sequential model");
  return (NVModel*)v.ptr;
}

static NVOptimizer* nv_expect_opt(NVal v) {
  if (v.tag != NV_OPTIMIZER) nv_panic("Expected Optimizer");
  return (NVOptimizer*)v.ptr;
}

static const NVTensor* nv_expect_tensor(NVal v) {
  if (v.tag != NV_TENSOR) nv_panic("Expected Tensor");
  return (const NVTensor*)v.ptr;
}

NVal nv_call_builtin(const char* raw_name, int32_t argc, const NVal* args) {
  const char* name = strip_module(raw_name);

  if (strcmp(name, "print") == 0) {
    for (int32_t i = 0; i < argc; i++) {
      if (i > 0) printf(" ");
      printf("%s", nv_to_str(args[i]));
    }
    printf("\n");
    return nv_none();
  }
  if (strcmp(name, "str") == 0) {
    return nv_str(argc > 0 ? nv_to_str(args[0]) : "None");
  }
  if (strcmp(name, "int") == 0) {
    if (argc == 0) return nv_int(0);
    if (args[0].tag == NV_INT) return args[0];
    if (args[0].tag == NV_FLOAT) return nv_int((int64_t)args[0].f);
    return nv_int(strtoll(nv_to_str(args[0]), NULL, 10));
  }
  if (strcmp(name, "float") == 0) {
    if (argc == 0) return nv_float(0.0);
    if (args[0].tag == NV_FLOAT) return args[0];
    if (args[0].tag == NV_INT) return nv_float((double)args[0].i);
    return nv_float(strtod(nv_to_str(args[0]), NULL));
  }
  if (strcmp(name, "len") == 0 && argc > 0) {
    if (args[0].tag == NV_STR) return nv_int((int64_t)strlen(nv_to_str(args[0])));
    if (args[0].tag == NV_ARRAY) return nv_int(((NVArray*)args[0].ptr)->len);
    if (args[0].tag == NV_MAP) return nv_int(((NVMap*)args[0].ptr)->len);
    if (args[0].tag == NV_TENSOR) return nv_int(((const NVTensor*)args[0].ptr)->len);
    nv_panic("len() requires String, Array, Map, or Tensor");
  }
  if (strcmp(name, "type_of") == 0 && argc > 0) {
    switch (args[0].tag) {
      case NV_INT: return nv_str("Int");
      case NV_FLOAT: return nv_str("Float");
      case NV_BOOL: return nv_str("Bool");
      case NV_CHAR: return nv_str("Char");
      case NV_STR: return nv_str("String");
      case NV_ARRAY: return nv_str("Array");
      case NV_MAP: return nv_str("Map");
      case NV_RESULT: return nv_str("Result");
      case NV_STRUCT: return nv_str(((NVStruct*)args[0].ptr)->type_name);
      case NV_TENSOR: return nv_str("Tensor");
      case NV_GRADVAR: return nv_str("GradVar");
      case NV_MODEL: return nv_str("Model");
      case NV_OPTIMIZER: return nv_str("Optimizer");
      default: return nv_str("None");
    }
  }
  if (strcmp(name, "is_ok") == 0 && argc > 0) {
    return nv_bool(args[0].tag == NV_RESULT && ((NVResult*)args[0].ptr)->is_ok);
  }
  if (strcmp(name, "is_error") == 0 && argc > 0) {
    return nv_bool(args[0].tag == NV_RESULT && !((NVResult*)args[0].ptr)->is_ok);
  }
  if (strcmp(name, "unwrap") == 0 && argc > 0) {
    if (args[0].tag == NV_RESULT) {
      NVResult* r = (NVResult*)args[0].ptr;
      if (r->is_ok) return r->value;
      fprintf(stderr, "Runtime error: Unwrap called on Error(%s)\n", nv_to_str(r->value));
      exit(1);
    }
    nv_panic("unwrap() requires a Result<T, E> value");
  }

  /* nova.math */
  if (strcmp(name, "sqrt") == 0 && argc > 0) return nv_float(sqrt(nv_num(args[0])));
  if (strcmp(name, "abs") == 0 && argc > 0) return nv_float(fabs(nv_num(args[0])));
  if (strcmp(name, "pow") == 0 && argc > 1) return nv_float(pow(nv_num(args[0]), nv_num(args[1])));
  if (strcmp(name, "sin") == 0 && argc > 0) return nv_float(sin(nv_num(args[0])));
  if (strcmp(name, "cos") == 0 && argc > 0) return nv_float(cos(nv_num(args[0])));
  if (strcmp(name, "exp") == 0 && argc > 0) return nv_float(exp(nv_num(args[0])));
  if (strcmp(name, "log") == 0 && argc > 0) return nv_float(log(nv_num(args[0])));
  if (strcmp(name, "floor") == 0 && argc > 0) return nv_int((int64_t)floor(nv_num(args[0])));
  if (strcmp(name, "ceil") == 0 && argc > 0) return nv_int((int64_t)ceil(nv_num(args[0])));
  if (strcmp(name, "min") == 0 && argc > 1) return nv_float(fmin(nv_num(args[0]), nv_num(args[1])));
  if (strcmp(name, "max") == 0 && argc > 1) return nv_float(fmax(nv_num(args[0]), nv_num(args[1])));

  /* nova.string */
  if (strcmp(name, "upper") == 0 && argc > 0) return str_upper(nv_to_str(args[0]));
  if (strcmp(name, "lower") == 0 && argc > 0) return str_lower(nv_to_str(args[0]));
  if (strcmp(name, "trim") == 0 && argc > 0) return str_trim(nv_to_str(args[0]));
  if (strcmp(name, "contains") == 0 && argc > 1) {
    return nv_bool(strstr(nv_to_str(args[0]), nv_to_str(args[1])) != NULL);
  }

  /* nova.array */
  if (strcmp(name, "range") == 0 && argc > 1) {
    int64_t s = (int64_t)nv_num(args[0]);
    int64_t e = (int64_t)nv_num(args[1]);
    int32_t count = e > s ? (int32_t)(e - s) : 0;
    NVal* els = count > 0 ? (NVal*)malloc(sizeof(NVal) * (size_t)count) : NULL;
    for (int32_t i = 0; i < count; i++) els[i] = nv_int(s + i);
    NVal arr = nv_alloc_array(count, els);
    free(els);
    return arr;
  }
  if (strcmp(name, "sum") == 0 && argc > 0) {
    if (args[0].tag != NV_ARRAY) nv_panic("sum() requires an Array");
    NVArray* arr = (NVArray*)args[0].ptr;
    double total = 0.0;
    for (int32_t i = 0; i < arr->len; i++) total += nv_num(arr->elements[i]);
    return nv_float(total);
  }
  if (strcmp(name, "now_ms") == 0) {
    return nv_int((int64_t)time(NULL) * 1000LL);
  }

  /* nova.ai — CPU Tensors */
  if (strcmp(name, "tensor") == 0 && argc > 0) return nv_wrap_tensor(nvt_from_nval(args[0]));
  if (strcmp(name, "zeros") == 0 && argc > 0) {
    int32_t sh[NV_MAX_DIMS];
    int32_t nd = nv_read_shape_array(args[0], sh);
    return nv_wrap_tensor(nvt_alloc(nd, sh));
  }
  if (strcmp(name, "ones") == 0 && argc > 0) {
    int32_t sh[NV_MAX_DIMS];
    int32_t nd = nv_read_shape_array(args[0], sh);
    return nv_wrap_tensor(nvt_full(nd, sh, 1.0));
  }
  if (strcmp(name, "full") == 0 && argc > 1) {
    int32_t sh[NV_MAX_DIMS];
    int32_t nd = nv_read_shape_array(args[0], sh);
    return nv_wrap_tensor(nvt_full(nd, sh, nv_num(args[1])));
  }
  if (strcmp(name, "randn") == 0 && argc > 0) {
    int32_t sh[NV_MAX_DIMS];
    int32_t nd = nv_read_shape_array(args[0], sh);
    uint32_t seed = argc > 1 ? (uint32_t)((int64_t)nv_num(args[1])) : 42u;
    return nv_wrap_tensor(nvt_rand_uniform(nd, sh, seed, 1.0));
  }
  if (strcmp(name, "matmul") == 0 && argc > 1) {
    return nv_wrap_tensor(nvt_matmul(nv_expect_tensor(args[0]), nv_expect_tensor(args[1])));
  }
  if (strcmp(name, "dot") == 0 && argc > 1) {
    return nv_float(nvt_dot(nv_expect_tensor(args[0]), nv_expect_tensor(args[1])));
  }
  if (strcmp(name, "transpose") == 0 && argc > 0) {
    return nv_wrap_tensor(nvt_transpose(nv_expect_tensor(args[0])));
  }
  if (strcmp(name, "reshape") == 0 && argc > 1) {
    int32_t sh[NV_MAX_DIMS];
    int32_t nd = nv_read_shape_array(args[1], sh);
    return nv_wrap_tensor(nvt_reshape(nv_expect_tensor(args[0]), nd, sh));
  }
  if (strcmp(name, "relu") == 0 && argc > 0) {
    return nv_wrap_tensor(nvt_relu(nv_expect_tensor(args[0])));
  }
  if (strcmp(name, "sigmoid") == 0 && argc > 0) {
    return nv_wrap_tensor(nvt_sigmoid(nv_expect_tensor(args[0])));
  }
  if (strcmp(name, "tanh") == 0 && argc > 0) {
    return nv_wrap_tensor(nvt_tanh(nv_expect_tensor(args[0])));
  }
  if (strcmp(name, "softmax") == 0 && argc > 0) {
    return nv_wrap_tensor(nvt_softmax(nv_expect_tensor(args[0])));
  }
  if (strcmp(name, "mse_loss") == 0 && argc > 1) {
    return nv_float(nvt_mse_loss(nv_expect_tensor(args[0]), nv_expect_tensor(args[1])));
  }
  if (strcmp(name, "bce_loss") == 0 && argc > 1) {
    return nv_float(nvt_bce_loss(nv_expect_tensor(args[0]), nv_expect_tensor(args[1])));
  }

  /* nova.ai — Reverse-Mode Tensor Autograd */
  if (strcmp(name, "grad_var") == 0 && argc > 0) {
    NVTensor* t = nvt_from_nval(args[0]);
    NVGradVar* g = nvg_new(t, NV_GOP_LEAF, NULL, NULL, NULL);
    return nv_wrap_grad(g);
  }
  if (strcmp(name, "grad_add") == 0 && argc > 1) {
    return nv_wrap_grad(nvg_add(nv_expect_grad(args[0]), nv_expect_grad(args[1])));
  }
  if (strcmp(name, "grad_sub") == 0 && argc > 1) {
    return nv_wrap_grad(nvg_sub(nv_expect_grad(args[0]), nv_expect_grad(args[1])));
  }
  if (strcmp(name, "grad_mul") == 0 && argc > 1) {
    return nv_wrap_grad(nvg_mul(nv_expect_grad(args[0]), nv_expect_grad(args[1])));
  }
  if (strcmp(name, "grad_matmul") == 0 && argc > 1) {
    return nv_wrap_grad(nvg_matmul(nv_expect_grad(args[0]), nv_expect_grad(args[1])));
  }
  if (strcmp(name, "grad_relu") == 0 && argc > 0) {
    return nv_wrap_grad(nvg_relu(nv_expect_grad(args[0])));
  }
  if (strcmp(name, "grad_sigmoid") == 0 && argc > 0) {
    return nv_wrap_grad(nvg_sigmoid(nv_expect_grad(args[0])));
  }
  if (strcmp(name, "grad_tanh") == 0 && argc > 0) {
    return nv_wrap_grad(nvg_tanh(nv_expect_grad(args[0])));
  }
  if (strcmp(name, "grad_mse") == 0 && argc > 1) {
    NVGradVar* pred = nv_expect_grad(args[0]);
    NVTensor* target = nvt_from_nval(args[1]);
    NVGradVar* res = nvg_mse_loss(pred, target);
    free(target->data); free(target);
    return nv_wrap_grad(res);
  }
  if (strcmp(name, "backward") == 0 && argc > 0) {
    nvg_backward(nv_expect_grad(args[0]));
    return nv_none();
  }
  if (strcmp(name, "grad_of") == 0 && argc > 0) {
    return nv_wrap_tensor(nvt_clone(nv_expect_grad(args[0])->grad));
  }
  if (strcmp(name, "value_of") == 0 && argc > 0) {
    return nv_wrap_tensor(nvt_clone(nv_expect_grad(args[0])->data));
  }
  if (strcmp(name, "zero_grad") == 0 && argc > 0) {
    nvg_zero_grad(nv_expect_grad(args[0]));
    return nv_none();
  }

  /* nova.nn — Neural Network Layers, Optimizers & Model Training */
  if (strcmp(name, "model_new") == 0) {
    return nv_wrap_model(nvm_new());
  }
  if (strcmp(name, "model_add_linear") == 0 && argc > 2) {
    NVModel* m = nv_expect_model(args[0]);
    int32_t in_f = (int32_t)nv_num(args[1]);
    int32_t out_f = (int32_t)nv_num(args[2]);
    uint32_t seed = argc > 3 ? (uint32_t)((int64_t)nv_num(args[3])) : 42u;
    nvm_add_linear(m, in_f, out_f, seed);
    return args[0];
  }
  if (strcmp(name, "model_add_activation") == 0 && argc > 1) {
    NVModel* m = nv_expect_model(args[0]);
    const char* act = nv_to_str(args[1]);
    if (strcmp(act, "relu") == 0 || strcmp(act, "ReLU") == 0) nvm_add_activation(m, NV_ACT_RELU);
    else if (strcmp(act, "sigmoid") == 0 || strcmp(act, "Sigmoid") == 0) nvm_add_activation(m, NV_ACT_SIGMOID);
    else if (strcmp(act, "tanh") == 0 || strcmp(act, "Tanh") == 0) nvm_add_activation(m, NV_ACT_TANH);
    else nv_panic("Unsupported activation");
    return args[0];
  }
  if (strcmp(name, "model_forward") == 0 && argc > 1) {
    NVModel* m = nv_expect_model(args[0]);
    NVTensor* x = nvt_from_nval(args[1]);
    NVGradVar* out = nvm_forward(m, nvg_new(x, NV_GOP_LEAF, NULL, NULL, NULL));
    return nv_wrap_tensor(nvt_clone(out->data));
  }
  if (strcmp(name, "model_param_count") == 0 && argc > 0) {
    return nv_int(nv_model_param_count_internal(nv_expect_model(args[0])));
  }
  if (strcmp(name, "optimizer_sgd") == 0 && argc > 0) {
    double lr = nv_num(args[0]);
    double mom = argc > 1 ? nv_num(args[1]) : 0.0;
    return nv_wrap_opt(nvo_new_sgd(lr, mom));
  }
  if (strcmp(name, "optimizer_adam") == 0 && argc > 0) {
    double lr = nv_num(args[0]);
    return nv_wrap_opt(nvo_new_adam(lr));
  }
  if (strcmp(name, "train_step") == 0 && argc > 3) {
    NVModel* m = nv_expect_model(args[0]);
    NVOptimizer* opt = nv_expect_opt(args[1]);
    NVTensor* x = nvt_from_nval(args[2]);
    NVTensor* y = nvt_from_nval(args[3]);
    bool use_bce = argc > 4 && (strcmp(nv_to_str(args[4]), "bce") == 0 || strcmp(nv_to_str(args[4]), "BCE") == 0);
    double lv = nv_train_step_internal(m, opt, x, y, use_bce);
    free(x->data); free(x);
    free(y->data); free(y);
    return nv_float(lv);
  }
  if (strcmp(name, "train_model") == 0 && argc > 4) {
    NVModel* m = nv_expect_model(args[0]);
    NVOptimizer* opt = nv_expect_opt(args[1]);
    NVTensor* x = nvt_from_nval(args[2]);
    NVTensor* y = nvt_from_nval(args[3]);
    int32_t epochs = (int32_t)nv_num(args[4]);
    bool use_bce = argc > 5 && (strcmp(nv_to_str(args[5]), "bce") == 0 || strcmp(nv_to_str(args[5]), "BCE") == 0);
    NVal report = nv_train_model_internal(m, opt, x, y, epochs, use_bce, true);
    free(x->data); free(x);
    free(y->data); free(y);
    return report;
  }
  if (strcmp(name, "autograd_demo") == 0 && argc > 1) {
    double xv = nv_num(args[0]);
    double wv = nv_num(args[1]);
    NVGradVar* x = nvg_new(nvt_scalar(xv), NV_GOP_LEAF, NULL, NULL, NULL);
    NVGradVar* w = nvg_new(nvt_scalar(wv), NV_GOP_LEAF, NULL, NULL, NULL);
    NVGradVar* b = nvg_new(nvt_scalar(1.5), NV_GOP_LEAF, NULL, NULL, NULL);
    NVGradVar* y = nvg_relu(nvg_add(nvg_mul(x, w), b));
    nvg_backward(y);
    NVal keys[4] = { nv_str("output"), nv_str("dx"), nv_str("dw"), nv_str("db") };
    NVal vals[4] = {
      nv_from_rounded_js(y->data->data[0], 4),
      nv_from_rounded_js(x->grad->data[0], 4),
      nv_from_rounded_js(w->grad->data[0], 4),
      nv_from_rounded_js(b->grad->data[0], 4)
    };
    return nv_alloc_map(4, keys, vals);
  }
  if (strcmp(name, "train_xor") == 0 && argc > 1) {
    int32_t epochs = (int32_t)nv_num(args[0]);
    double learning_rate = nv_num(args[1]);
    double lr = learning_rate > 0.0 ? (learning_rate < 0.1 ? learning_rate : 0.1) : 0.05;
    int32_t xsh[2] = { 4, 2 };
    int32_t ysh[2] = { 4, 1 };
    NVTensor* x = nvt_alloc(2, xsh);
    double xdata[8] = { 0, 0, 0, 1, 1, 0, 1, 1 };
    memcpy(x->data, xdata, sizeof(xdata));
    NVTensor* y = nvt_alloc(2, ysh);
    double ydata[4] = { 0, 1, 1, 0 };
    memcpy(y->data, ydata, sizeof(ydata));
    NVModel* m = nvm_new();
    nvm_add_linear(m, 2, 4, 42u);
    nvm_add_activation(m, NV_ACT_TANH);
    nvm_add_linear(m, 4, 1, 137u);
    nvm_add_activation(m, NV_ACT_SIGMOID);
    NVOptimizer* opt = nvo_new_adam(lr);
    NVal report = nv_train_model_internal(m, opt, x, y, epochs, false, false);
    free(x->data); free(x);
    free(y->data); free(y);
    return report;
  }

  nv_panic("Unknown builtin function in native runtime");
  return nv_none();
}
