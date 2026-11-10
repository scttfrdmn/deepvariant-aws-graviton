// SPDX-License-Identifier: Apache-2.0
// LD_PRELOAD shim (deepvariant-aws-graviton#10): override the deflate level that callers ask for, from DV_ZLEVEL (0-9).
// Load it before the zlib that should do the work (it forwards with RTLD_NEXT).
// Compressed bytes change; the decompressed stream (the records) does not.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <zlib.h>
static int lvl(int asked) { const char *e = getenv("DV_ZLEVEL"); return (e && *e) ? atoi(e) : asked; }
int deflateInit2_(z_streamp s, int level, int method, int wbits, int mem, int strategy, const char *v, int sz) {
  static int (*real)(z_streamp, int, int, int, int, int, const char *, int);
  if (!real) real = dlsym(RTLD_NEXT, "deflateInit2_");
  return real(s, lvl(level), method, wbits, mem, strategy, v, sz);
}
int deflateInit_(z_streamp s, int level, const char *v, int sz) {
  static int (*real)(z_streamp, int, const char *, int);
  if (!real) real = dlsym(RTLD_NEXT, "deflateInit_");
  return real(s, lvl(level), v, sz);
}
int deflateParams(z_streamp s, int level, int strategy) {
  static int (*real)(z_streamp, int, int);
  if (!real) real = dlsym(RTLD_NEXT, "deflateParams");
  return real(s, lvl(level), strategy);
}
