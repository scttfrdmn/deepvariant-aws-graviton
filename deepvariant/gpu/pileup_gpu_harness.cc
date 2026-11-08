// Gate and timing harness for the GPU pileup encoder (aws-realign#130).
//
// Input: a DV_DUMP_PILEUP capture from make_examples (one record per
// BuildPileupForOneSample call: options, reference window, sampled reads in
// sampled order, and the CPU's output rows). Output: per-call images built
//   (1) on the host with the shared encoder (pileup_encode.h), and
//   (2) on the GPU with the same encoder,
// each compared byte-for-byte with the CPU rows laid out as FillPileupArray
// does (HWC). Optionally also compared with images exported from the
// make_examples TFRecords (--tfrecord_images FILE: concatenated HWC bytes in
// record order).
//
// Build (AMD):    hipcc -O3 -std=c++17 -ffp-contract=off -x hip -I. deepvariant/gpu/pileup_gpu_harness.cc -o pileup_gpu_harness
// Build (NVIDIA): nvcc -O3 -std=c++17 --fmad=false -x cu -I. deepvariant/gpu/pileup_gpu_harness.cc -o pileup_gpu_harness
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "deepvariant/gpu/gpu_compat.h"
#include "deepvariant/gpu/pileup_encode.h"

using namespace dvgpu;

#define CK(x) do { gpuError_t e_ = (x); if (e_ != gpuSuccess) { \
  fprintf(stderr, "GPU error %s at %s:%d\n", gpuGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

struct Capture {
  Options opt{};
  std::vector<CallDesc> calls;
  std::vector<ReadDesc> reads;
  std::vector<uint8_t> cig_op;
  std::vector<int32_t> cig_len;
  std::vector<char> seq;
  std::vector<uint8_t> qual;
  std::vector<char> ref;          // width bytes per call
  std::vector<uint8_t> cpu_hwc;   // CPU rows as FillPileupArray lays them out
};

template <typename T> static bool Rd(FILE* f, T* v) { return fread(v, sizeof(T), 1, f) == 1; }

static Capture Load(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) { perror(path); exit(1); }
  Capture cap;
  std::vector<std::tuple<std::string, int>> keys;  // per read, for ranks
  std::vector<uint8_t> rows;
  uint32_t magic;
  bool first = true;
  while (Rd(f, &magic)) {
    if (magic != 0x454c4950) { fprintf(stderr, "bad magic at call %zu\n", cap.calls.size()); exit(1); }
    Options o{};
    int32_t nrows;
    Rd(f, &o.width); Rd(f, &nrows); Rd(f, &o.ref_band); Rd(f, &o.nch);
    if (o.nch > kMaxChannels) { fprintf(stderr, "too many channels\n"); exit(1); }
    for (int i = 0; i < o.nch; ++i) Rd(f, &o.ch[i]);
    Rd(f, &o.off_ag); Rd(f, &o.off_tc); Rd(f, &o.stride); Rd(f, &o.ref_bq);
    Rd(f, &o.bq_cap); Rd(f, &o.mq_cap); Rd(f, &o.pos_color); Rd(f, &o.neg_color);
    Rd(f, &o.a_supp); Rd(f, &o.a_unsupp); Rd(f, &o.a_other); Rd(f, &o.a_match); Rd(f, &o.a_mismatch);
    Rd(f, &o.anchor); Rd(f, &o.min_mq); Rd(f, &o.min_bq);
    int32_t n_blank; Rd(f, &n_blank);
    if (n_blank != 0) { fprintf(stderr, "blanked channels not implemented\n"); exit(1); }
    o.height = nrows;
    if (first) { cap.opt = o; first = false; }
    else if (memcmp(&o, &cap.opt, sizeof(Options)) != 0) { fprintf(stderr, "options vary between calls\n"); exit(1); }
    CallDesc c{};
    Rd(f, &c.vstart); Rd(f, &c.img_start);
    size_t r0 = cap.ref.size(); cap.ref.resize(r0 + o.width);
    fread(cap.ref.data() + r0, 1, o.width, f);
    Rd(f, &c.max_reads); Rd(f, &c.n_sampled);
    c.read0 = cap.reads.size();
    for (int i = 0; i < c.n_sampled; ++i) {
      ReadDesc r{};
      uint8_t rev, support; uint16_t nl; int32_t read_number;
      Rd(f, &r.sort_pos); Rd(f, &r.hap); Rd(f, &r.align_pos); Rd(f, &r.mq); Rd(f, &rev);
      Rd(f, &r.frag_len); Rd(f, &read_number); Rd(f, &nl);
      std::string name(nl, '\0'); fread(name.data(), 1, nl, f);
      Rd(f, &support);
      r.rev = rev; r.support = support;
      Rd(f, &r.ncig); r.cig0 = cap.cig_op.size();
      for (int k = 0; k < r.ncig; ++k) { uint8_t op; int32_t len; Rd(f, &op); Rd(f, &len); cap.cig_op.push_back(op); cap.cig_len.push_back(len); }
      Rd(f, &r.seq_len); r.seq0 = cap.seq.size();
      cap.seq.resize(r.seq0 + r.seq_len); fread(cap.seq.data() + r.seq0, 1, r.seq_len, f);
      int32_t ql; Rd(f, &ql);
      if (ql != r.seq_len) { fprintf(stderr, "qual length != seq length\n"); exit(1); }
      cap.qual.resize(r.seq0 + ql); fread(cap.qual.data() + r.seq0, 1, ql, f);
      keys.emplace_back(std::move(name), read_number);
      cap.reads.push_back(r);
    }
    // CPU rows: channel-major per row -> HWC.
    const size_t img = size_t(nrows) * o.width * o.nch;
    rows.resize(img);
    fread(rows.data(), 1, img, f);
    size_t h0 = cap.cpu_hwc.size(); cap.cpu_hwc.resize(h0 + img);
    for (int row = 0; row < nrows; ++row)
      for (int col = 0; col < o.width; ++col)
        for (int ch = 0; ch < o.nch; ++ch)
          cap.cpu_hwc[h0 + (size_t(row) * o.width + col) * o.nch + ch] =
              rows[(size_t(row) * o.nch + ch) * o.width + col];
    cap.calls.push_back(c);
  }
  fclose(f);
  // (fragment_name, read_number) ranks: equal keys share a rank, so the
  // comparator is the CPU's tuple<string,int> comparison.
  std::vector<int> order(keys.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](int a, int b) { return keys[a] < keys[b]; });
  int rank = -1;
  for (size_t i = 0; i < order.size(); ++i) {
    if (i == 0 || keys[order[i - 1]] < keys[order[i]]) ++rank;
    cap.reads[order[i]].rank = rank;
  }
  return cap;
}

// Assemble one call's image from encoded rows: first max_reads ok reads in
// sampled order, sorted ByAlignment, below the reference band; zero rows after.
DV_HD inline int Assemble(const Options& o, const CallDesc& c, const ReadDesc* reads,
                          const uint8_t* ok, int* keep) {
  int n = 0;
  for (int i = 0; i < c.n_sampled && n < c.max_reads; ++i)
    if (ok[c.read0 + i]) keep[n++] = c.read0 + i;
  ByAlignment comp{reads};
  return StdSort(keep, keep + n, comp) * 1000 + n;  // fallbacks*1000 + rows
}

// ---------------- host reference ----------------
static bool g_stable = false;  // --stable: std::stable_sort instead of std::sort (contrast only)
static size_t g_tied_calls = 0;

static void HostEncode(const Capture& cap, std::vector<uint8_t>& out, std::vector<int>& heaps) {
  g_tied_calls = 0;
  const Options& o = cap.opt;
  const size_t row_bytes = size_t(o.width) * o.nch, img = row_bytes * o.height;
  out.assign(cap.calls.size() * img, 0);
  std::vector<uint8_t> scratch(cap.reads.size() * row_bytes, 0), ok(cap.reads.size(), 0);
  std::vector<int> keep(4096);
  for (size_t ci = 0; ci < cap.calls.size(); ++ci) {
    const CallDesc& c = cap.calls[ci];
    const char* ref = cap.ref.data() + ci * o.width;
    for (int i = 0; i < c.n_sampled; ++i) {
      const ReadDesc& r = cap.reads[c.read0 + i];
      ok[c.read0 + i] = r.mq >= o.min_mq &&
          EncodeReadRow(o, c, r, cap.cig_op.data(), cap.cig_len.data(), cap.seq.data() + r.seq0,
                        cap.qual.data() + r.seq0, ref, scratch.data() + (c.read0 + i) * row_bytes);
    }
    if (size_t(c.n_sampled) > keep.size()) keep.resize(c.n_sampled);
    int res = Assemble(o, c, cap.reads.data(), ok.data(), keep.data());
    heaps[ci] = res / 1000;
    {
      ByAlignment comp{cap.reads.data()};
      const int n = res % 1000;
      for (int k = 1; k < n; ++k)
        if (!comp(keep[k - 1], keep[k]) && !comp(keep[k], keep[k - 1])) { g_tied_calls++; break; }
      if (g_stable) {  // redo the selection with a stable sort
        int m = 0;
        for (int i = 0; i < c.n_sampled && m < c.max_reads; ++i) if (ok[c.read0 + i]) keep[m++] = c.read0 + i;
        std::stable_sort(keep.begin(), keep.begin() + m, comp);
      }
    }
    uint8_t* im = out.data() + ci * img;
    for (int row = 0; row < o.ref_band; ++row)
      for (int col = 0; col < o.width; ++col) EncodeRefRow(o, ref, col, im + row * row_bytes + col * o.nch);
    for (int k = 0; k < res % 1000; ++k)
      memcpy(im + (o.ref_band + k) * row_bytes, scratch.data() + size_t(keep[k]) * row_bytes, row_bytes);
  }
}

// ---------------- device ----------------
__global__ void EncodeKernel(Options o, const CallDesc* calls, const int* read_call, int n_reads,
                             const ReadDesc* reads, const uint8_t* cig_op, const int32_t* cig_len,
                             const char* seq, const uint8_t* qual, const char* ref,
                             uint8_t* scratch, uint8_t* ok) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_reads) return;
  const ReadDesc r = reads[i];
  const int ci = read_call[i];
  const size_t row_bytes = size_t(o.width) * o.nch;
  uint8_t* row = scratch + size_t(i) * row_bytes;
  for (size_t b = 0; b < row_bytes; ++b) row[b] = 0;
  ok[i] = r.mq >= o.min_mq &&
          EncodeReadRow(o, calls[ci], r, cig_op, cig_len, seq + r.seq0, qual + r.seq0,
                        ref + size_t(ci) * o.width, row);
}

__global__ void AssembleKernel(Options o, const CallDesc* calls, const ReadDesc* reads,
                               const uint8_t* ok, const char* ref, const uint8_t* scratch,
                               int* keep_all, int* heaps, uint8_t* out) {
  const int ci = blockIdx.x;
  const CallDesc c = calls[ci];
  int* keep = keep_all + c.read0;
  __shared__ int nkeep;
  if (threadIdx.x == 0) {
    int res = Assemble(o, c, reads, ok, keep);
    heaps[ci] = res / 1000;
    nkeep = res % 1000;
  }
  __syncthreads();
  const size_t row_bytes = size_t(o.width) * o.nch;
  uint8_t* im = out + size_t(ci) * row_bytes * o.height;
  const char* rf = ref + size_t(ci) * o.width;
  for (int row = 0; row < o.height; ++row) {
    for (int col = threadIdx.x; col < o.width; col += blockDim.x) {
      uint8_t* px = im + row * row_bytes + col * o.nch;
      if (row < o.ref_band) {
        EncodeRefRow(o, rf, col, px);
      } else if (row - o.ref_band < nkeep) {
        const uint8_t* src = scratch + size_t(keep[row - o.ref_band]) * row_bytes + col * o.nch;
        for (int k = 0; k < o.nch; ++k) px[k] = src[k];
      } else {
        for (int k = 0; k < o.nch; ++k) px[k] = 0;
      }
    }
  }
}

template <typename T> static T* Up(const std::vector<T>& v) {
  T* d = nullptr;
  CK(gpuMalloc(&d, std::max<size_t>(1, v.size()) * sizeof(T)));
  if (!v.empty()) CK(gpuMemcpy(d, v.data(), v.size() * sizeof(T), gpuMemcpyHostToDevice));
  return d;
}

static size_t Diff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, size_t img, const char* what) {
  size_t bad = 0, first = SIZE_MAX;
  for (size_t ci = 0; ci * img < a.size(); ++ci)
    if (memcmp(a.data() + ci * img, b.data() + ci * img, img) != 0) { if (first == SIZE_MAX) first = ci; bad++; }
  printf("%-34s %s (%zu of %zu images differ%s)\n", what, bad ? "DIFFERENT" : "IDENTICAL", bad, a.size() / img,
         bad ? (", first call " + std::to_string(first)).c_str() : "");
  return bad;
}

#ifndef DV_HARNESS_NO_MAIN  // fused_cv.cc includes this file for Load/Assemble/Up
int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s DUMP.bin [--tfrecord_images FILE] [--reps N]\n", argv[0]); return 1; }
  const char* tfimg = nullptr; int reps = 5; double gpu_seconds = 0;
  for (int i = 2; i < argc; ++i) {
    if (!strcmp(argv[i], "--tfrecord_images") && i + 1 < argc) tfimg = argv[++i];
    else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--stable")) g_stable = true;
    // --gpu_seconds S: after the timed reps, keep the kernels running back to
    // back for S seconds so 1 Hz DCGM sampling (tach) sees a steady state.
    else if (!strcmp(argv[i], "--gpu_seconds") && i + 1 < argc) gpu_seconds = atof(argv[++i]);
  }
  auto t0 = std::chrono::steady_clock::now();
  Capture cap = Load(argv[1]);
  auto t1 = std::chrono::steady_clock::now();
  const Options& o = cap.opt;
  const size_t row_bytes = size_t(o.width) * o.nch, img = row_bytes * o.height;
  size_t max_sampled = 0;
  for (auto& c : cap.calls) max_sampled = std::max<size_t>(max_sampled, c.n_sampled);
  printf("capture: %zu calls, %zu sampled reads (max %zu per call), image %dx%dx%d, channels", cap.calls.size(),
         cap.reads.size(), max_sampled, o.height, o.width, o.nch);
  for (int i = 0; i < o.nch; ++i) printf(" %d", o.ch[i]);
  printf("; load %.2f s\n", std::chrono::duration<double>(t1 - t0).count());

  // Host reference (single thread), best of reps.
  std::vector<uint8_t> host; std::vector<int> host_heaps(cap.calls.size());
  double host_best = 1e30;
  for (int r = 0; r < reps; ++r) {
    auto a = std::chrono::steady_clock::now();
    HostEncode(cap, host, host_heaps);
    host_best = std::min(host_best, std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count());
  }
  int hh = 0; for (int h : host_heaps) hh += h;
  printf("host encoder (1 thread%s): %.3f s best of %d, %.1f us/call; heap-sort fallbacks %d; calls with tied sort keys %zu\n",
         g_stable ? ", stable_sort" : "", host_best, reps, 1e6 * host_best / cap.calls.size(), hh, g_tied_calls);
  size_t bad = Diff(cap.cpu_hwc, host, img, "host encoder vs make_examples rows:");

  // Device.
  gpuDeviceProp prop; CK(gpuGetDeviceProperties(&prop, 0));
  printf("GPU: %s, %d CUs/SMs\n", prop.name, prop.multiProcessorCount);
  std::vector<int> read_call(cap.reads.size());
  for (size_t ci = 0; ci < cap.calls.size(); ++ci)
    for (int i = 0; i < cap.calls[ci].n_sampled; ++i) read_call[cap.calls[ci].read0 + i] = ci;
  auto h0 = std::chrono::steady_clock::now();
  CallDesc* d_calls = Up(cap.calls); ReadDesc* d_reads = Up(cap.reads); int* d_rc = Up(read_call);
  uint8_t* d_op = Up(cap.cig_op); int32_t* d_len = Up(cap.cig_len); char* d_seq = Up(cap.seq);
  uint8_t* d_qual = Up(cap.qual); char* d_ref = Up(cap.ref);
  CK(gpuDeviceSynchronize());
  double h2d = std::chrono::duration<double>(std::chrono::steady_clock::now() - h0).count();
  size_t in_bytes = cap.calls.size() * sizeof(CallDesc) + cap.reads.size() * (sizeof(ReadDesc) + sizeof(int)) +
                    cap.cig_op.size() * 5 + cap.seq.size() * 2 + cap.ref.size();
  uint8_t *d_scratch, *d_ok, *d_out; int *d_keep, *d_heaps;
  CK(gpuMalloc(&d_scratch, cap.reads.size() * row_bytes)); CK(gpuMalloc(&d_ok, cap.reads.size()));
  CK(gpuMalloc(&d_out, cap.calls.size() * img)); CK(gpuMalloc(&d_keep, cap.reads.size() * sizeof(int)));
  CK(gpuMalloc(&d_heaps, cap.calls.size() * sizeof(int)));
  gpuEvent_t e0, e1, e2; CK(gpuEventCreate(&e0)); CK(gpuEventCreate(&e1)); CK(gpuEventCreate(&e2));
  float best_enc = 1e30f, best_asm = 1e30f;
  const int n_reads = cap.reads.size();
  for (int r = 0; r < reps + 1; ++r) {  // first rep warms up
    CK(gpuMemset(d_out, 0xAB, cap.calls.size() * img));  // poison: every byte must be written
    CK(gpuEventRecord(e0));
    EncodeKernel<<<(n_reads + 127) / 128, 128>>>(o, d_calls, d_rc, n_reads, d_reads, d_op, d_len, d_seq, d_qual,
                                                 d_ref, d_scratch, d_ok);
    CK(gpuEventRecord(e1));
    AssembleKernel<<<cap.calls.size(), 64>>>(o, d_calls, d_reads, d_ok, d_ref, d_scratch, d_keep, d_heaps, d_out);
    CK(gpuEventRecord(e2));
    CK(gpuEventSynchronize(e2)); CK(gpuGetLastError());
    float a, b; CK(gpuEventElapsedTime(&a, e0, e1)); CK(gpuEventElapsedTime(&b, e1, e2));
    if (r > 0) { best_enc = std::min(best_enc, a); best_asm = std::min(best_asm, b); }
  }
  if (gpu_seconds > 0) {
    auto utc = [] { char b[32]; time_t t = time(nullptr); strftime(b, sizeof b, "%FT%TZ", gmtime(&t)); return std::string(b); };
    std::string w0 = utc();
    auto s0 = std::chrono::steady_clock::now();
    long n = 0;
    CK(gpuEventRecord(e0));
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count() < gpu_seconds) {
      for (int k = 0; k < 16; ++k, ++n) {
        EncodeKernel<<<(n_reads + 127) / 128, 128>>>(o, d_calls, d_rc, n_reads, d_reads, d_op, d_len, d_seq, d_qual,
                                                     d_ref, d_scratch, d_ok);
        AssembleKernel<<<cap.calls.size(), 64>>>(o, d_calls, d_reads, d_ok, d_ref, d_scratch, d_keep, d_heaps, d_out);
      }
      CK(gpuDeviceSynchronize());
    }
    CK(gpuEventRecord(e1)); CK(gpuEventSynchronize(e1)); CK(gpuGetLastError());
    float ms; CK(gpuEventElapsedTime(&ms, e0, e1));
    printf("GPU sustained: %ld passes in %.1f s (%s to %s) = %.2f us/call mean\n", n, ms / 1e3, w0.c_str(),
           utc().c_str(), 1e3 * ms / (double(n) * cap.calls.size()));
  }
  std::vector<uint8_t> dev(cap.calls.size() * img);
  auto d0 = std::chrono::steady_clock::now();
  CK(gpuMemcpy(dev.data(), d_out, dev.size(), gpuMemcpyDeviceToHost));
  double d2h = std::chrono::duration<double>(std::chrono::steady_clock::now() - d0).count();
  std::vector<int> dev_heaps(cap.calls.size());
  CK(gpuMemcpy(dev_heaps.data(), d_heaps, dev_heaps.size() * sizeof(int), gpuMemcpyDeviceToHost));
  int dh = 0; for (int h : dev_heaps) dh += h;
  printf("GPU encode kernel %.3f ms + assemble kernel %.3f ms (best of %d) = %.2f us/call; heap fallbacks %d\n",
         best_enc, best_asm, reps, 1e3 * (best_enc + best_asm) / cap.calls.size(), dh);
  printf("transfers (not on the fused path's critical inputs): H2D inputs %.1f MB in %.3f s; D2H images %.1f MB in %.3f s\n",
         in_bytes / 1e6, h2d, dev.size() / 1e6, d2h);
  bad += Diff(cap.cpu_hwc, dev, img, "GPU encoder vs make_examples rows:");
  bad += Diff(host, dev, img, "GPU encoder vs host encoder:");
  if (tfimg) {
    FILE* f = fopen(tfimg, "rb");
    if (!f) { perror(tfimg); return 1; }
    std::vector<uint8_t> tf(cap.calls.size() * img);
    size_t got = fread(tf.data(), 1, tf.size(), f);
    fclose(f);
    if (got != tf.size()) printf("tfrecord images: %zu bytes, expected %zu\n", got, tf.size());
    else bad += Diff(tf, dev, img, "GPU encoder vs TFRecord images:");
  }
  printf("GATE %s\n", bad ? "FAIL" : "IDENTICAL");
  return bad ? 3 : 0;
}
#endif  // DV_HARNESS_NO_MAIN
