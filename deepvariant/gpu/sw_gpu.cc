// Exact GPU Smith-Waterman for DeepVariant's realigner (aws-realign#130 (b),
// deepvariant-aws-graviton#10). Reproduces the forward and reverse passes of
// libssw's ssw_align as DeepVariant runs it (sw_scan16, libssw-scan.patch,
// itself byte-identical to SSE2 libssw on all of chr20, #9):
//   Gotoh local alignment, H floored at 0, E/F with saturating (floor-0)
//   subtraction, F from H' = max(diag + s, E) (true Gotoh);
//   end_ref  = the column (in iteration order) where the running max last
//              strictly increased; end_read = smallest query index with
//              H == max in that column; max == 0 -> end_ref 0, end_read 0;
//   stop after the first column whose max == terminate (reverse pass:
//              terminate = score1, over ref[0..ref_end1] right to left with
//              the reversed read[0..read_end1]).
// banded_sw (the CIGAR) stays on the CPU: given identical score/ends/begins it
// is the same function of the same inputs.
//
// Input: an SSW_DUMP file (libssw hook; one record per ssw_align_scan call,
// inputs and outputs). Output: per-record comparison, timing, and an
// SSW_GPU_IN file for the end-to-end gate (fpa_replay with SSW_GPU_IN set
// replaces both passes with these results; the gate is its Read comparison).
//
// One thread per alignment, H/E in global scratch interleaved across threads
// (coalesced); records sorted by work (refLen x readLen) so warps are uniform.
//
// Build (AMD):    hipcc -O3 -std=c++17 -x hip -I. deepvariant/gpu/sw_gpu.cc -o sw_gpu
// Build (NVIDIA): nvcc -O3 -std=c++17 -x cu -I. deepvariant/gpu/sw_gpu.cc -o sw_gpu
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#include "deepvariant/gpu/gpu_compat.h"

#define CK(x) do { gpuError_t e_ = (x); if (e_ != gpuSuccess) { \
  fprintf(stderr, "GPU error %s at %s:%d\n", gpuGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

struct Rec {  // device-side view
  int32_t ref0, refLen, read0, readLen, mat0, n;
  uint8_t gapO, gapE, flag, pad;
  uint16_t filters, pad2;
};
struct Out {
  int32_t score1, ref_end1, read_end1, ref_begin1, read_begin1, rev_score, rev_ran;
};
struct Want {
  int32_t score1, ref_begin1, ref_end1, read_begin1, read_end1, rflag;
  uint16_t rev_score;
};

// One SW pass. ref is indexed by column i in [0, refLen); dir 0: i ascending, 1: descending.
// read(j) for j in [0, readLen): forward = read[j], reverse = read[readLen - 1 - j].
DV_HD inline void Pass(const int8_t* ref, int refLen, int dir, const int8_t* read, int readLen, bool rev_read,
                       const int8_t* mat, int n, int gapO, int gapE, int terminate, int16_t* H, int16_t* E,
                       size_t stride, int* o_max, int* o_ref, int* o_read) {
  for (int j = 0; j < readLen; ++j) { H[j * stride] = 0; E[j * stride] = 0; }
  int max = 0, end_ref = 0, end_read = 0;
  for (int k = 0; k < refLen; ++k) {
    const int i = dir == 0 ? k : refLen - 1 - k;
    const int8_t* row = mat + ref[i] * n;
    int diag = 0, F = 0, colmax = 0, colarg = 0;
    for (int j = 0; j < readLen; ++j) {
      const int q = rev_read ? read[readLen - 1 - j] : read[j];
      const int hold = H[j * stride];
      const int e = E[j * stride];
      int hp = diag + row[q];
      if (hp < e) hp = e;                    // H' = max(diag + s, E)  (>= 0 since E >= 0)
      const int h = hp > F ? hp : F;         // H = max(H', F)
      // F for the next query position from H' (true Gotoh; equals using H since gapO >= gapE).
      int f1 = F - gapE; if (f1 < 0) f1 = 0;
      int f2 = hp - gapO; if (f2 < 0) f2 = 0;
      F = f1 > f2 ? f1 : f2;
      int e1 = e - gapE; if (e1 < 0) e1 = 0;
      int e2 = h - gapO; if (e2 < 0) e2 = 0;
      E[j * stride] = int16_t(e1 > e2 ? e1 : e2);
      H[j * stride] = int16_t(h);
      diag = hold;
      if (h > colmax) { colmax = h; colarg = j; }  // first j reaching the column max
    }
    if (colmax > max) { max = colmax; end_ref = i; end_read = colarg; }
    if (colmax == terminate) break;
  }
  *o_max = max; *o_ref = end_ref; *o_read = end_read;
}

__global__ void SwKernel(const Rec* recs, const int* order, int nrec, const int8_t* refs, const int8_t* reads,
                         const int8_t* mats, int16_t* scratch, size_t stride, Out* out) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nrec) return;
  const Rec r = recs[order[t]];
  // H and E planes: element (j, thread t) at j * T + t, E after all of H.
  const size_t T = size_t(gridDim.x) * blockDim.x;
  int16_t* H = scratch + t;
  int16_t* E = scratch + T * stride + t;
  const size_t s = T;
  Out o{};
  int mx, er, ed;
  Pass(refs + r.ref0, r.refLen, 0, reads + r.read0, r.readLen, false, mats + r.mat0, r.n, r.gapO, r.gapE, 0x10000,
       H, E, s, &mx, &er, &ed);
  o.score1 = mx; o.ref_end1 = er; o.read_end1 = ed; o.ref_begin1 = -1; o.read_begin1 = -1;
  const bool run_rev = !(r.flag == 0 || (r.flag == 2 && mx < r.filters));
  if (run_rev) {
    // ssw: read_reverse = reverse(read[0..read_end1]); refLen = ref_end1 + 1; ref_dir 1; terminate score1.
    int rm, rr, rd;
    Pass(refs + r.ref0, er + 1, 1, reads + r.read0, ed + 1, true, mats + r.mat0, r.n, r.gapO, r.gapE, mx, H, E, s,
         &rm, &rr, &rd);
    o.rev_score = rm; o.ref_begin1 = rr; o.read_begin1 = ed - rd; o.rev_ran = 1;
  }
  out[order[t]] = o;
}


// The same pass, strip-mined over the query: K query rows live in registers
// while the pass sweeps every column; the strip's bottom row (H and the F
// carry) is handed to the next strip through per-column arrays Hb/Fb, and the
// per-column (max, first row) is folded into Key = max << 16 | (0xFFFF - row)
// (larger max wins; on a tie the earlier strip's smaller row stays). The
// sequential selection (strict increase, stop at terminate) then runs over
// the column keys in iteration order: it depends only on column maxima, so
// computing columns past the stop changes nothing.
template <int K>
DV_HD inline void PassStrip(const int8_t* ref, int refLen, int dir, const int8_t* read, int readLen, bool rev_read,
                            const int8_t* mat, int n, int gapO, int gapE, int terminate, int16_t* Hb, int16_t* Fb,
                            int32_t* Key, size_t stride, int* o_max, int* o_ref, int* o_read) {
  for (int j0 = 0; j0 < readLen; j0 += K) {
    const int kk = readLen - j0 < K ? readLen - j0 : K;
    int h[K], e[K], q[K];
#pragma unroll
    for (int r = 0; r < K; ++r) {
      h[r] = 0; e[r] = 0;
      const int j = j0 + r;
      q[r] = r < kk ? (rev_read ? read[readLen - 1 - j] : read[j]) : 0;
    }
    int diag_top = 0;
    for (int k = 0; k < refLen; ++k) {
      const int i = dir == 0 ? k : refLen - 1 - k;
      const size_t ix = size_t(i) * stride;
      const int htop = j0 ? Hb[ix] : 0;
      int F = j0 ? Fb[ix] : 0;
      const int8_t* row = mat + ref[i] * n;
      int diag = diag_top, cmax = 0, carg = 0, hbot = 0;
#pragma unroll
      for (int r = 0; r < K; ++r) {
        if (r < kk) {
          const int hold = h[r];
          int hp = diag + row[q[r]];
          if (hp < e[r]) hp = e[r];
          const int hn = hp > F ? hp : F;
          int f1 = F - gapE; if (f1 < 0) f1 = 0;
          int f2 = hp - gapO; if (f2 < 0) f2 = 0;
          F = f1 > f2 ? f1 : f2;
          int e1 = e[r] - gapE; if (e1 < 0) e1 = 0;
          int e2 = hn - gapO; if (e2 < 0) e2 = 0;
          e[r] = e1 > e2 ? e1 : e2;
          h[r] = hn; diag = hold; hbot = hn;
          if (hn > cmax) { cmax = hn; carg = j0 + r; }
        }
      }
      diag_top = htop;
      Hb[ix] = int16_t(hbot); Fb[ix] = int16_t(F);
      const int32_t key = (cmax << 16) | (0xFFFF - carg);
      Key[ix] = (j0 == 0 || key > Key[ix]) ? key : Key[ix];
    }
  }
  int max = 0, end_ref = 0, end_read = 0;
  for (int k = 0; k < refLen; ++k) {
    const int i = dir == 0 ? k : refLen - 1 - k;
    const int32_t key = Key[size_t(i) * stride];
    const int cm = key >> 16;
    if (cm > max) { max = cm; end_ref = i; end_read = 0xFFFF - (key & 0xFFFF); }
    if (cm == terminate) break;
  }
  *o_max = max; *o_ref = end_ref; *o_read = end_read;
}

template <int K>
__global__ void SwStripKernel(const Rec* recs, const int* order, int nrec, const int8_t* refs, const int8_t* reads,
                              const int8_t* mats, int16_t* hb, int16_t* fb, int32_t* key, Out* out) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nrec) return;
  const Rec r = recs[order[t]];
  const size_t T = size_t(gridDim.x) * blockDim.x;
  Out o{};
  int mx, er, ed;
  PassStrip<K>(refs + r.ref0, r.refLen, 0, reads + r.read0, r.readLen, false, mats + r.mat0, r.n, r.gapO, r.gapE,
               0x10000, hb + t, fb + t, key + t, T, &mx, &er, &ed);
  o.score1 = mx; o.ref_end1 = er; o.read_end1 = ed; o.ref_begin1 = -1; o.read_begin1 = -1;
  if (!(r.flag == 0 || (r.flag == 2 && mx < r.filters))) {
    int rm, rr, rd;
    PassStrip<K>(refs + r.ref0, er + 1, 1, reads + r.read0, ed + 1, true, mats + r.mat0, r.n, r.gapO, r.gapE, mx,
                 hb + t, fb + t, key + t, T, &rm, &rr, &rd);
    o.rev_score = rm; o.ref_begin1 = rr; o.read_begin1 = ed - rd; o.rev_ran = 1;
  }
  out[order[t]] = o;
}

static int g_k = 32, g_threads = 256;

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s SSW_DUMP.bin [--gpu_in OUT.bin] [--reps N]\n", argv[0]); return 1; }
  const char* gin = nullptr; int reps = 3; bool naive = false;
  for (int i = 2; i < argc; ++i) {
    if (!strcmp(argv[i], "--gpu_in") && i + 1 < argc) gin = argv[++i];
    else if (!strcmp(argv[i], "--naive")) naive = true;
    else if (!strcmp(argv[i], "--k") && i + 1 < argc) g_k = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--threads") && i + 1 < argc) g_threads = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
  }
  FILE* f = fopen(argv[1], "rb");
  if (!f) { perror(argv[1]); return 1; }
  std::vector<Rec> recs; std::vector<Want> want; std::vector<int8_t> refs, reads, mats;
  uint32_t magic;
  double cells = 0;
  int maxlen = 0, maxref = 0;
  auto t0 = std::chrono::steady_clock::now();
  while (fread(&magic, 4, 1, f) == 1) {
    if (magic != 0x52575353) { fprintf(stderr, "bad magic at record %zu\n", recs.size()); return 1; }
    int32_t hdr[5]; uint8_t pen[4]; uint16_t filters;
    fread(hdr, 4, 5, f); fread(pen, 1, 4, f); fread(&filters, 2, 1, f);
    Rec r{};
    r.refLen = hdr[0]; r.readLen = hdr[1]; r.n = hdr[2];
    r.gapO = pen[0]; r.gapE = pen[1]; r.flag = pen[2]; r.filters = filters;
    r.ref0 = refs.size(); refs.resize(refs.size() + r.refLen); fread(refs.data() + r.ref0, 1, r.refLen, f);
    r.read0 = reads.size(); reads.resize(reads.size() + r.readLen); fread(reads.data() + r.read0, 1, r.readLen, f);
    // Matrices are tiny and identical across calls in practice; dedupe the last one.
    std::vector<int8_t> m(r.n * r.n); fread(m.data(), 1, m.size(), f);
    if (!mats.empty() && recs.size() && recs.back().n == r.n && memcmp(mats.data() + recs.back().mat0, m.data(), m.size()) == 0)
      r.mat0 = recs.back().mat0;
    else { r.mat0 = mats.size(); mats.insert(mats.end(), m.begin(), m.end()); }
    int32_t o[6]; uint16_t rev; int32_t clen;
    fread(o, 4, 6, f); fread(&rev, 2, 1, f); fread(&clen, 4, 1, f);
    if (clen > 0) fseek(f, 4L * clen, SEEK_CUR);
    want.push_back({o[0], o[1], o[2], o[3], o[4], o[5], rev});
    recs.push_back(r);
    cells += double(r.refLen) * r.readLen;
    maxlen = std::max(maxlen, r.readLen); maxref = std::max(maxref, r.refLen);
  }
  fclose(f);
  const int N = recs.size();
  printf("records %d, forward-pass cells %.3g, max query %d, load %.2f s\n", N, cells, maxlen,
         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  // Order by work so a warp's threads loop alike.
  std::vector<int> order(N); std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    long wa = long(recs[a].refLen) * recs[a].readLen, wb = long(recs[b].refLen) * recs[b].readLen;
    return wa != wb ? wa > wb : a < b; });

  gpuDeviceProp prop; CK(gpuGetDeviceProperties(&prop, 0));
  printf("GPU: %s, %d CUs/SMs\n", prop.name, prop.multiProcessorCount);
  auto up = [](const void* p, size_t n) { void* d; CK(gpuMalloc(&d, std::max<size_t>(n, 1))); if (n) CK(gpuMemcpy(d, p, n, gpuMemcpyHostToDevice)); return d; };
  Rec* d_recs = (Rec*)up(recs.data(), recs.size() * sizeof(Rec));
  int* d_order = (int*)up(order.data(), order.size() * sizeof(int));
  int8_t* d_refs = (int8_t*)up(refs.data(), refs.size());
  int8_t* d_reads = (int8_t*)up(reads.data(), reads.size());
  int8_t* d_mats = (int8_t*)up(mats.data(), mats.size());
  const int threads = g_threads, blocks = (N + threads - 1) / threads;
  const size_t T = size_t(blocks) * threads;
  int16_t* d_scratch = nullptr; int16_t *d_hb = nullptr, *d_fb = nullptr; int32_t* d_key = nullptr;
  if (naive) CK(gpuMalloc(&d_scratch, 2 * T * size_t(maxlen) * sizeof(int16_t)));
  else {
    CK(gpuMalloc(&d_hb, T * size_t(maxref) * sizeof(int16_t))); CK(gpuMalloc(&d_fb, T * size_t(maxref) * sizeof(int16_t)));
    CK(gpuMalloc(&d_key, T * size_t(maxref) * sizeof(int32_t)));
  }
  Out* d_out; CK(gpuMalloc(&d_out, N * sizeof(Out)));
  gpuEvent_t e0, e1; CK(gpuEventCreate(&e0)); CK(gpuEventCreate(&e1));
  float best = 1e30f;
  for (int r = 0; r < reps; ++r) {
    CK(gpuMemset(d_out, 0xFF, N * sizeof(Out)));
    CK(gpuEventRecord(e0));
    if (naive) SwKernel<<<blocks, threads>>>(d_recs, d_order, N, d_refs, d_reads, d_mats, d_scratch, size_t(maxlen), d_out);
    else if (g_k == 8) SwStripKernel<8><<<blocks, threads>>>(d_recs, d_order, N, d_refs, d_reads, d_mats, d_hb, d_fb, d_key, d_out);
    else if (g_k == 64) SwStripKernel<64><<<blocks, threads>>>(d_recs, d_order, N, d_refs, d_reads, d_mats, d_hb, d_fb, d_key, d_out);
    else if (g_k == 48) SwStripKernel<48><<<blocks, threads>>>(d_recs, d_order, N, d_refs, d_reads, d_mats, d_hb, d_fb, d_key, d_out);
    else if (g_k == 32) SwStripKernel<32><<<blocks, threads>>>(d_recs, d_order, N, d_refs, d_reads, d_mats, d_hb, d_fb, d_key, d_out);
    else SwStripKernel<16><<<blocks, threads>>>(d_recs, d_order, N, d_refs, d_reads, d_mats, d_hb, d_fb, d_key, d_out);
    CK(gpuEventRecord(e1)); CK(gpuEventSynchronize(e1)); CK(gpuGetLastError());
    float ms; CK(gpuEventElapsedTime(&ms, e0, e1)); best = std::min(best, ms);
  }
  std::vector<Out> out(N);
  CK(gpuMemcpy(out.data(), d_out, N * sizeof(Out), gpuMemcpyDeviceToHost));
  long bad_fwd = 0, bad_rev = 0, rev_n = 0; int first = -1;
  for (int i = 0; i < N; ++i) {
    const Out& o = out[i]; const Want& w = want[i];
    if (o.score1 != w.score1 || o.ref_end1 != w.ref_end1 || o.read_end1 != w.read_end1) { ++bad_fwd; if (first < 0) first = i; }
    if (o.rev_ran) {
      ++rev_n;
      if (o.ref_begin1 != w.ref_begin1 || o.read_begin1 != w.read_begin1 || o.rev_score != w.rev_score) { ++bad_rev; if (first < 0) first = i; }
    } else if (w.ref_begin1 != -1) { ++bad_rev; if (first < 0) first = i; }
  }
  printf("forward pass  (score1, ref_end1, read_end1): %s (%ld of %d differ)\n", bad_fwd ? "DIFFERENT" : "IDENTICAL", bad_fwd, N);
  printf("reverse pass  (ref_begin1, read_begin1, score): %s (%ld of %ld differ)\n", bad_rev ? "DIFFERENT" : "IDENTICAL", bad_rev, rev_n);
  if (first >= 0) {
    const Out& o = out[first]; const Want& w = want[first];
    printf("first difference: record %d refLen %d readLen %d: gpu score %d ref_end %d read_end %d begin %d/%d rev %d; cpu %d %d %d %d/%d rev %d\n",
           first, recs[first].refLen, recs[first].readLen, o.score1, o.ref_end1, o.read_end1, o.ref_begin1, o.read_begin1,
           o.rev_score, w.score1, w.ref_end1, w.read_end1, w.ref_begin1, w.read_begin1, w.rev_score);
  }
  if (naive) printf("naive kernel "); else printf("strip-%d kernel, %d threads/block: ", g_k, g_threads);
  printf("GPU kernel %.3f ms best of %d: %.1f GCUPS (forward cells), %.2f us/alignment\n", best, reps,
         cells / (best * 1e-3) / 1e9, 1e3 * best / N);
  if (gin) {
    FILE* g = fopen(gin, "wb");
    for (int i = 0; i < N; ++i) {
      struct { uint32_t magic; int32_t readLen, refLen; uint16_t score1, rev_score; int32_t ref_end1, read_end1, ref_begin1, read_begin1; } x;
      x.magic = 0x47575353; x.readLen = recs[i].readLen; x.refLen = recs[i].refLen;
      x.score1 = uint16_t(out[i].score1); x.rev_score = uint16_t(out[i].rev_ran ? out[i].rev_score : 0);
      x.ref_end1 = out[i].ref_end1; x.read_end1 = out[i].read_end1; x.ref_begin1 = out[i].ref_begin1; x.read_begin1 = out[i].read_begin1;
      fwrite(&x, sizeof x, 1, g);
    }
    fclose(g);
    printf("wrote %s (%d records)\n", gin, N);
  }
  printf("GATE %s\n", (bad_fwd || bad_rev) ? "FAIL" : "IDENTICAL");
  return (bad_fwd || bad_rev) ? 3 : 0;
}
