// SPDX-License-Identifier: Apache-2.0
//
// Standalone oneDNN v2.7.3 executor for a span of the call_variants
// InceptionV3 graph (spec from blockspec.py), reproducing TF 2.13.1's oneDNN
// primitives (native NHWC, _MklNativeFusedConv2D with the FusedBatchNorm +
// Relu post-op chain, pooling_v2) so the arithmetic is unchanged. Only data
// movement differs, each change switchable for an A/B:
//   --reorder-each   re-reorder every filter on every run (TF stock behaviour)
//   --pool-inf       max pool as forward_inference (no workspace)
//   --concat-direct  convs write straight into their channel slice of the
//                    concat output (strided dst); refuses if the impl changes
//   --rsqrt eigen|div  BN rsqrt like TF (Eigen, AVX) or 1/sqrtf
//
//   cvdnn DIR [--reps N] [options]   (OMP_NUM_THREADS sets threads)
// Prints each primitive's impl, the span's median wall time and whether the
// output equals DIR/ref.f32 bit-for-bit.
//   cvdnn DIR --images DUMP --n N [options]   (whole network spec only)
// Runs N dumped images (images.u8, call_variants' (x - 128) / 128) in batches
// of the spec's N, timing each batch including the input conversion, and
// compares the probabilities with DUMP/probs_tf213.npy bit-for-bit.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/resource.h>
#include <unistd.h>

#include <unsupported/Eigen/CXX11/Tensor>

#include "dnnl.hpp"

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

static std::vector<float> load(const std::string& f, size_t n) {
  std::vector<float> v(n);
  std::ifstream in(f, std::ios::binary);
  if (!in.read(reinterpret_cast<char*>(v.data()), n * sizeof(float))) {
    fprintf(stderr, "short read %s\n", f.c_str());
    exit(2);
  }
  return v;
}

// TF: Eigen::Tensor<float,1,RowMajor> r = (var + eps).rsqrt();  (mkl_conv_ops.cc:1632)
__attribute__((noinline)) static void rsqrt_eigen(const float* var, float eps, float* out, int n) {
  Eigen::TensorMap<Eigen::Tensor<const float, 1, Eigen::RowMajor>> v(var, n);
  Eigen::Tensor<float, 1, Eigen::RowMajor> r = (v + eps).rsqrt();
  std::memcpy(out, r.data(), n * sizeof(float));
}

struct Tensor {
  memory::dims nchw;  // logical dims
  std::shared_ptr<std::vector<float>> buf;
  size_t off = 0;     // channel offset in a concat buffer (concat-direct)
  memory::dim ctot = 0;  // channel stride
  float* ptr() { return buf->data() + off; }
  // memory handle: a submemory desc carries the channel offset itself
  float* handle() { return ctot == nchw[1] ? ptr() : buf->data(); }
};

struct Op {
  std::string kind, out, in;
  primitive prim;
  std::unordered_map<int, memory> args;
  // conv
  memory wei_user, wei;
  bool reorder_each = false;
  std::vector<std::vector<float>> bn;  // mean, rsqrt, scale, offset
  // concat (copy)
  std::vector<std::string> ins;
  std::string impl;
};

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: cvdnn DIR [opts]\n"); return 2; }
  std::string D = argv[1];
  int reps = 10;
  bool reorder_each = false, pool_inf = false, concat_direct = false;
  std::string rs = "eigen", images;
  long nimg = 0;
  bool profile = false, reuse = true;
  long batch = 0;  // override the spec's N
  double start_at = 0;  // concurrent runs: read all images, then wait until this epoch time
  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--reps") reps = atoi(argv[++i]);
    else if (a == "--reorder-each") reorder_each = true;
    else if (a == "--pool-inf") pool_inf = true;
    else if (a == "--concat-direct") concat_direct = true;
    else if (a == "--rsqrt") rs = argv[++i];
    else if (a == "--images") images = argv[++i];
    else if (a == "--n") nimg = atol(argv[++i]);
    else if (a == "--profile") profile = true;
    else if (a == "--batch") batch = atol(argv[++i]);
    else if (a == "--no-reuse") reuse = false;
    else if (a == "--start-at") start_at = atof(argv[++i]);
    else { fprintf(stderr, "unknown %s\n", a.c_str()); return 2; }
  }
  engine eng(engine::kind::cpu, 0);
  stream s(eng);
  std::map<std::string, Tensor> T;
  std::vector<std::vector<std::string>> lines;
  {
    std::ifstream sp(D + "/spec.txt");
    std::string l;
    while (std::getline(sp, l)) {
      std::istringstream is(l);
      std::vector<std::string> w;
      std::string x;
      while (is >> x) w.push_back(x);
      if (!w.empty()) lines.push_back(w);
    }
  }
  std::string inname, outname;
  // concat-direct: allocate each concat output first and point its inputs into it.
  std::map<std::string, std::pair<std::string, memory::dim>> slice;  // tensor -> (concat, channel offset)
  std::map<std::string, memory::dim> oc_of;
  for (auto& w : lines)
    if (w[0] == "conv") oc_of[w[1]] = std::stol(w[6]);
    else if (w[0] == "pool") oc_of[w[2]] = -1;  // channels = input's, resolved below
  auto dims_of = [&](const std::string& n) { return T.at(n).nchw; };
  auto alloc = [&](const std::string& n, memory::dims d) {
    Tensor t; t.nchw = d; t.ctot = d[1];
    t.buf = std::make_shared<std::vector<float>>(d[0] * d[1] * d[2] * d[3]);
    T[n] = t;
  };
  auto md_of = [&](const std::string& n) {
    Tensor& t = T.at(n);
    if (t.ctot == t.nchw[1]) return memory::desc(t.nchw, dt::f32, tag::nhwc);
    memory::desc full({t.nchw[0], t.ctot, t.nchw[2], t.nchw[3]}, dt::f32, tag::nhwc);
    return full.submemory_desc(t.nchw, {0, (memory::dim)t.off, 0, 0});
  };
  std::vector<Op> ops;
  std::vector<float> in;
  // Shapes: one pass to know every tensor's dims before allocating concat buffers.
  std::map<std::string, memory::dims> shp;
  for (auto& w : lines) {
    if (w[0] == "input") {
      inname = w[1];
      shp[w[1]] = {batch ? batch : std::stol(w[2]), std::stol(w[5]), std::stol(w[3]), std::stol(w[4])};
    } else if (w[0] == "conv" || w[0] == "pool") {
      bool cv = w[0] == "conv";
      auto o = cv ? w[1] : w[2]; auto i = cv ? w[2] : w[3];
      auto id = shp.at(i);
      int k0 = cv ? 3 : 4;
      long kh = std::stol(w[k0]), kw = std::stol(w[k0 + 1]);
      int s0 = cv ? 7 : 6;
      long sh = std::stol(w[s0]), sw = std::stol(w[s0 + 1]);
      long pt = std::stol(w[s0 + 2]), pl = std::stol(w[s0 + 3]), pb = std::stol(w[s0 + 4]), pr = std::stol(w[s0 + 5]);
      long oh = (id[2] + pt + pb - kh) / sh + 1, ow = (id[3] + pl + pr - kw) / sw + 1;
      shp[o] = {id[0], cv ? std::stol(w[6]) : id[1], oh, ow};
    } else if (w[0] == "concat") {
      auto d = shp.at(w[2]); memory::dim c = 0;
      for (size_t k = 2; k < w.size(); k++) c += shp.at(w[k])[1];
      d[1] = c; shp[w[1]] = d;
    } else if (w[0] == "gap") {
      auto d = shp.at(w[2]); shp[w[1]] = {d[0], d[1], 1, 1};
    } else if (w[0] == "fc") {
      shp[w[1]] = {shp.at(w[2])[0], std::stol(w[4]), 1, 1};
    } else if (w[0] == "softmax") {
      shp[w[1]] = shp.at(w[2]);
    } else if (w[0] == "output") outname = w[1];
  }
  for (auto& w : lines)
    if (w[0] == "concat" && concat_direct) {
      memory::dim c = 0;
      for (size_t k = 2; k < w.size(); k++) { slice[w[k]] = {w[1], c}; c += shp.at(w[k])[1]; }
    }
  auto elems = [](const memory::dims& d) { return (size_t)(d[0] * d[1] * d[2] * d[3]); };
  if (reuse && slice.empty()) {
    // Liveness plan in execution order: a tensor takes the smallest free buffer that
    // fits (else a new one) and returns it after its last consumer has run; an op's
    // inputs are released only after its output is placed (no in-place aliasing).
    std::map<std::string, size_t> last;
    std::vector<std::pair<std::string, std::vector<std::string>>> seq;  // (out, ins)
    for (auto& w : lines) {
      if (w[0] == "conv" || w[0] == "gap" || w[0] == "fc" || w[0] == "softmax") seq.push_back({w[1], {w[2]}});
      else if (w[0] == "pool") seq.push_back({w[2], {w[3]}});
      else if (w[0] == "concat") seq.push_back({w[1], std::vector<std::string>(w.begin() + 2, w.end())});
    }
    for (size_t k = 0; k < seq.size(); k++) for (auto& i : seq[k].second) last[i] = k;
    std::vector<size_t> cap; std::vector<int> freel; std::map<std::string, int> bid;
    auto place = [&](const std::string& n) {
      size_t need = elems(shp.at(n)); int best = -1;
      for (int b : freel) if (cap[b] >= need && (best < 0 || cap[b] < cap[best])) best = b;
      if (best < 0) for (int b : freel) if (best < 0 || cap[b] > cap[best]) best = b;  // grow the largest
      if (best < 0) { cap.push_back(0); best = (int)cap.size() - 1; }
      else freel.erase(std::find(freel.begin(), freel.end(), best));
      cap[best] = std::max(cap[best], need); bid[n] = best;
    };
    place(inname);
    for (size_t k = 0; k < seq.size(); k++) {
      place(seq[k].first);
      for (auto& i : seq[k].second)
        if (last[i] == k && i != outname && i != inname && std::find(freel.begin(), freel.end(), bid[i]) == freel.end()) freel.push_back(bid[i]);
    }
    std::vector<std::shared_ptr<std::vector<float>>> bufs;
    size_t tot = 0;
    for (size_t c : cap) { bufs.push_back(std::make_shared<std::vector<float>>(c)); tot += c; }
    for (auto& [n, d] : shp) { Tensor t; t.nchw = d; t.ctot = d[1]; t.buf = bufs[bid.at(n)]; T[n] = t; }
    fprintf(stderr, "plan: %zu buffers, %.2f GB (no reuse: %.2f GB)\n", cap.size(), tot * 4e-9,
            [&] { size_t a = 0; for (auto& [n, d] : shp) a += elems(d); return a * 4e-9; }());
  } else
    for (auto& [n, d] : shp) {
      if (slice.count(n)) continue;
      alloc(n, d);
    }
  for (auto& [n, sl] : slice) {
    Tensor t; t.nchw = shp.at(n); t.buf = T.at(sl.first).buf; t.off = sl.second; t.ctot = shp.at(sl.first)[1];
    T[n] = t;
  }
  if (images.empty()) {
    auto d = shp.at(inname);
    auto v = load(D + "/in.f32", elems(d));
    std::memcpy(T.at(inname).ptr(), v.data(), v.size() * sizeof(float));
  }
  for (auto& w : lines) {
    Op op; op.kind = w[0];
    if (w[0] == "conv") {
      op.out = w[1]; op.in = w[2];
      long kh = std::stol(w[3]), kw = std::stol(w[4]), ic = std::stol(w[5]), oc = std::stol(w[6]);
      memory::dims strides{std::stol(w[7]), std::stol(w[8])}, padl{std::stol(w[9]), std::stol(w[10])},
          padr{std::stol(w[11]), std::stol(w[12])}, dil{0, 0};
      float eps = std::stof(w[13]);
      auto filt = load(D + "/" + w[14], kh * kw * ic * oc);
      auto mean = load(D + "/" + w[15], oc), var = load(D + "/" + w[16], oc), sc = load(D + "/" + w[17], oc),
           off = load(D + "/" + w[18], oc);
      std::vector<float> r(oc);
      if (rs == "eigen") rsqrt_eigen(var.data(), eps, r.data(), oc);
      else for (long c = 0; c < oc; c++) r[c] = 1.0f / std::sqrt(var[c] + eps);
      op.bn = {mean, r, sc, off};
      memory::desc bnmd({1, oc, 1, 1}, dt::f32, tag::nchw);
      post_ops po;
      po.append_binary(algorithm::binary_sub, bnmd);
      po.append_binary(algorithm::binary_mul, bnmd);
      po.append_binary(algorithm::binary_mul, bnmd);
      po.append_binary(algorithm::binary_add, bnmd);
      po.append_eltwise(1.0f, algorithm::eltwise_relu, 0.0f, 0.0f);
      primitive_attr attr; attr.set_post_ops(po);
      memory::desc src = md_of(op.in), dst = md_of(op.out);
      memory::desc wany({oc, ic, kh, kw}, dt::f32, tag::any);
      convolution_forward::desc cd(prop_kind::forward_training, algorithm::convolution_direct, src, wany, dst,
                                   strides, dil, padl, padr);
      convolution_forward::primitive_desc pd(cd, attr, eng);
      op.impl = pd.impl_info_str();
      op.wei_user = memory({{oc, ic, kh, kw}, dt::f32, tag::hwio}, eng);
      std::memcpy(op.wei_user.get_data_handle(), filt.data(), filt.size() * sizeof(float));
      op.wei = memory(pd.weights_desc(), eng);
      reorder(op.wei_user, op.wei).execute(s, op.wei_user, op.wei);
      op.reorder_each = reorder_each;
      op.prim = convolution_forward(pd);
      op.args[DNNL_ARG_SRC] = memory(src, eng, T.at(op.in).handle());
      op.args[DNNL_ARG_DST] = memory(dst, eng, T.at(op.out).handle());
      op.args[DNNL_ARG_WEIGHTS] = op.wei;
      for (int k = 0; k < 4; k++)
        op.args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(k) | DNNL_ARG_SRC_1] = memory(bnmd, eng, op.bn[k].data());
      char fmt[64];
      snprintf(fmt, sizeof fmt, " wei=%s", pd.weights_desc().data.format_kind == dnnl_blocked ? "blocked" : "?");
      op.impl += fmt;
    } else if (w[0] == "pool") {
      op.out = w[2]; op.in = w[3];
      bool mx = w[1] == "max";
      memory::dims k{std::stol(w[4]), std::stol(w[5])}, st{std::stol(w[6]), std::stol(w[7])},
          padl{std::stol(w[8]), std::stol(w[9])}, padr{std::stol(w[10]), std::stol(w[11])}, dil{0, 0};
      auto pk = (mx && pool_inf) ? prop_kind::forward_inference : prop_kind::forward_training;
      auto alg = mx ? algorithm::pooling_max : algorithm::pooling_avg_exclude_padding;
      memory::desc src = md_of(op.in), dst = md_of(op.out);
      pooling_v2_forward::desc d(pk, alg, src, dst, st, k, dil, padl, padr);
      pooling_v2_forward::primitive_desc pd(d, eng);
      op.impl = pd.impl_info_str();
      op.prim = pooling_v2_forward(pd);
      op.args[DNNL_ARG_SRC] = memory(src, eng, T.at(op.in).handle());
      op.args[DNNL_ARG_DST] = memory(dst, eng, T.at(op.out).handle());
      if (pk == prop_kind::forward_training && pd.workspace_desc().get_size())
        op.args[DNNL_ARG_WORKSPACE] = memory(pd.workspace_desc(), eng);
    } else if (w[0] == "gap") {
      op.out = w[1]; op.in = w[2]; op.impl = "eigen-order sum/div";
    } else if (w[0] == "fc") {
      // TF _MklNativeFusedMatMul(BiasAdd): inner_product, weights in the user's
      // [K, M] layout (oneDNN {M, K} as "ba") since the weight is not constant.
      op.out = w[1]; op.in = w[2];
      long K = std::stol(w[3]), M = std::stol(w[4]), N = shp.at(w[2])[0];
      auto wt = load(D + "/" + w[5], K * M), b = load(D + "/" + w[6], M);
      memory::desc src({N, K}, dt::f32, tag::ab), wmd({M, K}, dt::f32, tag::ba), bmd({M}, dt::f32, tag::a),
          dst({N, M}, dt::f32, tag::ab);
      inner_product_forward::desc d(prop_kind::forward_training, src, wmd, bmd, dst);
      inner_product_forward::primitive_desc pd(d, eng);
      op.impl = pd.impl_info_str();
      op.prim = inner_product_forward(pd);
      op.wei = memory(wmd, eng); std::memcpy(op.wei.get_data_handle(), wt.data(), wt.size() * 4);
      op.bn = {b};
      op.args[DNNL_ARG_SRC] = memory(src, eng, T.at(op.in).ptr());
      op.args[DNNL_ARG_WEIGHTS] = op.wei;
      op.args[DNNL_ARG_BIAS] = memory(bmd, eng, op.bn[0].data());
      op.args[DNNL_ARG_DST] = memory(dst, eng, T.at(op.out).ptr());
    } else if (w[0] == "softmax") {
      op.out = w[1]; op.in = w[2];
      auto d = shp.at(w[2]);
      memory::desc md({d[0], d[1]}, dt::f32, tag::ab);
      softmax_v2_forward::desc sd(prop_kind::forward_inference, algorithm::softmax_accurate, md, md, 1);
      softmax_v2_forward::primitive_desc pd(sd, eng);
      op.impl = pd.impl_info_str();
      op.prim = softmax_v2_forward(pd);
      op.args[DNNL_ARG_SRC] = memory(md, eng, T.at(op.in).ptr());
      op.args[DNNL_ARG_DST] = memory(md, eng, T.at(op.out).ptr());
    } else if (w[0] == "concat") {
      op.out = w[1];
      op.ins.assign(w.begin() + 2, w.end());
      op.impl = concat_direct ? "direct (in place)" : "copy";
    } else continue;
    ops.push_back(std::move(op));
  }
  for (auto& op : ops) printf("%-6s %-24s %s\n", op.kind.c_str(), op.impl.c_str(), op.out.c_str());
  std::map<std::string, double> prof;
  auto run = [&]() {
    for (auto& op : ops) {
      auto p0 = std::chrono::steady_clock::now();
      if (op.kind == "gap") {
        // TF Mean: in.reduce({H*W}, SumReducer) / float(H*W), sequential over H*W per output.
        Tensor& t = T.at(op.in); Tensor& o = T.at(op.out);
        long N = t.nchw[0], C = t.nchw[1], P = t.nchw[2] * t.nchw[3];
        const float* x = t.ptr(); float* y = o.ptr(); float cnt = (float)P;
#pragma omp parallel for
        for (long n = 0; n < N; n++)
          for (long c = 0; c < C; c++) {
            float acc = 0.0f;
            for (long p = 0; p < P; p++) acc += x[(n * P + p) * C + c];
            y[n * C + c] = acc / cnt;
          }
        if (profile) prof[op.kind] += std::chrono::duration<double>(std::chrono::steady_clock::now() - p0).count();
        continue;
      }
      if (op.kind == "concat") {
        if (concat_direct) continue;
        Tensor& o = T.at(op.out);
        long N = o.nchw[0], P = o.nchw[2] * o.nchw[3], C = o.nchw[1];
        std::vector<std::pair<float*, long>> src;
        for (auto& n : op.ins) src.push_back({T.at(n).ptr(), T.at(n).nchw[1]});
        float* dst = o.ptr();
#pragma omp parallel for
        for (long p = 0; p < N * P; p++) {
          float* d = dst + p * C;
          for (auto& [sp, c] : src) { std::memcpy(d, sp + p * c, c * sizeof(float)); d += c; }
        }
        if (profile) prof[op.kind] += std::chrono::duration<double>(std::chrono::steady_clock::now() - p0).count();
        continue;
      }
      if (op.reorder_each) reorder(op.wei_user, op.wei).execute(s, op.wei_user, op.wei);
      op.prim.execute(s, op.args);
      if (profile) { s.wait(); prof[op.kind + " " + op.impl.substr(0, op.impl.find(' '))] += std::chrono::duration<double>(std::chrono::steady_clock::now() - p0).count(); }
    }
    s.wait();
  };
  auto print_prof = [&](double scale) {
    double tot = 0; for (auto& [k, v] : prof) tot += v;
    for (auto& [k, v] : prof) printf("  prof %-34s %8.4fs %5.1f%%\n", k.c_str(), v * scale, 100 * v / tot);
  };
  if (!images.empty()) {
    // Whole network over dumped images.
    Tensor& x = T.at(inname); Tensor& y = T.at(outname);
    long B = x.nchw[0], C = x.nchw[1], H = x.nchw[2], W = x.nchw[3], E = H * W * C, M = y.nchw[1];
    std::ifstream f(images + "/images.u8", std::ios::binary);
    std::vector<unsigned char> u(start_at ? nimg * E : B * E);
    if (start_at) f.read(reinterpret_cast<char*>(u.data()), nimg * E);
    std::vector<float> probs(nimg * M);
    run();  // warm-up (primitive JIT, first-touch), not timed
    prof.clear();
    double wall = 0;
    auto epoch = [] { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); };
    if (start_at) while (epoch() < start_at) usleep(1000);
    double t_start = epoch();
    auto cpu_s = [] { rusage ru; getrusage(RUSAGE_SELF, &ru);
      return ru.ru_utime.tv_sec + ru.ru_stime.tv_sec + 1e-6 * (ru.ru_utime.tv_usec + ru.ru_stime.tv_usec); };
    double cpu0 = cpu_s();
    for (long s0 = 0; s0 < nimg; s0 += B) {
      long nb = std::min(B, nimg - s0);
      if (!start_at) f.read(reinterpret_cast<char*>(u.data()), nb * E);
      const unsigned char* ub = u.data() + (start_at ? s0 * E : 0);
      auto t0 = std::chrono::steady_clock::now();
      float* xp = x.ptr();
#pragma omp parallel for
      for (long i = 0; i < nb * E; i++) xp[i] = ((float)ub[i] - 128.0f) / 128.0f;
      if (nb < B) std::memset(xp + nb * E, 0, (B - nb) * E * sizeof(float));
      run();
      std::memcpy(probs.data() + s0 * M, y.ptr(), nb * M * sizeof(float));
      wall += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    double cpu = cpu_s() - cpu0;  // includes the (untimed) image reads
    double t_end = epoch();
    // probs_tf213.npy: float32 [n, M]; skip the .npy header.
    std::ifstream r(images + "/probs_tf213.npy", std::ios::binary);
    char hdr[10]; r.read(hdr, 10);
    unsigned hl = (unsigned char)hdr[8] | ((unsigned char)hdr[9] << 8);
    r.seekg(10 + hl);
    std::vector<float> ref(nimg * M);
    r.read(reinterpret_cast<char*>(ref.data()), ref.size() * 4);
    long rows = 0; double mx = 0;
    for (long i = 0; i < nimg; i++) {
      bool d = false;
      for (long j = 0; j < M; j++)
        if (std::memcmp(&ref[i * M + j], &probs[i * M + j], 4)) { d = true; mx = std::max(mx, (double)std::fabs(ref[i * M + j] - probs[i * M + j])); }
      rows += d;
    }
    printf("network n=%ld batch=%ld wall=%.3fs cpu=%.1fs  %s", nimg, B, wall, cpu, rows ? "DIFFER" : "IDENTICAL");
    if (rows) printf(" rows=%ld/%ld maxabs=%.3g", rows, nimg, mx);
    printf(" t_start=%.3f t_end=%.3f\n", t_start, t_end);
    if (profile) print_prof(1.0);
    std::ofstream(D + "/probs_cvdnn.f32", std::ios::binary).write(reinterpret_cast<char*>(probs.data()), probs.size() * 4);
    return rows ? 1 : 0;
  }
  run();
  prof.clear();
  std::vector<double> ts;
  for (int r = 0; r < reps; r++) {
    auto t0 = std::chrono::steady_clock::now();
    run();
    ts.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }
  std::sort(ts.begin(), ts.end());
  if (profile) print_prof(1.0 / reps);
  Tensor& o = T.at(outname);
  size_t n = o.nchw[0] * o.nchw[1] * o.nchw[2] * o.nchw[3];
  auto ref = load(D + "/ref.f32", n);
  size_t diff = 0; double mx = 0;
  for (size_t i = 0; i < n; i++)
    if (std::memcmp(&ref[i], o.ptr() + i, 4)) { diff++; mx = std::max(mx, (double)std::fabs(ref[i] - o.ptr()[i])); }
  printf("span wall median=%.4fs min=%.4fs reps=%d  %s", ts[ts.size() / 2], ts[0], reps,
         diff ? "DIFFER" : "IDENTICAL");
  if (diff) printf(" elems=%zu/%zu maxabs=%.3g", diff, n, mx);
  printf("\n");
  return diff ? 1 : 0;
}
