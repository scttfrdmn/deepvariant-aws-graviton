// SPDX-License-Identifier: Apache-2.0
//
// libcvdnn: call_variants inference without TensorFlow's runtime
// (deepvariant-aws-graviton#11). Runs the DeepVariant WGS InceptionV3 with the
// oneDNN v2.7.3 primitives TF 2.13.1 creates for it (native NHWC brgconv /
// brgconv_1x1 with the FusedBatchNorm + Relu post-op chain, pooling_v2,
// gemm:jit inner product, softmax_v2), so the probabilities are bit-for-bit
// those of stock call_variants. Only data movement differs: filters are
// reordered once, buffers come from a liveness plan, and each Inception branch
// conv writes straight into its channel slice of the concat output (needs
// oneDNN with tools/cv_onednn/onednn-v2.7.3-brgconv-channel-slice-dst.patch).
//
// The network is read from a spec directory written by engine.py
// (spec.txt + raw float32 weights). C API:
//   void* cvdnn_open(const char* dir, long batch, char* err, long errlen)
//   int   cvdnn_run(void* h, const float* x, long n, float* probs)  // n <= batch
//   long  cvdnn_batch(void* h); const char* cvdnn_info(void* h); void cvdnn_close(void* h)
// Threads: OpenMP (OMP_NUM_THREADS, OMP_PROC_BIND / OMP_PLACES).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <unsupported/Eigen/CXX11/Tensor>

#include "dnnl.hpp"

namespace {

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

std::vector<float> load(const std::string& f, size_t n) {
  std::vector<float> v(n);
  std::ifstream in(f, std::ios::binary);
  if (!in.read(reinterpret_cast<char*>(v.data()), n * sizeof(float)))
    throw std::runtime_error("short read " + f);
  return v;
}

// TF: Eigen::Tensor<float,1,RowMajor> r = (var + eps).rsqrt();  (mkl_conv_ops.cc:1632).
// Built -mavx without FMA like TF's x86 wheel, so the packet rsqrt is the same.
__attribute__((noinline)) void rsqrt_eigen(const float* var, float eps, float* out, int n) {
  Eigen::TensorMap<Eigen::Tensor<const float, 1, Eigen::RowMajor>> v(var, n);
  Eigen::Tensor<float, 1, Eigen::RowMajor> r = (v + eps).rsqrt();
  std::memcpy(out, r.data(), n * sizeof(float));
}

struct Tensor {
  memory::dims nchw;
  std::shared_ptr<std::vector<float>> buf;
  size_t off = 0;        // channel offset inside a concat output
  memory::dim ctot = 0;  // channel stride (== nchw[1] unless a concat slice)
  float* ptr() { return buf->data() + off; }
};

struct Op {
  std::string kind, out, in;
  primitive prim;
  std::unordered_map<int, memory> args;
  memory wei;
  std::vector<std::vector<float>> bn;
  std::vector<std::tuple<std::string, long>> copies;  // concat inputs not written in place: (tensor, channel offset)
};

size_t elems(const memory::dims& d) { return (size_t)(d[0] * d[1] * d[2] * d[3]); }

struct Net {
  engine eng{engine::kind::cpu, 0};
  stream s{eng};
  std::map<std::string, Tensor> T;
  std::vector<Op> ops;
  std::string inname, outname, info;
  long B = 0;

  Net(const std::string& D, long batch) : B(batch) {
    std::vector<std::vector<std::string>> lines;
    {
      std::ifstream sp(D + "/spec.txt");
      if (!sp) throw std::runtime_error("no " + D + "/spec.txt");
      std::string l;
      while (std::getline(sp, l)) {
        std::istringstream is(l);
        std::vector<std::string> w;
        std::string x;
        while (is >> x) w.push_back(x);
        if (!w.empty()) lines.push_back(w);
      }
    }
    // Shapes.
    std::map<std::string, memory::dims> shp;
    for (auto& w : lines) {
      if (w[0] == "input") {
        inname = w[1];
        shp[w[1]] = {B, std::stol(w[5]), std::stol(w[3]), std::stol(w[4])};
      } else if (w[0] == "conv" || w[0] == "pool") {
        bool cv = w[0] == "conv";
        auto o = cv ? w[1] : w[2];
        auto id = shp.at(cv ? w[2] : w[3]);
        int k0 = cv ? 3 : 4, s0 = cv ? 7 : 6;
        long kh = std::stol(w[k0]), kw = std::stol(w[k0 + 1]), sh = std::stol(w[s0]), sw = std::stol(w[s0 + 1]);
        long pt = std::stol(w[s0 + 2]), pl = std::stol(w[s0 + 3]), pb = std::stol(w[s0 + 4]), pr = std::stol(w[s0 + 5]);
        shp[o] = {id[0], cv ? std::stol(w[6]) : id[1], (id[2] + pt + pb - kh) / sh + 1, (id[3] + pl + pr - kw) / sw + 1};
      } else if (w[0] == "concat") {
        auto d = shp.at(w[2]);
        memory::dim c = 0;
        for (size_t k = 2; k < w.size(); k++) c += shp.at(w[k])[1];
        d[1] = c;
        shp[w[1]] = d;
      } else if (w[0] == "gap") {
        auto d = shp.at(w[2]);
        shp[w[1]] = {d[0], d[1], 1, 1};
      } else if (w[0] == "fc") {
        shp[w[1]] = {shp.at(w[2])[0], std::stol(w[4]), 1, 1};
      } else if (w[0] == "softmax") {
        shp[w[1]] = shp.at(w[2]);
      } else if (w[0] == "output") {
        outname = w[1];
      }
    }
    // In-place concat: a conv output whose only consumer is one concat.
    std::map<std::string, int> uses;
    std::set<std::string> conv_out;
    for (auto& w : lines) {
      if (w[0] == "conv") conv_out.insert(w[1]);
      if (w[0] == "conv" || w[0] == "gap" || w[0] == "fc" || w[0] == "softmax") uses[w[2]]++;
      else if (w[0] == "pool") uses[w[3]]++;
      else if (w[0] == "concat") for (size_t k = 2; k < w.size(); k++) uses[w[k]]++;
    }
    std::map<std::string, std::pair<std::string, memory::dim>> slot, slice;
    for (auto& w : lines)
      if (w[0] == "concat") {
        memory::dim c = 0;
        for (size_t k = 2; k < w.size(); k++) {
          slot[w[k]] = {w[1], c};
          if (conv_out.count(w[k]) && uses[w[k]] == 1) slice[w[k]] = {w[1], c};
          c += shp.at(w[k])[1];
        }
      }
    // Liveness plan in execution order; the input and output are never reused.
    std::map<std::string, size_t> last;
    std::vector<std::pair<std::string, std::vector<std::string>>> seq;
    for (auto& w : lines) {
      if (w[0] == "conv" || w[0] == "gap" || w[0] == "fc" || w[0] == "softmax") seq.push_back({w[1], {w[2]}});
      else if (w[0] == "pool") seq.push_back({w[2], {w[3]}});
      else if (w[0] == "concat") seq.push_back({w[1], std::vector<std::string>(w.begin() + 2, w.end())});
    }
    for (size_t k = 0; k < seq.size(); k++) for (auto& i : seq[k].second) last[i] = k;
    std::vector<size_t> cap;
    std::vector<int> freel;
    std::map<std::string, int> bid;
    std::function<void(const std::string&)> place = [&](const std::string& n) {
      if (bid.count(n)) return;
      if (slice.count(n)) { place(slice.at(n).first); bid[n] = bid.at(slice.at(n).first); return; }
      size_t need = elems(shp.at(n));
      int best = -1;
      for (int b : freel) if (cap[b] >= need && (best < 0 || cap[b] < cap[best])) best = b;
      if (best < 0) for (int b : freel) if (best < 0 || cap[b] > cap[best]) best = b;
      if (best < 0) { cap.push_back(0); best = (int)cap.size() - 1; }
      else freel.erase(std::find(freel.begin(), freel.end(), best));
      cap[best] = std::max(cap[best], need);
      bid[n] = best;
    };
    place(inname);
    for (size_t k = 0; k < seq.size(); k++) {
      place(seq[k].first);
      for (auto& i : seq[k].second)
        if (last[i] == k && i != outname && i != inname && !slice.count(i) &&
            std::find(freel.begin(), freel.end(), bid[i]) == freel.end())
          freel.push_back(bid[i]);
    }
    std::vector<std::shared_ptr<std::vector<float>>> bufs;
    size_t tot = 0;
    for (size_t c : cap) { bufs.push_back(std::make_shared<std::vector<float>>(c)); tot += c; }
    for (auto& [n, d] : shp) {
      Tensor t;
      t.nchw = d; t.ctot = d[1]; t.buf = bufs[bid.at(n)];
      if (slice.count(n)) { t.off = slice.at(n).second; t.ctot = shp.at(slice.at(n).first)[1]; }
      T[n] = t;
    }
    auto md_of = [&](const std::string& n) {
      Tensor& t = T.at(n);
      if (t.ctot == t.nchw[1]) return memory::desc(t.nchw, dt::f32, tag::nhwc);
      const auto H = t.nchw[2], W = t.nchw[3], L = t.ctot;
      return memory::desc(t.nchw, dt::f32, memory::dims{H * W * L, 1, W * L, L});
    };
    int nconv = 0, nslice = 0;
    std::map<std::string, int> impls;
    for (auto& w : lines) {
      Op op;
      op.kind = w[0];
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
        rsqrt_eigen(var.data(), eps, r.data(), oc);
        op.bn = {mean, r, sc, off};
        memory::desc bnmd({1, oc, 1, 1}, dt::f32, tag::nchw);
        post_ops po;
        po.append_binary(algorithm::binary_sub, bnmd);
        po.append_binary(algorithm::binary_mul, bnmd);
        po.append_binary(algorithm::binary_mul, bnmd);
        po.append_binary(algorithm::binary_add, bnmd);
        po.append_eltwise(1.0f, algorithm::eltwise_relu, 0.0f, 0.0f);
        primitive_attr attr;
        attr.set_post_ops(po);
        memory::desc src = md_of(op.in), dst = md_of(op.out);
        memory::desc wany({oc, ic, kh, kw}, dt::f32, tag::any);
        convolution_forward::desc cd(prop_kind::forward_training, algorithm::convolution_direct, src, wany, dst,
                                     strides, dil, padl, padr);
        convolution_forward::primitive_desc pd(cd, attr, eng);
        std::string impl = pd.impl_info_str();
        // Exactness needs TF's kernels: refuse anything but brgconv (e.g. the
        // ref:any fallback a stock oneDNN takes for a channel-slice dst).
        if (impl.rfind("brgconv", 0) != 0)
          throw std::runtime_error("conv " + op.out + " got impl " + impl + " (oneDNN without the channel-slice patch?)");
        impls[impl]++;
        memory wu({{oc, ic, kh, kw}, dt::f32, tag::hwio}, eng, filt.data());
        op.wei = memory(pd.weights_desc(), eng);
        reorder(wu, op.wei).execute(s, wu, op.wei);
        s.wait();
        op.prim = convolution_forward(pd);
        op.args[DNNL_ARG_SRC] = memory(src, eng, T.at(op.in).ptr());
        op.args[DNNL_ARG_DST] = memory(dst, eng, T.at(op.out).ptr());
        op.args[DNNL_ARG_WEIGHTS] = op.wei;
        for (int k = 0; k < 4; k++)
          op.args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(k) | DNNL_ARG_SRC_1] = memory(bnmd, eng, op.bn[k].data());
        nconv++;
        nslice += slice.count(op.out);
      } else if (w[0] == "pool") {
        op.out = w[2]; op.in = w[3];
        bool mx = w[1] == "max";
        memory::dims k{std::stol(w[4]), std::stol(w[5])}, st{std::stol(w[6]), std::stol(w[7])},
            padl{std::stol(w[8]), std::stol(w[9])}, padr{std::stol(w[10]), std::stol(w[11])}, dil{0, 0};
        memory::desc src = md_of(op.in), dst = md_of(op.out);
        pooling_v2_forward::desc d(prop_kind::forward_training,
                                   mx ? algorithm::pooling_max : algorithm::pooling_avg_exclude_padding, src, dst, st,
                                   k, dil, padl, padr);
        pooling_v2_forward::primitive_desc pd(d, eng);
        impls[pd.impl_info_str()]++;
        op.prim = pooling_v2_forward(pd);
        op.args[DNNL_ARG_SRC] = memory(src, eng, T.at(op.in).ptr());
        op.args[DNNL_ARG_DST] = memory(dst, eng, T.at(op.out).ptr());
        if (pd.workspace_desc().get_size()) op.args[DNNL_ARG_WORKSPACE] = memory(pd.workspace_desc(), eng);
      } else if (w[0] == "gap") {
        op.out = w[1]; op.in = w[2];
      } else if (w[0] == "fc") {
        // TF _MklNativeFusedMatMul(BiasAdd) with a non-constant weight: user [K, M] layout.
        op.out = w[1]; op.in = w[2];
        long K = std::stol(w[3]), M = std::stol(w[4]), N = shp.at(w[2])[0];
        auto wt = load(D + "/" + w[5], K * M), b = load(D + "/" + w[6], M);
        memory::desc src({N, K}, dt::f32, tag::ab), wmd({M, K}, dt::f32, tag::ba), bmd({M}, dt::f32, tag::a),
            dst({N, M}, dt::f32, tag::ab);
        inner_product_forward::desc d(prop_kind::forward_training, src, wmd, bmd, dst);
        inner_product_forward::primitive_desc pd(d, eng);
        impls[pd.impl_info_str()]++;
        op.prim = inner_product_forward(pd);
        op.wei = memory(wmd, eng);
        std::memcpy(op.wei.get_data_handle(), wt.data(), wt.size() * 4);
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
        impls[pd.impl_info_str()]++;
        op.prim = softmax_v2_forward(pd);
        op.args[DNNL_ARG_SRC] = memory(md, eng, T.at(op.in).ptr());
        op.args[DNNL_ARG_DST] = memory(md, eng, T.at(op.out).ptr());
      } else if (w[0] == "concat") {
        op.out = w[1];
        for (size_t k = 2; k < w.size(); k++)
          if (!slice.count(w[k])) op.copies.push_back({w[k], slot.at(w[k]).second});
      } else {
        continue;
      }
      ops.push_back(std::move(op));
    }
    std::ostringstream is;
    is << "batch=" << B << " ops=" << ops.size() << " convs=" << nconv << " in_place=" << nslice
       << " buffers=" << cap.size() << " buffer_gb=" << tot * 4e-9 << " impls:";
    for (auto& [k, v] : impls) is << " " << k << "x" << v;
    info = is.str();
  }

  void run() {
    for (auto& op : ops) {
      if (op.kind == "gap") {
        // TF Mean: in.reduce({H*W}, SumReducer) / float(H*W), sequential over H*W.
        Tensor& t = T.at(op.in); Tensor& o = T.at(op.out);
        long N = t.nchw[0], C = t.nchw[1], P = t.nchw[2] * t.nchw[3];
        const float* x = t.ptr();
        float* y = o.ptr();
        float cnt = (float)P;
#pragma omp parallel for
        for (long n = 0; n < N; n++)
          for (long c = 0; c < C; c++) {
            float acc = 0.0f;
            for (long p = 0; p < P; p++) acc += x[(n * P + p) * C + c];
            y[n * C + c] = acc / cnt;
          }
      } else if (op.kind == "concat") {
        if (op.copies.empty()) continue;
        Tensor& o = T.at(op.out);
        long N = o.nchw[0], P = o.nchw[2] * o.nchw[3], C = o.nchw[1];
        std::vector<std::tuple<float*, long, long>> src;
        for (auto& [n, co] : op.copies) src.push_back({T.at(n).ptr(), T.at(n).nchw[1], co});
        float* dst = o.ptr();
#pragma omp parallel for
        for (long p = 0; p < N * P; p++)
          for (auto& [sp, c, co] : src) std::memcpy(dst + p * C + co, sp + p * c, c * sizeof(float));
      } else {
        op.prim.execute(s, op.args);
      }
    }
    s.wait();
  }

  // x: n preprocessed images ((u8 - 128) / 128, NHWC float32), n <= B; probs: n x classes.
  void infer(const float* x, long n, float* probs) {
    Tensor& xi = T.at(inname); Tensor& yo = T.at(outname);
    size_t E = elems(xi.nchw) / B, M = yo.nchw[1];
    std::memcpy(xi.ptr(), x, n * E * sizeof(float));
    // Examples are independent; a partial batch is padded (gated identical on a 467-example tail).
    if (n < B) std::memset(xi.ptr() + n * E, 0, (B - n) * E * sizeof(float));
    run();
    std::memcpy(probs, yo.ptr(), n * M * sizeof(float));
  }
};

}  // namespace

extern "C" {

void* cvdnn_open(const char* dir, long batch, char* err, long errlen) {
  try {
    return new Net(dir, batch);
  } catch (const std::exception& e) {
    if (err && errlen > 0) { std::strncpy(err, e.what(), errlen - 1); err[errlen - 1] = 0; }
    return nullptr;
  }
}

int cvdnn_run(void* h, const float* x, long n, float* probs) {
  auto* net = static_cast<Net*>(h);
  if (n < 0 || n > net->B) return 1;
  try {
    net->infer(x, n, probs);
  } catch (const std::exception& e) {
    fprintf(stderr, "cvdnn_run: %s\n", e.what());
    return 2;
  }
  return 0;
}

long cvdnn_batch(void* h) { return static_cast<Net*>(h)->B; }
const char* cvdnn_info(void* h) { return static_cast<Net*>(h)->info.c_str(); }
void cvdnn_close(void* h) { delete static_cast<Net*>(h); }

}  // extern "C"
