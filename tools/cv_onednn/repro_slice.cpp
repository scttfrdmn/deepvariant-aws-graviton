// SPDX-License-Identifier: Apache-2.0
// Per-channel binary post-op with a strided (channel-slice / leading-dimension) dst,
// oneDNN v3.x C++ API. Each case runs the primitive into a dense dst and into a
// strided dst, and checks both against a float64 reference (relative error 1e-4):
// "bitwise" = strided equals dense bit for bit; "ref-ok" = within tolerance of the
// reference (e.g. a different impl with another summation order); "WRONG" = not.
//   repro_v3
#include <cstdio>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include "oneapi/dnnl/dnnl.hpp"
using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

static std::vector<float> rnd(size_t n, unsigned seed) {
  std::mt19937 g(seed); std::uniform_real_distribution<float> u(-1, 1);
  std::vector<float> v(n); for (auto& x : v) x = u(g); return v;
}

// conv NHWC, OC output channels written into channels [OFF, OFF+OC) of an L-channel tensor.
static int conv_case(engine& eng, stream& s, long OC, long L, long OFF, long K = 3, long IC = 32, long H = 6, long W = 7) {
  const long N = 2;
  auto x = rnd(N * H * W * IC, 1), w = rnd(K * K * IC * OC, 2), b = rnd(OC, 3);
  memory::desc src({N, IC, H, W}, dt::f32, tag::nhwc), wany({OC, IC, K, K}, dt::f32, tag::any);
  memory::desc bmd({1, OC, 1, 1}, dt::f32, tag::nchw);
  memory::desc dd({N, OC, H, W}, dt::f32, tag::nhwc);
  memory::desc ds({N, OC, H, W}, dt::f32, memory::dims{H * W * L, 1, W * L, L});
  post_ops po; po.append_binary(algorithm::binary_add, bmd);  // per output channel
  primitive_attr attr; attr.set_post_ops(po);
  std::vector<float> od(N * H * W * OC), os(N * H * W * L, -7.0f);
  std::string impl[2];
  memory::desc dsts[2] = {dd, ds};
  float* outs[2] = {od.data(), os.data() + OFF};
  for (int v = 0; v < 2; v++) {
    convolution_forward::primitive_desc pd(eng, prop_kind::forward_inference, algorithm::convolution_direct, src,
                                           wany, dsts[v], {1, 1}, {K / 2, K / 2}, {K / 2, K / 2}, attr);
    impl[v] = pd.impl_info_str();
    memory wu({{OC, IC, K, K}, dt::f32, tag::hwio}, eng, w.data()), wm(pd.weights_desc(), eng);
    reorder(wu, wm).execute(s, wu, wm);
    convolution_forward(pd).execute(s, {{DNNL_ARG_SRC, memory(src, eng, x.data())}, {DNNL_ARG_WEIGHTS, wm},
                                        {DNNL_ARG_DST, memory(dsts[v], eng, outs[v])},
                                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, memory(bmd, eng, b.data())}});
    s.wait();
  }
  long bits = 0, wrong = 0;
  for (long n = 0; n < N; n++)
    for (long h = 0; h < H; h++)
      for (long x0 = 0; x0 < W; x0++)
        for (long c = 0; c < OC; c++) {
          double r = b[c];
          for (long kh = 0; kh < K; kh++)
            for (long kw = 0; kw < K; kw++) {
              long ih = h + kh - K / 2, iw = x0 + kw - K / 2;
              if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
              for (long ic = 0; ic < IC; ic++)
                r += (double)x[((n * H + ih) * W + iw) * IC + ic] * w[((kh * K + kw) * IC + ic) * OC + c];
            }
          long p = (n * H + h) * W + x0;
          float d = od[p * OC + c], v = os[p * L + OFF + c];
          bits += std::memcmp(&d, &v, 4) != 0;
          wrong += std::fabs(v - r) > 1e-4 * (1 + std::fabs(r));
        }
  printf("conv %ldx%ld IC=%-3ld OC=%-3ld slice [%3ld,%3ld) of L=%-3ld: dense=%-22s strided=%-22s %s\n", K, K, IC, OC, OFF, OFF + OC, L,
         impl[0].c_str(), impl[1].c_str(), wrong ? "WRONG" : bits ? "ref-ok (differs from dense in rounding)" : "bitwise");
  return wrong != 0;
}

// matmul [M, K] x [K, N] + per-column bias as a binary post-op, dst rows with leading dimension LD >= N.
static int matmul_case(engine& eng, stream& s, long Nc, long LD) {
  const long M = 64, K = 48;
  auto a = rnd(M * K, 4), w = rnd(K * Nc, 5), b = rnd(Nc, 6);
  memory::desc amd({M, K}, dt::f32, tag::ab), wmd({K, Nc}, dt::f32, tag::ab), bmd({1, Nc}, dt::f32, tag::ab);
  memory::desc dd({M, Nc}, dt::f32, tag::ab), ds({M, Nc}, dt::f32, memory::dims{LD, 1});
  post_ops po; po.append_binary(algorithm::binary_add, bmd);  // per column
  primitive_attr attr; attr.set_post_ops(po);
  std::vector<float> od(M * Nc), os(M * LD, -7.0f);
  std::string impl[2];
  memory::desc dsts[2] = {dd, ds};
  float* outs[2] = {od.data(), os.data()};
  for (int v = 0; v < 2; v++) {
    matmul::primitive_desc pd(eng, amd, wmd, dsts[v], attr);
    impl[v] = pd.impl_info_str();
    matmul(pd).execute(s, {{DNNL_ARG_SRC, memory(amd, eng, a.data())}, {DNNL_ARG_WEIGHTS, memory(wmd, eng, w.data())},
                           {DNNL_ARG_DST, memory(dsts[v], eng, outs[v])},
                           {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, memory(bmd, eng, b.data())}});
    s.wait();
  }
  long bits = 0, wrong = 0;
  for (long m = 0; m < M; m++)
    for (long n = 0; n < Nc; n++) {
      double r = b[n];
      for (long k = 0; k < K; k++) r += (double)a[m * K + k] * w[k * Nc + n];
      float d = od[m * Nc + n], v = os[m * LD + n];
      bits += std::memcmp(&d, &v, 4) != 0;
      wrong += std::fabs(v - r) > 1e-4 * (1 + std::fabs(r));
    }
  printf("matmul N=%-3ld dst leading dimension %-6ld: dense impl=%-24s strided impl=%-24s %s\n", Nc, LD,
         impl[0].c_str(), impl[1].c_str(), wrong ? "WRONG" : bits ? "ref-ok (differs from dense in rounding)" : "bitwise");
  return wrong != 0;
}

int main() {
  engine eng(engine::kind::cpu, 0); stream s(eng);
  printf("oneDNN %d.%d.%d\n", dnnl_version()->major, dnnl_version()->minor, dnnl_version()->patch);
  int bad = 0;
  bad += conv_case(eng, s, 48, 48, 0);    // dense control
  bad += conv_case(eng, s, 48, 96, 0);    // L % OC == 0
  bad += conv_case(eng, s, 48, 112, 16);  // L % OC != 0
  bad += conv_case(eng, s, 64, 64, 0, 1, 192, 17, 17);    // 1x1 dense control
  bad += conv_case(eng, s, 64, 256, 0, 1, 192, 17, 17);   // Inception-style 1x1 branch into concat
  bad += conv_case(eng, s, 96, 256, 64, 3, 64, 17, 17);   // 3x3 branch at offset 64
  bad += conv_case(eng, s, 32, 256, 224, 1, 192, 17, 17); // last slice
  bad += matmul_case(eng, s, 24, 24);     // dense control
  bad += matmul_case(eng, s, 24, 48);     // LD % N == 0
  bad += matmul_case(eng, s, 24, 40);     // LD % N != 0
  bad += matmul_case(eng, s, 48, 80);
  return bad ? 1 : 0;
}
