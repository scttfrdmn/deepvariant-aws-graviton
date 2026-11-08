// SPDX-License-Identifier: Apache-2.0
// Time a conv writing a dense dst vs a channel slice of a wider nhwc tensor.
#include <chrono>
#include <cstdio>
#include <vector>
#include <string>
#include <unordered_map>
#include "oneapi/dnnl/dnnl.hpp"
using namespace dnnl; using tag = memory::format_tag; using dt = memory::data_type;
static double run(engine& eng, stream& s, long N, long IC, long OC, long H, long W, long K, long L, long OFF, bool strided, std::string& impl) {
  memory::desc src({N, IC, H, W}, dt::f32, tag::nhwc), wany({OC, IC, K, K}, dt::f32, tag::any), bmd({1, OC, 1, 1}, dt::f32, tag::nchw);
  memory::desc dst = strided ? memory::desc({N, OC, H, W}, dt::f32, memory::dims{H * W * L, 1, W * L, L}) : memory::desc({N, OC, H, W}, dt::f32, tag::nhwc);
  post_ops po; po.append_binary(algorithm::binary_add, bmd); po.append_eltwise(algorithm::eltwise_relu, 0, 0);
  primitive_attr attr; attr.set_post_ops(po);
  convolution_forward::primitive_desc pd(eng, prop_kind::forward_inference, algorithm::convolution_direct, src, wany, dst, {1, 1}, {K / 2, K / 2}, {K / 2, K / 2}, attr);
  impl = pd.impl_info_str();
  std::vector<float> x(N * H * W * IC, 0.5f), b(OC, 0.1f), y(N * H * W * (strided ? L : OC));
  memory wm(pd.weights_desc(), eng);
  convolution_forward c(pd);
  std::unordered_map<int, memory> args{{DNNL_ARG_SRC, memory(src, eng, x.data())}, {DNNL_ARG_WEIGHTS, wm},
      {DNNL_ARG_DST, memory(dst, eng, y.data() + (strided ? OFF : 0))}, {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, memory(bmd, eng, b.data())}};
  for (int i = 0; i < 3; i++) c.execute(s, args); s.wait();
  const int R = 20; auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < R; i++) c.execute(s, args); s.wait();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / R;
}
int main() {
  engine eng(engine::kind::cpu, 0); stream s(eng);
  struct C { long IC, OC, H, W, K, L, OFF; const char* name; } cs[] = {
    {192, 64, 35, 35, 1, 256, 0, "1x1 192->64, slice [0,64) of 256"},
    {48, 64, 35, 35, 5, 256, 64, "5x5 48->64, slice [64,128) of 256"},
    {96, 96, 35, 35, 3, 256, 128, "3x3 96->96, slice [128,224) of 256"}};
  for (auto& c : cs) {
    std::string i0, i1; double d = run(eng, s, 64, c.IC, c.OC, c.H, c.W, c.K, c.L, c.OFF, false, i0), t = run(eng, s, 64, c.IC, c.OC, c.H, c.W, c.K, c.L, c.OFF, true, i1);
    printf("%-36s dense %-24s %8.2f ms | slice %-24s %8.2f ms\n", c.name, i0.c_str(), d, i1.c_str(), t);
  }
}
