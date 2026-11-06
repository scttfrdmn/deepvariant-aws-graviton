// SPDX-License-Identifier: Apache-2.0
// ldtest: one brgconv with the BN post-op chain, dense nhwc dst vs a channel slice
// of a wider nhwc buffer; reports whether the slice matches the dense result.
//   ldtest MB IC OC K PAD H W L OFF
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "dnnl.hpp"
using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;
int main(int argc, char** argv) {
  long MB = atol(argv[1]), IC = atol(argv[2]), OC = atol(argv[3]), K = atol(argv[4]), P = atol(argv[5]),
       H = atol(argv[6]), W = atol(argv[7]), L = atol(argv[8]), OFF = atol(argv[9]);
  engine eng(engine::kind::cpu, 0); stream s(eng);
  std::mt19937 g(1); std::normal_distribution<float> nd;
  std::vector<float> x(MB * H * W * IC), wt(K * K * IC * OC), bn[4];
  for (auto& v : x) v = nd(g);
  for (auto& v : wt) v = nd(g) * 0.1f;
  for (int k = 0; k < 4; k++) { bn[k].resize(OC); for (auto& v : bn[k]) v = k == 1 ? 1.0f : nd(g) * 0.1f; }
  memory::desc bnmd({1, OC, 1, 1}, dt::f32, tag::nchw);
  post_ops po;
  po.append_binary(algorithm::binary_sub, bnmd); po.append_binary(algorithm::binary_mul, bnmd);
  po.append_binary(algorithm::binary_mul, bnmd); po.append_binary(algorithm::binary_add, bnmd);
  po.append_eltwise(1.0f, algorithm::eltwise_relu, 0.0f, 0.0f);
  primitive_attr attr; attr.set_post_ops(po);
  memory::desc src({MB, IC, H, W}, dt::f32, tag::nhwc), wany({OC, IC, K, K}, dt::f32, tag::any);
  memory::desc dd({MB, OC, H, W}, dt::f32, tag::nhwc), ds({MB, OC, H, W}, dt::f32, memory::dims{H * W * L, 1, W * L, L});
  std::vector<float> outd(MB * H * W * OC), outs(MB * H * W * L, -7.0f);
  float* res[2] = {outd.data(), outs.data() + OFF};
  memory::desc dsts[2] = {dd, ds};
  for (int v = 0; v < 2; v++) {
    convolution_forward::desc cd(prop_kind::forward_training, algorithm::convolution_direct, src, wany, dsts[v],
                                 {1, 1}, {0, 0}, {P, P}, {P, P});
    convolution_forward::primitive_desc pd(cd, attr, eng);
    memory wu({{OC, IC, K, K}, dt::f32, tag::hwio}, eng, wt.data()), w(pd.weights_desc(), eng);
    reorder(wu, w).execute(s, wu, w);
    std::unordered_map<int, memory> args{{DNNL_ARG_SRC, memory(src, eng, x.data())}, {DNNL_ARG_WEIGHTS, w},
                                         {DNNL_ARG_DST, memory(dsts[v], eng, res[v])}};
    for (int k = 0; k < 4; k++) args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(k) | DNNL_ARG_SRC_1] = memory(bnmd, eng, bn[k].data());
    convolution_forward(pd).execute(s, args); s.wait();
    printf("%s impl=%s\n", v ? "slice" : "dense", pd.impl_info_str());
  }
  long bad = 0, clob = 0;
  for (long p = 0; p < MB * H * W; p++) {
    for (long c = 0; c < OC; c++) bad += memcmp(&outd[p * OC + c], &outs[p * L + OFF + c], 4) != 0;
    for (long c = 0; c < L; c++) if ((c < OFF || c >= OFF + OC) && outs[p * L + c] != -7.0f) clob++;
  }
  printf("MB=%ld IC=%ld OC=%ld K=%ld HxW=%ldx%ld L=%ld OFF=%ld: slice %s (%ld wrong, %ld writes outside the slice)\n",
         MB, IC, OC, K, H, W, L, OFF, bad || clob ? "WRONG" : "IDENTICAL", bad, clob);
  return bad || clob;
}
