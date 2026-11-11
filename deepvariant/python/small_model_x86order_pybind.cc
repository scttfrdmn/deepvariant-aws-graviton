/*
 * Copyright 2026 Google LLC.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

// The small model's dense stack (70 -> 750 relu -> 750 relu -> 3 softmax) in
// the floating-point order that stock DeepVariant 1.9.0 computes it on x86-64
// (TF 2.13.1 pip wheel, oneDNN on, AVX-512 host), so another architecture
// gets the same bits (deepvariant-aws-graviton#1). Measured against TF on a
// Zen 5 host, bit for bit, for batch sizes 1..1024:
//   MatMul: each output is one fused multiply-add chain over k, from 0. With
//     one row (oneDNN's gemv) the chain covers all of K; with two or more
//     rows, K = 750 is two chains of 375 whose sums are added (K = 70: one).
//   BiasAdd: a separate float add. Relu: max(x, 0).
//   Softmax (oneDNN jit, axis of 3): e = exp(x - max) with oneDNN's exp
//     (Cody-Waite reduction, degree-5 polynomial, 2 * 2^(n-1) scaling),
//     s = (e0 + e2) + e1, p = e * (1 / s); flush-to-zero / denormals-are-zero
//     as TF sets them (x86: tininess after rounding, emulated in software).
// Built with -ffp-contract=off (settings_arm64.sh), so only the std::fma
// calls fuse.

#if true  // Trick to stop tooling from moving the #include around.
// MUST appear before any standard headers are included.
#include <pybind11/pybind11.h>
#endif

#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace py = pybind11;

namespace {

constexpr float kFltMin = 1.17549435e-38f;

inline float Bits(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
// x86 FTZ/DAZ: a result (or input) below FLT_MIN in magnitude is zero.
inline float Ftz(float x) { return std::fabs(x) < kFltMin ? 0.0f * x : x; }

// Hardware flush-to-zero for the matrix products (TF runs ops with FTZ+DAZ).
class ScopedFlushToZero {
 public:
  ScopedFlushToZero() {
#if defined(__aarch64__)
    uint64_t v;
    asm volatile("mrs %0, fpcr" : "=r"(v));
    saved_ = v;
    v |= (1ull << 24);  // FPCR.FZ
    asm volatile("msr fpcr, %0" : : "r"(v));
#elif defined(__x86_64__)
    uint32_t v;
    asm volatile("stmxcsr %0" : "=m"(v));
    saved_ = v;
    v |= 0x8040;  // FTZ | DAZ
    asm volatile("ldmxcsr %0" : : "m"(v));
#endif
  }
  ~ScopedFlushToZero() {
#if defined(__aarch64__)
    uint64_t v = saved_;
    asm volatile("msr fpcr, %0" : : "r"(v));
#elif defined(__x86_64__)
    uint32_t v = static_cast<uint32_t>(saved_);
    asm volatile("ldmxcsr %0" : : "m"(v));
#endif
  }

 private:
  uint64_t saved_ = 0;
};

// d[j] = fma(a, w[j], d[j]) for j < n; NEON lanes fuse exactly like std::fma.
inline void FmaRow(float a, const float* __restrict w, float* __restrict d,
                   int n) {
  int j = 0;
#if defined(__aarch64__)
  const float32x4_t av = vdupq_n_f32(a);
  for (; j + 16 <= n; j += 16) {
    vst1q_f32(d + j, vfmaq_f32(vld1q_f32(d + j), av, vld1q_f32(w + j)));
    vst1q_f32(d + j + 4,
              vfmaq_f32(vld1q_f32(d + j + 4), av, vld1q_f32(w + j + 4)));
    vst1q_f32(d + j + 8,
              vfmaq_f32(vld1q_f32(d + j + 8), av, vld1q_f32(w + j + 8)));
    vst1q_f32(d + j + 12,
              vfmaq_f32(vld1q_f32(d + j + 12), av, vld1q_f32(w + j + 12)));
  }
  for (; j + 4 <= n; j += 4)
    vst1q_f32(d + j, vfmaq_f32(vld1q_f32(d + j), av, vld1q_f32(w + j)));
#endif
  for (; j < n; ++j) d[j] = std::fma(a, w[j], d[j]);
}

// y[i][j] = act(sum_k x[i][k] * w[k][j] + b[j]) in the order above.
// Rows are processed in tiles so each weight row streams once per tile.
void Dense(const float* x, int rows, int K, const float* w, const float* b,
           int N, bool relu, float* y) {
  const int kb = (rows == 1 || K <= 375) ? K : (K + 1) / 2;
  constexpr int kTile = 8;
  std::vector<float> acc(static_cast<size_t>(kTile) * N);
  std::vector<float> part(static_cast<size_t>(kTile) * N);
  for (int i0 = 0; i0 < rows; i0 += kTile) {
    const int tr = std::min(kTile, rows - i0);
    for (int k0 = 0; k0 < K; k0 += kb) {
      const int k1 = std::min(K, k0 + kb);
      float* dst = (k0 == 0) ? acc.data() : part.data();
      std::fill(dst, dst + static_cast<size_t>(tr) * N, 0.0f);
      for (int k = k0; k < k1; ++k) {
        const float* wk = w + static_cast<size_t>(k) * N;
        for (int t = 0; t < tr; ++t) {
          const float a = x[static_cast<size_t>(i0 + t) * K + k];
          float* d = dst + static_cast<size_t>(t) * N;
          FmaRow(a, wk, d, N);
        }
      }
      if (k0 != 0) {
        for (size_t q = 0; q < static_cast<size_t>(tr) * N; ++q)
          acc[q] = acc[q] + part[q];
      }
    }
    for (int t = 0; t < tr; ++t) {
      float* out = y + static_cast<size_t>(i0 + t) * N;
      const float* a = acc.data() + static_cast<size_t>(t) * N;
      for (int j = 0; j < N; ++j) {
        const float v = a[j] + b[j];
        out[j] = relu ? (v > 0.0f ? v : 0.0f) : v;
      }
    }
  }
}

// oneDNN's jit exp (eltwise injector), scalar, with x86 FTZ on the scaling.
float OneDnnExp(float x) {
  const float log2ef = Bits(0x3fb8aa3b), ln2f = Bits(0x3f317218);
  const float lmax = Bits(0x42b17218), lmin = Bits(0xc2aeac50);
  const bool zero = x < lmin;
  x = std::fmin(x, lmax);
  x = std::fmax(x, lmin);
  float r = x;
  float fx = x * log2ef;
  fx = fx + 0.5f;
  const float t = std::floor(fx);
  r = std::fma(-t, ln2f, r);
  const float n1 = t - 1.0f;
  const int32_t e = static_cast<int32_t>(std::lrint(n1));
  const float p2 = zero ? 0.0f : Bits(static_cast<uint32_t>(e + 127) << 23);
  float y = Bits(0x3c07cfce);
  y = std::fma(y, r, Bits(0x3d2b9d0d));
  y = std::fma(y, r, Bits(0x3e2aad40));
  y = std::fma(y, r, Bits(0x3efffee3));
  y = std::fma(y, r, Bits(0x3f7ffffb));
  y = std::fma(y, r, 1.0f);
  y = Ftz(y * p2);
  return y * 2.0f;
}

void Softmax3(const float* lg, int rows, float* p) {
  for (int i = 0; i < rows; ++i) {
    const float* x = lg + 3 * i;
    const float x0 = Ftz(x[0]), x1 = Ftz(x[1]), x2 = Ftz(x[2]);
    const float m = std::fmax(std::fmax(x0, x1), x2);
    const float e0 = Ftz(OneDnnExp(x0 - m)), e1 = Ftz(OneDnnExp(x1 - m)),
                e2 = Ftz(OneDnnExp(x2 - m));
    const float s = (e0 + e2) + e1;
    const float inv = 1.0f / s;
    p[3 * i] = Ftz(e0 * inv);
    p[3 * i + 1] = Ftz(e1 * inv);
    p[3 * i + 2] = Ftz(e2 * inv);
  }
}

using Arr = py::array_t<float, py::array::c_style | py::array::forcecast>;

void Need(const Arr& a, int d0, int d1, const char* what) {
  const bool ok = (d1 < 0) ? (a.ndim() == 1 && a.shape(0) == d0)
                           : (a.ndim() == 2 && a.shape(0) == d0 &&
                              (d1 == 0 || a.shape(1) == d1));
  if (!ok) throw std::invalid_argument(std::string("bad shape: ") + what);
}

// forward(x[B,70], w1[70,750], b1, w2[750,750], b2, w3[750,3], b3) -> [B,3]
py::array_t<float> Forward(const Arr& x, const Arr& w1, const Arr& b1,
                           const Arr& w2, const Arr& b2, const Arr& w3,
                           const Arr& b3) {
  if (x.ndim() != 2 || x.shape(1) != 70)
    throw std::invalid_argument("bad shape: x");
  const int rows = static_cast<int>(x.shape(0));
  Need(w1, 70, 750, "w1");
  Need(b1, 750, -1, "b1");
  Need(w2, 750, 750, "w2");
  Need(b2, 750, -1, "b2");
  Need(w3, 750, 3, "w3");
  Need(b3, 3, -1, "b3");
  py::array_t<float> out({rows, 3});
  std::vector<float> h1(static_cast<size_t>(rows) * 750),
      h2(static_cast<size_t>(rows) * 750), lg(static_cast<size_t>(rows) * 3);
  {
    py::gil_scoped_release release;
    {
      ScopedFlushToZero ftz;
      Dense(x.data(), rows, 70, w1.data(), b1.data(), 750, true, h1.data());
      Dense(h1.data(), rows, 750, w2.data(), b2.data(), 750, true, h2.data());
      Dense(h2.data(), rows, 750, w3.data(), b3.data(), 3, false, lg.data());
    }
    Softmax3(lg.data(), rows, out.mutable_data());
  }
  return out;
}

}  // namespace

PYBIND11_MODULE(small_model_x86order, m) {
  m.doc() =
      "Small-model dense stack in stock x86-64 DeepVariant 1.9.0's FP order.";
  m.def("forward", &Forward, py::arg("x"), py::arg("w1"), py::arg("b1"),
        py::arg("w2"), py::arg("b2"), py::arg("w3"), py::arg("b3"));
}
