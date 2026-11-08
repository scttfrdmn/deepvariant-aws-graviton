// Fused GPU hand-off prototype (aws-realign#130, deepvariant-aws-graviton#10):
// make_examples' compact read records -> exact GPU pileup encoder -> uint8
// images in device memory -> (x - 128) / 128 on the device -> TensorRT
// inference on the same buffer -> probabilities to the host (12 bytes per
// example). No image ever crosses PCIe or touches host memory.
//
// Gates, per capture (a DV_DUMP_PILEUP file, see pileup_gpu_harness.cc):
//   images:  every fused batch, copied back once for the gate, is
//            byte-identical to make_examples' rows (the TFRecord images);
//   probs:   fused probabilities are bit-identical to the unfused GPU path
//            (make_examples' images copied host->device, same normalize
//            kernel, same engine), run twice to show the engine itself is
//            deterministic;
//   calls:   against --ref_probs (float32 [n,3], CPU stock call_variants on
//            the same examples): argmax agreement and max |dp|. FP32 TensorRT
//            vs oneDNN is tier 1 (different summation order), so this is a
//            report, not a bit gate.
//
// Build (NVIDIA, TensorRT 10):
//   nvcc -O3 -std=c++17 --fmad=false -x cu -I. deepvariant/gpu/fused_cv.cc \
//        -o fused_cv -lnvinfer -lnvonnxparser
// Usage: fused_cv CAPTURE.bin MODEL.onnx [--ref_probs F.f32] [--batch 1024]
//                 [--plan FILE] [--precision fp32|tf32] [--reps N] [--out_probs F.f32]
#define DV_HARNESS_NO_MAIN
#include "deepvariant/gpu/pileup_gpu_harness.cc"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <cmath>
#include <fstream>
#include <memory>

namespace {

struct Logger : nvinfer1::ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) fprintf(stderr, "[trt] %s\n", msg);
  }
} g_log;

// Encode the sampled reads of calls [c0, c1) (reads [r0, r1)).
__global__ void EncodeRange(Options o, const CallDesc* calls, const int* read_call, int r0, int r1,
                            const ReadDesc* reads, const uint8_t* cig_op, const int32_t* cig_len,
                            const char* seq, const uint8_t* qual, const char* ref, uint8_t* scratch,
                            uint8_t* ok) {
  int i = r0 + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= r1) return;
  const ReadDesc r = reads[i];
  const int ci = read_call[i];
  const size_t row_bytes = size_t(o.width) * o.nch;
  uint8_t* row = scratch + size_t(i) * row_bytes;
  for (size_t b = 0; b < row_bytes; ++b) row[b] = 0;
  ok[i] = r.mq >= o.min_mq &&
          EncodeReadRow(o, calls[ci], r, cig_op, cig_len, seq + r.seq0, qual + r.seq0,
                        ref + size_t(ci) * o.width, row);
}

// Assemble calls [c0, c0 + gridDim.x) into a batch buffer (image k = call c0 + k).
__global__ void AssembleRange(Options o, int c0, const CallDesc* calls, const ReadDesc* reads,
                              const uint8_t* ok, const char* ref, const uint8_t* scratch, int* keep_all,
                              uint8_t* batch) {
  const int ci = c0 + blockIdx.x;
  const CallDesc c = calls[ci];
  int* keep = keep_all + c.read0;
  __shared__ int nkeep;
  if (threadIdx.x == 0) nkeep = Assemble(o, c, reads, ok, keep) % 1000;
  __syncthreads();
  const size_t row_bytes = size_t(o.width) * o.nch;
  uint8_t* im = batch + size_t(blockIdx.x) * row_bytes * o.height;
  const char* rf = ref + size_t(ci) * o.width;
  for (int row = 0; row < o.height; ++row)
    for (int col = threadIdx.x; col < o.width; col += blockDim.x) {
      uint8_t* px = im + row * row_bytes + col * o.nch;
      if (row < o.ref_band) EncodeRefRow(o, rf, col, px);
      else if (row - o.ref_band < nkeep) {
        const uint8_t* src = scratch + size_t(keep[row - o.ref_band]) * row_bytes + col * o.nch;
        for (int k = 0; k < o.nch; ++k) px[k] = src[k];
      } else for (int k = 0; k < o.nch; ++k) px[k] = 0;
    }
}

// call_variants' preprocessing (dv_utils.preprocess_images): (float(x) - 128) / 128.
// Both steps are exact in float32 (integers < 2^24, division by a power of two).
__global__ void Normalize(const uint8_t* in, float* out, size_t n, size_t valid) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = i < valid ? (float(in[i]) - 128.0f) / 128.0f : 0.0f;
}

double Now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s CAPTURE.bin MODEL.onnx [--ref_probs F] [--batch B] [--plan F] [--precision fp32|tf32] [--reps N] [--out_probs F]\n", argv[0]); return 1; }
  const char *ref_probs = nullptr, *plan_path = nullptr, *out_probs = nullptr;
  std::string prec = "fp32";
  int B = 1024, reps = 3;
  for (int i = 3; i < argc; ++i) {
    if (!strcmp(argv[i], "--ref_probs") && i + 1 < argc) ref_probs = argv[++i];
    else if (!strcmp(argv[i], "--batch") && i + 1 < argc) B = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--plan") && i + 1 < argc) plan_path = argv[++i];
    else if (!strcmp(argv[i], "--precision") && i + 1 < argc) prec = argv[++i];
    else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--out_probs") && i + 1 < argc) out_probs = argv[++i];
  }
  Capture cap = Load(argv[1]);
  const Options& o = cap.opt;
  const int N = cap.calls.size();
  const size_t row_bytes = size_t(o.width) * o.nch, img = row_bytes * o.height;
  printf("capture %s: %d examples, %zu sampled reads, image %dx%dx%d; batch %d, %s\n", argv[1], N, cap.reads.size(),
         o.height, o.width, o.nch, B, prec.c_str());

  // ---- TensorRT engine (built once; cached plan if --plan exists) ----
  std::vector<char> plan;
  if (plan_path) { std::ifstream f(plan_path, std::ios::binary); if (f) plan.assign(std::istreambuf_iterator<char>(f), {}); }
  double tb = Now();
  if (plan.empty()) {
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(g_log));
    std::unique_ptr<nvinfer1::INetworkDefinition> net(builder->createNetworkV2(0));
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*net, g_log));
    if (!parser->parseFromFile(argv[2], int(nvinfer1::ILogger::Severity::kWARNING))) { fprintf(stderr, "ONNX parse failed\n"); return 2; }
    std::unique_ptr<nvinfer1::IBuilderConfig> cfg(builder->createBuilderConfig());
    if (prec == "tf32") cfg->setFlag(nvinfer1::BuilderFlag::kTF32); else cfg->clearFlag(nvinfer1::BuilderFlag::kTF32);
    auto* prof = builder->createOptimizationProfile();
    nvinfer1::Dims4 d(B, o.height, o.width, o.nch);
    const char* in_name = net->getInput(0)->getName();
    for (auto sel : {nvinfer1::OptProfileSelector::kMIN, nvinfer1::OptProfileSelector::kOPT, nvinfer1::OptProfileSelector::kMAX})
      prof->setDimensions(in_name, sel, d);
    cfg->addOptimizationProfile(prof);
    std::unique_ptr<nvinfer1::IHostMemory> ser(builder->buildSerializedNetwork(*net, *cfg));
    if (!ser) { fprintf(stderr, "engine build failed\n"); return 2; }
    plan.assign((const char*)ser->data(), (const char*)ser->data() + ser->size());
    if (plan_path) { std::ofstream f(plan_path, std::ios::binary); f.write(plan.data(), plan.size()); }
  }
  std::unique_ptr<nvinfer1::IRuntime> rt(nvinfer1::createInferRuntime(g_log));
  std::unique_ptr<nvinfer1::ICudaEngine> eng(rt->deserializeCudaEngine(plan.data(), plan.size()));
  std::unique_ptr<nvinfer1::IExecutionContext> ctx(eng->createExecutionContext());
  const char* in_name = eng->getIOTensorName(0);
  const char* out_name = eng->getIOTensorName(1);
  ctx->setInputShape(in_name, nvinfer1::Dims4(B, o.height, o.width, o.nch));
  printf("engine: %s -> %s, %.1f s to build/load, TensorRT %d.%d.%d\n", in_name, out_name, Now() - tb,
         NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);

  // ---- device inputs: compact records only ----
  std::vector<int> read_call(cap.reads.size());
  for (int ci = 0; ci < N; ++ci)
    for (int i = 0; i < cap.calls[ci].n_sampled; ++i) read_call[cap.calls[ci].read0 + i] = ci;
  double th = Now();
  CallDesc* d_calls = Up(cap.calls); ReadDesc* d_reads = Up(cap.reads); int* d_rc = Up(read_call);
  uint8_t* d_op = Up(cap.cig_op); int32_t* d_len = Up(cap.cig_len); char* d_seq = Up(cap.seq);
  uint8_t* d_qual = Up(cap.qual); char* d_ref = Up(cap.ref);
  CK(cudaDeviceSynchronize());
  const size_t in_bytes = cap.calls.size() * sizeof(CallDesc) + cap.reads.size() * (sizeof(ReadDesc) + sizeof(int)) +
                          cap.cig_op.size() * 5 + cap.seq.size() * 2 + cap.ref.size();
  printf("H2D compact inputs: %.1f MB (%.0f B/example) in %.3f s\n", in_bytes / 1e6, double(in_bytes) / N, Now() - th);
  uint8_t *d_scratch, *d_ok, *d_u8; int* d_keep; float *d_in, *d_out;
  CK(cudaMalloc(&d_scratch, cap.reads.size() * row_bytes)); CK(cudaMalloc(&d_ok, cap.reads.size()));
  CK(cudaMalloc(&d_keep, cap.reads.size() * sizeof(int)));
  CK(cudaMalloc(&d_u8, size_t(B) * img)); CK(cudaMalloc(&d_in, size_t(B) * img * sizeof(float)));
  CK(cudaMalloc(&d_out, size_t(B) * 3 * sizeof(float)));
  ctx->setTensorAddress(in_name, d_in); ctx->setTensorAddress(out_name, d_out);
  cudaStream_t st; CK(cudaStreamCreate(&st));
  std::vector<int> call_read0(N + 1);
  for (int ci = 0; ci < N; ++ci) call_read0[ci] = cap.calls[ci].read0;
  call_read0[N] = cap.reads.size();
  std::vector<uint8_t> pinned_batch(size_t(B) * img);  // unfused path: host images per batch

  enum Mode { FUSED, UNFUSED };
  // One pass over all examples; returns wall seconds; fills probs (n x 3) and, if gate, images.
  auto pass = [&](Mode m, std::vector<float>& probs, std::vector<uint8_t>* images, double* t_enc, double* t_inf) {
    probs.assign(size_t(N) * 3, 0.f);
    cudaEvent_t a, b, c; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); CK(cudaEventCreate(&c));
    double enc = 0, inf = 0;
    double t0 = Now();
    for (int c0 = 0; c0 < N; c0 += B) {
      const int k = std::min(B, N - c0);
      CK(cudaEventRecord(a, st));
      if (m == FUSED) {
        const int r0 = call_read0[c0], r1 = call_read0[c0 + k];
        if (r1 > r0)
          EncodeRange<<<(r1 - r0 + 127) / 128, 128, 0, st>>>(o, d_calls, d_rc, r0, r1, d_reads, d_op, d_len, d_seq,
                                                               d_qual, d_ref, d_scratch, d_ok);
        AssembleRange<<<k, 64, 0, st>>>(o, c0, d_calls, d_reads, d_ok, d_ref, d_scratch, d_keep, d_u8);
      } else {
        memcpy(pinned_batch.data(), cap.cpu_hwc.data() + size_t(c0) * img, size_t(k) * img);
        CK(cudaMemcpyAsync(d_u8, pinned_batch.data(), size_t(k) * img, cudaMemcpyHostToDevice, st));
      }
      const size_t n = size_t(B) * img;
      Normalize<<<(n + 255) / 256, 256, 0, st>>>(d_u8, d_in, n, size_t(k) * img);
      CK(cudaEventRecord(b, st));
      if (!ctx->enqueueV3(st)) { fprintf(stderr, "enqueueV3 failed\n"); exit(2); }
      CK(cudaEventRecord(c, st));
      CK(cudaMemcpyAsync(probs.data() + size_t(c0) * 3, d_out, size_t(k) * 3 * sizeof(float), cudaMemcpyDeviceToHost, st));
      if (images) CK(cudaMemcpyAsync(images->data() + size_t(c0) * img, d_u8, size_t(k) * img, cudaMemcpyDeviceToHost, st));
      CK(cudaStreamSynchronize(st)); CK(cudaGetLastError());
      float x, y; CK(cudaEventElapsedTime(&x, a, b)); CK(cudaEventElapsedTime(&y, b, c));
      enc += x / 1e3; inf += y / 1e3;
    }
    double w = Now() - t0;
    if (t_enc) *t_enc = enc;
    if (t_inf) *t_inf = inf;
    return w;
  };

  // Gate run: fused with images copied back, unfused twice.
  std::vector<float> pf, pu1, pu2;
  std::vector<uint8_t> fimg(size_t(N) * img);
  pass(FUSED, pf, &fimg, nullptr, nullptr);
  pass(UNFUSED, pu1, nullptr, nullptr, nullptr);
  pass(UNFUSED, pu2, nullptr, nullptr, nullptr);
  size_t bad_img = 0;
  for (int ci = 0; ci < N; ++ci) bad_img += memcmp(fimg.data() + size_t(ci) * img, cap.cpu_hwc.data() + size_t(ci) * img, img) != 0;
  auto bits_eq = [](const std::vector<float>& x, const std::vector<float>& y) {
    size_t d = 0; for (size_t i = 0; i < x.size(); i += 3) d += memcmp(&x[i], &y[i], 12) != 0; return d; };
  size_t d_uu = bits_eq(pu1, pu2), d_fu = bits_eq(pf, pu1);
  printf("gate images  (fused vs make_examples rows): %s (%zu of %d differ)\n", bad_img ? "DIFFERENT" : "IDENTICAL", bad_img, N);
  printf("gate engine  (unfused run 1 vs run 2):      %s (%zu of %d prob vectors differ)\n", d_uu ? "NONDETERMINISTIC" : "IDENTICAL", d_uu, N);
  printf("gate fused   (fused vs unfused probs):      %s (%zu of %d prob vectors differ)\n", d_fu ? "DIFFERENT" : "IDENTICAL", d_fu, N);
  if (out_probs) { FILE* f = fopen(out_probs, "wb"); fwrite(pf.data(), 4, pf.size(), f); fclose(f); }
  if (ref_probs) {
    std::vector<float> rp(size_t(N) * 3);
    FILE* f = fopen(ref_probs, "rb");
    size_t got = f ? fread(rp.data(), 4, rp.size(), f) : 0;
    if (f) fclose(f);
    if (got != rp.size()) printf("ref probs: %zu floats, expected %zu\n", got, rp.size());
    else {
      size_t argmax_diff = 0, bit_same = 0; double maxd = 0;
      for (int i = 0; i < N; ++i) {
        const float* x = &pf[size_t(i) * 3]; const float* y = &rp[size_t(i) * 3];
        int ax = std::max_element(x, x + 3) - x, ay = std::max_element(y, y + 3) - y;
        argmax_diff += ax != ay; bit_same += memcmp(x, y, 12) == 0;
        for (int k = 0; k < 3; ++k) maxd = std::max(maxd, double(std::fabs(x[k] - y[k])));
      }
      printf("vs CPU stock: argmax differs %zu of %d; bit-identical vectors %zu; max |dp| %.3g\n", argmax_diff, N, bit_same, maxd);
    }
  }
  // Timing: best of reps, no image copy-back.
  double best_f = 1e30, best_u = 1e30, ef = 0, inff = 0, eu = 0, infu = 0;
  for (int r = 0; r < reps; ++r) {
    std::vector<float> p; double e, i;
    double w = pass(FUSED, p, nullptr, &e, &i); if (w < best_f) { best_f = w; ef = e; inff = i; }
    w = pass(UNFUSED, p, nullptr, &e, &i); if (w < best_u) { best_u = w; eu = e; infu = i; }
  }
  printf("fused:   %.3f s = %.0f examples/s (encode+normalize %.3f s, inference %.3f s)\n", best_f, N / best_f, ef, inff);
  printf("unfused: %.3f s = %.0f examples/s (host copy+H2D+normalize %.3f s, inference %.3f s; H2D %.1f MB)\n", best_u, N / best_u, eu, infu,
         double(N) * img / 1e6);
  printf("GATE %s\n", (bad_img || d_fu) ? "FAIL" : "IDENTICAL");
  return (bad_img || d_fu) ? 3 : 0;
}
