// Fused GPU hand-off, production path (aws-realign#130, deepvariant-aws-graviton#10):
// call_variants' model step over make_examples' compact records
// (DV_FUSED_OUT, deepvariant/make_examples_native.cc) instead of its examples.
//
// Per record (TFRecord, GZIP): "FUS1", a rows flag, variant/encoded,
// alt_allele_indices/encoded, then the pileup record written by
// PileupImageEncoderNative::DumpPileupCall (options, reference window, every
// sampled read in sampled order; the CPU's rows when the flag is set). The
// exact GPU encoder (pileup_encode.h) builds each image in device memory, a
// device kernel applies call_variants' (x - 128) / 128, and TensorRT runs on
// the same buffer; only 12 bytes of probabilities per example come back.
//
// Output (--out): per example, u32 length + variant/encoded, u32 length +
// alt_allele_indices/encoded, float32[3] probabilities, in record order.
// fused_cvo_writer.py turns that into CallVariantsOutput records with stock
// call_variants' own write_variant_call, so postprocess_variants is unchanged.
//
// --gate: records written with DV_FUSED_ROWS=1 carry the CPU's rows; every
// GPU image is compared byte-for-byte with them (make_examples' images).
//
// A reader thread parses the next chunk while the GPU runs the current one.
//
// Build (NVIDIA, TensorRT 10, in the NGC TensorRT container):
//   nvcc -O3 -std=c++17 --fmad=false -x cu -I. deepvariant/gpu/fused_call_variants.cc \
//        -o fused_call_variants -lnvinfer -lnvonnxparser -lz -lpthread
// Usage: fused_call_variants --model wgs.onnx --out preds.bin [--plan F] [--precision fp32|tf32]
//                            [--batch 1024] [--chunk 32768] [--gate] RECORDS.gz...
#define DV_HARNESS_NO_MAIN
#include "deepvariant/gpu/pileup_gpu_harness.cc"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <zlib.h>

#include <condition_variable>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>

namespace {

struct Logger : nvinfer1::ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) fprintf(stderr, "[trt] %s\n", msg);
  }
} g_log;

double Now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// One chunk of examples, laid out as the encoder kernels take them.
struct Chunk {
  Capture cap;                      // opt, calls, reads, cigars, seq, qual, ref, cpu_hwc (with rows)
  std::vector<std::string> var, alt;
  bool rows = false;
  bool last = false;
};

// TFRecord reader over a gzip stream (TensorFlow's RecordWriter, GZIP).
class TfRecordGz {
 public:
  explicit TfRecordGz(const std::string& path) : f_(gzopen(path.c_str(), "rb")) {
    if (!f_) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    gzbuffer(f_, 1 << 20);
  }
  ~TfRecordGz() { gzclose(f_); }
  bool Next(std::string* rec) {
    uint64_t len; uint32_t crc;
    int n = gzread(f_, &len, 8);
    if (n == 0) return false;
    if (n != 8 || gzread(f_, &crc, 4) != 4) { fprintf(stderr, "truncated record header\n"); exit(1); }
    rec->resize(len);
    if (gzread(f_, rec->data(), len) != int(len) || gzread(f_, &crc, 4) != 4) { fprintf(stderr, "truncated record\n"); exit(1); }
    return true;
  }
 private:
  gzFile f_;
};

struct Cursor {
  const char* p; const char* e;
  template <typename T> T Get() {
    T v; if (p + sizeof(T) > e) { fprintf(stderr, "record overrun\n"); exit(1); }
    memcpy(&v, p, sizeof(T)); p += sizeof(T); return v;
  }
  const char* Take(size_t n) { if (p + n > e) { fprintf(stderr, "record overrun\n"); exit(1); } const char* q = p; p += n; return q; }
};

// Appends one FUS1 record to the chunk; keys collect (fragment_name, read_number) per read.
void ParseRecord(const std::string& rec, Chunk* ch, std::vector<std::tuple<std::string, int>>* keys, bool* first) {
  Capture& cap = ch->cap;
  Cursor c{rec.data(), rec.data() + rec.size()};
  if (c.Get<uint32_t>() != 0x31535546) { fprintf(stderr, "bad FUS1 magic\n"); exit(1); }
  const bool rows = c.Get<uint8_t>() != 0;
  uint32_t lv = c.Get<uint32_t>(); ch->var.emplace_back(c.Take(lv), lv);
  uint32_t la = c.Get<uint32_t>(); ch->alt.emplace_back(c.Take(la), la);
  if (c.Get<uint32_t>() != 0x454c4950) { fprintf(stderr, "bad PILE magic\n"); exit(1); }
  Options o{};
  o.width = c.Get<int32_t>(); const int32_t nrows = c.Get<int32_t>(); o.ref_band = c.Get<int32_t>(); o.nch = c.Get<int32_t>();
  if (o.nch > kMaxChannels) { fprintf(stderr, "too many channels\n"); exit(1); }
  for (int i = 0; i < o.nch; ++i) o.ch[i] = c.Get<int32_t>();
  o.off_ag = c.Get<int32_t>(); o.off_tc = c.Get<int32_t>(); o.stride = c.Get<int32_t>(); o.ref_bq = c.Get<int32_t>();
  o.bq_cap = c.Get<float>(); o.mq_cap = c.Get<float>(); o.pos_color = c.Get<int32_t>(); o.neg_color = c.Get<int32_t>();
  o.a_supp = c.Get<float>(); o.a_unsupp = c.Get<float>(); o.a_other = c.Get<float>(); o.a_match = c.Get<float>(); o.a_mismatch = c.Get<float>();
  o.anchor = c.Get<uint8_t>(); o.min_mq = c.Get<int32_t>(); o.min_bq = c.Get<int32_t>();
  if (c.Get<int32_t>() != 0) { fprintf(stderr, "blanked channels not implemented\n"); exit(1); }
  o.height = nrows;
  if (*first) { cap.opt = o; ch->rows = rows; *first = false; }
  else if (memcmp(&o, &cap.opt, sizeof(Options)) != 0 || rows != ch->rows) { fprintf(stderr, "options vary between records\n"); exit(1); }
  CallDesc cd{};
  cd.vstart = c.Get<int64_t>(); cd.img_start = c.Get<int32_t>();
  size_t r0 = cap.ref.size(); cap.ref.resize(r0 + o.width);
  memcpy(cap.ref.data() + r0, c.Take(o.width), o.width);
  cd.max_reads = c.Get<int32_t>(); cd.n_sampled = c.Get<int32_t>();
  cd.read0 = cap.reads.size();
  for (int i = 0; i < cd.n_sampled; ++i) {
    ReadDesc r{};
    r.sort_pos = c.Get<int32_t>(); r.hap = c.Get<int32_t>(); r.align_pos = c.Get<int32_t>(); r.mq = c.Get<int32_t>();
    r.rev = c.Get<uint8_t>(); r.frag_len = c.Get<int32_t>(); const int32_t read_number = c.Get<int32_t>();
    const uint16_t nl = c.Get<uint16_t>(); std::string name(c.Take(nl), nl);
    r.support = c.Get<uint8_t>();
    r.ncig = c.Get<int32_t>(); r.cig0 = cap.cig_op.size();
    for (int k = 0; k < r.ncig; ++k) { cap.cig_op.push_back(c.Get<uint8_t>()); cap.cig_len.push_back(c.Get<int32_t>()); }
    r.seq_len = c.Get<int32_t>(); r.seq0 = cap.seq.size();
    const char* sq = c.Take(r.seq_len);
    cap.seq.insert(cap.seq.end(), sq, sq + r.seq_len);
    const int32_t ql = c.Get<int32_t>();
    if (ql != r.seq_len) { fprintf(stderr, "qual length != seq length\n"); exit(1); }
    const char* q = c.Take(ql);
    cap.qual.insert(cap.qual.end(), q, q + ql);
    keys->emplace_back(std::move(name), read_number);
    cap.reads.push_back(r);
  }
  if (rows) {  // channel-major per row -> HWC, as FillPileupArray lays the image out
    const char* src = c.Take(size_t(nrows) * o.width * o.nch);
    const size_t img = size_t(nrows) * o.width * o.nch, h0 = cap.cpu_hwc.size();
    cap.cpu_hwc.resize(h0 + img);
    for (int row = 0; row < nrows; ++row)
      for (int col = 0; col < o.width; ++col)
        for (int k = 0; k < o.nch; ++k)
          cap.cpu_hwc[h0 + (size_t(row) * o.width + col) * o.nch + k] = src[(size_t(row) * o.nch + k) * o.width + col];
  }
  if (c.p != c.e) { fprintf(stderr, "trailing bytes in record\n"); exit(1); }
  cap.calls.push_back(cd);
}

// (fragment_name, read_number) ranks within the chunk: equal keys share a rank, so the
// tie-break is the CPU's tuple<string,int> comparison (only reads of one call are compared).
void Rank(Capture* cap, std::vector<std::tuple<std::string, int>>* keys) {
  std::vector<int> order(keys->size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](int a, int b) { return (*keys)[a] < (*keys)[b]; });
  int rank = -1;
  for (size_t i = 0; i < order.size(); ++i) {
    if (i == 0 || (*keys)[order[i - 1]] < (*keys)[order[i]]) ++rank;
    cap->reads[order[i]].rank = rank;
  }
}

class Queue {
 public:
  void Push(std::unique_ptr<Chunk> c) {
    std::unique_lock<std::mutex> l(m_); cv_.wait(l, [&] { return q_.size() < 2; });
    q_.push_back(std::move(c)); cv_.notify_all();
  }
  std::unique_ptr<Chunk> Pop() {
    std::unique_lock<std::mutex> l(m_); cv_.wait(l, [&] { return !q_.empty(); });
    auto c = std::move(q_.front()); q_.pop_front(); cv_.notify_all(); return c;
  }
 private:
  std::mutex m_; std::condition_variable cv_; std::deque<std::unique_ptr<Chunk>> q_;
};

void Reader(const std::vector<std::string>& files, int chunk, Queue* q, double* busy_s, size_t* raw_bytes) {
  auto ch = std::make_unique<Chunk>();
  std::vector<std::tuple<std::string, int>> keys;
  bool first = true;
  std::string rec;
  double busy = 0;
  auto flush = [&](bool last) {
    double t = Now();
    Rank(&ch->cap, &keys);
    busy += Now() - t;
    ch->last = last;
    q->Push(std::move(ch));
    ch = std::make_unique<Chunk>(); keys.clear(); first = true;
  };
  for (const auto& path : files) {
    TfRecordGz in(path);
    double t = Now();
    while (in.Next(&rec)) {
      *raw_bytes += rec.size();
      ParseRecord(rec, ch.get(), &keys, &first);
      if (int(ch->cap.calls.size()) >= chunk) { busy += Now() - t; flush(false); t = Now(); }
    }
    busy += Now() - t;
  }
  flush(true);
  *busy_s = busy;
}

__global__ void EncodeRange(Options o, const CallDesc* calls, const int* read_call, int r0, int r1,
                            const ReadDesc* reads, const uint8_t* cig_op, const int32_t* cig_len,
                            const char* seq, const uint8_t* qual, const char* ref, uint8_t* scratch,
                            uint8_t* ok) {
  int i = r0 + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= r1) return;
  const ReadDesc r = reads[i];
  const int ci = read_call[i];
  const size_t row_bytes = size_t(o.width) * o.nch;
  uint8_t* row = scratch + size_t(i - r0) * row_bytes;
  for (size_t b = 0; b < row_bytes; ++b) row[b] = 0;
  ok[i] = r.mq >= o.min_mq &&
          EncodeReadRow(o, calls[ci], r, cig_op, cig_len, seq + r.seq0, qual + r.seq0,
                        ref + size_t(ci) * o.width, row);
}

__global__ void AssembleRange(Options o, int c0, int r0, const CallDesc* calls, const ReadDesc* reads,
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
        const uint8_t* src = scratch + size_t(keep[row - o.ref_band] - r0) * row_bytes + col * o.nch;
        for (int k = 0; k < o.nch; ++k) px[k] = src[k];
      } else for (int k = 0; k < o.nch; ++k) px[k] = 0;
    }
}

// call_variants' preprocessing (dv_utils.preprocess_images): (float(x) - 128) / 128, exact in float32.
__global__ void Normalize(const uint8_t* in, float* out, size_t n, size_t valid) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = i < valid ? (float(in[i]) - 128.0f) / 128.0f : 0.0f;
}

template <typename T> T* DevCopy(const std::vector<T>& v, cudaStream_t st) {
  T* d = nullptr;
  CK(cudaMallocAsync(&d, std::max<size_t>(1, v.size()) * sizeof(T), st));
  if (!v.empty()) CK(cudaMemcpyAsync(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, st));
  return d;
}

}  // namespace

int main(int argc, char** argv) {
  const char *model = nullptr, *out_path = nullptr, *plan_path = nullptr;
  std::string prec = "fp32";
  int B = 1024, chunk = 32768;
  bool gate = false;
  std::vector<std::string> files;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
    else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
    else if (!strcmp(argv[i], "--plan") && i + 1 < argc) plan_path = argv[++i];
    else if (!strcmp(argv[i], "--precision") && i + 1 < argc) prec = argv[++i];
    else if (!strcmp(argv[i], "--batch") && i + 1 < argc) B = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--chunk") && i + 1 < argc) chunk = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--gate")) gate = true;
    else files.push_back(argv[i]);
  }
  if (!model || !out_path || files.empty()) {
    fprintf(stderr, "usage: %s --model M.onnx --out preds.bin [--plan F] [--precision fp32|tf32] [--batch B] [--chunk N] [--gate] RECORDS.gz...\n", argv[0]);
    return 1;
  }
  const double t_start = Now();
  Queue q;
  double reader_busy = 0; size_t raw_bytes = 0;
  std::thread reader(Reader, files, chunk, &q, &reader_busy, &raw_bytes);
  std::unique_ptr<Chunk> ch = q.Pop();
  const Options o = ch->cap.opt;
  const size_t row_bytes = size_t(o.width) * o.nch, img = row_bytes * o.height;
  printf("records: %zu files; image %dx%dx%d; batch %d; chunk %d; %s%s\n", files.size(), o.height, o.width, o.nch, B, chunk,
         prec.c_str(), gate ? "; gate on" : "");

  // ---- TensorRT engine (cached plan if --plan exists) ----
  std::vector<char> plan;
  if (plan_path) { std::ifstream f(plan_path, std::ios::binary); if (f) plan.assign(std::istreambuf_iterator<char>(f), {}); }
  double tb = Now();
  if (plan.empty()) {
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(g_log));
    std::unique_ptr<nvinfer1::INetworkDefinition> net(builder->createNetworkV2(0));
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*net, g_log));
    if (!parser->parseFromFile(model, int(nvinfer1::ILogger::Severity::kWARNING))) { fprintf(stderr, "ONNX parse failed\n"); return 2; }
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
  printf("engine: %.1f s to build/load, TensorRT %d.%d.%d\n", Now() - tb, NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);

  cudaStream_t st; CK(cudaStreamCreate(&st));
  uint8_t* d_u8; float *d_in, *d_out;
  CK(cudaMalloc(&d_u8, size_t(B) * img)); CK(cudaMalloc(&d_in, size_t(B) * img * sizeof(float)));
  CK(cudaMalloc(&d_out, size_t(B) * 3 * sizeof(float)));
  ctx->setTensorAddress(in_name, d_in); ctx->setTensorAddress(out_name, d_out);
  float* h_probs; CK(cudaMallocHost(&h_probs, size_t(B) * 3 * sizeof(float)));
  uint8_t* h_img = nullptr;
  if (gate) CK(cudaMallocHost(&h_img, size_t(B) * img));
  FILE* out = fopen(out_path, "wb");
  if (!out) { perror(out_path); return 1; }

  size_t n_total = 0, n_reads = 0, h2d = 0, bad_img = 0, gated = 0;
  double gpu_s = 0, wait_s = 0;
  while (true) {
    Capture& cap = ch->cap;
    const int N = cap.calls.size();
    if (N > 0) {
      if (memcmp(&cap.opt, &o, sizeof(Options)) != 0) { fprintf(stderr, "options vary between chunks\n"); return 2; }
      if (gate && !ch->rows) { fprintf(stderr, "--gate needs records written with DV_FUSED_ROWS=1\n"); return 2; }
      const double tg = Now();
      std::vector<int> read_call(cap.reads.size());
      for (int ci = 0; ci < N; ++ci)
        for (int i = 0; i < cap.calls[ci].n_sampled; ++i) read_call[cap.calls[ci].read0 + i] = ci;
      CallDesc* d_calls = DevCopy(cap.calls, st); ReadDesc* d_reads = DevCopy(cap.reads, st); int* d_rc = DevCopy(read_call, st);
      uint8_t* d_op = DevCopy(cap.cig_op, st); int32_t* d_len = DevCopy(cap.cig_len, st); char* d_seq = DevCopy(cap.seq, st);
      uint8_t* d_qual = DevCopy(cap.qual, st); char* d_ref = DevCopy(cap.ref, st);
      h2d += cap.calls.size() * sizeof(CallDesc) + cap.reads.size() * (sizeof(ReadDesc) + sizeof(int)) +
             cap.cig_op.size() * 5 + cap.seq.size() * 2 + cap.ref.size();
      // Scratch rows for one batch's reads (max over the chunk's batches).
      size_t max_batch_reads = 0;
      for (int c0 = 0; c0 < N; c0 += B) {
        const int k = std::min(B, N - c0);
        const int r0 = cap.calls[c0].read0, r1 = c0 + k < N ? cap.calls[c0 + k].read0 : int(cap.reads.size());
        max_batch_reads = std::max<size_t>(max_batch_reads, r1 - r0);
      }
      uint8_t *d_scratch, *d_ok; int* d_keep;
      CK(cudaMallocAsync(&d_scratch, std::max<size_t>(1, max_batch_reads) * row_bytes, st));
      CK(cudaMallocAsync(&d_ok, std::max<size_t>(1, cap.reads.size()), st));
      CK(cudaMallocAsync(&d_keep, std::max<size_t>(1, cap.reads.size()) * sizeof(int), st));
      for (int c0 = 0; c0 < N; c0 += B) {
        const int k = std::min(B, N - c0);
        const int r0 = cap.calls[c0].read0, r1 = c0 + k < N ? cap.calls[c0 + k].read0 : int(cap.reads.size());
        if (r1 > r0)
          EncodeRange<<<(r1 - r0 + 127) / 128, 128, 0, st>>>(o, d_calls, d_rc, r0, r1, d_reads, d_op, d_len, d_seq,
                                                               d_qual, d_ref, d_scratch, d_ok);
        AssembleRange<<<k, 64, 0, st>>>(o, c0, r0, d_calls, d_reads, d_ok, d_ref, d_scratch, d_keep, d_u8);
        const size_t n = size_t(B) * img;
        Normalize<<<(n + 255) / 256, 256, 0, st>>>(d_u8, d_in, n, size_t(k) * img);
        if (!ctx->enqueueV3(st)) { fprintf(stderr, "enqueueV3 failed\n"); return 2; }
        CK(cudaMemcpyAsync(h_probs, d_out, size_t(k) * 3 * sizeof(float), cudaMemcpyDeviceToHost, st));
        if (gate) CK(cudaMemcpyAsync(h_img, d_u8, size_t(k) * img, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st)); CK(cudaGetLastError());
        for (int i = 0; i < k; ++i) {
          const std::string& v = ch->var[c0 + i]; const std::string& a = ch->alt[c0 + i];
          uint32_t lv = v.size(), la = a.size();
          fwrite(&lv, 4, 1, out); fwrite(v.data(), 1, lv, out);
          fwrite(&la, 4, 1, out); fwrite(a.data(), 1, la, out);
          fwrite(h_probs + size_t(i) * 3, 4, 3, out);
        }
        if (gate)
          for (int i = 0; i < k; ++i, ++gated)
            bad_img += memcmp(h_img + size_t(i) * img, cap.cpu_hwc.data() + size_t(c0 + i) * img, img) != 0;
      }
      for (void* p : {(void*)d_calls, (void*)d_reads, (void*)d_rc, (void*)d_op, (void*)d_len, (void*)d_seq, (void*)d_qual,
                      (void*)d_ref, (void*)d_scratch, (void*)d_ok, (void*)d_keep})
        CK(cudaFreeAsync(p, st));
      CK(cudaStreamSynchronize(st));
      gpu_s += Now() - tg;
      n_total += N; n_reads += cap.reads.size();
    }
    if (ch->last) break;
    const double tw = Now();
    ch = q.Pop();
    wait_s += Now() - tw;
  }
  reader.join();
  fclose(out);
  const double wall = Now() - t_start;
  printf("examples %zu, sampled reads %zu (%.1f per example); record bytes %.2f GB (%.0f B/example) uncompressed; H2D %.2f GB\n",
         n_total, n_reads, double(n_reads) / std::max<size_t>(1, n_total), raw_bytes / 1e9, double(raw_bytes) / std::max<size_t>(1, n_total),
         h2d / 1e9);
  printf("wall %.1f s = %.0f examples/s; GPU loop %.1f s; waiting on the reader %.1f s; reader busy %.1f s\n", wall,
         n_total / wall, gpu_s, wait_s, reader_busy);
  if (gate) printf("gate images (GPU vs make_examples rows): %s (%zu of %zu differ)\n", bad_img ? "DIFFERENT" : "IDENTICAL", bad_img, gated);
  return gate && bad_img ? 3 : 0;
}
