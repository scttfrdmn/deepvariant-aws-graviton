# Standalone oneDNN executor for call_variants (deepvariant-aws-graviton#11)

Runs the DeepVariant 1.9 WGS InceptionV3 with the same oneDNN v2.7.3 primitives that
TF 2.13.1 creates for it (native NHWC brgconv / brgconv_1x1, FusedBatchNorm + Relu as
binary + eltwise post-ops, pooling_v2, gemm:jit inner product, softmax_v2), so the
arithmetic is the same and the probabilities stay bit-for-bit identical. The difference
is the executor: TF's graph runtime is replaced by a fixed op sequence over pre-reordered
filters and a liveness-planned buffer pool.

| File | Role |
|---|---|
| `export.py` | SavedModel → frozen GraphDef with every weight turned back into a Placeholder (so TF's oneDNN layout pass still sees non-constant filters), `weights.npz`, `ops.json` |
| `tfref.py` | TF reference over the exported graph (v1 Session), optional intermediate fetches; gates against `probs_tf213.npy` |
| `blockspec.py` | Writes a line spec plus raw weights for a span (one Inception block, or `input_1:0` → Softmax), with TF's own tensors for the span |
| `cvdnn.cpp` | The executor; `--images DUMP --n N` runs the whole network over a dump and gates against `probs_tf213.npy` |
| `sat.sh` | Concurrent-process throughput, synchronized start, TF vs the executor |

Build: `make ONEDNN=<oneDNN v2.7.3 tree built with cmake> TFINC=<TF 2.13.1 wheel>/tensorflow/include`.
Built `-mavx` without FMA so Eigen's BatchNorm `rsqrt` matches TF's x86 wheel bit-for-bit;
a plain `1/sqrtf` changes 27% of the block's outputs.

Batch size: 1024 (the stock batch) is gated identical, and so are 256, 512 and 32. 128 and 64 are not:
6,394 of 8,192 rows differ by up to 2.4e-7, with the same impls and weight formats, so
brgconv blocks differently internally at those batch sizes.
