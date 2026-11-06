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
| `sat.sh` | Concurrent-process throughput with a synchronized start, TF vs the executor; uses aws-realign `hack/ie/run.py` with `aws-realign-run-start-at.patch` |
| `onednn-v2.7.3-brgconv-channel-slice-dst.patch` | Lets brgconv / brgconv_1x1 write into a channel slice of a wider nhwc tensor (`dst_ld`), and fixes the per-channel binary post-op offset for such a dst. Dense dsts take the unchanged code path |
| `ldtest.cpp` | Single-conv check: dense dst vs channel-slice dst, bit-for-bit, and no writes outside the slice |

Build: `make cvdnn ONEDNN=<oneDNN v2.7.3, cmake -DDNNL_CPU_RUNTIME=OMP> TFINC=<TF 2.13.1 wheel>/tensorflow/include`;
`make cvdnn-ld ldtest ONEDNN_LD=<the same tree with the patch applied>`. `--concat-direct` needs `cvdnn-ld`.
Without the patch, brgconv rejects the strided dst and oneDNN falls back to `ref:any`. That
path is 160× slower and not identical.
Built `-mavx` without FMA so Eigen's BatchNorm `rsqrt` matches TF's x86 wheel bit-for-bit;
a plain `1/sqrtf` changes 27% of the block's outputs.

Batch size: 1024 (the stock batch) is gated identical, and so are 256, 512 and 32. 128 and 64 are not:
6,394 of 8,192 rows differ by up to 2.4e-7, with the same impls and weight formats, so
brgconv blocks differently internally at those batch sizes.
