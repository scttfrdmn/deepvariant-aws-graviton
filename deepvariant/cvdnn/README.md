# libcvdnn: call_variants inference without TensorFlow's runtime (opt-in)

`DV_CV_ENGINE=cvdnn` makes call_variants run the WGS InceptionV3 through `libcvdnn.so`. libcvdnn uses the same oneDNN v2.7.3 primitives that TF 2.13.1 creates for this model, so the probabilities are bit-for-bit stock's. Two things stay stock: reading examples (tf.data) and writing CallVariantsOutput. In this mode:
- the `(x - 128) / 128` preprocessing moves into libcvdnn;
- tf.data autotuning is off. Its model thread waits up to 60 s between optimizations, and with inference outside TF that stalled the end of iteration.

It needs a SavedModel checkpoint, no debug info, no channel ablation and examples from files; anything else is refused.

| File | Role |
|---|---|
| `cvdnn_lib.cpp` | The executor with a C API. Filters are reordered once, buffers come from a liveness plan, and Inception concat is done in place. It refuses any conv impl other than brgconv |
| `engine.py` | Freezes the SavedModel at start-up, writes the spec and weights to a temporary directory, and opens libcvdnn through ctypes |
| `patch_call_variants.py` | The call_variants.py change, applicable to DeepVariant 1.9.0's file (`call_variants.py` here already has it) |
| `build.sh` | Builds the patched oneDNN v2.7.3 and libcvdnn inside the DeepVariant image, and packages an unpacked call_variants build (`cv-perf.tar.zst`) |

Build: `deepvariant/cvdnn/build.sh google/deepvariant:1.9.0@sha256:07e9… OUT`. It needs Docker, and network access for the oneDNN tag.
The oneDNN patch is `tools/cv_onednn/onednn-v2.7.3-brgconv-channel-slice-dst.patch`.

Threads come from OpenMP (`OMP_NUM_THREADS`). Don't set `OMP_PROC_BIND`: libgomp would bind the main thread to one core, and every TF thread created after that inherits the binding.

## Measured (deepvariant-aws-graviton#11)

**chr20 on antares** (Zen 5, cores 0–15, 77,267 examples): the CallVariantsOutput multiset is **IDENTICAL** to stock at batch 1024 and at 512.

| call_variants | Wall | CPU-s | Peak RSS (main process) |
|---|---|---|---|
| stock, batch 1024 | 73 s | 852–893 | 8.64 GB |
| stock, batch 512 | 75 s | 834 | 5.06 GB |
| cvdnn, batch 1024 | 74–80 s | 906–973 | 6.45 GB |
| cvdnn, batch 512 | 76 s | 977 | 4.27 GB |

**Whole genome on c8a.16xlarge** (4 processes × 16 CPUs, pinned): gVCF records **IDENTICAL** to stock. call_variants took 597 s and 9.68 CPU-h, against 575 s and 9.06 CPU-h for stock (`probe-dv-tuned-20261106T042738Z-57a9`). Peak RSS was 6.5 GiB per process.

**Inside call_variants the engine is not faster than TF.**
- The standalone executor's +5.5% saturated throughput (`tools/cv_onednn`) disappears.
- tf.data's parse and decode threads share the cores with the OpenMP team, and every oneDNN primitive waits for its slowest thread.
- What remains is lower memory per process: −25% at batch 1024 and −16% at batch 512 against stock at the same batch.
- Stock call_variants at batch 512 already halves its own memory with identical output.
