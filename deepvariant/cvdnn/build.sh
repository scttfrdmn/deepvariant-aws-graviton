#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Build libcvdnn.so (+ the patched oneDNN v2.7.3 libdnnl.so.2) inside the
# DeepVariant image it will run in, then package an unpacked call_variants
# build with the opt-in engine (deepvariant-aws-graviton#11):
#
#   OUT/cv-perf/__main__.py        IsRunningFromZip False (runs unpacked)
#   OUT/cv-perf/runfiles/...       the image's call_variants.zip, call_variants.py
#                                  patched (patch_call_variants.py), plus
#                                  com_google_deepvariant/deepvariant/cvdnn/
#   OUT/cv-perf.tar.zst(.sha256)   the same as a tarball (probe-dv-tuned CV_URI)
#
#   deepvariant/cvdnn/build.sh IMAGE OUT
set -euo pipefail
IMAGE=${1:?image}; OUT=${2:?out dir}
SRC=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
rm -rf "$OUT/cv-perf" "$OUT/lib"; mkdir -p "$OUT/lib"
docker run --rm -u 0:0 -v "$SRC:/src:ro" -v "$OUT:/out" --entrypoint bash "$IMAGE" -c '
set -euo pipefail
apt-get -qq update >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get -qq -y install cmake ninja-build git unzip >/dev/null
git clone -q --depth 1 -b v2.7.3 https://github.com/uxlfoundation/oneDNN.git /tmp/onednn
cd /tmp/onednn && git apply /src/tools/cv_onednn/onednn-v2.7.3-brgconv-channel-slice-dst.patch
cmake -G Ninja -S . -B b -DCMAKE_BUILD_TYPE=Release -DDNNL_CPU_RUNTIME=OMP -DDNNL_BUILD_TESTS=OFF -DDNNL_BUILD_EXAMPLES=OFF >/dev/null
ninja -C b >/dev/null
TFINC=$(python3 -c "import os, tensorflow as tf; print(os.path.join(os.path.dirname(tf.__file__), \"include\"))" 2>/dev/null)
# -mavx without FMA: Eigen'"'"'s BatchNorm rsqrt must match the TF wheel bit-for-bit.
g++ -O2 -mavx -fopenmp -std=c++17 -fPIC -shared -I/tmp/onednn/include -I/tmp/onednn/b/include -I"$TFINC" \
  -o /out/lib/libcvdnn.so /src/deepvariant/cvdnn/cvdnn_lib.cpp -L/tmp/onednn/b/src -ldnnl -Wl,-rpath,"\$ORIGIN"
cp -L /tmp/onednn/b/src/libdnnl.so.2 /out/lib/
# unzip, not zipfile: the zip stores symlinks (e.g. the nucleus .so files)
mkdir -p /out/cv-perf && cd /out/cv-perf && unzip -q /opt/deepvariant/bin/call_variants.zip
sed -i "/^def IsRunningFromZip():/{n;s/return True/return False/}" __main__.py
ln -s runfiles __main__.py.runfiles
D=runfiles/com_google_deepvariant/deepvariant
python3 /src/deepvariant/cvdnn/patch_call_variants.py $D/call_variants.py
mkdir -p $D/cvdnn && cp /src/deepvariant/cvdnn/__init__.py /src/deepvariant/cvdnn/engine.py /out/lib/libcvdnn.so /out/lib/libdnnl.so.2 $D/cvdnn/
chown -R '"$(id -u):$(id -g)"' /out
'
grep -A1 "^def IsRunningFromZip" "$OUT/cv-perf/__main__.py" | grep -q "return False"
(cd "$OUT/cv-perf" && tar c . | zstd -q -T0 -19 -o "$OUT/cv-perf.tar.zst" -f)
(cd "$OUT" && sha256sum cv-perf.tar.zst > cv-perf.tar.zst.sha256)
echo "built $OUT/cv-perf.tar.zst ($(du -h "$OUT/cv-perf.tar.zst" | cut -f1)) from $IMAGE at $(git -C "$SRC" rev-parse --short HEAD)"
