#!/bin/bash
# Copyright 2017 Google LLC.
# Modifications Copyright 2024 deepvariant-linux-arm64 contributors.
#
# ARM64-specific settings for DeepVariant build.
# Sources the upstream settings.sh and overrides x86-specific values.

source "$(dirname "$0")/settings.sh"

# Detect architecture
ARCH=$(uname -m)

if [[ "${ARCH}" == "aarch64" ]]; then
  echo "========== Configuring for ARM64 (aarch64) build"

  # Override x86-specific CUDNN path
  export CUDNN_INSTALL_PATH="/usr/lib/aarch64-linux-gnu"

  # Remove x86-specific compiler flags (-march=corei7) and add ARM64 flags
  # deepvariant-aws-graviton perf/arm64-exact: Graviton4/5 codegen with no FMA contraction, so float
# results keep the same rounding as the x86 build (-march=x86-64-v4 -ffp-contract=off). DV_ARM_MCPU
# overrides the CPU target (e.g. armv8.4-a for a portable build).
  export DV_COPT_FLAGS="--copt=-mcpu=${DV_ARM_MCPU:-neoverse-v2} --copt=-ffp-contract=off --copt=-Wno-sign-compare --copt=-Wno-write-strings --experimental_build_setting_api --java_runtime_version=remotejdk_11"

  # Bazel output directory on aarch64 uses "aarch64-opt" instead of "k8-opt"
  export DV_BAZEL_OUTPUT_DIR="aarch64-opt"

  # TensorFlow OneDNN+ACL optimizations for ARM64
  export TF_ENABLE_ONEDNN_OPTS=1

  # Enable BF16 fast math on Graviton3+ (Neoverse V1/V2 with BF16 support)
  if [[ -f /proc/cpuinfo ]] && grep -q bf16 /proc/cpuinfo; then
    echo "========== BF16 support detected, enabling BF16 fast math"
    : # perf/arm64-exact: never force BF16 (changes small-model outputs; see #1)
  fi

  # Thread configuration for optimal ARM64 inference
  export OMP_NUM_THREADS=$(nproc)
  export OMP_PROC_BIND=false
  export OMP_PLACES=cores
else
  echo "========== Architecture ${ARCH} detected, using default x86 settings"
  export DV_BAZEL_OUTPUT_DIR="k8-opt"
fi
