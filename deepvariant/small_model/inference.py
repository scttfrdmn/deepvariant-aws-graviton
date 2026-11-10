# Copyright 2024 Google LLC.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Module for calling variants on examples using a trained keras model."""
from collections.abc import Sequence
import os
import numpy as np
# pylint: disable=g-import-not-at-top
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

from deepvariant.protos import deepvariant_pb2
from deepvariant.small_model import keras_config
from deepvariant.small_model import make_small_model_examples
from third_party.nucleus.util import genomics_math
from third_party.nucleus.util import variant_utils
from third_party.nucleus.util import variantcall_utils


SMALL_MODEL_ID = "small_model"

GENOTYPE_BY_ENCODING = {
    make_small_model_examples.GenotypeEncoding.REF.value: (0, 0),
    make_small_model_examples.GenotypeEncoding.HET.value: (0, 1),
    make_small_model_examples.GenotypeEncoding.HOM_ALT.value: (1, 1),
}


def _dense_stack_forward(stack, batch):
  """Keras' eager Sequential-of-Dense forward pass, op for op."""
  if _RAW_OPS:
    return _dense_stack_forward_raw(stack, batch)
  x = tf.cast(tf.convert_to_tensor(batch), tf.float32)
  for kernel, bias, activation in stack:
    x = tf.matmul(x, kernel)
    x = tf.nn.bias_add(x, bias)
    x = tf.nn.relu(x) if activation == 'relu' else tf.nn.softmax(x, axis=-1)
  return x


# The same eager ops without the Python dispatch layers around them: for rank-2
# float32 inputs tf.cast, tf.matmul, tf.nn.bias_add, tf.nn.relu and
# tf.nn.softmax(axis=-1) each end in exactly these gen_* calls (TF 2.13), so the
# kernels and their inputs are unchanged (deepvariant-aws-graviton#10).
# DV_SMALL_MODEL_RAWOPS=0 keeps the public API calls.
_RAW_OPS = os.environ.get('DV_SMALL_MODEL_RAWOPS') != '0'
from tensorflow.python.ops import gen_math_ops as _gen_math_ops  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
from tensorflow.python.ops import gen_nn_ops as _gen_nn_ops  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


def _dense_stack_forward_raw(stack, batch):
  x = tf.convert_to_tensor(batch)
  if x.dtype != tf.float32:
    x = _gen_math_ops.cast(x, tf.float32, Truncate=False)
  for kernel, bias, activation in stack:
    x = _gen_math_ops.mat_mul(x, kernel, transpose_a=False, transpose_b=False)
    x = _gen_nn_ops.bias_add(x, bias, data_format='NHWC')
    x = _gen_nn_ops.relu(x) if activation == 'relu' else _gen_nn_ops.softmax(x)
  return x

class SmallModelVariantCaller:
  """Class responsible for calling variants on candidate examples.

  It is a stateful object that loads the trained model from `model_path`
  and holds a set of quality predicates that are applied to each call result to
  determine if we can accept to write it as CVO directly or if it should be
  passed to the regular DeepVariant model as a pileup image.
  """
  MAX_CONFIDENCE = 1 - 1e-7

  def __init__(
      self,
      classifier: tf.keras.Model,
      snp_gq_threshold: float,
      indel_gq_threshold: float,
      batch_size: int,
  ):
    self.classifier = classifier
    self.snp_gq_threshold = snp_gq_threshold
    self.indel_gq_threshold = indel_gq_threshold
    self.batch_size = batch_size

  @classmethod
  def from_model_path(
      cls,
      model_path: str,
      snp_gq_threshold: float,
      indel_gq_threshold: float,
      batch_size: int,
  ) -> "SmallModelVariantCaller":
    """Init class with a path to a pickled model."""
    return cls(
        keras_config.load_keras_model(model_path, compile_model=False),
        snp_gq_threshold,
        indel_gq_threshold,
        batch_size,
    )

  def _accept_call_result(
      self, candidate, alt_allele_indices, probability
  ) -> bool:
    """Determine if the given probability is above the GQ threshold."""
    if variant_utils.is_snp(
        candidate.variant,
        exclude_alleles=make_small_model_examples.get_exclude_alleles(
            candidate, alt_allele_indices
        ),
    ):
      threshold = self.snp_gq_threshold
    else:
      threshold = self.indel_gq_threshold
    return (
        genomics_math.ptrue_to_bounded_phred(
            max(probability), self.MAX_CONFIDENCE
        )
        >= threshold
    )

  def _get_call_variant_outputs(
      self,
      candidate,
      genotype_probabilities,
      alt_allele_indices: tuple[int, ...],
  ) -> deepvariant_pb2.CallVariantsOutput:
    """Returns a CallVariantsOutput for the given candidate and prediction."""
    variant = candidate.variant
    variant_call = candidate.variant.calls[0]
    variantcall_utils.set_model_id(variant_call, SMALL_MODEL_ID)
    del variant.calls[:]
    variant.calls.append(variant_call)
    return deepvariant_pb2.CallVariantsOutput(
        variant=variant,
        genotype_probabilities=genotype_probabilities,
        debug_info=None,
        alt_allele_indices=deepvariant_pb2.CallVariantsOutput.AltAlleleIndices(
            indices=alt_allele_indices
        ),
    )

  def _classify_tagged(self, examples: np.ndarray):
    """classify(), with eager float32 outputs as numpy rows, tagged.

    Returns (predictions, eager): eager[i] is True when predictions[i] came
    from self.classifier(...) (an EagerTensor in classify()) and is now a
    float32 numpy row. Values are identical; only the container changes, and
    _eager_row_passes reproduces TF's float32 scalar arithmetic for them.
    (deepvariant-aws-graviton)
    """
    predictions, eager = [], []
    for i in range(0, len(examples), self.batch_size):
      batch = examples[i : i + self.batch_size]
      if len(batch) < self.batch_size:
        stack = self._dense_stack()
        if stack is not None:
          out = _dense_stack_forward(stack, batch)
        else:
          out = self.classifier(batch, training=False)
        if getattr(out, 'dtype', None) == tf.float32 and not os.environ.get(
            'DV_SMALL_MODEL_EAGER'
        ):
          rows = list(out.numpy())
          predictions.extend(rows)
          eager.extend([True] * len(rows))
        else:
          rows = list(out)
          predictions.extend(rows)
          eager.extend([False] * len(rows))
      else:
        rows = list(self.classifier.predict_on_batch(batch))
        predictions.extend(rows)
        eager.extend([False] * len(rows))
    return predictions, eager

  def _dense_stack(self):
    """The classifier as (kernel, bias, activation) per Dense layer, or None.

    For a Sequential of float32 Dense layers with bias, ReLU hidden layers and a
    softmax output, an eager call is: cast to float32, then per layer
    tf.matmul, tf.nn.bias_add and tf.nn.relu / tf.nn.softmax(axis=-1). That is
    what Keras' eager __call__ runs (deepvariant-aws-graviton#10); calling the
    same ops directly skips the per-call Keras machinery. DV_SMALL_MODEL_KERAS=1
    keeps the Keras call.
    """
    if hasattr(self, '_stack_cache'):
      return self._stack_cache
    stack = None
    try:
      layers = list(self.classifier.layers)
      if (
          not os.environ.get('DV_SMALL_MODEL_KERAS')
          and type(self.classifier).__name__ == 'Sequential'
          and layers
          and all(
              type(l).__name__ == 'Dense'
              and l.use_bias
              and l.dtype == 'float32'
              and l.compute_dtype == 'float32'
              for l in layers
          )
          and all(l.activation.__name__ == 'relu' for l in layers[:-1])
          and layers[-1].activation.__name__ == 'softmax'
          and all(len(l.kernel.shape) == 2 for l in layers)
      ):
        stack = [(l.kernel, l.bias, l.activation.__name__) for l in layers]
    except AttributeError:
      stack = None
    self._stack_cache = stack
    return stack

  def _eager_row_passes(self, candidate, alt_allele_indices, row) -> bool:
    """_accept_call_result for a float32 row as if it were an EagerTensor.

    ptrue_to_bounded_phred(max(t), MAX) on an EagerTensor computes:
      max: float32 comparisons, first maximum kept (same on numpy float32);
      min(p, MAX): `MAX < p` dispatches to tf.greater(p, float32(MAX));
      1.0 - p: tf.subtract in float32 (np.float32(1.0) - p here);
      1.0 - MAX: Python float.
    """
    if variant_utils.is_snp(
        candidate.variant,
        exclude_alleles=make_small_model_examples.get_exclude_alleles(
            candidate, alt_allele_indices
        ),
    ):
      threshold = self.snp_gq_threshold
    else:
      threshold = self.indel_gq_threshold
    p = max(row)
    if not 0 <= p <= 1:
      raise ValueError('ptrue must be between zero and one: {}'.format(p))
    if np.float32(self.MAX_CONFIDENCE) < p:
      perror = 1.0 - self.MAX_CONFIDENCE
    else:
      perror = np.float32(1.0) - p
    return genomics_math.perror_to_phred(float(perror)) >= threshold

  def classify(self, examples: np.ndarray) -> list[tuple[float, ...]]:
    """Classifies the given example."""
    predictions = []
    for i in range(0, len(examples), self.batch_size):
      batch = examples[i : i + self.batch_size]
      if len(batch) < self.batch_size:
        predictions.extend(self.classifier(batch, training=False))
      else:
        predictions.extend(self.classifier.predict_on_batch(batch))
    return predictions

  def call_variants(
      self,
      candidates_with_alt_allele_indices: Sequence[
          tuple[deepvariant_pb2.DeepVariantCall, tuple[int, ...]]
      ],
      examples: Sequence[Sequence[int]],
  ) -> tuple[
      list[deepvariant_pb2.CallVariantsOutput],
      list[deepvariant_pb2.DeepVariantCall],
  ]:
    """Calls variants on the given examples."""
    genotype_probabilities, eager = self._classify_tagged(np.array(examples))

    call_variant_outputs = []
    candidates_not_called_by_id = {}
    for (candidate, alt_allele_indices), probability, is_eager in zip(
        candidates_with_alt_allele_indices,
        genotype_probabilities,
        eager,
    ):
      if (
          self._eager_row_passes(candidate, alt_allele_indices, probability)
          if is_eager
          else self._accept_call_result(
              candidate, alt_allele_indices, probability
          )
      ):
        call_variant_outputs.append(
            self._get_call_variant_outputs(
                candidate, probability, alt_allele_indices
            )
        )
      else:
        candidate.make_examples_alt_allele_indices.append(
            deepvariant_pb2.CallVariantsOutput.AltAlleleIndices(
                indices=alt_allele_indices
            )
        )
        candidates_not_called_by_id[id(candidate)] = candidate

    candidates_not_called = list(candidates_not_called_by_id.values())
    return call_variant_outputs, candidates_not_called
