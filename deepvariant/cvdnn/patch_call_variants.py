# SPDX-License-Identifier: Apache-2.0
"""Add the opt-in libcvdnn engine to a call_variants.py (DeepVariant 1.9.0 or this branch).

  python3 patch_call_variants.py PATH/TO/deepvariant/call_variants.py

DV_CV_ENGINE=cvdnn: inference through deepvariant/cvdnn (engine.py + libcvdnn.so);
reading examples and writing CallVariantsOutput stay stock; the (x - 128) / 128
preprocessing moves into libcvdnn (same IEEE arithmetic, 4x less data per batch).
Also logs the main process's peak RSS at the end (either engine).
"""
import sys

p = sys.argv[1]
s = open(p).read()
if 'DV_CV_ENGINE' in s:
  sys.exit(f'{p}: already patched')


def sub(old, new):
  global s
  assert s.count(old) == 1, ('anchor', old[:60])
  s = s.replace(old, new)


sub("""    image = tf.reshape(image, example_shape)
    image = tf.cast(image, tf.float32)
    image = dv_utils.preprocess_images(image, channel_indices)
    variant = parsed_features['variant/encoded']
""", """    image = tf.reshape(image, example_shape)
    if not cvdnn_u8:  # DV_CV_ENGINE=cvdnn preprocesses the raw bytes itself
      image = tf.cast(image, tf.float32)
      image = dv_utils.preprocess_images(image, channel_indices)
    variant = parsed_features['variant/encoded']
""")
sub("""  def _parse_example(example):
    \"\"\"Parses a serialized tf.Example.\"\"\"
""", """  cvdnn_u8 = (os.environ.get('DV_CV_ENGINE') == 'cvdnn' and not include_debug_info
              and not channel_indices)

  def _parse_example(example):
    \"\"\"Parses a serialized tf.Example.\"\"\"
""")
# The file dataset's tail: with DV_CV_ENGINE=cvdnn, tf.data autotuning is off. Its
# model thread waits up to 60 s between optimizations, and with inference outside
# TF the end of iteration then stalled 18-60 s in 2 of 6 chr20 runs (0 of 6 with
# autotune off). Parallelism AUTOTUNE then means the process's CPU count.
old_tail = """    enc_image_variant_alt_allele_ds = enc_image_variant_alt_allele_ds.batch(
        batch_size=batch_size
    ).prefetch(tf.data.AUTOTUNE)
    return enc_image_variant_alt_allele_ds
"""
assert s.count(old_tail) == 2, 'dataset tail anchor'
i = s.rindex(old_tail)
s = s[:i] + old_tail.replace("""    return enc_image_variant_alt_allele_ds
""", """    if cvdnn_u8:
      options = tf.data.Options()
      options.autotune.enabled = False
      enc_image_variant_alt_allele_ds = enc_image_variant_alt_allele_ds.with_options(options)
    return enc_image_variant_alt_allele_ds
""") + s[i + len(old_tail):]
sub("""  logging.info('example_shape: %s', example_shape)
""", """  logging.info('example_shape: %s', example_shape)
  cvdnn_engine = None
  if os.environ.get('DV_CV_ENGINE') == 'cvdnn':
    if not use_saved_model or include_debug_info or channel_indices or _STREAM_EXAMPLES.value:
      raise ValueError('DV_CV_ENGINE=cvdnn needs a SavedModel, no debug info, '
                       'no channel ablation and examples from files')
    # Same oneDNN kernels as TF, without TF's runtime (deepvariant-aws-graviton#11).
    from deepvariant.cvdnn import engine as cvdnn_lib  # pylint: disable=g-import-not-at-top
    cvdnn_engine = cvdnn_lib.Engine(checkpoint_path, batch_size)
    logging.info('call_variants: DV_CV_ENGINE=cvdnn (%d ops): %s', cvdnn_engine.n_ops, cvdnn_engine.info)
""")
sub("""    # specifically the `_parse_example` function within it.
    if use_saved_model:
""", """    # specifically the `_parse_example` function within it.
    if cvdnn_engine is not None:
      predictions = cvdnn_engine(images_in_batch.numpy())
    elif use_saved_model:
""")
sub("""  # Put none values in each queue so the running processes can terminate.
  for queue in writer_queues:
""", """  import resource  # pylint: disable=g-import-not-at-top
  logging.info('call_variants: peak RSS %.2f GB (main process)',
               resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
  # Put none values in each queue so the running processes can terminate.
  for queue in writer_queues:
""")
open(p, 'w').write(s)
print(f'{p}: patched')
