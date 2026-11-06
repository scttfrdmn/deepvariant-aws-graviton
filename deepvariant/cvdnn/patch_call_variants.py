# SPDX-License-Identifier: Apache-2.0
"""Add the opt-in libcvdnn engine to a call_variants.py (DeepVariant 1.9.0 or this branch).

  python3 patch_call_variants.py PATH/TO/deepvariant/call_variants.py

DV_CV_ENGINE=cvdnn: inference through deepvariant/cvdnn (engine.py + libcvdnn.so);
reading examples, preprocessing and writing CallVariantsOutput stay stock.
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


sub("""  logging.info('example_shape: %s', example_shape)
""", """  logging.info('example_shape: %s', example_shape)
  cvdnn_engine = None
  if use_saved_model and os.environ.get('DV_CV_ENGINE') == 'cvdnn':
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
