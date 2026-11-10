# Fused GPU hand-off (aws-realign#130): turn fused_call_variants' output into
# CallVariantsOutput records with stock call_variants' own writer path
# (round_gls, _create_cvo_proto, tfrecord.Writer GZIP), so the records are
# what call_variants would write for the same float32 probabilities.
#
# Run with DeepVariant's runfiles on PYTHONPATH (e.g. an unpacked
# call_variants.zip), as postprocess_variants' input producer:
#   python3 fused_cvo_writer.py preds.bin cvo.tfrecord.gz
import struct
import sys

import numpy as np

from deepvariant import call_variants
from third_party.nucleus.io import tfrecord


def main(src, dst):
  writer = tfrecord.Writer(dst, compression_type='GZIP')
  n = 0
  with open(src, 'rb') as f:
    data = f.read()
  p, end = 0, len(data)
  while p < end:
    (lv,) = struct.unpack_from('<I', data, p); p += 4
    v = data[p:p + lv]; p += lv
    (la,) = struct.unpack_from('<I', data, p); p += 4
    a = data[p:p + la]; p += la
    probs = np.frombuffer(data, dtype=np.float32, count=3, offset=p).copy(); p += 12
    pred = {'probabilities': probs, 'variant': v, 'alt_allele_indices': a,
            'image_encoded': b'', 'layer_outputs_encoded': {}}
    call_variants.write_variant_call(writer, pred, False, False)
    n += 1
  writer.close()
  print(f'fused_cvo_writer: {n} CallVariantsOutput records to {dst}')


if __name__ == '__main__':
  main(sys.argv[1], sys.argv[2])
