"""Write make_examples TFRecord images as concatenated raw HWC bytes, in record order (aws-realign#130).

Usage (DV image): python3 export_tfrecord_images.py 'me.tfrecord*.gz' OUT.bin
"""
import glob
import sys

import tensorflow as tf

files = sorted(glob.glob(sys.argv[1]))
n = 0
with open(sys.argv[2], 'wb') as out:
  for raw in tf.data.TFRecordDataset(files, compression_type='GZIP'):
    f = tf.train.Example.FromString(raw.numpy()).features.feature
    out.write(f['image/encoded'].bytes_list.value[0])
    n += 1
print(f'{n} images, shape {list(f["image/shape"].int64_list.value)}')
