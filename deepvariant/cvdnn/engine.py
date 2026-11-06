# SPDX-License-Identifier: Apache-2.0
"""call_variants inference through libcvdnn (deepvariant-aws-graviton#11).

Opt-in: DV_CV_ENGINE=cvdnn in call_variants. The SavedModel's serving
signature is frozen and walked once to write a spec (Conv2D ->
FusedBatchNormV3 -> Relu chains, pools, ConcatV2, Mean, MatMul + BiasAdd,
Softmax) plus raw float32 weights into a private temporary directory, which
libcvdnn reads at open; the directory is removed afterwards. libcvdnn runs the
same oneDNN v2.7.3 primitives TF 2.13.1 uses, so the probabilities are
bit-for-bit stock's (gate: the CallVariantsOutput multiset).

  eng = Engine(checkpoint_path, batch_size)
  probs = eng(images)   # float32 [n, H, W, C], already (x - 128) / 128; n <= batch_size

libcvdnn.so and its libdnnl.so.2 live next to this file (DV_CVDNN_LIB overrides).
Threads: OpenMP (OMP_NUM_THREADS; set OMP_PROC_BIND/OMP_PLACES only when the
process has its own cores).
"""
import ctypes
import os
import shutil
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import convert_to_constants

_HERE = os.path.dirname(os.path.abspath(__file__))


def _pads(insz, k, s, padding):
  if padding == 'VALID':
    return 0, 0, (insz - k) // s + 1
  out = -(-insz // s)
  tot = max((out - 1) * s + k - insz, 0)
  return tot // 2, tot - tot // 2, out


def write_spec(checkpoint_path, d):
  """Writes spec.txt and weights for the SavedModel at checkpoint_path into d."""
  model = tf.saved_model.load(checkpoint_path)
  sig = model.signatures['serving_default']
  frozen = convert_to_constants.convert_variables_to_constants_v2(sig)
  gd = frozen.graph.as_graph_def()
  ops = {n.name: n for n in gd.node}
  src = lambda i: i.split(':')[0].lstrip('^')

  def real(n):
    while ops[n].op == 'Identity':
      n = src(ops[n].input[0])
    return n

  tname = lambda i: real(src(i)) + ':0'
  consts = {}

  def wfile(i):
    n = real(src(i))
    assert ops[n].op == 'Const', (n, ops[n].op)
    if n not in consts:
      v = tf.make_ndarray(ops[n].attr['value'].tensor).astype(np.float32)
      f = 'w%04d.f32' % len(consts)
      v.tofile(os.path.join(d, f))
      consts[n] = (f, v.shape)
    return consts[n]

  inp = frozen.inputs[0]
  in_name = real(src(inp.name)) + ':0'
  _, h, w, c = inp.shape.as_list()
  shape = {in_name: (h, w, c)}
  out_name = tname(frozen.outputs[0].name)
  lines = [f'input {in_name} 0 {h} {w} {c}']
  for n in gd.node:  # graph order is topological for a frozen function graph
    a = n.attr
    if n.op == 'Relu':
      bn = ops[real(src(n.input[0]))]
      cv = ops[real(src(bn.input[0]))]
      assert bn.op == 'FusedBatchNormV3' and cv.op == 'Conv2D', (n.name, bn.op, cv.op)
      assert cv.attr['data_format'].s == b'NHWC' and list(cv.attr['dilations'].list.i) in ([], [1, 1, 1, 1])
      i = tname(cv.input[0])
      ih, iw, _ = shape[i]
      f, fs = wfile(cv.input[1])
      kh, kw, ic, oc = fs
      sh, sw = list(cv.attr['strides'].list.i)[1:3]
      pad = cv.attr['padding'].s.decode()
      pt, pb, oh = _pads(ih, kh, sh, pad)
      pl, pr, ow = _pads(iw, kw, sw, pad)
      sc, mn, va, of = (wfile(bn.input[k])[0] for k in (1, 3, 4, 2))
      o = n.name + ':0'
      shape[o] = (oh, ow, oc)
      lines.append(f"conv {o} {i} {kh} {kw} {ic} {oc} {sh} {sw} {pt} {pl} {pb} {pr} "
                   f"{bn.attr['epsilon'].f!r} {f} {mn} {va} {sc} {of}")
    elif n.op in ('MaxPool', 'AvgPool'):
      i = tname(n.input[0])
      ih, iw, ic = shape[i]
      kh, kw = list(a['ksize'].list.i)[1:3]
      sh, sw = list(a['strides'].list.i)[1:3]
      pad = a['padding'].s.decode()
      pt, pb, oh = _pads(ih, kh, sh, pad)
      pl, pr, ow = _pads(iw, kw, sw, pad)
      o = n.name + ':0'
      shape[o] = (oh, ow, ic)
      lines.append(f"pool {'max' if n.op == 'MaxPool' else 'avg'} {o} {i} {kh} {kw} {sh} {sw} {pt} {pl} {pb} {pr}")
    elif n.op == 'ConcatV2':
      ins = [tname(i) for i in n.input[:-1]]
      o = n.name + ':0'
      s0 = shape[ins[0]]
      shape[o] = s0[:2] + (sum(shape[i][2] for i in ins),)
      lines.append('concat ' + o + ' ' + ' '.join(ins))
    elif n.op == 'Mean':
      i = tname(n.input[0])
      axes = tf.make_ndarray(ops[real(src(n.input[1]))].attr['value'].tensor).tolist()
      assert sorted(axes) == [1, 2] and not a['keep_dims'].b, (axes, a['keep_dims'].b)
      o = n.name + ':0'
      shape[o] = (1, 1, shape[i][2])
      lines.append(f'gap {o} {i}')
    elif n.op == 'BiasAdd':
      mm = ops[real(src(n.input[0]))]
      assert mm.op == 'MatMul' and not mm.attr['transpose_a'].b and not mm.attr['transpose_b'].b
      i = tname(mm.input[0])
      f, fs = wfile(mm.input[1])
      b, _ = wfile(n.input[1])
      o = n.name + ':0'
      shape[o] = (1, 1, fs[1])
      lines.append(f'fc {o} {i} {fs[0]} {fs[1]} {f} {b}')
    elif n.op == 'Softmax':
      i = tname(n.input[0])
      o = n.name + ':0'
      shape[o] = shape[i]
      lines.append(f'softmax {o} {i}')
    elif n.op in ('Conv2D', 'FusedBatchNormV3', 'MatMul', 'Identity', 'Const', 'Placeholder', 'NoOp'):
      pass  # folded into the ops above
    else:
      raise ValueError(f'cvdnn: unsupported op {n.op} ({n.name})')
  lines.append(f'output {out_name}')
  open(os.path.join(d, 'spec.txt'), 'w').write('\n'.join(lines) + '\n')
  return len(lines) - 2


class Engine:
  """Batched inference; one oneDNN network sized for batch_size examples."""

  def __init__(self, checkpoint_path, batch_size):
    lib = os.environ.get('DV_CVDNN_LIB', os.path.join(_HERE, 'libcvdnn.so'))
    self._lib = ctypes.CDLL(lib)
    self._lib.cvdnn_open.restype = ctypes.c_void_p
    self._lib.cvdnn_open.argtypes = [ctypes.c_char_p, ctypes.c_long, ctypes.c_char_p, ctypes.c_long]
    self._lib.cvdnn_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_void_p]
    self._lib.cvdnn_info.restype = ctypes.c_char_p
    self._lib.cvdnn_info.argtypes = [ctypes.c_void_p]
    self._lib.cvdnn_close.argtypes = [ctypes.c_void_p]
    d = tempfile.mkdtemp(prefix='cvdnn-')
    try:
      self.n_ops = write_spec(checkpoint_path, d)
      err = ctypes.create_string_buffer(1024)
      self._h = self._lib.cvdnn_open(d.encode(), batch_size, err, len(err))
      if not self._h:
        raise RuntimeError('cvdnn_open: ' + err.value.decode())
    finally:
      shutil.rmtree(d, ignore_errors=True)
    self.batch_size = batch_size
    self.info = self._lib.cvdnn_info(self._h).decode()
    self.classes = 3

  def __call__(self, images):
    x = np.ascontiguousarray(images, dtype=np.float32)
    n = x.shape[0]
    out = np.empty((n, self.classes), np.float32)
    for s0 in range(0, n, self.batch_size):  # a batch larger than planned is split
      xb = x[s0:s0 + self.batch_size]
      rc = self._lib.cvdnn_run(self._h, xb.ctypes.data, xb.shape[0], out[s0:].ctypes.data)
      if rc:
        raise RuntimeError(f'cvdnn_run failed ({rc})')
    return out

  def close(self):
    if getattr(self, '_h', None):
      self._lib.cvdnn_close(self._h)
      self._h = None
