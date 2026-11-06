# SPDX-License-Identifier: Apache-2.0
"""Write a line spec (plus raw float32 weights) for a span of the exported graph,
and TF's own input/output tensors for it, for the standalone oneDNN executor.

The span runs from the tensor IN (fed) to the tensor OUT (fetched); every Conv2D
-> FusedBatchNormV3 -> Relu chain, pool and ConcatV2 between them is emitted in
graph order. TF reference: v1 Session over export.py's placeholder-weight graph,
feeding IN and fetching OUT (stock grappler + oneDNN layout pass).

  python blockspec.py EXPORT DUMP N IN OUT DIR
Spec lines (NHWC tensors, names are graph tensor names):
  input NAME N H W C
  conv OUT IN KH KW IC OC SH SW PT PL PB PR EPS FILTER MEAN VAR SCALE OFFSET
  pool max|avg OUT IN KH KW SH SW PT PL PB PR
  concat OUT IN1 IN2 ...
  gap OUT IN                       (Mean over H, W; Eigen sum then divide)
  fc OUT IN K M WEIGHT BIAS        (MatMul [K, M] + BiasAdd)
  softmax OUT IN
  output NAME
"""
import json, os, sys, time
import numpy as np
import tensorflow as tf

exp, dump, N, IN, OUT, d = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
os.makedirs(d, exist_ok=True)
meta = json.load(open(os.path.join(exp, "ops.json"))); ops = {o["name"]: o for o in meta["ops"]}
wz = np.load(os.path.join(exp, "weights.npz"))
src = lambda i: i.split(":")[0].lstrip("^")


def real(n):  # skip Identity chains
    while ops[n]["op"] == "Identity":
        n = src(ops[n]["inputs"][0])
    return n


def tname(n):
    return real(src(n)) + ":0"


def wfile(n):
    k = real(src(n)).replace("/", "|"); f = os.path.join(d, k.replace("|", "_") + ".f32")
    if not os.path.exists(f):
        wz[k].astype(np.float32).tofile(f)
    return os.path.basename(f), wz[k].shape


def pads(insz, k, s, padding):
    if padding == "VALID":
        return 0, 0, (insz - k) // s + 1
    out = -(-insz // s); tot = max((out - 1) * s + k - insz, 0)
    return tot // 2, tot - tot // 2, out


tf.compat.v1.disable_eager_execution()
gd = tf.compat.v1.GraphDef(); gd.ParseFromString(open(os.path.join(exp, "graph.pb"), "rb").read())
g = tf.Graph()
with g.as_default():
    tf.import_graph_def(gd, name="")
sess = tf.compat.v1.Session(graph=g, config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=16))
feed_w = {g.get_tensor_by_name(k.replace("|", "/") + ":0"): wz[k] for k in wz.files}
info = json.load(open(os.path.join(dump, "info.json"))); H, W, C = info["shape"]
imgs = np.memmap(os.path.join(dump, "images.u8"), dtype=np.uint8, mode="r", shape=(info["n"], H, W, C))
x = (imgs[:N].astype(np.float32) - np.float32(128.0)) / np.float32(128.0)
xin = sess.run(g.get_tensor_by_name(IN), {g.get_tensor_by_name(meta["inputs"][0]): x, **feed_w})
t_in = g.get_tensor_by_name(IN); t_out = g.get_tensor_by_name(OUT)
ref = sess.run(t_out, {t_in: xin, **feed_w})
ts = []
for _ in range(5):
    t0 = time.perf_counter(); sess.run(t_out, {t_in: xin, **feed_w}); ts.append(time.perf_counter() - t0)
xin.tofile(os.path.join(d, "in.f32")); ref.tofile(os.path.join(d, "ref.f32"))

# Emit the span in graph order: nodes reachable backwards from OUT, stopping at IN.
need, stack = set(), [real(src(OUT))]
stop = real(src(IN))
while stack:
    n = stack.pop()
    if n in need or n == stop:
        continue
    need.add(n); o = ops[n]
    if o["op"] in ("Placeholder", "Const"):
        continue
    stack += [real(src(i)) for i in o["inputs"]]
shape = {IN: tuple(xin.shape)}
L = [f"input {IN} " + " ".join(map(str, xin.shape))]
for o in meta["ops"]:
    n = o["name"]
    if n not in need:
        continue
    if o["op"] == "Relu":
        bn = ops[real(src(o["inputs"][0]))]; cv = ops[real(src(bn["inputs"][0]))]
        assert bn["op"] == "FusedBatchNormV3" and cv["op"] == "Conv2D", (n, bn["op"], cv["op"])
        i = tname(cv["inputs"][0]); Nn, h, w, c = shape[i]
        f, fs = wfile(cv["inputs"][1]); kh, kw, ic, oc = fs; sh, sw = cv["attrs"]["strides"][1:3]
        pt, pb, oh = pads(h, kh, sh, cv["attrs"]["padding"]); pl, pr, ow = pads(w, kw, sw, cv["attrs"]["padding"])
        sc, mn, va, of = (wfile(bn["inputs"][k])[0] for k in (1, 3, 4, 2))
        o_ = n + ":0"; shape[o_] = (Nn, oh, ow, oc)
        L.append(f"conv {o_} {i} {kh} {kw} {ic} {oc} {sh} {sw} {pt} {pl} {pb} {pr} {bn['attrs']['epsilon']!r} {f} {mn} {va} {sc} {of}")
    elif o["op"] in ("MaxPool", "AvgPool"):
        i = tname(o["inputs"][0]); Nn, h, w, c = shape[i]
        kh, kw = o["attrs"]["ksize"][1:3]; sh, sw = o["attrs"]["strides"][1:3]
        pt, pb, oh = pads(h, kh, sh, o["attrs"]["padding"]); pl, pr, ow = pads(w, kw, sw, o["attrs"]["padding"])
        o_ = n + ":0"; shape[o_] = (Nn, oh, ow, c)
        L.append(f"pool {'max' if o['op'] == 'MaxPool' else 'avg'} {o_} {i} {kh} {kw} {sh} {sw} {pt} {pl} {pb} {pr}")
    elif o["op"] == "Mean":
        i = tname(o["inputs"][0]); Nn, h, w, c = shape[i]; o_ = n + ":0"; shape[o_] = (Nn, c)
        L.append(f"gap {o_} {i}")
    elif o["op"] == "BiasAdd":
        mm = ops[real(src(o["inputs"][0]))]; assert mm["op"] == "MatMul", mm["op"]
        assert not mm["attrs"]["transpose_a"] and not mm["attrs"]["transpose_b"]
        i = tname(mm["inputs"][0]); f, fs = wfile(mm["inputs"][1]); b, _ = wfile(o["inputs"][1])
        o_ = n + ":0"; shape[o_] = (shape[i][0], fs[1])
        L.append(f"fc {o_} {i} {fs[0]} {fs[1]} {f} {b}")
    elif o["op"] == "Softmax":
        i = tname(o["inputs"][0]); o_ = n + ":0"; shape[o_] = shape[i]
        L.append(f"softmax {o_} {i}")
    elif o["op"] == "ConcatV2":
        ins = [tname(i) for i in o["inputs"][:-1]]
        o_ = n + ":0"; s0 = shape[ins[0]]; shape[o_] = s0[:3] + (sum(shape[i][3] for i in ins),)
        L.append("concat " + o_ + " " + " ".join(ins))
L.append(f"output {OUT}")
open(os.path.join(d, "spec.txt"), "w").write("\n".join(L) + "\n")
assert shape[OUT] == tuple(ref.shape), (shape[OUT], ref.shape)
print(f"span ops={len(L) - 2} in={xin.shape} out={ref.shape} tf_span_wall_s={sorted(ts)[2]:.4f} (median of 5, intra 16)")
