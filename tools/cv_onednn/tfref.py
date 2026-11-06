# SPDX-License-Identifier: Apache-2.0
"""Run the exported placeholder-weight graph in a v1 Session (stock grappler and
oneDNN layout pass), optionally fetching intermediate tensors.

  python tfref.py EXPORT DUMP N [--fetch name:0 ...] [--save OUT.npz] [--threads 16]
Prints whether probs over the first N images equal DUMP/probs_tf213.npy bit-for-bit.
"""
import argparse, json, os, time
import numpy as np
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("export"); ap.add_argument("dump"); ap.add_argument("n", type=int)
ap.add_argument("--fetch", nargs="*", default=[]); ap.add_argument("--save"); ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--start", type=int, default=0)
a = ap.parse_args()
tf.compat.v1.disable_eager_execution()
info = json.load(open(os.path.join(a.dump, "info.json"))); H, W, C = info["shape"]; B = info["batch"]
imgs = np.memmap(os.path.join(a.dump, "images.u8"), dtype=np.uint8, mode="r", shape=(info["n"], H, W, C))
meta = json.load(open(os.path.join(a.export, "ops.json")))
gd = tf.compat.v1.GraphDef(); gd.ParseFromString(open(os.path.join(a.export, "graph.pb"), "rb").read())
wz = np.load(os.path.join(a.export, "weights.npz"))
g = tf.Graph()
with g.as_default():
    tf.import_graph_def(gd, name="")
cfg = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=a.threads)
sess = tf.compat.v1.Session(graph=g, config=cfg)
feed_w = {g.get_tensor_by_name(k.replace("|", "/") + ":0"): wz[k] for k in wz.files}
x_t = g.get_tensor_by_name(meta["inputs"][0]); y_t = g.get_tensor_by_name(meta["outputs"][0])
fetch = [y_t] + [g.get_tensor_by_name(n) for n in a.fetch]
out = np.empty((a.n, 3), np.float32); saved = {n: [] for n in a.fetch}; wall = 0.0
for s0 in range(a.start, a.start + a.n, B):
    x = (imgs[s0:min(s0 + B, a.start + a.n)].astype(np.float32) - np.float32(128.0)) / np.float32(128.0)
    t0 = time.perf_counter(); r = sess.run(fetch, {x_t: x, **feed_w}); wall += time.perf_counter() - t0
    out[s0 - a.start:s0 - a.start + x.shape[0]] = r[0]
    for n, v in zip(a.fetch, r[1:]):
        saved[n].append(v)
ref = np.load(os.path.join(a.dump, "probs_tf213.npy"))[a.start:a.start + a.n]
d = np.any(out != ref, axis=1)
print("probs", "IDENTICAL" if not d.any() else f"DIFFER rows={int(d.sum())} maxabs={float(np.abs(out - ref).max()):.3g}", f"wall={wall:.2f}s")
if a.save:
    np.savez(a.save, probs=out, **{n.replace("/", "|"): np.concatenate(v) for n, v in saved.items()})
