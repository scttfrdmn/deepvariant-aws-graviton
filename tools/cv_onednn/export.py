# SPDX-License-Identifier: Apache-2.0
"""Export the call_variants SavedModel as a placeholder-weight v1 graph plus weights.

Freezes the serving signature, then turns every weight Const back into a
Placeholder so the oneDNN layout pass sees non-constant filters, exactly as
with the stock ReadVariableOp graph (no filter cache, matmul on gemm:jit).
Writes OUT/graph.pb (GraphDef), OUT/weights.npz and OUT/ops.json (the node
list in graph order with attrs needed by the standalone executor).

  python export.py SAVEDMODEL OUT
"""
import json, os, sys
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

sm, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
m = tf.saved_model.load(sm)  # keep the object alive: the signature references its variables
f = m.signatures["serving_default"]
frozen = convert_variables_to_constants_v2(f)
gd = frozen.graph.as_graph_def()
weights = {}
for n in gd.node:
    if n.op == "Const":
        v = tf.make_ndarray(n.attr["value"].tensor)
        if v.size > 1:  # weights; keep scalars/shape constants (axis, epsilon) as Const
            weights[n.name] = v
            dt = n.attr["dtype"].type
            n.op = "Placeholder"
            for k in list(n.attr.keys()):
                del n.attr[k]
            n.attr["dtype"].type = dt
            n.attr["shape"].shape.CopyFrom(tf.TensorShape(v.shape).as_proto())
open(os.path.join(out, "graph.pb"), "wb").write(gd.SerializeToString())
np.savez(os.path.join(out, "weights.npz"), **{k.replace("/", "|"): v for k, v in weights.items()})


def attrs(n):
    a = {}
    for k, v in n.attr.items():
        w = v.WhichOneof("value")
        if w == "list":
            a[k] = list(v.list.i) or [s.decode() for s in v.list.s]
        elif w == "s":
            a[k] = v.s.decode()
        elif w in ("i", "f", "b"):
            a[k] = getattr(v, w)
        elif w == "type":
            a[k] = int(v.type)
    return a


ops = [dict(name=n.name, op=n.op, inputs=list(n.input), attrs=attrs(n)) for n in gd.node]
json.dump(dict(inputs=[t.name for t in frozen.inputs], outputs=[t.name for t in frozen.outputs],
               ops=ops), open(os.path.join(out, "ops.json"), "w"), indent=0)
from collections import Counter
print(Counter(n.op for n in gd.node).most_common(), "weights", len(weights),
      "inputs", [t.name for t in frozen.inputs], "outputs", [t.name for t in frozen.outputs])
