# weights.py: the MTP head's tensors from a GGUF, dequantized exactly as ggml stores them (gguf-py's dequantizers), with
# optional per-tensor overrides (e.g. the BF16 source in src-bf16/, or a trained head). numpy (out, in) = torch Linear.
import json, os, sys
import numpy as np
sys.path.insert(0, "/work/src/llama.cpp/gguf-py")
import gguf
from .spec import SPECS, hparams_from_gguf


class GGUFHead:
    def __init__(self, path, spec=None):
        self.r = gguf.GGUFReader(path)
        arch = self.r.fields["general.architecture"].contents()
        self.spec = spec or SPECS[arch]
        self.hp = hparams_from_gguf(self.r, self.spec)
        self.T = {t.name: t for t in self.r.tensors}
        self.prefix = self.spec.block.format(L=self.hp.L)

    def tensor(self, name, rows=None, dtype=np.float32):
        """dequantized (ne1, ne0) array; rows = optional row indices (embedding lookups)."""
        t = self.T[name]
        raw = np.asarray(t.data)
        if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            a = raw.reshape(list(reversed([int(x) for x in t.shape])))
            return (a[rows] if rows is not None else a).astype(dtype)
        ne0 = int(t.shape[0])
        raw = raw.reshape(-1, raw.shape[-1]) if raw.ndim > 1 else raw.reshape(int(np.prod(t.shape[1:])), -1)
        if rows is not None:
            raw = raw[rows]
        out = np.empty((raw.shape[0], ne0), dtype)
        step = 4096
        for i in range(0, raw.shape[0], step):
            out[i:i + step] = gguf.quants.dequantize(raw[i:i + step], t.tensor_type).reshape(-1, ne0)
        return out

    def type_of(self, name):
        return self.T[name].tensor_type.name

    def block_types(self):
        return {full[len(self.prefix):]: t.tensor_type.name for full, t in self.T.items() if full.startswith(self.prefix)}

    def block(self, overrides=None):
        """dict of the block's tensors (relative names), with overrides {relative name: array} applied."""
        out = {}
        for full, t in self.T.items():
            if full.startswith(self.prefix):
                rel = full[len(self.prefix):]
                out[rel] = self.tensor(full)
        for k, v in (overrides or {}).items():
            assert k in out and out[k].shape == v.shape, k
            out[k] = v.astype(np.float32)
        return out


def bf16_overrides(src_dir, prefix):
    """{relative name: f32 array} from gguf_remote.py's raw tensor dump (BF16 / F32)."""
    infos = json.load(open(os.path.join(src_dir, "infos.json")))["tensors"]
    out = {}
    for t in infos:
        raw = np.fromfile(os.path.join(src_dir, t["name"] + ".bin"), np.uint8)
        a = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32) if t["type"] == 30 else raw.view(np.float32)
        out[t["name"][len(prefix):]] = a.reshape(list(reversed(t["dims"])))
    return out
