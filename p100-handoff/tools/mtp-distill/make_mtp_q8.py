# make_mtp_q8.py (task 1b): a copy of the production GGUF whose MTP-block tensors that are below Q8_0 (blk.64 attn_q,
# attn_output, ffn_gate/up/down, nextn.eh_proj: Q6_K) are re-quantized to Q8_0 from the BF16 source (unsloth's BF16 GGUF,
# fetched tensor by tensor with gguf_remote.py into src-bf16/). Every other tensor and all metadata are copied byte for
# byte. Re-quantizing from the Q6_K data itself would add no information, hence the BF16 source.
#   python3 make_mtp_q8.py check            source lineage + quantizer checks only
#   python3 make_mtp_q8.py write <out.gguf> check, then write
import hashlib, json, os, sys
import numpy as np
sys.path.insert(0, "/work/src/llama.cpp/gguf-py")
import gguf
from gguf import GGMLQuantizationType as QT

PROD = "/work/models/Qwen3.8-27B-UD-Q6_K.gguf"
SRC = "/work/bench/mtp5/src-bf16"
UPGRADE = ["blk.64.attn_q.weight", "blk.64.attn_output.weight", "blk.64.ffn_gate.weight", "blk.64.ffn_up.weight",
           "blk.64.ffn_down.weight", "blk.64.nextn.eh_proj.weight"]


def src_f32(name, infos):
    t = infos[name]
    raw = np.fromfile(os.path.join(SRC, name + ".bin"), np.uint8)
    dims = t["dims"]
    if t["type"] == 30:  # BF16
        a = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif t["type"] == 0:
        a = raw.view(np.float32)
    else:
        raise ValueError(t)
    return a.reshape(list(reversed(dims)))


def check(r, infos):
    T = {t.name: t for t in r.tensors}
    ok = True
    for name in infos:
        t = T[name]; s = src_f32(name, infos)
        if t.tensor_type == QT.F32:
            same = np.array_equal(np.asarray(t.data).reshape(s.shape), s)
            print(f"{name}: F32 in both, identical {same}"); ok &= same
        elif t.tensor_type == QT.Q8_0:
            q = gguf.quants.quantize(s, QT.Q8_0)
            same = np.array_equal(np.asarray(t.data).reshape(-1), q.reshape(-1))
            print(f"{name}: production Q8_0 == gguf-py Q8_0 of the BF16 source: {same}"); ok &= same
        else:
            d = gguf.quants.dequantize(np.asarray(t.data), t.tensor_type).reshape(s.shape)
            rel = np.sqrt(((d - s) ** 2).mean() / (s ** 2).mean())
            q = gguf.quants.quantize(s, QT.Q8_0)
            d8 = gguf.quants.dequantize(q, QT.Q8_0).reshape(s.shape)
            rel8 = np.sqrt(((d8 - s) ** 2).mean() / (s ** 2).mean())
            print(f"{name}: production {t.tensor_type.name} rel. RMS error vs BF16 {rel:.4%}; Q8_0 {rel8:.4%}")
            ok &= rel < 0.05
    return ok


def write(r, infos, out):
    w = gguf.GGUFWriter(out, arch=r.get_field(gguf.Keys.General.ARCHITECTURE).contents(), endianess=r.endianess)
    for field in r.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(field.name, field.contents(), vt, sub_type=st)
    new = {}
    for name in UPGRADE:
        new[name] = gguf.quants.quantize(src_f32(name, infos), QT.Q8_0)
    for t in r.tensors:
        if t.name in new:
            q = new[t.name]
            w.add_tensor_info(t.name, q.shape, q.dtype, q.nbytes, QT.Q8_0)
        else:
            w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for t in r.tensors:
        w.write_tensor_data(new[t.name] if t.name in new else t.data, tensor_endianess=r.endianess)
    w.close()


def write_f32test(r, infos, out):
    """parity test file (not for serving): every 2D MTP-block tensor F32 from the BF16 source and the LM head F32
    (its Q8_0 values, dequantized), so llama.cpp's CPU path computes the head without quantized activations."""
    w = gguf.GGUFWriter(out, arch=r.get_field(gguf.Keys.General.ARCHITECTURE).contents(), endianess=r.endianess)
    for field in r.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(field.name, field.contents(), vt, sub_type=st)
    T = {t.name: t for t in r.tensors}
    new = {n: src_f32(n, infos) for n in infos if len(infos[n]["dims"]) == 2}
    lm = T["output.weight"]
    for t in r.tensors:
        if t.name in new:
            a = new[t.name]; w.add_tensor_info(t.name, a.shape, a.dtype, a.nbytes, QT.F32)
        elif t.name == "output.weight":
            shp = (int(lm.shape[1]), int(lm.shape[0]))
            w.add_tensor_info(t.name, shp, np.dtype(np.float32), shp[0] * shp[1] * 4, QT.F32)
        else:
            w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for t in r.tensors:
        if t.name in new:
            w.write_tensor_data(new[t.name])
        elif t.name == "output.weight":
            raw = np.asarray(t.data)
            full = np.empty((raw.shape[0], int(lm.shape[0])), np.float32)
            for i in range(0, raw.shape[0], 8192):
                full[i:i + 8192] = gguf.quants.dequantize(raw[i:i + 8192], t.tensor_type)
            w.write_tensor_data(full)
            del full
        else:
            w.write_tensor_data(t.data, tensor_endianess=r.endianess)
    w.close()


def tensor_hashes(path):
    r = gguf.GGUFReader(path)
    return {t.name: (t.tensor_type.name, hashlib.sha256(np.asarray(t.data).tobytes()).hexdigest()) for t in r.tensors}, r


if __name__ == "__main__":
    infos = {t["name"]: t for t in json.load(open(os.path.join(SRC, "infos.json")))["tensors"]}
    r = gguf.GGUFReader(PROD)
    ok = check(r, infos)
    print("checks", "PASS" if ok else "FAIL")
    if sys.argv[1] == "write-f32test" and ok:
        write_f32test(r, infos, sys.argv[2])
    if sys.argv[1] == "write" and ok:
        out = sys.argv[2]
        write(r, infos, out)
        h0, _ = tensor_hashes(PROD)
        h1, r1 = tensor_hashes(out)
        same = [n for n in h0 if h0[n] == h1.get(n)]
        diff = [n for n in h0 if h0[n] != h1.get(n)]
        print(f"tensors: {len(h0)} in production, {len(h1)} in the copy; byte-identical {len(same)}; changed {diff}")
        assert sorted(diff) == sorted(UPGRADE) and len(h0) == len(h1)
        kv0 = {f.name: f.contents() for f in r.fields.values() if not f.name.startswith("GGUF.")}
        kv1 = {f.name: f.contents() for f in r1.fields.values() if not f.name.startswith("GGUF.")}
        print("metadata identical:", kv0 == kv1)
        json.dump(dict(production=h0, copy=h1), open(out + ".tensor-sha256.json", "w"), indent=0)
