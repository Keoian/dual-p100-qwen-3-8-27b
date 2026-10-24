# vocab.py: dump the GGUF tokenizer's token strings (read-only, metadata only) to cache/vocab.json
import sys, json, os
sys.path.insert(0, "/work/src/llama.cpp/gguf-py")
from gguf.gguf_reader import GGUFReader
r = GGUFReader("/work/models/Qwen3.8-27B-UD-Q6_K.gguf", "r")
f = r.fields["tokenizer.ggml.tokens"]
toks = [bytes(f.parts[i]).decode("utf-8", "replace") for i in f.data]
os.makedirs(os.path.join(os.path.dirname(__file__), "cache"), exist_ok=True)
json.dump(toks, open(os.path.join(os.path.dirname(__file__), "cache", "vocab.json"), "w"))
print(len(toks), toks[:5], toks[248046], toks[248068] if len(toks) > 248068 else None)
