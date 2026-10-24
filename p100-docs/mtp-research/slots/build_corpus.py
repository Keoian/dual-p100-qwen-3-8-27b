# build_corpus.py: the frozen corpus behind the long-context slot library (corpus.txt, ~1.9 MB, ~588k tokens).
# Run from the repo root. The slots were built from the files as of commit 9acfc08e2 plus /work/bench/kld-text.txt
# (corpus sha256 prefix 3370d9b38137f157, recorded in manifest.json); later edits to these files change the corpus,
# so rebuild the slots after regenerating it. Writes corpus-new.txt (never the frozen corpus.txt) and prints
# its sha256 prefix to compare with the manifest. Check out 9acfc08e2 first to reproduce the original.
import hashlib, os
files = ["/work/bench/kld-text.txt", "p100-docs/QUICKSTART.md", "p100-docs/FINDINGS.md", "p100-docs/CHANGES.md",
         "tools/server/README.md", "docs/build.md", "docs/speculative.md", "docs/function-calling.md", "OPTLOG.md",
         "src/llama-context.cpp", "src/llama-kv-cache.cpp", "src/llama-graph.cpp", "common/sampling.cpp",
         "common/speculative.cpp", "ggml/src/ggml-cuda/mmvq.cu", "ggml/src/ggml-cuda/fattn-q4p.cuh"]
out = []
for f in files:
    t = open(f, errors="replace").read()
    out.append(f"\n\n===== {os.path.basename(f)} =====\n\n{t}")
s = "".join(out)
open("/work/bench/slots/corpus-new.txt", "w").write(s)
print(len(s), "chars, sha256", hashlib.sha256(s.encode()).hexdigest()[:16])
