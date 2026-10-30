# mtp-distill: the MTP draft head in PyTorch, its parity checks, and a distillation trainer

Tooling behind [p100-docs/MTP-DRAFT.md](../../../p100-docs/MTP-DRAFT.md) (2026-10-29/30). It reads dumps from the
server's `LLAMA_MTP_DUMP` hook (CHANGES §25) and the production GGUF. Written for the reference box's layout:
results, dumps and models live under `/work/bench/mtp5/` there, and a few scripts import the earlier MTP research
code from `/work/bench/mtp3/analysis` and `/work/bench/mtp4` (branch `mtp-adaptive-research`,
`p100-docs/mtp-research/analysis/`). Python 3.10, numpy 2, torch 2.6 with CUDA 12.4 (supports sm_60), and the repo's
`gguf-py`.

| file | what it does |
|---|---|
| `mtp/spec.py` | **everything model-specific** (tensor names, hparam keys, structure flags, trainable set, Hadamard sizes) as one `HeadSpec`; a new MTP model needs a new entry here |
| `mtp/weights.py` | the head's tensors from a GGUF, dequantized as ggml stores them, with per-tensor overrides (a BF16 source, a trained head) |
| `mtp/head.py` | `MTPHead`: the head as `src/models/qwen35.cpp` graph_mtp computes it; emulates the q4_0 / q8_0 Hadamard-rotated draft KV and the activation quantization of ggml-cpu (Q8_K / Q8_0) or CUDA (q8_1); batched draft rows for depths 2-4 |
| `mtp/dump.py` | reader for `LLAMA_MTP_DUMP` files (record layout in its header and in `common/speculative.cpp`); offsets in Python ints (numpy 2 int32 offsets wrap past 2 GiB) |
| `mtp/data.py` | `compact(dump, out_dir, sampling)`: a dump to one compact `.npz` shard per request (h fp16, the target's top-64 logits and lse, presence window, draft cycles with their teacher rows): ~10.8 KiB per position |
| `parity.py` | the torch head against llama.cpp's own draft logits, teacher-forced or chained (`--kv`, `--emulate cpu\|gpu`, `--draft-vocab`, `--draft-type`) |
| `ovl_eval.py` | OFFLINE, paired: exact-acceptance overlap sum(min(p, q)) of draft variants on the same logged rows |
| `train.py` | distillation (TV + 0.1 KL against the served distribution; depth 1 teacher-forced, depths 2-4 on logged rollouts with the head's own chained state). **Smoke-tested only** (`--smoke`); never trained |
| `make_mtp_q8.py`, `gguf_remote.py` | a new GGUF whose MTP block is Q8_0 from the BF16 source (range-fetched per tensor), every other tensor byte-identical (hash-checked), lineage checks first |
| `parity_sim.py`, `vocab_sim.py`, `hiids.py` | SIMULATIONS on the earlier counterfactual traces: sampler parity and the draft vocabulary's reach |
| `run_suite.py`, `analyze_k1.py`, `vram_depth.py` | the 72-prompt harness (with `LLAMA_SPEC_LOG` offsets), window K1's analysis, VRAM at production depth |
| `windowK1.sh`, `windowK1b.sh`, `chainK1.sh`, `srv/` | the production-downtime windows that measured MTP-DRAFT.md's knob table, and their server scripts. **They stop production** (`pkill -x llama-server`, after saving the chat), run test servers on :8090 bound to 127.0.0.1 with a temporary key, pause to cool at 70 C, and restore production and the chat from an EXIT trap |

Typical use (inside a GPU window, production stopped; Python from the box's venv, `/work/venv-mtp5/bin/python`):

```bash
# a test server in production's configuration with the dump on. BUILD must be a build of this branch from the
# hooks commit on (production's build-vision predates it and silently ignores LLAMA_MTP_DUMP); the srv/ scripts
# serve on :8090. Under -sm tensor the draft samples on the CPU, as the dump requires (elsewhere pass
# --no-spec-draft-backend-sampling).
BUILD=<build with the hooks> HOST=127.0.0.1 API_KEY_FILE=<temp key> SPEC_ACCEPT=exact \
  LLAMA_MTP_DUMP=/work/bench/mtp5/runs/dump.bin bash srv/start-qwen-vision.sh
# requests with "cache_prompt": false (run_suite.py: CACHE_PROMPT=0): the reader splits requests at position 0
# and expects every request to start there
CACHE_PROMPT=0 SPLITS=test SAMPLING=nothink MAXTOK=300 python run_suite.py D-chat
# then, with the server stopped
python parity.py /work/bench/mtp5/runs/dump.bin --device cuda:0 --emulate gpu
python ovl_eval.py /work/bench/mtp5/runs/dump.bin --reqs chat:1-25,think:25-49 --device cuda:1 --head-device cuda:0
python -c "from mtp import data; data.compact('<dump>', '<shard dir>', dict(temp=0.7, top_k=20, top_p=0.8, min_p=0.0, pres=1.5))"
python train.py --smoke --train <shard dir> --device cpu
```

The dump appends (`ab`): use a new path per run. About 32 KB per committed position raw (2.3 GB for 73.5k
positions), 10.8 KiB compacted.

Box-only dependencies: `/work/bench/mtp3/analysis` (`cfcore`, `load`; on branch `mtp-adaptive-research` as
`p100-docs/mtp-research/analysis/`) and `/work/bench/mtp4/{quality,qload}.py` for the simulations;
`/work/bench/prodctl.sh` (chat save/restore) for the windows; `/work/bench/slots/longctx.py` and
`/work/bench/vision1440/img0.png` for `vram_depth.py`; the 72-prompt suite `/work/bench/mtp3/suite/prompts.json`
(on `mtp-adaptive-research` as `p100-docs/mtp-research/suite/prompts.json`) for `run_suite.py`.

A dump from eval-suite prompts is for checks only: training prompts must be disjoint from the 72-prompt suite
(MTP-DRAFT.md "Data").
