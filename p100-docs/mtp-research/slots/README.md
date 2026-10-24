# Long-context slot library (built 2026-10-24)

Prefilled long contexts, saved as llama-server slot files. A test restores one in 2-20 s instead of prefilling it
(16k: 43 s ... 250k: ~20 min from scratch).

| depth | tokens | restore | files (GiB) | the last part of the document |
|---|---|---|---|---|
| 16k | 15,998 | 2.0 s | 0.9 | kld-text: a Python CSV statistics script |
| 25k | 25,001 | 3.2 s | 1.4 | kld-text: a retry decorator with backoff |
| 32k | 32,001 | 4.4 s | 1.9 | kld-text: entropy and the arrow of time |
| 64k | 64,001 | 7.1 s | 3.0 | kld-text: machine understanding of language |
| 128k | 128,001 | 12.0 s | 4.9 | tools/server/README.md |
| 200k | 199,999 | 16.6 s | 7.1 | OPTLOG.md |
| 250k | 250,001 | 20.5 s | 8.7 | OPTLOG.md |

Total ~28 GiB in /work/slots (`longctx-<depth>.bin` with `.draft`, `.spec` and `.ckpt`; keep each set together).

## What a slot holds

The conversation in `messages-<depth>.json`: one short system message, then the reference document as user messages
"Reference document, part 1..k". The parts come from `corpus.txt` (frozen; sha256 prefix in manifest.json): our
kld-text (prose and code), then the repo docs, Kmic's OPTLOG.md and some source files. Each depth's conversation is
an exact prefix of the next, so the library was built in one nested prefill.

Verified on a fresh server: restore, then a chat request with one more user message processed only that message
(`timings.prompt_n` 27 at every depth). The greedy answers describe the right last part.

## Using it

```python
import sys; sys.path.insert(0, "/work/bench/slots"); import longctx
longctx.restore("128k", key)              # server on :8090, slot 0
msgs = longctx.messages("128k") + [{"role": "user", "content": "your question"}]
# POST /v1/chat/completions {"messages": msgs, ...}
```

`python3 longctx.py list` prints the table; `python3 longctx.py restore 128k [keyfile]` restores one slot.

- **Follow-up questions don't need a restore** (measured 10-24, window 10). After a request, the slot holds the
  document plus that question and answer; the next question rolls back to the document's end through a context
  checkpoint and processes only itself (37-43 tokens, 0.6-1.0 s at 32k-200k), and its reply is identical to the one
  from a freshly restored slot. Restore once per depth (or after a request with a different document prefix).
- **Thinking on or off doesn't change the prefix.** `LLAMA_THINK_BY_BUDGET=1` always renders the template in thinking
  mode, so both reuse the slot.

## Compatibility

The files only restore into a server with the same:
- model (Qwen3.8-27B UD-Q6_K);
- KV types (q8_0 K / q4_0 V, draft q4_0 / q4_0);
- MTP head;
- slot file format (this fork as of c73976485);
- enough context (`-c 262144` for 250k).

They were built with production's binary (build-vision) and settings: `-ub 1024 -b 2048 -np 1 -ctxcp 32`, bf16
exchange wire. If the cache types, the model or the slot format change, rebuild:
1. In a test window, start a server with the new settings and the temp key on 127.0.0.1:8090.
2. Run `python3 make_slots.py build`, then `python3 make_slots.py verify`.

`window8.sh` does all of it: it stops production, builds, verifies and restores production, in about 25 minutes.
For other configurations (e.g. Kmic's q4_0/q4_0), use different file names; there is room (282 GB free on /work).
