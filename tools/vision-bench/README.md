# llama-vision-bench: time a vision projector alone

Times the vision encoder (the `--mmproj` projector) on its own and optionally dumps its output embeddings, for
comparing projector quantizations and encoder kernels without loading the text model's weights. The text model is
loaded vocab-only: mtmd needs its tokenizer. Built with the other tools (`build-*/bin/llama-vision-bench`).

```bash
llama-vision-bench --model Qwen3.8-27B-UD-Q6_K.gguf --mmproj mmproj-Qwen3.8-27B-Q8_0.gguf \
    --dev CUDA1 --max-tokens 3600 --reps 3 --out emb/q8_0 screenshot.png photo.jpg
```

| option | meaning |
|---|---|
| `--model FILE` | the text model GGUF (vocabulary only) |
| `--mmproj FILE` | the projector to time (required) |
| `--dev NAME` | device to run on, e.g. `CUDA1` (default: mtmd's default device, with memory read from the first GPU) |
| `--max-tokens N` | image token cap, as the server's `--image-max-tokens` (default: the projector's own limit) |
| `--reps R` | encodes per image (default 3) |
| `--fa on\|off` | flash attention in the encoder (default: auto) |
| `--out PREFIX` | write each image's embeddings as raw f32 to `PREFIX.<i>.f32` |
| `img...` | one or more image files |

Output: the load time and device memory after load and warmup, then per image its token count, the minimum and median
encode time over the R runs, and the device memory used by the projector.

The embedding dump assumes 5120 floats per token (Qwen3.8-27B's projector output); other models need that constant
in `vision-bench.cpp` changed. To compare two projectors, dump both and compare the files (e.g. relative L2), and
judge quantizations against the BF16 projector (FINDINGS "q8_0 K cache, 1440p images and the Q8_0 projector").
