# System-1 sensor bank (research, M1)

Linear heads over the hidden row Qwen already computes at the last prompt token. No extra transformer pass.

- `llama-hidden-dump`: dumps layer-input / final-norm rows at marked prompt positions (feature extraction for training).
  `--hd-prefix` decodes a shared prefix once and restores its sequence state for every row (deep-context tests).
- `llama-jev-multi`: asks JEV many noul questions about one state, sharing the state prefix (baseline comparison).
- Server: `--sensor-bank bank.json` (weights in `bank.f32` next to it).
  - `POST /v1/sensors {"messages", "tools"?, "heads": [ids] | "all"}` -> `{"sensors": {id: p}, "usage", "timings"}`
    (a 0-token chat completion on the generation slot: the cached prefix is reused).
  - `"sensors": [ids] | "all"` on any completion request adds a `sensors` object to the final response.
  - `GET /v1/sensors` returns the registry (descriptions, layer, calibration, validation metrics).
  Heads on the final-norm row cost only the dot products when MTP is on (MTP already copies that row for every token).
