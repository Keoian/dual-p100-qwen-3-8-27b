# mtp3 Phase 1 baselines (window 5, 2026-10-24 06:19-07:33)

Production binary (build-vision, code at c73976485, bf16 wire on unless stated), production setup, suite dev+val (48 prompts),
seed 11, max 300 tokens, natural length, thinking off, thinking-off sampling 0.7/0.8/20/0 presence 1.5 unless stated.
Sweep runs exclude the 6 long-context prompts; B0 and B1 ran all 48 (their sweep-category rows are shown). Different
settings produce different texts, so these are unpaired: treat differences under ~2% as noise.

| run | setting | tokens | committed t/s (sweep categories) | vs B1r | acceptance |
|---|---|---|---|---|---|
| B0 | no MTP | 8867 | 30.27 | -38.8% | 0.000 |
| B1 | production (n-max 4, p-min 0.2, sampled drafts) | 8809 | 50.28 | +1.6% | 0.512 |
| B3-n1 | n-max 1 | 8566 | 43.42 | -12.2% | 0.814 |
| B3-n2 | n-max 2 | 8756 | 49.60 | +0.3% | 0.691 |
| B3-n3 | n-max 3 | 8879 | 49.65 | +0.4% | 0.585 |
| B1r | production, repeated last | 8809 | 49.48 | +0.0% | 0.512 |
| B3-n6 | n-max 6 | 8830 | 40.75 | -17.6% | 0.394 |
| B3-n8 | n-max 8 | 8775 | 33.91 | -31.5% | 0.321 |
| B4-p0.0 | p-min 0 | 8767 | 48.94 | -1.1% | 0.498 |
| B4-p0.1 | p-min 0.1 | 8767 | 49.18 | -0.6% | 0.497 |
| B4-p0.4 | p-min 0.4 | 8570 | 49.96 | +1.0% | 0.588 |
| B4-p0.6 | p-min 0.6 | 8594 | 49.30 | -0.4% | 0.706 |
| B5-greedy | greedy drafts | 8390 | 49.31 | -0.3% | 0.509 |
| B1k | production, Discord sampling 1.0/0.95/20/0 | 8174 | 51.46 | +4.0% | 0.543 |
| B2k | Discord reference (adapted), its sampling | 9204 | 49.00 | -1.0% | 0.545 |

Per category, committed t/s (B0 / B1r / B3-n2 / B4-p0.4):

| category | B0 | B1r | n-max 2 | p-min 0.4 |
|---|---|---|---|---|
| casual | 30.6 | 44.9 | 47.2 | 45.6 |
| code | 30.2 | 61.8 | 56.1 | 62.4 |
| creative | 30.3 | 36.5 | 41.4 | 37.4 |
| json | 30.3 | 70.6 | 57.9 | 70.8 |
| reasoning | 30.1 | 56.0 | 53.6 | 55.6 |
| switch | 30.1 | 49.7 | 49.5 | 48.1 |
| tools | 30.2 | 64.8 | 53.7 | 64.7 |

Long-context prompts (~7k tokens): B0 29.6 t/s, B1 47.3 t/s.

B2k = the Discord reference adapted: GGML_CUDA_P2P=1 (no peer access on this board), q4_0/q4_0, -b 32768 -ub 2048, no vision,
no --reasoning off (requests send enable_thinking false), f32 exchange wire (code default), 127.0.0.1:8090 with a key.
