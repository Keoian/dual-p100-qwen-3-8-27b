#!/bin/bash
# Reproduce every table in ../CF_ANALYSIS.md (CPU only, one process at a time, ~15 min on the 4-core host).
# Traces: runs/trace-cfchat.jsonl, runs/trace-cfthink.jsonl (LLAMA_SPEC_TRACE_TOPK=128). Caches in cache/cf/, outputs in out/cf/.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
O=out/cf
mkdir -p $O cache/cf
N="nice -n 10 python3"
$N counterfactual.py selfcheck ../runs/trace-cfthink.jsonl > $O/selfcheck-cfthink.txt   # (cfchat's is in runs/window13.log)
$N cfload.py cfchat cfthink > $O/load.txt                       # traces -> cache/cf/*.npz, exclusion counts
$N cfvalidate.py cfchat > $O/validate-cfchat.txt                # vectorized chain vs recorded p and vs counterfactual.py
$N cfvalidate.py cfthink > $O/validate-cfthink.txt
$N cfgrid.py grid chat > $O/grid-chat.txt                       # step 2 (a_S cached in cache/cf/*.a.npy)
$N cfgrid.py grid think > $O/grid-think.txt
$N cfgrid.py grid pooled > $O/grid-pooled.txt
$N cfgrid.py repro > $O/repro.txt                               # step 2 check: production's tables
for s in chat think; do $N cfrepro_decomp.py $s; done > $O/repro-decomp.txt
$N cfshift.py > $O/shift.txt                                    # step 3
for s in chat think pooled; do $N cfapprox.py $s > $O/approx-$s.txt && $N cfapprox2.py $s > $O/approx2-$s.txt; done   # step 4
$N cfrecommend.py > $O/recommend.txt                            # candidates with CIs, extrapolation
$N cfcard.py > $O/card.txt                                      # candidates vs production at the card settings
$N cfwritec.py > /dev/null                                      # out/cf/recommended-c.txt
$N cfreport.py > $O/report.txt
echo done
