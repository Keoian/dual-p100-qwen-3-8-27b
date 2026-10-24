#!/bin/bash
# Reproduce every table in ../ANALYSIS.md (CPU only, single process, ~3 min on the 4-core host). Outputs in out/.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
mkdir -p out cache
[ -f cache/vocab.json ] || nice -n 10 python3 vocab.py            # tokenizer strings from the GGUF metadata (read-only)
nice -n 10 python3 load.py                                        # parse traces -> cache/*.npz
nice -n 10 python3 depth.py      > out/depth.txt                  # Q1 free-running depth curve
nice -n 10 python3 cost.py       > out/cost.txt                   # Q2 cost model -> cache/cost.json
nice -n 10 python3 calib.py      > out/calib.txt                  # Q2 replay of BASELINES
nice -n 10 python3 fidelity.py   > out/fidelity.txt               # Q3d/e per-token metrics, Q5 partly
nice -n 10 python3 policies.py n4p0     > out/policies-n4p0.txt   # Q3 policies, n-max <= 4
nice -n 10 python3 policies.py n8p0     > out/policies-n8p0.txt   # Q3 policies, n-max <= 8
nice -n 10 python3 policies.py n8p0-le4 > out/policies-n8p0-le4.txt
nice -n 10 python3 sens.py       > out/sens.txt                   # cost-model and estimator sensitivity
nice -n 10 python3 lenient.py    > out/lenient.txt                # Q4 lenient acceptance frontier
nice -n 10 python3 tasktype.py   > out/tasktype.txt               # Q5 task-type evidence
nice -n 10 python3 iso_tables.py > out/iso_tables.txt               # lookups of the recommended rule
echo done
