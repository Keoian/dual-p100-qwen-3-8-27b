#!/bin/bash
# The run_all.sh tables on the thinking-on traces tn4p0/tn8p0 (MTP3_TRACESET=think; CPU only, single process).
# Caches in cache/think/, outputs in out/think/ with out/'s file names; inside them "n4p0"/"n8p0" mean tn4p0/tn8p0.
# Left out: calib.py (replays the thinking-off BASELINES runs). cost.py only with MTP3_THINK_REFIT_COST=1; otherwise the
# chat set's cache/cost.json is used (timing, not sampling). iso_tables runs first; its two 24-point lookups are also
# written as C arrays (spec_iso_x/q/p1 of common/speculative.cpp) to out/think/iso_tables-c.txt.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MTP3_TRACESET=think
O=out/think
mkdir -p $O cache/think
[ -f cache/vocab.json ] || nice -n 10 python3 vocab.py            # shared with the chat set
if [ "${MTP3_THINK_REFIT_COST:-}" != 1 ] && [ ! -f cache/cost.json ]; then echo "cache/cost.json missing: run run_all.sh first" >&2; exit 1; fi
nice -n 10 python3 load.py                                        # parse traces -> cache/think/*.npz
if [ "${MTP3_THINK_REFIT_COST:-}" = 1 ]; then nice -n 10 python3 cost.py > $O/cost.txt; fi   # -> cache/think/cost.json
nice -n 10 python3 iso_tables.py > $O/iso_tables.txt               # lookups of the recommended rule
awk 'BEGIN { n = 0 } NR > 1 { x[n] = $1; q[n] = $2; p[n] = $3; n++ }
     function arr(name, v,   s, i) {
       s = sprintf("static const float %-11s[%d] = { ", name, n)
       for (i = 0; i < n; i++) s = s v[i] "f" (i == n - 1 ? " };" : i % 12 == 11 ? ",\n" sprintf("%39s", "") : ", ")
       print s }
     END { if (n != 24) { print "iso_tables: expected 24 grid points, got " n > "/dev/stderr"; exit 1 }
           arr("spec_iso_x", x); arr("spec_iso_q", q); arr("spec_iso_p1", p) }' $O/iso_tables.txt > $O/iso_tables-c.txt
nice -n 10 python3 depth.py      > $O/depth.txt                   # Q1 free-running depth curve
nice -n 10 python3 fidelity.py   > $O/fidelity.txt                # Q3d/e per-token metrics, Q5 partly
nice -n 10 python3 policies.py n4p0     > $O/policies-n4p0.txt    # Q3 policies, n-max <= 4
nice -n 10 python3 policies.py n8p0     > $O/policies-n8p0.txt    # Q3 policies, n-max <= 8
nice -n 10 python3 policies.py n8p0-le4 > $O/policies-n8p0-le4.txt
nice -n 10 python3 sens.py       > $O/sens.txt                    # cost-model and estimator sensitivity
nice -n 10 python3 lenient.py    > $O/lenient.txt                 # Q4 lenient acceptance frontier
nice -n 10 python3 tasktype.py   > $O/tasktype.txt                # Q5 task-type evidence
echo done
