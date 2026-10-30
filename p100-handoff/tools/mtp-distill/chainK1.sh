#!/bin/bash
# chainK1.sh: start windowK1.sh at 06:00 UTC on 2026-10-30 (00:00 MDT: the operator's pre-approved night hours)
t=$(date -u -d "2026-10-30 06:00:00" +%s); now=$(date -u +%s)
[ "$t" -gt "$now" ] && sleep $((t - now))
exec bash /work/bench/mtp5/windowK1.sh
