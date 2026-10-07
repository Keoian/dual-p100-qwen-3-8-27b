#!/bin/bash
# every 2 s, synced: VRAM, temp, power, PCIe tx/rx, replays (both GPUs), AER totals (GPUs + root ports 00:01.0, 00:1b.4)
OUT=${1:-/work/monitor.log}   # usage: bash monitor.sh [file] &  (kill when done)
aer() { for d in 0000:01:00.0 0000:00:01.0 0000:03:00.0 0000:00:1b.4; do printf "%s:" ${d:5}; for t in correctable nonfatal fatal; do printf "%s/" $(awk '/TOTAL/{print $2}' /sys/bus/pci/devices/$d/aer_dev_$t 2>/dev/null); done; printf " "; done; }
while true; do
  q=$(nvidia-smi --query-gpu=index,memory.used,temperature.gpu,power.draw,pcie.link.width.current,clocks_throttle_reasons.active --format=csv,noheader | tr '\n' '|')
  r=$(nvidia-smi -q | awk '/Replays Since Reset/{printf "rep=%s ",$NF} /Tx Throughput/{printf "tx=%s ",$(NF-1)} /Rx Throughput/{printf "rx=%s ",$(NF-1)}')
  echo "$(date -u +%T.%N | cut -c1-12) $q $r AER[$(aer)]" >> "$OUT"; sync; sleep 2
done
