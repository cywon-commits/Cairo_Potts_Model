#!/bin/bash
# fine gap scans (L = 4..10) around the crossings of the ordered layouts
cd "$(dirname "$0")/.."
out=data/layout_gap_refine.jsonl
g() { python3 -c "import numpy as np; print(' '.join('%.4f'%t for t in np.arange($1,$2+1e-9,$3)))"; }
run() { OMP_NUM_THREADS=1 python3 -W ignore scripts/layout_gap_scan.py --layouts "$1" --q $2 --L 4 6 8 10 --T $(g $3 $4 $5) --out $out; }
run 8:0,1,2,3,4,5,6 0.5  0.115 0.130 0.0025
run 8:0,1,2,3,4,5,6 1.05 0.225 0.245 0.0025
run 8:0,1,2,3,4     1.05 0.170 0.190 0.0025
run 6:0,1,2         1.05 0.165 0.185 0.0025
run 8:0,1,2,3,4     0.5  0.030 0.050 0.0025
run 6:0,1,2         0.5  0.030 0.050 0.0025
echo refine-done
