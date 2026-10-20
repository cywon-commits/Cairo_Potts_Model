#!/bin/bash
# usage: chain.sh q Tmin Tmax L1 L2 ...   (runs sizes sequentially)
q=$1; Tmin=$2; Tmax=$3; shift 3
cd "$(dirname "$0")/.."
for L in "$@"; do
  if [ $L -le 16 ]; then bs=2000; elif [ $L -le 32 ]; then bs=1000; else bs=500; fi
  OMP_NUM_THREADS=1 python3 -W ignore scripts/run_pt.py --L $L --q $q --Tmin $Tmin --Tmax $Tmax \
     --therm 20000 --bins 200 --bin_size $bs --seed $L \
     --out data/mc/pt_p1_q${q}_L${L}.npz > data/mc/pt_p1_q${q}_L${L}.log 2>&1
done
