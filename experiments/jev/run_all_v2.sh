#!/bin/bash
# Rung 1 pass: the 18 tasks added by prep_data_v2.py, test+calib, both readout formats.
# The v0 four are NOT re-run — their results are already committed and must stay byte-identical.
# CPU arms are pinned off the GPUs; only the smart lane touches :8041.
cd "$(dirname "$0")"
export HF_HOME=/mnt/models/huggingface HF_HUB_OFFLINE=1
PYQ=/home/ava/dev/quant-env/bin/python
NEW=agnews,appreviews,cola,dbpedia,emotion,finnews,hate,massive,mrpc,offensive,qnli,rte,snli,spam,sst5,tweetsent,yahoo,yelpstars
mkdir -p logs

( export CUDA_VISIBLE_DEVICES=""
  for fmt in name letter; do $PYQ decide.py --arm hf:Qwen/Qwen3.5-4B --suites $NEW --format $fmt --threads 10; done
) > logs/v2-4b.log 2>&1 &

( export CUDA_VISIBLE_DEVICES=""
  for fmt in name letter; do $PYQ decide.py --arm hf:Qwen/Qwen3.5-2B --suites $NEW --format $fmt --threads 10; done
) > logs/v2-2b.log 2>&1 &

( export CUDA_VISIBLE_DEVICES=""
  $PYQ decide.py --arm hf:Qwen/Qwen3.5-0.8B --suites $NEW --format name --threads 8
) > logs/v2-08b.log 2>&1 &

( for fmt in name letter; do $PYQ decide.py --arm smart --suites $NEW --format $fmt --workers 4; done
) > logs/v2-smart.log 2>&1 &

wait
echo "RUNG 1 DECISION RUNS DONE"
