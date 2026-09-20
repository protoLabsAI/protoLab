#!/bin/bash
# Full pass: test+calib, all suites, both readout formats. CPU arms never touch the prod GPUs.
cd "$(dirname "$0")"
export HF_HOME=/mnt/models/huggingface HF_HUB_OFFLINE=1
PYQ=/home/ava/dev/quant-env/bin/python
( export CUDA_VISIBLE_DEVICES=""
  for fmt in name letter; do $PYQ decide.py --arm hf:Qwen/Qwen3.5-2B   --format $fmt --threads 16; done
  $PYQ decide.py --arm hf:Qwen/Qwen3.5-0.8B --format name --threads 16 ) > logs/cpu-a.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=""
  for fmt in name letter; do $PYQ decide.py --arm hf:Qwen/Qwen3.5-4B   --format $fmt --threads 16; done ) > logs/cpu-b.log 2>&1 &
( for fmt in name letter; do $PYQ decide.py --arm smart --format $fmt --workers 4; done ) > logs/smart.log 2>&1 &
wait
echo "ALL DECISION RUNS DONE"
