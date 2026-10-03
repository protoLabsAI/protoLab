#!/usr/bin/env python3
"""Streaming FP8 quantize of Qwen3-Coder-Next-Opus-4.6-Reasoning-Distilled (bf16 → block-wise FP8).

Reuses the tensor-by-tensor streaming quantizer (peak RAM ~2x largest tensor, CPU-only, no GPU,
no fleet impact) from quantize_native_fp8_lowmem.py, but points at our LOCAL bf16 dir and writes
to /mnt/data (/mnt/models is full). This is the #398 "streaming-quantize a model you can't fit"
technique applied to a NEW arch: bf16 Qwen3-Next 80B MoE (74,391 individual 2D expert tensors),
vs the post's Mistral-FP8 Leanstral. Skip-patterns already cover the qwen3_next hybrid path
(linear_attn/SSM, MoE gates, mtp, conv1d).
"""
import gc, json, shutil, sys, time
from pathlib import Path

sys.path.insert(0, "/home/ava/dev/lab/experiments/quantize")
import quantize_native_fp8_lowmem as Q  # streaming functions + constants

SNAPSHOT = Path("/mnt/data/models-src/Coder-Opus-Distill-bf16")
OUT = Path("/mnt/data/models/Coder-Opus-Distill-FP8")
OUT.mkdir(parents=True, exist_ok=True)

index = json.load(open(SNAPSHOT / "model.safetensors.index.json"))
shard_names = sorted(set(index["weight_map"].values()))
print(f"streaming FP8: {len(shard_names)} shards, {len(index['weight_map'])} tensors -> {OUT}", flush=True)

t0 = time.time()
total_q = total_s = 0
new_weight_map = {}
shard_counter = [0]
all_skipped_modules = []

for shard_name in shard_names:
    wmap, q, s, mods = Q.process_shard_streaming(SNAPSHOT / shard_name, OUT, shard_counter)
    total_q += q; total_s += s
    new_weight_map.update(wmap)
    all_skipped_modules.extend(mods)
    gc.collect()

num_shards = shard_counter[0]
# rename PLACEHOLDER -> final N-of-M
for i in range(1, num_shards + 1):
    old = f"model-{i:05d}-of-PLACEHOLDER.safetensors"
    new = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
    (OUT / old).rename(OUT / new)
    for k, v in new_weight_map.items():
        if v == old:
            new_weight_map[k] = new
all_shards = [f"model-{i:05d}-of-{num_shards:05d}.safetensors" for i in range(1, num_shards + 1)]

total_size = sum((OUT / s).stat().st_size for s in all_shards)
json.dump({"metadata": {"total_size": total_size}, "weight_map": new_weight_map},
          open(OUT / "model.safetensors.index.json", "w"), indent=2)

cfg = json.load(open(SNAPSHOT / "config.json"))
cfg["quantization_config"] = {
    "activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
    "weight_block_size": [Q.BLOCK_SIZE, Q.BLOCK_SIZE],
    "modules_to_not_convert": sorted(set(all_skipped_modules)),
}
json.dump(cfg, open(OUT / "config.json", "w"), indent=2)

from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(str(SNAPSHOT), trust_remote_code=True).save_pretrained(str(OUT))
for fn in ["generation_config.json", "chat_template.jinja", "vocab.json"]:
    if (SNAPSHOT / fn).exists():
        shutil.copy2(SNAPSHOT / fn, OUT / fn)

out_gb = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1024**3
print(f"\n{'='*60}\n  streaming FP8 DONE\n  out: {OUT} ({out_gb:.1f} GB, {num_shards} shards)\n"
      f"  quantized {total_q} / skipped {total_s} tensors in {time.time()-t0:.0f}s\n"
      f"  skip-kept-bf16 modules: {len(set(all_skipped_modules))}\n{'='*60}", flush=True)
