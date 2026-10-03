#!/usr/bin/env python3
"""Fable-Fusion-711-Qwen3.6-27B NVFP4 requant.

DavidAU's Qwen3.6-27B merge (Heretic ARA decensor + Polaris/F451/Opus traces).
Same dense hybrid-GDN + VL + MTP arch as ThinkingCap, quantized with the same recipe
(`thinkingcap_27b_heretic_requant.py`). KEY DIFFERENCE: Fable-Fusion SHIPS its MTP head
(mtp.* tensors present, grafts natively on 0.25.0), so we quantize main weights here and
keep the MTP head bf16 in the SAME shot — no separate graft_mtp.py post-step.

Motivation: bf16 eval beat the ThinkingCap reasoning lane on claw (0.782 vs 0.704) and
LCB (0.633 vs 0.548), but at 82G/half-throughput and with a 2/3 reasoning-runaway defect.
This produces the ~26G NVFP4 that would actually be servable, to test whether the lift
survives quantization (abliteration does — [[project_abliterate_nvfp4]]) before any runaway fix.

  vision tower / GDN linear-attn / lm_head / embed / mtp stay bf16 (no sm120 W4A4 kernel).

Run on the free GPU1 window (prod fast lane owns GPU0):
  CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/ava/dev/quant-env/bin/python fablefusion_711_27b_requant.py
"""
import os
os.environ.setdefault("HF_HOME", "/mnt/models/huggingface")

import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MAIN = ("/mnt/models/huggingface/hub/models--DavidAU--Qwen3.6-27B-Fable-Fusion-711-"
        "Uncensored-Heretic-NM-DAU-MTP/snapshots/289d2ebbb4569a0f5367aee3020995cde88e6eae")
PROC_SRC = MAIN  # ships its own processor / chat_template / video preprocessor
OUT = "/mnt/data/quantized/Qwen3.6-27B-Fable-Fusion-711-NVFP4"
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 2048

with load_context(Qwen3_5ForConditionalGeneration):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MAIN, torch_dtype="auto", device_map="auto"
    )
processor = AutoProcessor.from_pretrained(PROC_SRC)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "re:.*lm_head",
        "re:visual.*",
        "re:model.visual.*",
        "re:.*embed_tokens$",
        "re:.*linear_attn.*",  # keep the GDN hybrid layers in bf16
        "re:.*mtp.*",          # keep the shipped MTP head bf16 (drafts; target verifies)
    ],
)

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.select_columns(["messages"]).shuffle(seed=42)

def preprocess_function(example):
    messages = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                for m in example["messages"]]
    return processor.apply_chat_template(
        messages, return_tensors="pt", padding=False, truncation=True,
        max_length=MAX_SEQUENCE_LENGTH, tokenize=True, add_special_tokens=False,
        return_dict=True, add_generation_prompt=False)

ds = ds.map(preprocess_function, batched=False, remove_columns=ds.column_names)

def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}

oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES, data_collator=data_collator)

model.save_pretrained(OUT, save_compressed=True)
processor.save_pretrained(OUT)
print(f"saved Fable-Fusion-711 NVFP4 (with bf16 MTP head) to {OUT}")
