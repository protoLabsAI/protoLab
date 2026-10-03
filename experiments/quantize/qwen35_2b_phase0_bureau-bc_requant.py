#!/usr/bin/env python3
"""Qwen3.5-2B NVFP4 requant — dense hybrid-GDN + VL variant.

Adapted from a1_requant.py (which targets the Qwen3.5 *MoE*). The 2B is
`Qwen3_5ForConditionalGeneration`: dense text backbone + a vision tower +
GDN linear-attention layers. So vs the A1 recipe:
  - dense model class (no MoE loader, no moe_calibrate_all_experts)
  - ignore list drops MoE gates; keeps vision / embed / lm_head / linear_attn
  - no MTP graft (base 2B has no draft head)

Purpose: a ~1.2GB on-disk NVFP4 peer to MiniCPM5-1B Q8_0 (1.1GB) for a fair
distillation-base comparison, and the format a shipped/distilled artifact runs in.

Run on the freed GPU1 window:
  CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/ava/dev/quant-env/bin/python qwen35_2b_requant.py
"""

import os

os.environ.setdefault("HF_HOME", "/mnt/models/huggingface")

import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MODEL_ID = "/mnt/data/checkpoints/phase0/bureau-bc-merged"
OUT = "/mnt/models/quantized/Qwen3.5-2B-phase0-bureau-bc-NVFP4"
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 2048

with load_context(Qwen3_5ForConditionalGeneration):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype="auto", device_map="auto"
    )
processor = AutoProcessor.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "re:.*lm_head",
        "re:visual.*",
        "re:model.visual.*",
        "re:.*embed_tokens$",
        "re:.*linear_attn.*",  # keep the GDN hybrid layers in bf16
        "re:.*mtp.*",
    ],
)

ds = load_dataset(
    "HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]"
)
ds = ds.select_columns(["messages"]).shuffle(seed=42)


def preprocess_function(example):
    messages = [
        {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
        for m in example["messages"]
    ]
    return processor.apply_chat_template(
        messages,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        tokenize=True,
        add_special_tokens=False,
        return_dict=True,
        add_generation_prompt=False,
    )


ds = ds.map(preprocess_function, batched=False, remove_columns=ds.column_names)


def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}


oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=data_collator,
)

model.save_pretrained(OUT, save_compressed=True)
processor.save_pretrained(OUT)
print(f"saved to {OUT}")
