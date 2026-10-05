#!/usr/bin/env python3
"""Daria (mistral-creative-base-v0) -> NVFP4, for a co-resident serving lane.

Source is the TEXT-ONLY extraction (`..._-text`), not the VL checkpoint: quantizing a
Mistral3ForConditionalGeneration silently rewrites the tokenizer and aux configs (the standing
VL-quant rule), and Daria is text-only anyway. Straight dense Mistral here — no vision tower,
no MTP head, no DeltaNet/GDN layers — so the recipe is far simpler than the Qwen3.8 one and
none of that checkpoint's post-save fixups apply.

CALIBRATION IS IN-DOMAIN ON PURPOSE. The usual ultrachat calibration set is assistant chat;
this model's job is long-form literary prose, and quantization error is minimised where the
calibration distribution sits. We calibrate on the same essay/passage corpus the voice SFT was
trained on (`register-sft.jsonl`), matching the serving distribution.

WHY THIS MATTERS BEYOND SIZE: the clamp's tau is an ABSOLUTE threshold on h.d_hat, calibrated
against bf16 activations. Quantization moves the activation distribution, so tau almost
certainly needs re-measuring against this checkpoint -- do NOT assume 0.41 carries over.

  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/ava/dev/quant-env/bin/python mistral_creative_24b_nvfp4.py
"""
import json
import os
os.environ.setdefault("HF_HOME", "/mnt/models/huggingface")

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MAIN = "/mnt/data/abliterate/mistral-creative-base-v0-text"
OUT = "/mnt/models/quantized/Daria-24B-NVFP4"
CALIB = "/mnt/data/datasets/creative/register-sft.jsonl"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048

model = AutoModelForCausalLM.from_pretrained(MAIN, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MAIN)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    # lm_head + embeddings stay bf16: no sm120 W4A4 kernel for them, and with a 131k vocab
    # they are ~2.7GB of the footprint either way.
    ignore=["re:.*lm_head", "re:.*embed_tokens$"],
)

rows = [json.loads(l) for l in open(CALIB)][:NUM_CALIBRATION_SAMPLES]
ds = Dataset.from_list([{"text": r["text"]} for r in rows])

def preprocess(example):
    return tokenizer(example["text"], truncation=True, max_length=MAX_SEQUENCE_LENGTH,
                     add_special_tokens=True)

ds = ds.map(preprocess, batched=False, remove_columns=ds.column_names)

def data_collator(batch):
    # Plain tokenizer() returns FLAT input_ids (no batch dim); the VL recipes this was adapted
    # from fed processor output that was already [1, seq]. Feeding 1-D here makes rotary blow
    # up with a head_dim mismatch deep in the attention call, which reads like a config bug
    # and is not one. Add the batch dim explicitly.
    assert len(batch) == 1
    out = {}
    for k, v in batch[0].items():
        t = torch.tensor(v)
        out[k] = t.unsqueeze(0) if t.dim() == 1 else t
    return out

oneshot(model=model, processor=tokenizer, dataset=ds, recipe=recipe, max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES, data_collator=data_collator)

model.save_pretrained(OUT, save_compressed=True)
tokenizer.save_pretrained(OUT)
for fn in ("chat_template.jinja", "generation_config.json"):
    src = os.path.join(MAIN, fn)
    if os.path.exists(src):
        import shutil
        shutil.copy(src, os.path.join(OUT, fn))
print(f"saved -> {OUT}")
