#!/usr/bin/env python3
"""Qwen3.8-27B NVFP4 requant (+ native bf16 MTP head, single shot).

Qwen3.8 is a NEW POST-TRAIN ON THE QWEN3.5 ARCHITECTURE, not a new arch — config.json
declares model_type=qwen3_5 / Qwen3_5ForConditionalGeneration. So this is the SAME dense
hybrid-GDN + VL + MTP shape as ThinkingCap and Fable-Fusion-711, and the proven recipe
(`fablefusion_711_27b_requant.py`) ports directly.

Like Fable-Fusion, the checkpoint SHIPS its MTP head (15 `mtp.*` tensors, verified in
model.safetensors.index.json) — so we quantize the main weights and keep the MTP head bf16
in the SAME shot. No graft_mtp.py post-step.

  vision tower / GDN linear-attn / lm_head / embed / mtp stay bf16 (no sm120 W4A4 kernel,
  and DeltaNet corrupts under low-precision activations — the standing finding on this arch).

Motivation: the bf16 eval (evals/results/QWEN38-27B-EVAL-2026-08-14.md) came in board-best
on LiveCodeBench (0.725, taking it from Gemma4-31B's 0.708) and 2nd on claw (0.795), but at
~5x DSV4's latency at the shipped reasoning_effort=xhigh default. NVFP4 + MTP is the attempt
to keep the quality and pay down the latency. WATCH: FableFusion-711 lost claw 0.782 -> 0.680
on requant — a wash-to-loss is a real possible outcome, which is why the gate runs before any
smart-lane cutover.

Needs both cards free (~60GB) — the DSV4 smart lane must be stopped for the duration.
  CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/ava/dev/quant-env/bin/python qwen38_27b_nvfp4_requant.py
"""
import os
os.environ.setdefault("HF_HOME", "/mnt/models/huggingface")

import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MAIN = ("/mnt/models/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/"
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
PROC_SRC = MAIN  # ships its own processor / chat_template / video preprocessor
OUT = "/mnt/models/quantized/Qwen3.8-27B-NVFP4"
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
    # 4-D DeltaNet crash without this — see the A1 war story in the quant-release skill.
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}

oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES, data_collator=data_collator)

model.save_pretrained(OUT, save_compressed=True)
processor.save_pretrained(OUT)

# ---------------------------------------------------------------------------
# POST-SAVE FIXUP — save_pretrained rewrites far more than the weights on a VL
# model, and every one of these failures produces a checkpoint that loads fine
# and serves TEXT correctly. All four were hit for real on 2026-08-15.
# ---------------------------------------------------------------------------
import shutil, json as _json
from pathlib import Path
_src, _out = Path(MAIN), Path(OUT)

# 1. MTP head is DROPPED. Qwen3_5ForConditionalGeneration does not own `mtp.*`,
#    so save_pretrained never writes them (15 tensors in, 0 out). Extract the
#    sidecar from the SAME checkpoint -- exact, not a cross-model graft:
#      python extract_mtp.py --model <MAIN> --output <OUT>/model-mtp.safetensors
#    ...and append the ignore entry, which llm-compressor cannot record because
#    it never saw those modules.
_cfg_p = _out / "config.json"
_cfg = _json.loads(_cfg_p.read_text())
_ig = _cfg.get("quantization_config", {}).get("ignore", [])
if _ig and not any("mtp" in x for x in _ig):
    _ig.append("re:.*mtp.*")
    _cfg_p.write_text(_json.dumps(_cfg, indent=2))
    print("fixup: appended re:.*mtp.* to quantization_config.ignore")

# 2. TOKENIZER IS MANGLED -> silently breaks VISION only. llm-compressor rewrites
#    tokenizer_config.json in a different shape, dropping `added_tokens_decoder`
#    and `additional_special_tokens`. Text and tools still work; images fail with
#    "Mismatch in `image` token count between text and `input_ids`. Got ids=[2047]
#    and text=[2451]" -- which reads like a truncation bug and is not.
#    Quantization does not change tokenization: restore the source files verbatim.
for _f in ("tokenizer_config.json", "tokenizer.json"):
    shutil.copy2(_src / _f, _out / _f)
    print(f"fixup: restored source {_f}")

# 3. BOGUS processor_config.json. Written by processor.save_pretrained, does NOT
#    exist upstream, and downgrades Qwen2VLImageProcessorFast -> ...Processor.
#    Remove so the loader falls back to preprocessor_config.json.
_pc = _out / "processor_config.json"
if _pc.exists():
    _pc.unlink()
    print("fixup: removed processor_config.json (not in source; downgrades the image processor)")

# 4. AUX CONFIGS MISSING. The vision/video preprocessors are load-bearing for VL.
for _f in ("preprocessor_config.json", "video_preprocessor_config.json",
           "vocab.json", "merges.txt", "generation_config.json"):
    if (_src / _f).exists() and not (_out / _f).exists():
        shutil.copy2(_src / _f, _out / _f)
        print(f"fixup: copied missing {_f}")

print(f"saved Qwen3.8-27B NVFP4 (with bf16 MTP head) to {OUT}")
print("VERIFY BEFORE SHIPPING: packed census (visual/linear_attn/mtp/lm_head == 0),")
print("  a real completion, a TOOL CALL, and an IMAGE -- text-only smoke tests pass")
print("  on a checkpoint whose vision is completely broken.")
