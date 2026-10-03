#!/usr/bin/env python3
"""Ornith-1.5-35B-A3B NVFP4 requant (MoE, VL, MTP tensors preserved).

Requested on the hub: protoLabsAI/Ornith-1.0-35B-NVFP4 discussion #1 -- a user who has been
running our 1.0 NVFP4 asked for the 1.5 equivalent. Upstream ornith-ai DOES publish
`Ornith-1.5-35B-A3B-NVFP4` (it is what our own prod smart lane serves), so this is not a gap
release; we build it because someone who uses our work asked for it. The card credits
upstream's build rather than implying we are first.

Arch: Qwen3_5MoeForConditionalGeneration -- 35B total / 3B active, 256 experts (8/tok),
40 layers, native VL. Recipe ports from `a1_requant.py` (the Agents-A1 35B hybrid-MoE that
finally worked after the #2848 MoE-loading fix).

MTP: the upstream checkpoint ships 785 mtp.* tensors. save_pretrained drops them, so they are
re-attached with the official util. Note they are NOT servable on this box today: MoE NVFP4
requires --moe-backend marlin on sm120 (trtllm's Sm120_SafeFP4 kernel segfaults), and marlin
cannot also serve an unquantized bf16 draft MoE. Ship them anyway -- other backends may.

SOURCE IS DELETED. The bf16 checkpoint this reads was removed on 2026-08-21 after the quant
was gated and cut over (67 GB reclaimed). To re-run:
  HF_HOME=/mnt/models/huggingface hf download ornith-ai/Ornith-1.5-35B-A3B   # ~10 min
The glob below will raise if it is missing rather than failing deep in the run.

MEASURED: 41 sequential stages at 128 samples took 13 min on 2x RTX PRO 6000, not the ~85 min
the Agents-A1 precedent suggested. Peak ~90 GB per card, so the smart lane must be stopped.

RESOURCES: this needs a real dual-GPU window (~72 GB of bf16 weights + per-expert calibration
copies). Host RAM is 61 GB total / ~27 GB available, so CPU offload is NOT a way around it --
it will thrash. Stop the smart lane first.

  CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/ava/dev/quant-env/bin/python ornith15_35b_nvfp4_requant.py
"""
import os
os.environ.setdefault("HF_HOME", "/mnt/models/huggingface")

import torch
from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

# Resolve the snapshot by glob rather than pasting a hash -- a truncated hash copied out of
# a download log cost one false start here (transformers reports it as "Repo id must be in
# the form 'repo_name'", which reads like an auth/offline problem, not a typo).
import glob as _glob
_snaps = _glob.glob("/mnt/models/huggingface/hub/models--ornith-ai--Ornith-1.5-35B-A3B/snapshots/*/")
if len(_snaps) != 1:
    raise SystemExit(f"expected exactly one snapshot dir, found {_snaps}")
MAIN = _snaps[0].rstrip("/")
OUT = "/mnt/models/quantized/Ornith-1.5-35B-A3B-NVFP4"
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 2048

with load_context(Qwen3_5MoeForConditionalGeneration):
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        MAIN, torch_dtype="auto", device_map="auto"
    )
processor = AutoProcessor.from_pretrained(MAIN)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "re:.*lm_head",
        "re:visual.*",
        "re:model.visual.*",
        "re:.*mlp.gate$",            # router stays bf16 -- INT4/FP4 routing corrupts MoE
        "re:.*embed_tokens$",
        "re:.*shared_expert_gate$",
        "re:.*linear_attn.*",        # GDN hybrid layers stay bf16
        "re:.*mtp.*",
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
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        moe_calibrate_all_experts=True, data_collator=data_collator)

model.save_pretrained(OUT, save_compressed=True)
processor.save_pretrained(OUT)
save_mtp_tensors_to_checkpoint(source_model=MAIN, dest_dir=OUT)

# --- POST-SAVE FIXUP (feedback_vl_quant_save_pretrained) --------------------
import shutil
from pathlib import Path
_src, _out = Path(MAIN), Path(OUT)
for _f in ("tokenizer_config.json", "tokenizer.json", "processor_config.json",
           "preprocessor_config.json", "video_preprocessor_config.json",
           "vocab.json", "merges.txt", "generation_config.json", "chat_template.jinja"):
    if (_src / _f).exists():
        shutil.copy2(_src / _f, _out / _f)
        print(f"fixup: restored {_f}")

print(f"saved Ornith-1.5-35B-A3B NVFP4 to {OUT}")
print("VERIFY: packed census (visual/linear_attn/mtp/mlp.gate/lm_head == 0, experts FUSED),")
print("  then serve with --moe-backend marlin and gate on completion + TOOL CALL + IMAGE.")
