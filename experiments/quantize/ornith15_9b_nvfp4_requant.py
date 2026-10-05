#!/usr/bin/env python3
"""Ornith-1.5-9B NVFP4 requant (+ our distilled bf16 MTP head, single shot).

Requested on the hub: protoLabsAI/Ornith-1.5-9B-MTP-GGUF discussion #1 ("NVFP4 build
request"). Upstream ships 1.5-9B as bf16 / GGUF / MLX only -- there is no NVFP4 anywhere,
and our own 1.5 GGUF card lists this as the blocking prerequisite for the NVFP4 GGUF rung.

Same dense hybrid-GDN + VL shape as Ornith-1.0-9B (`Qwen3_5ForConditionalGeneration`,
32 layers, 3:1 linear-attn:full, vision tower), so the proven recipe ports directly from
`qwen38_27b_nvfp4_requant.py`.

SOURCE IS OUR MTP CHECKPOINT, NOT UPSTREAM. ornith-ai ships 1.5-9B with
`mtp_num_hidden_layers: 1` and zero `mtp.*` tensors; /mnt/data/checkpoints/ornith-1.5-9b-mtp
is the grafted + KL-distilled head baked into the trunk (775 tensors, 15 of them mtp.*).
Quantizing that keeps the head bf16 in the SAME shot, so the NVFP4 release ships MTP-capable
the way Ornith-1.0-9B-NVFP4 does.

  vision tower / GDN linear-attn / lm_head / embed / mtp stay bf16 -- no sm120 W4A4 kernel,
  and DeltaNet corrupts under low-precision activations (standing finding on this arch).

Run (needs ~24GB, one card):
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/ava/dev/quant-env/bin/python ornith15_9b_nvfp4_requant.py
"""
import os
os.environ.setdefault("HF_HOME", "/mnt/models/huggingface")

import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MAIN = "/mnt/data/checkpoints/ornith-1.5-9b-mtp"
PROC_SRC = MAIN
OUT = "/mnt/models/quantized/Ornith-1.5-9B-NVFP4"
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
        "re:.*mtp.*",          # keep our distilled MTP head bf16 (drafts; target verifies)
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
    # 4-D DeltaNet crash without this.
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}

oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES, data_collator=data_collator)

model.save_pretrained(OUT, save_compressed=True)
processor.save_pretrained(OUT)

# ---------------------------------------------------------------------------
# POST-SAVE FIXUP -- see feedback_vl_quant_save_pretrained. Every one of these
# failures yields a checkpoint that loads fine and serves TEXT correctly.
# NOTE the deviation from the Qwen3.8 script: Ornith-1.5-9B DOES ship a real
# processor_config.json upstream, so it is restored, not deleted.
# ---------------------------------------------------------------------------
import shutil, json as _json
from pathlib import Path
_src, _out = Path(MAIN), Path(OUT)

# 1. MTP head is dropped by save_pretrained (Qwen3_5ForConditionalGeneration does
#    not own `mtp.*`). Ship the sidecar and record the ignore entry llm-compressor
#    could not record because it never saw those modules.
_cfg_p = _out / "config.json"
_cfg = _json.loads(_cfg_p.read_text())
_ig = _cfg.get("quantization_config", {}).get("ignore", [])
if _ig and not any("mtp" in x for x in _ig):
    _ig.append("re:.*mtp.*")
    _cfg_p.write_text(_json.dumps(_cfg, indent=2))
    print("fixup: appended re:.*mtp.* to quantization_config.ignore")
if (_src / "model-mtp.safetensors").exists():
    shutil.copy2(_src / "model-mtp.safetensors", _out / "model-mtp.safetensors")
    print("fixup: copied model-mtp.safetensors sidecar")

# 2. Tokenizer rewritten -> breaks VISION only (image token count mismatch that
#    reads like a truncation bug). Quantization never changes tokenization.
for _f in ("tokenizer_config.json", "tokenizer.json"):
    shutil.copy2(_src / _f, _out / _f)
    print(f"fixup: restored source {_f}")

# 3/4. Aux configs -- vision/video preprocessors are load-bearing. Restore from
#      source verbatim (processor.save_pretrained downgrades the image processor).
for _f in ("processor_config.json", "preprocessor_config.json",
           "video_preprocessor_config.json", "vocab.json", "merges.txt",
           "generation_config.json", "chat_template.jinja"):
    if (_src / _f).exists():
        shutil.copy2(_src / _f, _out / _f)
        print(f"fixup: restored {_f}")

print(f"saved Ornith-1.5-9B NVFP4 (with bf16 MTP head) to {OUT}")
print("VERIFY BEFORE SHIPPING: packed census (visual/linear_attn/mtp/lm_head == 0),")
print("  a real completion, a TOOL CALL, and an IMAGE -- text-only smoke tests pass")
print("  on a checkpoint whose vision is completely broken.")
