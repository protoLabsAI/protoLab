"""Qwen3-Coder-Next-Opus-4.6-Reasoning-Distilled → NVFP4 requant.

Apples-to-apples with our incumbent Qwen3-Coder-Next-NVFP4 (same qwen3_next arch: 48 layers,
512 experts/10 active, hybrid linear+full attention). Same ignore policy as the incumbent's
quantization_config: keep the GDN linear-attn projections, the MoE router gates, and lm_head in
bf16 (INT4/FP4 corrupts MoE routing + DeltaNet has no sm120 W4A4 kernel); NVFP4 everything else.

Text-only coder (no vision tower, no MTP head) → drop the visual/mtp handling from the 27B recipe.
Source is a LOCAL bf16 path (pre-downloaded to /mnt/data — /mnt/models is full and quantize would
OOM the disk if it re-pinned HF_HOME there). OUT lands on /mnt/data for the same reason.
"""
import os
# HF_HOME OFF the full /mnt/models disk — datasets (calibration) download here, not the source.
os.environ["HF_HOME"] = "/mnt/data/.hf-cache-challenger"

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "/mnt/data/models-src/Coder-Opus-Distill-bf16"
OUT = "/mnt/data/models/Coder-Opus-Distill-NVFP4"
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 2048

# 160GB bf16 model on a 61GB-RAM node → must disk-offload what won't fit on the (single, freed)
# GPU. Run with CUDA_VISIBLE_DEVICES=1 (coder's card, seen here as device 0). llm-compressor's
# sequential pipeline then streams layers GPU-ward for calibration, so this fits without both cards.
# If OOM/offload trouble, escalate to a brief full-fleet stop + device_map across BOTH GPUs.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    max_memory={0: "45GiB", "cpu": "44GiB"},  # GPU capped LOW: leaves ~50G headroom for linearize_moe
    # (512-expert MoE restructuring) + calibration forward passes. Rest disk-offloads. Was OOMing at
    # 84G because the loaded weights maxed the card (92G) with nothing left for the MoE-linearize step.
    offload_folder="/mnt/data/.offload-challenger",
    offload_state_dict=True,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "re:.*lm_head",
        "re:.*linear_attn.*",         # GDN hybrid layers stay bf16 (no sm120 W4A4 kernel)
        "re:.*mlp\\.gate$",           # MoE router gate — bf16 (FP4 corrupts routing)
        "re:.*shared_expert_gate$",   # shared-expert gate — bf16
    ],
)

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.select_columns(["messages"]).shuffle(seed=42)


def preprocess_function(example):
    return tokenizer.apply_chat_template(
        example["messages"],
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
tokenizer.save_pretrained(OUT)
print(f"saved NVFP4 to {OUT}")
