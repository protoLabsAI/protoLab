#!/usr/bin/env python3
"""Quantize an LTX-2.3 bf16 checkpoint to NVFP4 (fp4_mixed), replicating Lightricks'
exact layer-selection policy reverse-engineered from the shipped 19B fp4 checkpoint.

Policy (matches ltx-2-19b-dev-fp4.safetensors):
  - Quantize ONLY 2D Linear weights in model.diffusion_model.transformer_blocks
  - KEEP block 0 and the last 5 blocks (43..47) in bf16   [first/last precision retention]
  - KEEP all to_gate_logits gates in bf16                 [tiny, gating-sensitive]
  - KEEP everything else in bf16: patchify/proj_out, adaln, caption/embeddings
    connectors, VAE, audio_vae, vocoder, and all non-2D tensors.

Each quantized weight K becomes three keys (comfy_kitchen serialization, verified
identical to the shipped fp4): K (uint8 packed fp4), K_scale (fp8_e4m3 per-block),
K_scale_2 (fp32 per-tensor). ComfyUI's LTX loader auto-detects fp4 from these sidecars.

Run with ComfyUI's venv (has comfy_kitchen + torch cu130):
  CUDA_VISIBLE_DEVICES=1 ~/dev/ComfyUI/venv/bin/python ltx23_nvfp4_quantize.py \
      --in  /mnt/data/models-cold/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors \
      --out /mnt/data/models-cold/LTX-2.3/ltx-2.3-22b-distilled-1.1-fp4.safetensors
"""
import argparse, re, time, torch
from safetensors import safe_open
from safetensors.torch import save_file
from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout as NVFP4

KEEP_BLOCKS = {0, 43, 44, 45, 46, 47}          # bf16 (matches 19B: block 0 + last 5)
BLOCK_RE = re.compile(r"model\.diffusion_model\.transformer_blocks\.(\d+)\.")

def is_quant_target(key: str, shape) -> bool:
    if not key.endswith(".weight"):
        return False
    if len(shape) != 2:
        return False
    m = BLOCK_RE.match(key)
    if not m:                                   # only the main transformer_blocks
        return False
    if int(m.group(1)) in KEEP_BLOCKS:
        return False
    if "to_gate_logits" in key:                 # keep gates high-precision
        return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--limit", type=int, default=0, help="quantize only first N targets (dry test)")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "need CUDA for ck.quantize_nvfp4"
    t0 = time.time()
    out = {}
    quant_layers = {}                                # module_path -> {"format":"nvfp4"}
    n_q = n_kept = n_qbytes = 0
    with safe_open(args.src, framework="pt", device="cpu") as f:
        src_meta = f.metadata() or {}
        keys = list(f.keys())
        targets = [k for k in keys if is_quant_target(k, f.get_slice(k).get_shape())]
        if args.limit:
            targets = targets[:args.limit]
        tset = set(targets)
        print(f"{len(keys)} tensors total | {len(targets)} → NVFP4 | {len(keys)-len(targets)} → bf16 passthrough")
        for i, k in enumerate(keys):
            t = f.get_tensor(k)
            if k in tset:
                qdata, params = NVFP4.quantize(t.to("cuda"))
                sd = NVFP4.state_dict_tensors(qdata, params)  # {"":qdata,"_scale":bs,"_scale_2":s}
                for suf, qt in sd.items():
                    out[k + suf] = qt.cpu().contiguous()
                    n_qbytes += out[k + suf].numel() * out[k + suf].element_size()
                quant_layers[k[:-len(".weight")]] = {"format": "nvfp4"}   # module path, no .weight
                n_q += 1
                if n_q % 200 == 0:
                    print(f"  [{n_q}/{len(targets)}] quantized … {time.time()-t0:.0f}s", flush=True)
            else:
                out[k] = t.contiguous()
                n_kept += 1
            del t
    # Header the ComfyUI LTX loader reads to build NVFP4 layers (the load trigger).
    import json as _json
    meta = {k: v for k, v in src_meta.items() if k != "encrypted_wandb_properties"}
    meta["_quantization_metadata"] = _json.dumps({"format_version": "1.0", "layers": quant_layers})
    meta["quant_source"] = args.src.split("/")[-1]
    print(f"quantized {n_q} weights (~{n_qbytes/1e9:.1f} GB fp4), kept {n_kept} bf16 tensors")
    print(f"_quantization_metadata: {len(quant_layers)} layers tagged nvfp4")
    print(f"writing {args.dst} …")
    save_file(out, args.dst, metadata=meta)
    print(f"DONE in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
