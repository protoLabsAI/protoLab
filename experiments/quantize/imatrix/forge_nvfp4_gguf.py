#!/usr/bin/env python3
"""Forge the bundled NVFP4 GGUF rung from the BF16 master.

`llama-quantize` has NO NVFP4 target (it is absent from QUANT_OPTIONS even though
GGML_TYPE_NVFP4 = 40 and LLAMA_FTYPE_MOSTLY_NVFP4 = 39 both exist in this build), so the
rung has to be written through gguf-py's NVFP4 quantizer instead of a one-line CLI call.

Composition mirrors the shipped Ornith-1.0-9B-MTP-NVFP4.gguf (159 NVFP4 / 99 Q8_0 / 440 F32):

  NVFP4   the dense GEMMs -- ffn_{down,gate,up}, attn_qkv, attn_gate, attn_{q,k,v,output}
  Q8_0    token_embd + output (2 GB of the file on their own), every ssm_* DeltaNet tensor,
          and the ENTIRE MTP block. DeltaNet corrupts under low-precision -- the standing
          finding on this arch, and the same reason the vLLM recipe keeps linear_attn bf16.
          The MTP head is pinned for the reason the i-quant rungs pin it: it is the draft,
          and a degraded draft costs acceptance on every token.
  F32     norms and other 1-D tensors, untouched.

MEMORY: GGUFWriter with no temp file holds every packed tensor in RAM until the final
write, so this peaks at ~(output size) plus a transient float32 buffer for the largest
tensor -- ~6.5 GB + ~4 GB for token_embd on this model. Fine on a 61 GB box, not fine on
a small one; pass use_temp_file=True to GGUFWriter if you ever hit that.

  python forge_nvfp4_gguf.py --dry-run     # preview types + predicted size
  python forge_nvfp4_gguf.py
"""
import argparse, ctypes, re, sys
sys.path.insert(0, "/home/ava/dev/llama.cpp/gguf-py")
import numpy as np
import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as QT

# gguf-py's NVFP4 class implements dequantize_blocks ONLY -- there is no Python
# quantizer -- so go through ggml's own reference implementation via ctypes. This is
# the exact code path llama-quantize uses, so the output is byte-identical to what the
# CLI would produce if it exposed NVFP4. Verified by round-tripping the C output back
# through gguf-py's independent Python dequantize (corr 0.9955 NVFP4 / 0.999986 Q8_0).
_LIB = ctypes.CDLL("/home/ava/dev/llama.cpp/build-cuda/bin/libggml-base.so")
_LIB.ggml_quantize_chunk.restype = ctypes.c_size_t
_LIB.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                                     ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                     ctypes.c_int64, ctypes.POINTER(ctypes.c_float)]
_LIB.ggml_quantize_init.argtypes = [ctypes.c_int]
_GGML_TYPE = {QT.NVFP4: 40, QT.Q8_0: 8}


def quantize(arr_f32, qt):
    arr = np.ascontiguousarray(arr_f32, dtype=np.float32)
    nrows = int(np.prod(arr.shape[:-1])) if arr.ndim > 1 else 1
    nper = arr.shape[-1]
    blk, tsz = gguf.GGML_QUANT_SIZES[qt]
    if nper % blk:
        raise SystemExit(f"row of {nper} not divisible by {qt.name} block {blk}")
    row_size = nper // blk * tsz
    out = np.empty(nrows * row_size, dtype=np.uint8)
    _LIB.ggml_quantize_init(_GGML_TYPE[qt])
    n = _LIB.ggml_quantize_chunk(_GGML_TYPE[qt],
                                 arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                 out.ctypes.data_as(ctypes.c_void_p), 0, nrows, nper, None)
    assert n == nrows * row_size, (n, nrows * row_size)
    return out.reshape(nrows, row_size)

SRC = "/mnt/data/gguf-forge/Ornith-1.5-9B-MTP/out/Ornith-1.5-9B-MTP-BF16.gguf"
DST = "/mnt/data/gguf-forge/Ornith-1.5-9B-MTP/out/Ornith-1.5-9B-MTP-NVFP4.gguf"
NAME = "Ornith 1.5 9B MTP NVFP4"
MTP_BLOCK = 32  # the nextn block

NVFP4_SUFFIXES = ("ffn_down.weight", "ffn_gate.weight", "ffn_up.weight",
                  "attn_qkv.weight", "attn_gate.weight",
                  "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight")


def target_type(name: str, n_dims: int, src_type=None, row: int = 0) -> QT:
    # Never re-quantize what is already F32 (norms, ssm_a/dt/conv1d), and never try to
    # quantize a row that is not block-divisible -- ssm_conv1d.weight has rows of 4.
    # llama-quantize applies the same fallbacks.
    if src_type == QT.F32:
        return QT.F32
    if n_dims < 2:
        return QT.F32                      # norms etc. stay as-is
    blk = re.match(r"blk\.(\d+)\.", name)
    if blk and int(blk.group(1)) == MTP_BLOCK:
        return QT.Q8_0                     # pin the whole draft block
    if "ssm_" in name:
        return QT.Q8_0                     # DeltaNet
    if name in ("token_embd.weight", "output.weight"):
        return QT.Q8_0
    want = QT.NVFP4 if any(name.endswith(s) for s in NVFP4_SUFFIXES) else QT.Q8_0
    if row and row % gguf.GGML_QUANT_SIZES[want][0]:
        return QT.F32
    return want


def to_f32(t):
    d = t.data
    if t.tensor_type == QT.BF16:
        return (d.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    if t.tensor_type == QT.F16:
        return d.astype(np.float32)
    if t.tensor_type == QT.F32:
        return d
    raise SystemExit(f"unexpected source type {t.tensor_type} on {t.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    a = ap.parse_args()

    r = GGUFReader(a.src)
    plan, total = [], 0
    for t in r.tensors:
        row = int(t.shape[0])   # GGUF ne0 == elements per row
        tt = target_type(t.name, len(t.shape), t.tensor_type, row)
        # NB: gguf-py hands BF16 back as raw uint8, so data.size is 2x the element
        # count. Size math must use the logical shape or it double-counts.
        elems = 1
        for x in t.shape:
            elems *= int(x)
        if tt == QT.F32 and t.tensor_type == QT.F32:
            nbytes = elems * 4
        else:
            blk, sz = gguf.GGML_QUANT_SIZES[tt]
            nbytes = elems // blk * sz
        plan.append((t, tt))
        total += nbytes
    from collections import Counter
    print("target types:", dict(Counter(tt.name for _, tt in plan)))
    print(f"predicted size: {total/1e9:.2f} GB")
    if a.dry_run:
        return

    w = GGUFWriter(a.dst, arch=r.get_field("general.architecture").contents(),
                   endianess=r.endianess)
    for field in r.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        if field.name == gguf.Keys.General.FILE_TYPE:
            continue
        if field.name == gguf.Keys.General.NAME:
            continue
        vt = field.types[0]
        st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(field.name, field.contents(), vt, sub_type=st)
    w.add_name(NAME)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_NVFP4)

    for i, (t, tt) in enumerate(plan):
        if tt == QT.F32 and t.tensor_type == QT.F32:
            data = t.data
        else:
            data = quantize(to_f32(t), tt)
        # NB: do NOT pass raw_shape here. add_tensor_info() treats the shape of a uint8
        # payload as a BYTE shape and converts it back to elements itself; handing it the
        # logical shape makes it try to divide 4096 by the 34-byte Q8_0 block and throw.
        w.add_tensor(t.name, data, raw_dtype=tt)
        if i % 50 == 0:
            print(f"  [{i+1}/{len(plan)}] {t.name} -> {tt.name}", flush=True)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {a.dst}")


if __name__ == "__main__":
    main()
