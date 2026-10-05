#!/usr/bin/env python3
"""Assemble the imatrix calibration corpus for the Ornith-1.5-9B IQ rungs.

Composition is deliberate. An importance matrix is only as good as the activation
distribution it sees, and i-quants at IQ3/IQ2 punish anything the corpus never lit up:

  ~70%  Ornith-1.5-9B's OWN generations (agentic/business + coding + general chat) --
        the same 3942-sample corpus the MTP head was KL-distilled against, so the
        calibration distribution matches what the model actually emits in deployment.
  ~30%  literary prose passages -- breadth the self-generated corpus lacks. Instruct
        output is register-narrow; without this the low-bit rungs degrade on creative
        writing first, which is exactly where Q4_K_M's MTP gain already looked worst.

Both halves interleave user turns and assistant turns so prompt-side tokens are
represented, not just completions.

Usage:  python build_calibration.py --out /mnt/data/gguf-forge/.../calibration.txt
"""
import argparse, json, random

CORPUS = "/mnt/data/datasets/ornith-1.5-9b-mtp/corpus.jsonl"
PROSE = "/mnt/data/datasets/creative/passages.jsonl"


def load_self_gen(path, budget):
    out, n = [], 0
    for line in open(path):
        d = json.loads(line)
        parts = [m["content"] for m in d.get("messages", []) if m.get("content")]
        if d.get("text"):
            parts.append(d["text"])
        chunk = "\n\n".join(parts).strip()
        if len(chunk) < 200:
            continue
        out.append(chunk)
        n += len(chunk)
        if n >= budget:
            break
    return out


def load_prose(path, budget):
    out, n = [], 0
    for line in open(path):
        t = json.loads(line).get("text", "").strip()
        if len(t) < 400:
            continue
        out.append(t)
        n += len(t)
        if n >= budget:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--total-bytes", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=1985)
    a = ap.parse_args()

    random.seed(a.seed)
    blocks = load_self_gen(CORPUS, int(a.total_bytes * 0.70))
    prose = load_prose(PROSE, int(a.total_bytes * 0.30))
    print(f"self-gen blocks: {len(blocks)}  prose blocks: {len(prose)}")
    blocks += prose
    random.shuffle(blocks)

    with open(a.out, "w") as f:
        f.write("\n\n".join(blocks))
    import os
    print(f"wrote {a.out}  {os.path.getsize(a.out)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
