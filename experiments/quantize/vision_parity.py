#!/usr/bin/env python3
"""Vision parity: quantized VL checkpoint vs its bf16 source, on identical probes.

The point of a VL gate is NOT "can the model read this image well" -- it is "did
quantization take something away". Absolute OCR skill is a property of the base model:
on our wordmark probe the bf16 Ornith-1.5-9B itself hallucinates a trailing digit
(protoLabs VLM-42 -> "VLM-429") in most samples. So score both sides on the same probe
and test the DIFFERENCE, with enough trials to say anything at all.

n=5 per side is not enough: see feedback_underpowered_is_not_null. Default is 20/side and
a two-sided Fisher exact test on the match counts.

  python vision_parity.py --a 8063 ornith-1.5-9b-bf16 --b 8062 ornith-1.5-9b-nvfp4 -n 20
"""
import argparse, base64, json, urllib.request
from math import comb

VISION = "/home/ava/dev/lab/experiments/vision-eval"
PROBES = [
    # (name, image, prompt, scorer)
    ("ocr_exact", f"{VISION}/ocr_test.png",
     "Transcribe the text in this image exactly. Reply with the text only.",
     lambda t: "protolabs vlm-42" in t.lower().replace("**", "").replace("-", "-")),
    ("ocr_digits", f"{VISION}/ocr_test.png",
     "Transcribe the text in this image exactly. Reply with the text only.",
     # weaker: did it get the alphanumeric token right, ignoring the wordmark's casing?
     lambda t: "vlm-42" in t.lower().replace("**", "") and "429" not in t),
    ("shapes", f"{VISION}/shapes_test.png",
     "What shapes and colors are in this image? Answer in one short sentence.",
     lambda t: all(w in t.lower() for w in ("red", "blue")) and
               any(w in t.lower() for w in ("circle", "round")) and
               any(w in t.lower() for w in ("square", "rectangle"))),
]


def ask(port, model, img_b64, prompt, timeout=300):
    body = json.dumps({"model": model, "max_tokens": 4096, "temperature": 0.7,
                       "messages": [{"role": "user", "content": [
                           {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                           {"type": "text", "text": prompt}]}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    m = d["choices"][0]["message"]
    return (m.get("content") or "").strip() or (m.get("reasoning") or "").strip()


def fisher_two_sided(a1, a0, b1, b0):
    """Two-sided Fisher exact on [[a1,a0],[b1,b0]]."""
    n = a1 + a0 + b1 + b0
    row1, col1 = a1 + a0, a1 + b1
    def p(k):
        return comb(row1, k) * comb(n - row1, col1 - k) / comb(n, col1)
    obs = p(a1)
    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    return sum(p(k) for k in range(lo, hi + 1) if p(k) <= obs * (1 + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs=2, metavar=("PORT", "MODEL"), required=True)
    ap.add_argument("--b", nargs=2, metavar=("PORT", "MODEL"), required=True)
    ap.add_argument("-n", type=int, default=20)
    args = ap.parse_args()
    sides = [("A/" + args.a[1], int(args.a[0]), args.a[1]),
             ("B/" + args.b[1], int(args.b[0]), args.b[1])]

    cache = {}
    print(f"{'probe':12s} {'side':28s} {'hits':>8s}")
    table = {}
    for name, img, prompt, score in PROBES:
        if img not in cache:
            cache[img] = base64.b64encode(open(img, "rb").read()).decode()
        for label, port, model in sides:
            hits, samples = 0, []
            for _ in range(args.n):
                t = ask(port, model, cache[img], prompt)
                if score(t):
                    hits += 1
                samples.append(t[:80])
            table[(name, label)] = (hits, samples)
            print(f"{name:12s} {label:28s} {hits:3d}/{args.n}")
    print()
    print(f"{'probe':12s} {'A':>8s} {'B':>8s} {'delta':>8s} {'p(2-sided)':>12s}")
    for name, _, _, _ in PROBES:
        ah = table[(name, sides[0][0])][0]
        bh = table[(name, sides[1][0])][0]
        p = fisher_two_sided(ah, args.n - ah, bh, args.n - bh)
        print(f"{name:12s} {ah:5d}/{args.n} {bh:5d}/{args.n} {bh-ah:+8d} {p:12.4f}")
    print("\nrequired n for a real effect of size d (rule of thumb ~16/d^2) — report it "
          "rather than calling a null from a small sample.")
    for name, _, _, _ in PROBES:
        print(f"\n--- {name} samples ---")
        for label, _, _ in sides:
            print(f"  {label}: {table[(name,label)][1][:4]}")


if __name__ == "__main__":
    main()
