#!/usr/bin/env python3
"""Release gate for a VL quant: completion + TOOL CALL + IMAGE.

Text-only checks pass on a checkpoint whose vision is completely dead (see
feedback_vl_quant_save_pretrained) -- that is the whole reason this exists. Run it
against any OpenAI-compatible endpoint serving the quant.

  python gate_vl_quant.py --url http://127.0.0.1:8062/v1 --model ornith-1.5-9b-nvfp4
"""
import argparse, base64, json, sys, urllib.request
sys.path.insert(0, "/home/ava/dev/lab/evals/graders")
from probe_lib import extract_text, MIN_SANE_BUDGET  # shared field-drift + budget rules

OCR_IMAGE = "/home/ava/dev/lab/experiments/vision-eval/ocr_test.png"
SHAPES_IMAGE = "/home/ava/dev/lab/experiments/vision-eval/shapes_test.png"
OCR_TRUTH = "protolabs vlm-42"
# Ornith-1.5 thinks adaptively; anything under a prod-like budget measures the
# budget, not the model (feedback_eval_prod_token_budget). 400 tokens returned
MAX_TOKENS = 4096  # an EMPTY completion on a checkpoint that was fine.
assert MAX_TOKENS >= MIN_SANE_BUDGET
IMAGE_TRIALS = 5


def post(url, payload, timeout=180):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    ep = a.url.rstrip("/") + "/chat/completions"
    results = {}

    # ---- 1. completion -------------------------------------------------
    r = post(ep, {"model": a.model, "max_tokens": MAX_TOKENS, "temperature": 0.7,
                  "messages": [{"role": "user", "content":
                                "In three sentences, explain why a quantized model can serve text "
                                "correctly while its vision path is completely broken."}]})
    msg = r["choices"][0]["message"]
    text = extract_text(msg)
    ok = len(text) > 80 and text.count("�") == 0
    results["completion"] = ok
    print(f"[{'PASS' if ok else 'FAIL'}] completion ({len(text)} chars)\n  {text[:220]}...\n")

    # ---- 2. tool call --------------------------------------------------
    tools = [{"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"},
                                      "unit": {"type": "string", "enum": ["c", "f"]}},
                       "required": ["city"]}}}]
    r = post(ep, {"model": a.model, "max_tokens": MAX_TOKENS, "temperature": 0.7, "tools": tools,
                  "messages": [{"role": "user",
                                "content": "What's the weather in Reykjavik in celsius? Use the tool."}]})
    msg = r["choices"][0]["message"]
    tc = msg.get("tool_calls") or []
    ok = False
    if tc:
        try:
            args = json.loads(tc[0]["function"]["arguments"])
            ok = tc[0]["function"]["name"] == "get_weather" and "reykjav" in args.get("city", "").lower()
        except Exception:
            ok = False
    results["tool_call"] = ok
    print(f"[{'PASS' if ok else 'FAIL'}] tool call\n  {json.dumps(tc)[:300]}\n")

    # ---- 3. image ------------------------------------------------------
    # HARD GATE = shapes. It asks whether the vision path is ALIVE AND CORRECT, which is
    # what quantization actually breaks (a dead path answers confidently about an image it
    # never saw, or dies on an image-token count mismatch).
    #
    # The wordmark OCR probe is INFORMATIONAL ONLY, and that is a deliberate correction:
    # gating on it fails good checkpoints. Measured 2026-08-21, Ornith-1.5-9B, n=20/side --
    # bf16 1/20 exact, NVFP4 1/20 exact. The base model reads "protoLabs" as "protocolabs"
    # almost every time and invents a trailing digit ("VLM-429"). That is a base-model
    # weakness on a stylised wordmark, not quantization damage. For a real quant-vs-source
    # verdict use vision_parity.py, which tests the DIFFERENCE at adequate n.
    def img_probe(path, prompt, trials):
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        out = []
        for _ in range(trials):
            r = post(ep, {"model": a.model, "max_tokens": MAX_TOKENS, "temperature": 0.7,
                          "messages": [{"role": "user", "content": [
                              {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                              {"type": "text", "text": prompt}]}]})
            m = r["choices"][0]["message"]
            out.append(extract_text(m))
        return out

    shapes = img_probe(SHAPES_IMAGE,
                       "What shapes and colors are in this image? Answer in one short sentence.",
                       IMAGE_TRIALS)
    def shapes_ok(t):
        t = t.lower()
        return ("red" in t and "blue" in t
                and any(w in t for w in ("circle", "round"))
                and any(w in t for w in ("square", "rectangle")))
    hits = sum(shapes_ok(t) for t in shapes)
    ok = hits == IMAGE_TRIALS
    results["image"] = ok
    print(f"[{'PASS' if ok else 'FAIL'}] image / shapes {hits}/{IMAGE_TRIALS}")
    for t in shapes[:3]:
        print(f"    {t[:100]!r}")

    ocr = img_probe(OCR_IMAGE, "Transcribe the text in this image exactly. Reply with the text only.", 3)
    ex = sum(OCR_TRUTH in t.lower().replace("**", "") for t in ocr)
    print(f"[info] image / wordmark OCR {ex}/3 exact (NOT a gate — base model scores ~1/20)")
    for t in ocr:
        print(f"    {t[:100]!r}")
    print()

    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:12s} {'PASS' if v else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
