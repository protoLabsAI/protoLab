#!/usr/bin/env python3
"""Coherence + MTP acceptance + decode speed across the Ornith-1.5-9B GGUF rungs.

Our own 1.5 card measured MTP at 1.77x on Q8_0 but only 1.13x on Q4_K_M -- and SLOWER
than no-MTP on creative prose. That trend has to be re-measured, not extrapolated, before
the IQ rungs get a card: at 3 and 2 bits the answer may well be "turn MTP off".

Each rung is served twice (with and without --spec-type draft-mtp) over the same prompts.
Speed numbers are single-stream (C=1) and are therefore INTERNAL REFERENCE ONLY unless the
box is quiet -- see feedback_speed_numbers_honest.

  python bench_iq_rungs.py --rungs IQ4_XS IQ3_M IQ2_M --out results.json
"""
import argparse, json, subprocess, time, urllib.request, urllib.error, os, signal, sys

FORGE = "/mnt/data/gguf-forge/Ornith-1.5-9B-MTP"
BIN = "/home/ava/dev/llama.cpp/build-cuda/bin/llama-server"
PORT = 8099

PROMPTS = [
    ("code", "Write a Python function that merges two sorted lists into one sorted list. Explain the complexity."),
    ("prose", "Write an opening paragraph for a story about a lighthouse keeper who stops receiving mail."),
    ("factual", "Explain what an importance matrix does in llama.cpp quantization, and why it matters more at 3 bits than at 8."),
    ("runbook", "Give a numbered runbook for diagnosing a GPU process that will not release VRAM on Linux."),
    ("reasoning", "A train leaves at 14:20 travelling 80 km/h. A second leaves the same station at 15:05 travelling 110 km/h on the same track. When does the second catch the first?"),
    ("technical", "Explain the difference between speculative decoding acceptance rate and end-to-end speedup."),
]


def wait_ready(proc, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read()
            return True
        except Exception:
            time.sleep(2)
    return False


def run_arm(gguf, mtp, n_predict=200):
    cmd = [BIN, "--model", gguf, "--n-gpu-layers", "99", "--ctx-size", "8192",
           "--flash-attn", "on", "--jinja", "--port", str(PORT), "--host", "127.0.0.1",
           "--no-warmup"]
    if mtp:
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            env=env, preexec_fn=os.setsid)
    try:
        if not wait_ready(proc):
            return {"error": "server did not become ready"}
        rows = []
        for tag, prompt in PROMPTS:
            body = json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "n_predict": n_predict, "max_tokens": n_predict,
                "temperature": 0.0, "cache_prompt": False,
            }).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                         data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.load(r)
            tim = d.get("timings", {}) or d.get("usage", {}).get("timings", {}) or {}
            txt = d["choices"][0]["message"].get("content") or ""
            rows.append({"tag": tag, "tok_s": tim.get("predicted_per_second"),
                         "draft_n": tim.get("draft_n"), "draft_n_accepted": tim.get("draft_n_accepted"),
                         "text": txt})
        return {"rows": rows}
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            pass
        time.sleep(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", nargs="+", default=["IQ4_XS", "IQ3_M", "IQ2_M"])
    ap.add_argument("--out", default=f"{FORGE}/imatrix/iq-bench.json")
    a = ap.parse_args()

    out = {}
    for rung in a.rungs:
        gguf = f"{FORGE}/out/Ornith-1.5-9B-MTP-{rung}.gguf"
        if not os.path.exists(gguf):
            print(f"skip {rung}: missing"); continue
        out[rung] = {}
        for arm in ("base", "mtp"):
            print(f"=== {rung} / {arm} ===", flush=True)
            out[rung][arm] = run_arm(gguf, mtp=(arm == "mtp"))
            r = out[rung][arm]
            if "rows" in r:
                sp = [x["tok_s"] for x in r["rows"] if x["tok_s"]]
                dn = sum(x["draft_n"] or 0 for x in r["rows"])
                da = sum(x["draft_n_accepted"] or 0 for x in r["rows"])
                acc = (da / dn) if dn else None
                print(f"    mean {sum(sp)/len(sp):.1f} tok/s"
                      + (f"  acceptance {acc:.3f}" if acc is not None else ""), flush=True)
            else:
                print("    " + str(r), flush=True)
            json.dump(out, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
