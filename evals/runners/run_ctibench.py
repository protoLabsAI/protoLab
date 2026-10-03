#!/usr/bin/env python3
"""CTIBench runner — threat-intel capability, VERIFIABLE grading (no LLM judge).

Probes the threat-intel + MITRE part of the security SFT. Tracks (AI4Sec/cti-bench ship a
pre-formatted `Prompt` + ground-truth `GT`):
  cti-mcq       2500  CTI knowledge MCQ            → letter exact-match accuracy
  cti-rcm       1000  CVE→CWE (2024, post-cutoff)  → CWE exact-match accuracy
  cti-rcm-2021  1000  CVE→CWE (2021, pre-cutoff)   → exact-match; date-control PAIR vs cti-rcm
  cti-vsp       1000  CVE→CVSS v3.1 vector         → base-score MAD (Python cvss lib; lower=better)
  cti-ate         60  threat text→ATT&CK tech IDs  → micro-F1
(cti-taa skipped — human attribution grading, not automatable.)

Usage:
  python -m runners.run_ctibench --model heretic-sft-v1 --gateway-url http://localhost:8042/v1 \
      --limit 0 --workers 8 --output-dir results/ctibench-sft
"""
import argparse, json, os, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)  # not installed in this venv
from datasets import load_dataset
from openai import OpenAI
from cvss import CVSS3

from sampling import resolve, to_openai_kwargs

REPO = "AI4Sec/cti-bench"
GRADEABLE = ["cti-mcq", "cti-rcm", "cti-rcm-2021", "cti-vsp", "cti-ate"]


def call(client, model, prompt, max_tokens, no_think=False):
    kw = {}
    if no_think:  # abliterated security SFTs are no-think models; thinking-on token-starves them
        kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    s = resolve("CTI")
    ok = to_openai_kwargs(s)
    if "extra_body" in kw:                       # merge, don't clobber no-think
        ok["extra_body"] = {**ok["extra_body"], **kw.pop("extra_body")}
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, **ok, **kw)
    m = r.choices[0].message
    return (m.content or getattr(m, "reasoning_content", "") or "")


# --- verifiable extractors (parse the model's answer out of prose/thinking) ---
def x_letter(t):
    m = re.findall(r"(?:answer|option)\D{0,8}([A-D])\b", t, re.I) or re.findall(r"\b([A-D])\b", t.upper())
    return m[-1].upper() if m else None

def x_cwe(t):
    m = re.findall(r"CWE[-\s]?(\d+)", t, re.I)
    return f"CWE-{m[-1]}" if m else None

def x_cvss(t):
    m = re.findall(r"CVSS:3\.[01]/[A-Za-z:/.]+", t)
    return m[-1] if m else None

def x_ate(t):
    return {x.upper() for x in re.findall(r"\bT\d{4}(?:\.\d+)?\b", t)}

def base_score(vec):
    try:
        return CVSS3(vec).scores()[0]
    except Exception:
        return None


def grade_track(track, rows, client, model, workers, max_tokens, no_think=False):
    def one(row):
        resp = call(client, model, row["Prompt"], max_tokens, no_think)
        gt = (row.get("GT") or "").strip()
        if track == "cti-mcq":
            p = x_letter(resp); return {"correct": p == gt.upper(), "pred": p, "gt": gt}
        if track in ("cti-rcm", "cti-rcm-2021"):
            p = x_cwe(resp); return {"correct": (p or "").upper() == gt.upper(), "pred": p, "gt": gt}
        if track == "cti-vsp":
            p = x_cvss(resp); pb = base_score(p) if p else None; gb = base_score(gt)
            ad = abs(pb - gb) if (pb is not None and gb is not None) else None
            return {"pred": p, "gt": gt, "abs_dev": ad, "parsed": p is not None}
        if track == "cti-ate":
            ps = x_ate(resp); gs = {x.strip().upper() for x in gt.split(",") if x.strip()}
            return {"tp": len(ps & gs), "n_pred": len(ps), "n_gt": len(gs)}

    res = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(one, r) for r in rows]):
            res.append(f.result())

    if track in ("cti-mcq", "cti-rcm", "cti-rcm-2021"):
        return {"metric": "accuracy", "score": round(sum(r["correct"] for r in res) / len(res), 4),
                "n": len(res)}
    if track == "cti-vsp":
        devs = [r["abs_dev"] for r in res if r["abs_dev"] is not None]
        return {"metric": "MAD_base(lower=better)", "score": round(sum(devs) / len(devs), 4) if devs else None,
                "n": len(res), "parsed": len(devs)}
    if track == "cti-ate":
        tp = sum(r["tp"] for r in res); npred = sum(r["n_pred"] for r in res); ngt = sum(r["n_gt"] for r in res)
        prec = tp / npred if npred else 0.0; rec = tp / ngt if ngt else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {"metric": "micro-F1", "score": round(f1, 4), "n": len(res),
                "precision": round(prec, 4), "recall": round(rec, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gateway-url", required=True)
    ap.add_argument("--tracks", default=",".join(GRADEABLE))
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8192, help="headroom so a thinking model isn't false-zeroed")
    ap.add_argument("--api-key", default="sk-x")
    ap.add_argument("--no-think", action="store_true",
                    help="disable thinking (REQUIRED for the abliterated security SFT — thinking-on token-starves it)")
    ap.add_argument("--output-dir", default="results/ctibench")
    a = ap.parse_args()

    client = OpenAI(base_url=a.gateway_url, api_key=a.api_key)
    out = {"model": a.model, "url": a.gateway_url, "no_think": a.no_think,
           "when": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tracks": {}}
    for t in [x.strip() for x in a.tracks.split(",") if x.strip()]:
        rows = list(load_dataset(REPO, t, split="test"))
        if a.limit:
            rows = rows[:a.limit]
        print(f"[ctibench] {t}: {len(rows)} items (no_think={a.no_think}) ...", flush=True)
        res = grade_track(t, rows, client, a.model, a.workers, a.max_tokens, a.no_think)
        out["tracks"][t] = res
        print(f"  → {res}", flush=True)

    od = Path(a.output_dir); od.mkdir(parents=True, exist_ok=True)
    (od / "ctibench_results.json").write_text(json.dumps(out, indent=2))
    print("wrote", od / "ctibench_results.json")


if __name__ == "__main__":
    main()
