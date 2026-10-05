#!/usr/bin/env python3
"""Score a /v1/systemone server on TypeSafe's public workflow eval (build_typesafe.py first).

Agreement with the reference consensus (two frontier models), the same per-pair metric the
public viewer uses, beside the published Opus / Sol / TypeSafe (Jev) answers on the SAME pairs.
"strict common" = pairs where all three published models answered.

  python ts_eval.py --url http://localhost:8070 --label smart-s1
Writes results/<label>.jsonl (one row per pair) and prints the table.
"""
import argparse, json, math, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PUB = ["opus", "sol", "typesafe"]


def to_request(case):
    qs = {}
    for q in case["questions"]:
        d = q["descs"] or [None] * len(q["options"])
        if q["kind"] == "noul":
            crit = {"false": d[0], "true": d[1]} if any(d) else None
            qs[q["qid"]] = {"type": "noul", "instructions": q["text"], **({"criteria": crit} if crit else {})}
        elif q["kind"] == "choice":
            qs[q["qid"]] = {"type": "choice", "instructions": q["text"], "criteria": dict(zip(q["options"], d))}
        else:
            qs[q["qid"]] = {"type": "score", "instructions": q["text"], "criteria": [x or "" for x in d]}
    return {"state": case["state"], "questions": qs}


def ours(ans, kind):
    """-> (canonical value, prob dict over canonical options)."""
    if kind == "noul":
        p = ans["noul"]
        return ("yes" if p >= 0.5 else "no"), {"yes": p, "no": 1 - p}
    if kind == "choice":
        return ans["choice"], ans["probabilities"]
    probs = ans["probabilities"]
    return max(probs, key=probs.get), probs


def post(url, body):
    req = urllib.request.Request(f"{url}/v1/systemone", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8070")
    ap.add_argument("--label", required=True)
    ap.add_argument("--conc", type=int, default=2)
    a = ap.parse_args()
    data = json.load(open(os.path.join(HERE, "data", "typesafe.json")))
    cases = [(wf, c) for wf, d in data["workflows"].items() for c in d["cases"]]

    def run(item):
        wf, c = item
        resp, secs = post(a.url, to_request(c))
        rows = []
        for q in c["questions"]:
            ans = resp["answers"][q["qid"]]
            val, probs = ours(ans, q["kind"])
            rows.append({"wf": wf, "case": c["case_id"], "qid": q["qid"], "kind": q["kind"],
                         "ref": q["ref_value"], "ref_probs": q["ref_probs"], "ours": val,
                         "p_ours": probs.get(val, 0.0), "p_ref": probs.get(q["ref_value"], 0.0), "probs": probs,
                         "label_mass": ans.get("label_mass"),
                         **{f"pub_{m}": c["published"].get(m, {}).get(q["qid"]) for m in PUB},
                         "case_secs": round(secs, 2), "case_nq": len(c["questions"]),
                         "case_in_tokens": resp["usage"]["input_tokens"]})
        print(f"  {wf:26} {c['case_id'][:28]:28} {len(rows):3}q {secs:6.2f}s", flush=True)
        return rows

    with ThreadPoolExecutor(a.conc) as ex:
        rows = [r for rs in ex.map(run, cases) for r in rs]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"{a.label}.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    report(rows, a.label)


def report(rows, label):
    def acc(rs, key):
        rs = [r for r in rs if r[key] is not None]
        return (sum(r[key] == r["ref"] for r in rs) / len(rs), len(rs)) if rs else (float("nan"), 0)
    for r in rows:
        r["ours_v"] = r["ours"]
    strict = [r for r in rows if all(r[f"pub_{m}"] is not None for m in PUB)]
    print(f"\n{'subset':22} {'n':>4}  {label:>9}  {'opus':>6}  {'sol':>6}  {'jev':>6}")
    for name, rs in [("all pairs", rows), ("strict common", strict)] + \
            [(f"  {wf}", [r for r in strict if r["wf"] == wf]) for wf in sorted({r['wf'] for r in rows})] + \
            [(f"  kind={k}", [r for r in strict if r["kind"] == k]) for k in ("noul", "choice", "score")]:
        if not rs:
            continue
        o = acc(rs, "ours_v")[0]
        pubs = [acc(rs, f"pub_{m}")[0] for m in PUB]
        print(f"{name:22} {len(rs):4}  {o:9.3f}  " + "  ".join(f"{p:6.3f}" for p in pubs))
    # calibration of our confidence against agreement with the reference
    n = len(rows); brier = sum((r["p_ours"] - (r["ours"] == r["ref"])) ** 2 for r in rows) / n
    bins = [[] for _ in range(10)]
    for r in rows:
        bins[min(9, int(r["p_ours"] * 10))].append(r)
    ece = sum(len(b) / n * abs(sum(x["p_ours"] for x in b) / len(b) - sum(x["ours"] == x["ref"] for x in b) / len(b))
              for b in bins if b)
    nll = -sum(math.log(max(r["p_ref"], 1e-9)) for r in rows) / n
    lm = [r["label_mass"] for r in rows if r["label_mass"] is not None]
    secs = {(r["case"]): (r["case_secs"], r["case_nq"]) for r in rows}
    print(f"\nconfidence vs agreement: Brier {brier:.3f}  ECE {ece:.3f}  NLL(ref) {nll:.3f}  "
          f"label_mass min/median {min(lm):.3f}/{sorted(lm)[len(lm)//2]:.3f}")
    print("case latency (s): " + ", ".join(f"{s:.2f}/{q}q" for s, q in sorted(secs.values())))


if __name__ == "__main__":
    main()
