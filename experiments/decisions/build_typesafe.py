#!/usr/bin/env python3
"""Rebuild TypeSafe's public workflow eval (evals.typesafe.ai) as local JSON.

Parsing adapted from mithalouni/system-one-open `typesafe_eval.py` (MIT, (c) 2026 Rami
Mithalouni); Modal plumbing removed. The source data is TypeSafe's and is NOT redistributed:
this script fetches it (or reads data/typesafe-raw/) and writes data/typesafe.json.

Reference answers are the CONSENSUS of two frontier models (gpt-6-astra, claude-fable-5-1),
not human ground truth, so any score here is agreement with frontier consensus.
"""
import json, os, re, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
WORKFLOWS = ["security_incidents", "agent_trace_observability", "invoice_processing", "customer_service"]

def render(doc):
    return doc if isinstance(doc, str) else json.dumps(doc, indent=1, ensure_ascii=False)


def canonical(value, qtype):
    if value is None:
        return None
    if qtype == "noul":
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, (int, float)):
            return "yes" if float(value) >= 0.5 else "no"
        return "yes" if str(value).lower() in ("true", "yes", "1") else "no"
    if qtype == "score":
        try:
            return str(int(round(float(value))))
        except (TypeError, ValueError):
            return None
    return str(value)


def consensus(entry, qtype):
    sets = entry.get("sets", [])
    acc = {}; values = []
    for s in sets:
        for k, v in (s.get("probabilities") or {}).items():
            acc[k] = acc.get(k, 0.0) + float(v)
        if s.get("value") is not None:
            values.append(s["value"])
    if acc:
        probs = {k: v / len(sets) for k, v in acc.items()}
        best = max(probs, key=probs.get)
        return canonical(best if qtype != "noul" else (best in ("true", "yes", "True")), qtype), probs
    if values:
        canon = [c for c in (canonical(v, qtype) for v in values) if c is not None]
        if canon:
            return max(set(canon), key=canon.count), {}
    return None, {}


def build_from_viewer(raw_by_wf):
    out = {"workflows": {}}
    for wf in WORKFLOWS:
        raw = raw_by_wf[wf]
        m = re.search(r"__VIEWER_DATA__\((.*)\)\s*;?\s*$", raw, re.S) or re.search(r"__VIEWER_DATA__\((.*)\)", raw, re.S)
        ev = json.loads(m.group(1))["eval"]
        docs = ev["documents"]; catalog = ev["questions"]
        wf_qmap = {}
        for ex in ev["examples"]:
            case = ev["cases"][ex["case_id"]]
            for mv in case["models"].values():
                for n in mv.get("nodes", []):
                    for qid, idx in (n.get("questions") or {}).items():
                        wf_qmap.setdefault(qid, int(idx))
        cases = []
        for ex in ev["examples"]:
            cid = ex["case_id"]; case = ev["cases"][cid]
            doc_idx = set()
            for mv in case["models"].values():
                for n in mv.get("nodes", []):
                    if n.get("doc") is not None:
                        doc_idx.add(int(n["doc"]))
            reference = {}
            for node_answers in case.get("reference_answers", {}).values():
                for qid, entry in node_answers.items():
                    val, probs = consensus(entry, entry["type"])
                    reference[qid] = {"value": val, "probs": probs, "type": entry["type"]}
            published = {}
            for mkey, mv in case["models"].items():
                answers = {}
                for n in mv.get("nodes", []):
                    for qid, ans in (n.get("answers") or {}).items():
                        raw_v = ans.get(ans["type"])
                        answers[qid] = canonical(raw_v, ans["type"])
                published[mkey] = answers
            questions = []
            for qid, ref in reference.items():
                idx = wf_qmap.get(qid)
                if idx is None or not (0 <= idx < len(catalog)) or ref["value"] is None:
                    continue
                q = catalog[idx]; crit = q.get("criteria")
                if q["type"] == "noul":
                    opts = ["no", "yes"]; descs = None
                    if isinstance(crit, dict):
                        descs = [crit.get("false") or None, crit.get("true") or None]
                        if not any(descs):
                            descs = None
                elif q["type"] == "score":
                    lv = crit if isinstance(crit, list) else []
                    opts = [str(i) for i in range(len(lv))]; descs = list(lv)
                else:
                    if isinstance(crit, dict):
                        opts = list(crit.keys()); descs = [crit[k] for k in opts]
                    elif isinstance(crit, list):
                        opts = list(crit); descs = None
                    else:
                        continue
                if ref["value"] not in opts:
                    continue
                questions.append({"qid": qid, "kind": q["type"], "text": q["instructions"], "options": opts, "descs": descs, "ref_value": ref["value"], "ref_probs": ref["probs"]})
            state = "\n\n".join(f"## Document {i}\n{render(docs[i])}" for i in sorted(doc_idx) if 0 <= i < len(docs))
            cases.append({"case_id": cid, "name": ex.get("name", cid), "state": state, "state_chars": len(state), "questions": questions, "published": published})
        out["workflows"][wf] = {"cases": cases, "catalog_size": len(catalog)}
    return out


def main():
    raw = {}
    for wf in WORKFLOWS:
        p = os.path.join(HERE, "data", "typesafe-raw", f"{wf}-cases.js")
        if not os.path.exists(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            urllib.request.urlretrieve(f"https://evals.typesafe.ai/{wf}-cases.js", p)
        raw[wf] = open(p).read()
    out = build_from_viewer(raw)
    json.dump(out, open(os.path.join(HERE, "data", "typesafe.json"), "w"))
    tot = 0
    for wf, d in out["workflows"].items():
        qs = [q for c in d["cases"] for q in c["questions"]]
        kinds = {}
        for q in qs:
            kinds[q["kind"]] = kinds.get(q["kind"], 0) + 1
        cov = {m: sum(1 for c in d["cases"] for q in c["questions"]
                      if c["published"].get(m, {}).get(q["qid"]) is not None) for m in ["opus", "sol", "typesafe"]}
        chars = [c["state_chars"] for c in d["cases"]]
        print(f"{wf:28} cases={len(d['cases'])} pairs={len(qs)} kinds={kinds} "
              f"max_opts={max(len(q['options']) for q in qs)} state_chars={min(chars)}-{max(chars)} published={cov}")
        tot += len(qs)
    print("TOTAL reference pairs:", tot)


if __name__ == "__main__":
    main()
