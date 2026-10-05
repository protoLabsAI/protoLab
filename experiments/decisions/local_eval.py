#!/usr/bin/env python3
"""Score a local HF model (optionally + LoRA adapter) on TypeSafe's public eval, same prompts as
the served lane, same row format and report as ts_eval.py.

  CUDA_VISIBLE_DEVICES="" python local_eval.py --base Qwen/Qwen3.5-0.8B --label q08b-zs --threads 12
"""
import argparse, json, os, time

from s1model import S1Model, rows_from_typesafe_case
import ts_eval

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--label", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--batch", type=int, default=1)
    a = ap.parse_args()
    m = S1Model(a.base, device=a.device, adapter=a.adapter, threads=a.threads)
    data = json.load(open(os.path.join(HERE, "data", "typesafe.json")))
    out = []
    t_all = time.time()
    for wf, d in data["workflows"].items():
        for c in d["cases"]:
            t0 = time.time()
            rows = rows_from_typesafe_case(c)
            res = m.score(rows, batch_size=a.batch)
            secs = time.time() - t0
            by = {q["qid"]: q for q in c["questions"]}
            for r, s in zip(rows, res):
                qid = r["id"].split("/", 1)[1]
                q = by[qid]
                probs = s["probs"]
                if q["kind"] == "noul":
                    val = "yes" if probs["yes"] >= 0.5 else "no"
                else:
                    val = max(probs, key=probs.get)
                out.append({"wf": wf, "case": c["case_id"], "qid": qid, "kind": q["kind"],
                            "ref": q["ref_value"], "ref_probs": q["ref_probs"], "ours": val,
                            "p_ours": probs.get(val, 0.0), "p_ref": probs.get(q["ref_value"], 0.0),
                            "probs": probs, "label_mass": s["label_mass"],
                            **{f"pub_{mm}": c["published"].get(mm, {}).get(qid) for mm in ts_eval.PUB},
                            "case_secs": round(secs, 2), "case_nq": len(rows), "case_in_tokens": None})
            print(f"  {wf:26} {c['case_id'][:28]:28} {len(rows):3}q {secs:7.1f}s", flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"{a.label}.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in out)
    print(f"total {time.time() - t_all:.0f}s")
    ts_eval.report(out, a.label)


if __name__ == "__main__":
    main()
