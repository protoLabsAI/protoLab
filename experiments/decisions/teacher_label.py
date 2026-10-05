#!/usr/bin/env python3
"""Label decisions-v1 train rows with the 27B's full distribution, through s1serve.

Two passes per item, averaged: the lane isn't batch-invariant (~10% of answers flip run to run),
so one pass would bake that noise into the targets. Output: <data>/teacher/<task>.jsonl rows
{id, probs, passes, p_spread}; train_s1.py merges them by id. Resumable per run (rows already labelled are skipped); a run cut by the deadline writes nothing.

Guarded for prod: low concurrency, pauses while either smart lane has a waiting queue, and stops
at --deadline (UTC HH:MM).

  python teacher_label.py --per-task 400 --deadline 16:45
"""
import argparse, glob, json, os, random, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

DATA = "/mnt/data/datasets/decisions-v1"


def lane_waiting():
    w = 0
    for port in (8041, 8042):
        try:
            txt = urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=5).read().decode()
            w = max(w, max((int(float(l.split()[-1])) for l in txt.splitlines()
                            if l.startswith("vllm:num_requests_waiting{")), default=0))
        except Exception:
            pass
    return w


def ask(url, row):
    body = {"state": row["state"], "questions": {"q": row["question"]}}
    req = urllib.request.Request(f"{url}/v1/systemone", json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        a = json.load(r)["answers"]["q"]
    if a["type"] == "noul":
        return {"yes": a["noul"], "no": 1 - a["noul"]}
    return a["probabilities"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8070")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--per-task", type=int, default=400)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--max-wait", type=int, default=4, help="pause while a lane has more waiting than this")
    ap.add_argument("--deadline", default="16:45")
    ap.add_argument("--tasks", default=None, help="comma list (default: all non-heldout)")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.data, "teacher"), exist_ok=True)
    todo = []
    for p in sorted(glob.glob(os.path.join(a.data, "*.jsonl"))):
        task = os.path.basename(p)[:-6]
        if a.tasks and task not in a.tasks.split(","):
            continue
        rows = [json.loads(l) for l in open(p)]
        rows = [r for r in rows if r["split"] == "train" and not r.get("heldout")]
        random.Random(0).shuffle(rows)
        outp = os.path.join(a.data, "teacher", f"{task}.jsonl")
        done = {json.loads(l)["id"] for l in open(outp)} if os.path.exists(outp) else set()
        todo += [(task, r) for r in rows[: a.per_task] if r["id"] not in done]
    print(f"{len(todo)} rows to label x{a.passes} passes", flush=True)

    # Passes are separate SWEEPS in different shuffled orders, not back-to-back repeats: repeats of
    # one request see the same batch conditions and come back identical (measured: spread 0.0),
    # which would average away none of the batch noise this is meant to cancel.
    def ask_guarded(row):
        while lane_waiting() > a.max_wait:
            if time.strftime("%H:%M", time.gmtime()) > a.deadline:
                return None
            time.sleep(30)
        return ask(a.url, row)

    t0 = time.time()
    passes = []
    for k in range(a.passes):
        order = list(range(len(todo)))
        random.Random(k).shuffle(order)
        got = {}
        with ThreadPoolExecutor(a.conc) as ex:
            for i, probs in zip(order, ex.map(lambda i: ask_guarded(todo[i][1]), order)):
                if probs is None or time.strftime("%H:%M", time.gmtime()) > a.deadline:
                    print(f"deadline reached in sweep {k+1}; nothing from this run is written", flush=True)
                    return
                got[i] = probs
                if len(got) % 2000 == 0:
                    print(f"  sweep {k+1}: {len(got)}/{len(todo)}  {len(got)/(time.time()-t0):.1f} req/s", flush=True)
        passes.append(got)

    n = agree = 0
    spreads = []
    for i, (task, row) in enumerate(todo):
        ps = [p[i] for p in passes]
        avg = {o: sum(p.get(o, 0.0) for p in ps) / len(ps) for o in ps[0]}
        spread = max(abs(ps[0].get(o, 0) - p.get(o, 0)) for p in ps for o in ps[0])
        spreads.append(spread)
        rec = {"id": row["id"], "probs": avg, "passes": len(ps), "p_spread": round(spread, 4),
               "gold": row["gold"], "teacher_argmax": max(avg, key=avg.get)}
        with open(os.path.join(a.data, "teacher", f"{task}.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        n += 1
        agree += rec["teacher_argmax"] == rec["gold"]
    spreads.sort()
    print(f"done {n} rows, teacher agrees with gold {agree/max(n,1):.3f}; "
          f"pass spread median {spreads[len(spreads)//2]:.4f} p90 {spreads[int(.9*len(spreads))]:.4f}", flush=True)


if __name__ == "__main__":
    main()
