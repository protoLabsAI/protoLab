#!/usr/bin/env python3
"""LoRA fine-tune of a small Qwen3.5 into a one-forward-pass decision model.

Data: DATA-FORMAT.md rows from /mnt/data/datasets/decisions-v1 (heldout tasks never trained).
Prompt: s1prompt (byte-identical to what s1serve serves). Choice options shuffled per example.
Loss over each example's VALID labels only:
    CE(gold) + kl_w * KL(teacher || student)  [rows with teacher probs]  + brier_w * Brier
Then a per-kind temperature is fitted on the calib split and saved with the adapter.

  python train_s1.py --base Qwen/Qwen3.5-0.8B --out runs/q08b-v1 --device cuda
  python train_s1.py --base Qwen/Qwen3.5-0.8B --out runs/smoke --device cpu --smoke   # CPU smoke
"""
from __future__ import annotations

import argparse, glob, json, math, os, random, time

import torch
import torch.nn.functional as F

from s1model import LORA_TARGETS, S1Model

DATA = "/mnt/data/datasets/decisions-v1"


def load_rows(data_dir, split, include_heldout=False, tasks=None, cap=None, seed=0):
    rows = []
    for p in sorted(glob.glob(os.path.join(data_dir, "*.jsonl"))):
        task = os.path.basename(p)[:-6]
        if tasks and task not in tasks:
            continue
        rs = [json.loads(l) for l in open(p)]
        tp = os.path.join(data_dir, "teacher", f"{task}.jsonl")  # teacher_label.py output, merged by id
        if os.path.exists(tp):
            teach = {t["id"]: t for t in map(json.loads, open(tp))}
            for r in rs:
                if r["id"] in teach:
                    r["teacher"] = teach[r["id"]]
        rs = [r for r in rs if r["split"] == split and (include_heldout or not r.get("heldout"))]
        if cap:
            random.Random(seed).shuffle(rs)
            rs = rs[:cap]
        rows += rs
    return rows


def batches_by_tokens(enc, max_tokens, rng):
    """Length-bucketed batches whose padded size (B * longest) stays under max_tokens."""
    idx = sorted(range(len(enc)), key=lambda i: len(enc[i]["ids"]))
    out, cur, longest = [], [], 0
    for i in idx:
        L = len(enc[i]["ids"])
        if cur and max(longest, L) * (len(cur) + 1) > max_tokens:
            out.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, L)
    if cur:
        out.append(cur)
    rng.shuffle(out)
    return out


def loss_fn(logits, mask, batch, kl_w, brier_w):
    logp = F.log_softmax(logits, -1).masked_fill(~mask, 0.0)
    p = logp.exp() * mask
    gold = torch.tensor([b["outcomes"].index(b["gold"]) for b in batch], device=logits.device)
    ce = F.nll_loss(logp, gold)
    onehot = F.one_hot(gold, logits.shape[1]).float() * mask
    brier = ((p - onehot) ** 2).sum(1).mean()
    kl = torch.zeros((), device=logits.device)
    tm = [i for i, b in enumerate(batch) if b["teacher"]]
    if tm and kl_w > 0:
        t = torch.zeros_like(p)
        for i in tm:
            for j, o in enumerate(batch[i]["outcomes"]):
                t[i, j] = batch[i]["teacher"].get(o, 0.0)
        t = t / t.sum(1, keepdim=True).clamp_min(1e-9)
        sel = torch.tensor(tm, device=logits.device)
        kl = (t[sel] * (t[sel].clamp_min(1e-9).log() - logp[sel])).sum(1).mean()
    return ce + kl_w * kl + brier_w * brier, {"ce": ce.item(), "kl": kl.item(), "brier": brier.item()}


@torch.no_grad()
def evaluate(m, enc, max_tokens, temps=None):
    """acc / nll / brier / ece on gold, per kind. Returns metrics + raw (kind, logits, gold idx)."""
    m.model.eval()
    raw = []
    for bidx in batches_by_tokens(enc, max_tokens, random.Random(0)):
        batch = [enc[i] for i in bidx]
        lg, mask, _ = m.label_logits(batch)
        for b, l in zip(batch, lg):
            raw.append((b["kind"], l[: len(b["outcomes"])].cpu(), b["outcomes"].index(b["gold"])))
    return metrics(raw, temps), raw


def metrics(raw, temps=None):
    n = len(raw)
    acc = nll = brier = 0.0
    bins = [[0, 0.0, 0.0] for _ in range(10)]
    for kind, l, g in raw:
        p = torch.softmax(l / (temps or {}).get(kind, 1.0), -1)
        c = int(p.argmax())
        acc += c == g
        nll -= math.log(max(float(p[g]), 1e-9))
        brier += float(((p - F.one_hot(torch.tensor(g), len(p)).float()) ** 2).sum())
        b = bins[min(9, int(float(p[c]) * 10))]
        b[0] += 1; b[1] += float(p[c]); b[2] += c == g
    ece = sum(abs(b[1] - b[2]) for b in bins) / max(n, 1)
    return {"n": n, "acc": acc / n, "nll": nll / n, "brier": brier / n, "ece": ece}


def fit_temperatures(raw):
    temps = {}
    for kind in ("noul", "choice", "score"):
        sub = [r for r in raw if r[0] == kind]
        if not sub:
            continue
        best = min((metrics(sub, {kind: t})["nll"], t) for t in [0.5 + 0.05 * i for i in range(51)])
        temps[kind] = best[1]
    return temps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=16384, help="padded tokens per micro-batch")
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--kl-w", type=float, default=1.0)
    ap.add_argument("--brier-w", type=float, default=0.5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--train-cap", type=int, default=None, help="per-task cap on train rows")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run: few tasks, few rows, few steps")
    a = ap.parse_args()
    if a.smoke:
        a.train_cap, a.max_steps, a.eval_every, a.warmup, a.max_len, a.max_tokens = 8, 6, 3, 2, 1024, 4096
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "train.log"), "a")
    def say(s):
        line = f"[{time.strftime('%H:%M:%S')}] {s}"
        print(line, flush=True); log.write(line + "\n"); log.flush()

    m = S1Model(a.base, device=a.device, threads=a.threads)
    from peft import LoraConfig, get_peft_model
    m.model = get_peft_model(m.model, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0,
                                                 target_modules=LORA_TARGETS, bias="none"))
    if a.device == "cuda":
        m.model.gradient_checkpointing_enable()
        m.model.enable_input_require_grads()
    ntrain = sum(p.numel() for p in m.model.parameters() if p.requires_grad)
    say(f"base {a.base}  trainable {ntrain/1e6:.1f}M params  args {vars(a)}")

    tasks = None
    if a.smoke:
        tasks = sorted({os.path.basename(p)[:-6] for p in glob.glob(os.path.join(a.data, "*.jsonl"))})[:4]
    rng = random.Random(0)
    train = load_rows(a.data, "train", tasks=tasks, cap=a.train_cap)
    calib = load_rows(a.data, "calib", tasks=tasks, cap=8 if a.smoke else 150)
    held = load_rows(a.data, "test", include_heldout=True, tasks=tasks, cap=8 if a.smoke else 150)
    held = [r for r in held if r.get("heldout")]
    say(f"rows: train {len(train)}  calib {len(calib)}  heldout-test {len(held)}")
    enc_tr = [m.encode(r, rng=rng, max_len=a.max_len) for r in train]
    enc_ca = [m.encode(r, max_len=a.max_len) for r in calib]
    enc_he = [m.encode(r, max_len=a.max_len) for r in held]
    ntok = sum(len(e["ids"]) for e in enc_tr)

    batches = batches_by_tokens(enc_tr, a.max_tokens, rng)
    total = a.max_steps or int(len(batches) * a.epochs / a.accum)
    say(f"train tokens {ntok/1e6:.1f}M  micro-batches/epoch {len(batches)}  optimizer steps {total}")
    params = [p for p in m.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        return a.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - a.warmup) / max(1, total - a.warmup)))))

    step, micro, t0, agg = 0, 0, time.time(), {"ce": 0.0, "kl": 0.0, "brier": 0.0, "n": 0}
    ep = 0
    while step < total:
        for bidx in batches:
            m.model.train()
            batch = [enc_tr[i] for i in bidx]
            lg, mask, _ = m.label_logits(batch)
            loss, parts = loss_fn(lg, mask, batch, a.kl_w, a.brier_w)
            (loss / a.accum).backward()
            for k in ("ce", "kl", "brier"):
                agg[k] += parts[k]
            agg["n"] += 1
            micro += 1
            if micro % a.accum:
                continue
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0 or step == total or a.smoke:
                n = max(agg["n"], 1)
                el = time.time() - t0
                say(f"step {step}/{total} ep {ep} ce {agg['ce']/n:.4f} kl {agg['kl']/n:.4f} brier {agg['brier']/n:.4f} "
                    f"gn {float(gn):.2f} lr {lr_at(step):.2e} {el/60:.1f}min eta {(total-step)*el/step/60:.0f}min")
                agg = {"ce": 0.0, "kl": 0.0, "brier": 0.0, "n": 0}
            if step % a.eval_every == 0 or step == total:
                mc, _ = evaluate(m, enc_ca, a.max_tokens)
                mh, _ = evaluate(m, enc_he, a.max_tokens)
                say(f"eval step {step}: calib {json.dumps({k: round(v, 4) for k, v in mc.items()})}  "
                    f"heldout {json.dumps({k: round(v, 4) for k, v in mh.items()})}")
            if step >= total:
                break
        ep += 1

    _, raw = evaluate(m, enc_ca, a.max_tokens)
    temps = fit_temperatures(raw)
    mh, _ = evaluate(m, enc_he, a.max_tokens, temps)
    say(f"temperatures {temps}  heldout after T: {json.dumps({k: round(v, 4) for k, v in mh.items()})}")
    m.model.save_pretrained(a.out)
    json.dump({"base": a.base, "temperatures": temps, "steps": total, "train_tokens": ntok,
               "args": vars(a)}, open(os.path.join(a.out, "s1.json"), "w"), indent=1)
    say(f"saved adapter to {a.out}")


if __name__ == "__main__":
    main()
