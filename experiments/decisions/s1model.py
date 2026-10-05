"""Local HF decision model: renders the shared prompt, reads the label logits at the last position.

One class serves the zero-shot baselines (CPU), the trainer's forward pass (GPU), and evaluation of
a fine-tuned adapter. Batches are RIGHT-padded and each example's logits are read at its last real
token: in a causal model (DeltaNet recurrence included) padding after that position can't reach it.
"""
from __future__ import annotations

import math, random

import torch

from s1prompt import compile_question, render

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",            # full-attention layers
                "in_proj_qkv", "in_proj_z", "out_proj",             # Gated DeltaNet layers
                "gate_proj", "up_proj", "down_proj"]                # MLP


class S1Model:
    def __init__(self, base: str, device: str = "cpu", adapter: str | None = None,
                 threads: int | None = None, dtype=torch.bfloat16):
        if device == "cpu":
            if threads:
                torch.set_num_threads(threads)
            # fla's Triton kernels need a GPU; force the pure-torch DeltaNet path on CPU
            import transformers.models.qwen3_5.modeling_qwen3_5 as m
            m.chunk_gated_delta_rule = m.fused_recurrent_gated_delta_rule = m.FusedRMSNormGated = None
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(base)
        self.model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, device_map=device)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.label_id = {}
        self.temperature = {"noul": 1.0, "choice": 1.0, "score": 1.0}

    def lid(self, label: str) -> int:
        if label not in self.label_id:
            ids = self.tok.encode(label, add_special_tokens=False)
            assert len(ids) == 1, f"label {label!r} is not a single token"
            self.label_id[label] = ids[0]
        return self.label_id[label]

    def encode(self, row: dict, rng: random.Random | None = None, max_len: int = 8192) -> dict:
        q = row["question"]
        order = None
        if rng is not None and q["type"] == "choice":
            order = list(range(len(q["criteria"])))
            rng.shuffle(order)
        block, labels, outcomes = compile_question(q, order)
        ids = self.tok(render(self.tok, row["state"], block), add_special_tokens=False).input_ids
        if len(ids) > max_len:  # keep the question end (it carries the options); cut the state's middle
            ids = ids[: max_len // 2] + ids[-(max_len - max_len // 2):]
        return {"ids": ids, "label_ids": [self.lid(l) for l in labels], "outcomes": outcomes,
                "kind": q["type"], "gold": row.get("gold"),
                "teacher": (row.get("teacher") or {}).get("probs")}

    def label_logits(self, batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> (logits over each example's labels [B, K] with -inf padding, valid mask, raw label mass)."""
        B, T = len(batch), max(len(b["ids"]) for b in batch)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        x = torch.full((B, T), pad, dtype=torch.long)
        att = torch.zeros((B, T), dtype=torch.long)
        for i, b in enumerate(batch):
            x[i, : len(b["ids"])] = torch.tensor(b["ids"])
            att[i, : len(b["ids"])] = 1
        x, att = x.to(self.model.device), att.to(self.model.device)
        # decoder hidden states, then the LM head at each example's last real token only: full
        # [B, T, V] logits with a 248K vocab would be gigabytes per long example
        h = self.model.get_decoder()(input_ids=x, attention_mask=att).last_hidden_state  # [B, T, H]
        last = torch.tensor([len(b["ids"]) - 1 for b in batch], device=h.device)
        lg = self.model.get_output_embeddings()(h[torch.arange(B, device=h.device), last]).float()  # [B, V]
        K = max(len(b["label_ids"]) for b in batch)
        sel = torch.zeros((B, K), dtype=torch.long, device=lg.device)
        mask = torch.zeros((B, K), dtype=torch.bool, device=lg.device)
        for i, b in enumerate(batch):
            sel[i, : len(b["label_ids"])] = torch.tensor(b["label_ids"])
            mask[i, : len(b["label_ids"])] = True
        picked = lg.gather(1, sel).masked_fill(~mask, float("-inf"))
        mass = (torch.softmax(lg, -1).gather(1, sel) * mask).sum(1)
        return picked, mask, mass

    @torch.no_grad()
    def score(self, rows: list[dict], batch_size: int = 1) -> list[dict]:
        self.model.eval()
        enc = [self.encode(r) for r in rows]
        out = []
        for i in range(0, len(enc), batch_size):
            batch = enc[i: i + batch_size]
            lg, mask, mass = self.label_logits(batch)
            for b, l, m in zip(batch, lg, mass):
                t = self.temperature.get(b["kind"], 1.0)
                p = torch.softmax(l[: len(b["outcomes"])] / t, -1).tolist()
                out.append({"outcomes": b["outcomes"], "probs": dict(zip(b["outcomes"], p)),
                            "label_mass": float(m)})
        return out


def rows_from_typesafe_case(case: dict) -> list[dict]:
    """TypeSafe eval case (build_typesafe.py) -> DATA-FORMAT rows, so local models face the exact
    questions the served lane does."""
    import ts_eval
    req = ts_eval.to_request(case)
    by = {q["qid"]: q for q in case["questions"]}
    return [{"id": f"{case['case_id']}/{qid}", "state": req["state"], "question": q,
             "gold": by[qid]["ref_value"]} for qid, q in req["questions"].items()]
