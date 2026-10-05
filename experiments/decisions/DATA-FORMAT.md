# decisions-v1 data format

One decision per JSONL row, in TypeSafe's question shape, so the trainer, teacher labeller and
evals all render the exact prompt `s1serve.py` serves (`compile_question` + `Engine.render`).

Location: `/mnt/data/datasets/decisions-v1/` (working data, per the node storage rules):
`<task>.jsonl` per task, plus `manifest.json`.

```json
{"id": "banking77/train/000123",
 "task": "banking77", "family": "intent", "split": "train", "heldout": false,
 "state": "I still haven't received my new card, it's been two weeks.",
 "question": {"type": "choice",
              "instructions": "Which banking intent does this message express?",
              "criteria": {"card_arrival": "the customer is waiting for a card to arrive", "...": "..."}},
 "gold": "card_arrival",
 "source": "PolyAI/banking77", "license": "cc-by-4.0"}
```

- `question.type` is `noul` | `choice` | `score`. `criteria` follows TypeSafe exactly:
  - noul: `{"true": str|null, "false": str|null}` or omitted; `gold` is `"yes"` / `"no"`.
  - choice: `{option_name: description|null}`, 2–20 options; `gold` is an option name.
  - score: list of level descriptions (≤10); `gold` is the level index as a string, `"0"`..`"9"`.
- `split`: `train` / `calib` / `test`. `heldout: true` marks a whole task that is never trained on
  (its rows are all eval).
- Teacher labels are added later as `"teacher": {"probs": {outcome: p}, "passes": 2}`.
- Option order is stored as the dataset gives it. The trainer shuffles per example.

`manifest.json`: per task, the source dataset id + revision/config, license, family, heldout
flag, row counts per split, and the per-task cap applied.
