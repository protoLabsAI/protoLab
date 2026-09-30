# creative-writer — Daria / de-slop steering

Landed here for [protoLab#30](https://github.com/protoLabsAI/protoLab/issues/30) so
[protoContent#486](https://github.com/protoLabsAI/protoContent/issues/486) and
[#494](https://github.com/protoLabsAI/protoContent/issues/494) can draft against source rather
than a transcribed numbers block — which is how #486's draft ended up publishing figures that
had already been retired upstream.

## Read this first if you are drafting

`RESULTS.md` is chronological and **later rounds retire earlier ones**. Do not lift a number
from Rounds 1–3 without checking what Round 4 and Round 5 did to it.

- **Rounds 1–3 rubric numbers are retired.** Two independent defects: the bench judged with
  `max_tokens=4096`, so a thinking judge burned the budget on reasoning and the piece went
  unscored (reported rubrics silently averaged over 19–29 of 32 pieces); and a base arm that
  silently inherited the serving lane's `repetition_penalty 1.15`, making "base vs Daria"
  really "base+rep vs base+rep+clamp".
- **Round 4** is the first controlled comparison: 3 arms × 3 runs, every knob stated, uniform
  16k judging. `repetition_penalty 1.15` supplies +6.38 rubric and eliminates degeneration;
  the clamp supplies −0.16, i.e. nothing.
- **Round 5** is the powered A/B: the clamp's remaining claim tested at n=9/arm, judge-free,
  two concurrently-served lanes differing in one environment variable. −24.2% deterministic
  slop, exact permutation p = 0.0082.

The current defensible one-line spec: **the craft and stability of `base + repetition_penalty
1.15`, with ~24% less deterministic slop.**

## Files

    RESULTS.md          Rounds 3/4/5 with what was retired and why. The blocking artifact.
    DARIA.md            The served artifact: config, two serving paths, six serving traps.
    clamp_decision.py   Per-run re-analysis + exact permutation test (--ab, --holdout).
    clamp_ab.py         Two-lane concurrent A/B driver (generation only, no judge).
    refusal_probe.py    Wilson-interval probe for the nonsense-refusal artifact.
    PLAN.md             Original Jul-6 scoping. Superseded by RESULTS.md; kept for lineage.

## Provenance and the divergence risk, stated plainly

These are **copies**. The working tree is `~/dev/atelier/creative-corpus/` on the protoLab box
(a local git repo with no remote), which also holds the direction-extraction, SFT, DPO and
serving scripts that these five depend on historically but do not import. Nothing keeps the two
in sync automatically. **If you change a number here, change it there**, or the next handover
reintroduces exactly the problem #30 was filed about.

## Data the scripts read

Not in git — large, and judge-free re-analysable, which is the point. On the box:

    /mnt/data/abliterate/creative-vectors/clamp_ab_pieces.json          576 pieces, Round 5 powered A/B
    /mnt/data/abliterate/creative-vectors/repeat_scored.json            Round 4, base+rep1.15 + tau0+rep1.15
    /mnt/data/abliterate/creative-vectors/repeat_base_clean_scored.json Round 4, the true base arm
    /mnt/data/abliterate/creative-vectors/rejudge_all_scored.json       Round 3 re-judge at 16k
    /mnt/data/abliterate/creative-vectors/refusal_probe.json            288 short-prompt calls
    /mnt/data/abliterate/creative-vectors/v6_mean_sentiment_dir.pt      the direction the clamp projects on

The slop index is built from `creative-writing-bench/data/slop_phrase_prob_adjustments.json`
(top 600 entries), which the bench checkout supplies.

## Reproduce

    python clamp_decision.py              # per-run re-analysis of the banked Round 4 arms
    python clamp_decision.py --ab         # the powered n=9 A/B + exact permutation test
    python clamp_decision.py --holdout    # same effect on phrases the clamp was never tuned on

`clamp_ab.py` and `refusal_probe.py` need two live lanes and are the only ones that use a GPU;
everything else re-analyses banked text and runs anywhere.
