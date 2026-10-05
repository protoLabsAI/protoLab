"""Degeneration detectors for the Ornith looping investigation.

Three independent signals, deliberately overlapping so a single detector bug
can't manufacture a result:

  tail_cycle  smallest period p whose block repeats >=3x verbatim at the tail.
              This is the hard "stuck in a loop" case a user sees.
  rep15       fraction of word-level 15-grams that occur more than once.
              Soft/paraphrase looping that never becomes exactly periodic.
  max_block   longest verbatim block repeated back-to-back >=3x anywhere.

A sample is LOOPED if tail_cycle is not None, or rep15 >= 0.35.
"""


def tail_cycle(text, tail_chars=800, max_period=400, min_reps=3):
    t = text[-tail_chars:]
    if len(t) < min_reps * 4:
        return None
    for p in range(2, min(max_period, len(t) // min_reps) + 1):
        block = t[-p:]
        n = len(t) // p
        if n < min_reps:
            continue
        if t[-p * n:] == block * n:
            return {"period": p, "reps": n, "block": block[:120]}
    return None


def rep15(text, n=15):
    w = text.split()
    if len(w) < n * 2:
        return 0.0
    grams = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    seen, dup = set(), 0
    for g in grams:
        if g in seen:
            dup += 1
        else:
            seen.add(g)
    return dup / len(grams)


def max_block(text, max_period=300, min_reps=3):
    best = None
    L = len(text)
    for p in range(4, min(max_period, L // min_reps) + 1):
        for start in range(0, L - p * min_reps + 1, max(1, p // 2)):
            block = text[start:start + p]
            reps = 1
            while text[start + reps * p: start + (reps + 1) * p] == block:
                reps += 1
            if reps >= min_reps and (best is None or p * reps > best["span"]):
                best = {"period": p, "reps": reps, "span": p * reps,
                        "block": block[:120]}
    return best


def analyse(text):
    tc = tail_cycle(text)
    r = rep15(text)
    mb = max_block(text)
    return {
        "chars": len(text),
        "tail_cycle": tc,
        "rep15": round(r, 4),
        "max_block": mb,
        "looped": bool(tc) or r >= 0.35,
    }


if __name__ == "__main__":
    # self-test: the detector must fire on a loop and stay silent on clean prose
    loop = "The answer is 42. " * 40
    clean = ("Rain fell on the tin roof all afternoon and the dog refused to "
             "come inside, so we left the door open and let the weather in. "
             "By evening the yard had turned to soup and the fenceline was a "
             "row of dark posts leaning out of the water like bad teeth. ")
    a, b = analyse(loop), analyse(clean)
    assert a["looped"], a
    assert not b["looped"], b
    print("self-test OK")
    print("loop :", a)
    print("clean:", b)
