"""Score whether an answer is actually a copy of its source.

Scoring logic follows syv-ai/qwen38-27b-rtx3090 bench/verbatim.py (Apache-2.0), which
established the method: the SAME broken prompt length has produced an empty answer, a
one-character answer, and 400 tokens of fluent invented content. Any detector keyed on
one signature files at least one of the three as healthy. What survives all three is
coverage -- how much of the answer is verbatim from the source -- judged against what
the OTHER prompt lengths on the same server managed.
"""

WINDOW = 40
STRIDE = 20


def coverage(ans, doc):
    """Fraction of the answer's 40-char windows that appear verbatim in the source."""
    if not ans:
        return 0.0
    stop = max(1, len(ans) - WINDOW + 1)
    wins = [ans[i:i + WINDOW] for i in range(0, stop, STRIDE)]
    return sum(w in doc for w in wins) / len(wins)


def prefix_match(ans, doc):
    n = 0
    while n < len(ans) and ans[:n + 1] in doc:
        n += 1
    return n


def invented_repeats(ans, doc):
    """Repetition the MODEL added, beyond what the source itself repeats."""
    return max((ans.count(ans[i:i + WINDOW]) - doc.count(ans[i:i + WINDOW])
                for i in range(0, max(1, len(ans) - WINDOW), WINDOW)), default=0)


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if s else 0.0


def classify(ans, doc, ref=None, min_len=40, floor=0.5, rel=0.6, max_loop=3):
    """-> (flag, coverage, reason). Never judge on a failure signature."""
    cov = coverage(ans, doc)
    if len(ans) < min_len:
        return "BROKEN", cov, f"answer {len(ans)} chars < {min_len}"
    if cov < floor:
        return "BROKEN", cov, f"coverage {cov:.2f} < absolute floor {floor}"
    if ref is not None and cov < rel * ref:
        return "BROKEN", cov, f"coverage {cov:.2f} < {rel} x neighbours {ref:.2f}"
    if invented_repeats(ans, doc) > max_loop:
        return "BROKEN", cov, f"model-invented repetition x{invented_repeats(ans, doc)}"
    return "ok", cov, ""
