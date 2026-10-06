"""Classify every draw against the bracketed ground truth, with Wilson intervals."""
import json, sys
import numpy as np

TOL = 1e-6


def wilson(k, n, z=1.0):
    """Wilson score interval; z=1 for a 1-sigma band. Behaves at k=0."""
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def pct(k, n):
    p, lo, hi = wilson(k, n)
    return f"{100*p:6.3f}%  [{100*lo:5.3f}, {100*hi:5.3f}]"


def classify(rec, prefix):
    """correct / spurious / phantom / missed / agreed-none, vs ground truth."""
    if not rec.get("brk_ok") and not rec.get(prefix + "_ok"):
        return "agreed_none"
    if rec.get("brk_ok") and not rec.get(prefix + "_ok"):
        return "missed"
    if not rec.get("brk_ok") and rec.get(prefix + "_ok"):
        return "phantom"
    r_ref, r_got = rec["brk_r0"], rec[prefix + "_r0"]
    if not np.isfinite(r_got) or r_got <= 0:
        return "missed"
    return "correct" if abs(r_got / r_ref - 1.0) < TOL else "spurious"


for kind in ("wide", "fold"):
    R = [r for r in json.load(open(f"campaign_{kind}.json")) if "fatal" not in r]
    n = len(R)
    print(f"\n{'='*78}\n{kind.upper()} prior: {n} draws   "
          f"(ground truth has a solution in {sum(r['brk_ok'] for r in R)})")
    print(f"{'='*78}")

    for prefix, name in (("ramp", "16-stage ramp"), ("n1", "plain Newton, no ramp")):
        cls = [classify(r, prefix) for r in R]
        from collections import Counter
        cnt = Counter(cls)
        wrong = cnt["spurious"] + cnt["phantom"]
        print(f"\n  {name}:")
        for k in ("correct", "agreed_none", "spurious", "phantom", "missed"):
            print(f"     {k:14s} {cnt[k]:5d}")
        print(f"     WRONG ANSWER RATE (spurious+phantom): {pct(wrong, n)}")

        # how big are the errors?
        bad = [r for r, c in zip(R, cls) if c == "spurious"]
        if bad:
            f = [abs(r[prefix + "_r0"] / r["brk_r0"]) for r in bad]
            print(f"     r0 wrong by factors {min(f):.3g} to {max(f):.3g}; "
                  f"residuals {min(r.get('ramp_res', np.nan) for r in bad):.1e} "
                  f"to {max(r.get('ramp_res', np.nan) for r in bad):.1e}"
                  if prefix == "ramp" else
                  f"     r0 wrong by factors {min(f):.3g} to {max(f):.3g}")

    # ---- do the two screens catch the wrong answers? ----
    print(f"\n  SCREENS, on the 16-stage ramp's output:")
    cls = [classify(r, "ramp") for r in R]
    for screen, label in (("rampv_ok", "schedule independence (costs a 2nd solve)"),
                          ("cert_ok", "branch certificate (this work)")):
        # Only cases where the ramp returned something are screenable.
        sub = [(r, c) for r, c in zip(R, cls) if r.get("ramp_ok")]
        wrongs = [(r, c) for r, c in sub if c in ("spurious", "phantom")]
        goods = [(r, c) for r, c in sub if c == "correct"]
        caught = sum(1 for r, c in wrongs if r.get(screen) is False)
        rejected = sum(1 for r, c in goods if r.get(screen) is False)
        print(f"\n    {label}")
        print(f"      wrong answers caught : {caught}/{len(wrongs)}"
              + (f"   (miss rate {pct(len(wrongs)-caught, len(wrongs))})" if wrongs else ""))
        print(f"      good answers rejected: {rejected}/{len(goods)}"
              f"   (false positive {pct(rejected, len(goods))})")

    ok = [r for r in R if r.get("ramp_ok")]
    print(f"\n  COST, median wall time per solve (s):")
    print(f"      plain Newton            {np.median([r['t_n1'] for r in R]):.4f}")
    print(f"      16-stage ramp           {np.median([r['t_ramp'] for r in R]):.4f}")
    print(f"      ramp + schedule screen  {np.median([r['t_rampv'] for r in R]):.4f}")
    print(f"      certificate alone       {np.median([r.get('t_cert', 0) for r in ok]):.6f}"
          f"   (mean {np.mean([r.get('t_cert', 0) for r in ok]):.6f})")
    print(f"      bracketed solve         {np.median([r['t_brk'] for r in R]):.4f}")
    fast = sum(1 for r in ok if str(r.get('cert_why', '')).startswith('u1<'))
    print(f"      certificate took the free u1<U_SAFE path in {fast}/{len(ok)}"
          f" = {100*fast/max(1,len(ok)):.2f}% of cases")
