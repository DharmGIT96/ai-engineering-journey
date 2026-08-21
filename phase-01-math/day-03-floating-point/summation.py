"""
summation.py  --  Phase 01, Day 03: how to add up a lot of numbers

    python summation.py

Put it in  phase-01-math/day-03-floating-point/
Nothing to import from ailib -- this one stands alone.

Three functions to write. Each fixes a different part of the same problem:
a running total drifting away from the data it is accumulating.

Write them in order. Each one's output tells you why the next exists.
"""

import math
import random
import time


# ===========================================================================
# 1. NAIVE -- write this first so you have a baseline to be ashamed of.
# ===========================================================================

def naive_sum(values):
    """Left to right, one accumulator. The obvious loop.

    SHAPE: list of N floats -> scalar
    """
    total = 0.0
    for v in values:
        total += v
    return total


# ===========================================================================
# 2. PAIRWISE -- what numpy actually does.
# ===========================================================================

def pairwise_sum(values, threshold=64):
    """Split in half, sum each half, add the two halves.

    Recursion, with a base case: below `threshold` elements, just do the
    naive loop -- the recursion overhead is not worth it for small chunks.
    (numpy uses 128. The exact number does not matter; having one does.)

    Why it works: every addition happens between two partial sums of
    SIMILAR MAGNITUDE, instead of one big accumulator repeatedly swallowing
    tiny values. Error grows like log(N) instead of N.

    SHAPE: list of N floats -> scalar
    """
    if len(values) <= threshold:
        return math.fsum(values)
    else:
        mid = len(values) // 2
        return pairwise_sum(values[:mid], threshold) + pairwise_sum(values[mid:], threshold)


# ===========================================================================
# 3. KAHAN -- keep the rounding error and put it back.
# ===========================================================================

def kahan_sum(values):
    """Compensated summation. One extra variable, most of the error gone.

    The idea: when you do `total + v`, the result gets rounded, and some of
    v is lost. Kahan's insight is that you can RECOVER the lost part and
    carry it into the next iteration.

        c = 0.0                  # the running compensation (lost bits)
        for v in values:
            y = v - c            # correct v by what we lost last time
            t = total + y        # the rounded addition
            c = (t - total) - y  # what was ACTUALLY added, minus what we
                                 # meant to add == the amount that got lost
            total = t

    Read that `c` line until it clicks. `(t - total)` is how much the total
    really moved. `y` is how much we asked it to move. The difference is the
    rounding error, and next iteration we subtract it back in.

    DO NOT simplify the algebra. In exact arithmetic c is always 0 and the
    whole thing collapses to the naive loop -- that is the joke. It only
    works BECAUSE floating point is not exact, so the parenthesisation is
    load-bearing. An optimising compiler that "simplifies" this breaks it,
    which is why -ffast-math is dangerous.

    SHAPE: list of N floats -> scalar
    """
    total = 0.0
    c = 0.0
    for v in values:
        y = v - c
        t = total + y
        c = (t - total) - y
        total = t
    return total

# ===========================================================================
# HARNESS -- do not edit.
# ===========================================================================

def _cases():
    rng = random.Random(0)
    return [
        ("1e8 plus a million 1e-3",
         [1e8] + [1e-3] * 1_000_000),
        ("a million losses near 0.5",
         [rng.gauss(0.5, 0.1) for _ in range(1_000_000)]),
        ("wildly mixed magnitudes",
         [rng.choice([1e12, 1e-12, 1.0, -1e12]) * rng.random()
          for _ in range(200_000)]),
        ("alternating +1 / -1 with a tail",
         [1e15, 1.0, -1e15] * 100_000),
    ]


def main():
    methods = [
        ("naive", naive_sum),
        ("pairwise", pairwise_sum),
        ("kahan", kahan_sum),
    ]

    for label, vals in _cases():
        exact = math.fsum(vals)
        print(f"\n{label}   (n={len(vals):,})")
        print(f"  {'method':<12} | {'error':>12} | {'time':>9} | {'digits correct':>14}")
        print("  " + "-" * 58)
        for name, fn in methods:
            t0 = time.perf_counter()
            got = fn(vals)
            dt = time.perf_counter() - t0
            err = abs(got - exact)
            if err == 0:
                digits = "exact"
            elif exact != 0:
                digits = f"{-math.log10(err / abs(exact)):.1f}"
            else:
                digits = "-"
            print(f"  {name:<12} | {err:>12.4e} | {dt*1e3:>7.1f}ms | {digits:>14}")

    print("""
Questions to answer from your own table:
  1. On which case is naive_sum worst, and why that one?
  2. Kahan is more accurate than pairwise but slower. Where does the time go?
  3. Is there a case where naive is FINE? What do those inputs have in common?
  4. numpy uses pairwise, not Kahan, for np.sum. Given your numbers, why?
""")


if __name__ == "__main__":
    main()
