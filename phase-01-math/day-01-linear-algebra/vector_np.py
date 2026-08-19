"""
vector_np.py  --  Phase 01, Day 01, Step 3: VECTORIZE

Put this NEXT TO your linalg_scratch.py -- it imports from it, and uses your
pure-Python Vector as the source of truth.

Your job: fill in the eight functions below. Each is one line. Then run:

    python vector_np.py

It will (1) prove your NumPy version agrees with your pure-Python version to
1e-6 on random inputs, and (2) time them both across growing dimensions.

RULES
  - NumPy only. No Python-level `for` loop over elements. If you write
    `for x in arr:` you have re-implemented the slow version with extra steps.
  - Agreement is checked FIRST. A fast wrong answer scores zero.
  - Before each function, say the output shape out loud. Same habit.
    np.dot returns a numpy scalar -- shape (). Not shape (1,).
"""

import time
import numpy as np

from linalg_scratch import Vector


# ===========================================================================
# YOUR JOB -- eight one-liners. Arrays in, arrays or scalars out.
# ===========================================================================

def add(a, b):
    """SHAPE: (N,) and (N,) -> (N,)"""
    return np.add(a, b)


def sub(a, b):
    """SHAPE: (N,) and (N,) -> (N,)"""
    return np.subtract(a, b)


def scale(a, k):
    """Scalar multiplication. SHAPE: (N,) and scalar -> (N,)

    NumPy does the 'stretch the scalar across every slot' step for you.
    That is broadcasting -- the thing you met by hand in __mul__.
    """
    return np.multiply(a, k)


def hadamard(a, b):
    """Element-wise product. SHAPE: (N,) and (N,) -> (N,)

    Note which Python operator NumPy assigns to this, and which one it
    assigns to matrix multiplication. They are not the same operator.
    """
    return np.multiply(a, b)


def dot(a, b):
    """SHAPE: (N,) and (N,) -> scalar, shape ()"""
    return np.dot(a, b)


def norm(a):
    """SHAPE: (N,) -> scalar

    np.linalg.norm exists. So does sqrt(dot(a, a)). Try both, then look at
    the benchmark: one of them is measurably faster at small N, and the
    reason is dispatch overhead, not arithmetic.
    """
    return np.linalg.norm(a)


def cosine_similarity(a, b):
    """SHAPE: (N,) and (N,) -> scalar"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def project_onto(a, b):
    """Projection of a onto b. SHAPE: (N,) and (N,) -> (N,)"""
    return np.multiply(b, np.dot(a, b) / np.dot(b, b))


# ===========================================================================
# HARNESS -- do not edit.
# ===========================================================================

TOL = 1e-6


def _close(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return x.shape == y.shape and np.allclose(x, y, atol=TOL, rtol=0)


def check_agreement(trials=200, dim=64, seed=0):
    """Random inputs, both implementations, must match to 1e-6."""
    rng = np.random.default_rng(seed)
    cases = [
        ("add",      lambda u, v: add(u.arr, v.arr),               lambda u, v: (u.vec + v.vec).c),
        ("sub",      lambda u, v: sub(u.arr, v.arr),               lambda u, v: (u.vec - v.vec).c),
        ("scale",    lambda u, v: scale(u.arr, 3.7),               lambda u, v: (u.vec * 3.7).c),
        ("hadamard", lambda u, v: hadamard(u.arr, v.arr),          lambda u, v: u.vec.hadamard(v.vec).c),
        ("dot",      lambda u, v: dot(u.arr, v.arr),               lambda u, v: u.vec.dot(v.vec)),
        ("norm",     lambda u, v: norm(u.arr),                     lambda u, v: u.vec.norm()),
        ("cosine",   lambda u, v: cosine_similarity(u.arr, v.arr), lambda u, v: u.vec.cosine_similarity(v.vec)),
        ("project",  lambda u, v: project_onto(u.arr, v.arr),      lambda u, v: u.vec.project_onto(v.vec).c),
    ]

    class Pair:
        def __init__(self, data):
            self.arr = np.asarray(data, dtype=float)
            self.vec = Vector(data)

    print(f"AGREEMENT -- {trials} random trials, dim={dim}, tol={TOL:g}")
    worst = {}
    for name, fast, slow in cases:
        bad = 0
        peak = 0.0
        for _ in range(trials):
            u = Pair(rng.normal(size=dim) * rng.uniform(0.01, 100))
            v = Pair(rng.normal(size=dim) * rng.uniform(0.01, 100))
            got, want = fast(u, v), slow(u, v)
            peak = max(peak, float(np.max(np.abs(np.asarray(got, dtype=float)
                                                 - np.asarray(want, dtype=float)))))
            if not _close(got, want):
                bad += 1
                if bad == 1:
                    print(f"  FAIL  {name}\n        numpy  {got!r}\n        python {want!r}")
        worst[name] = peak
        if bad == 0:
            print(f"  pass  {name:<9} max abs diff {peak:.3e}")
        else:
            print(f"  FAIL  {name:<9} {bad}/{trials} disagreed")
            return False
    print(f"\n  all eight agree. Worst error anywhere: {max(worst.values()):.3e}\n")
    return True


def _time(fn, reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def benchmark():
    print(f"{'dim':>7} | {'pure python':>13} | {'numpy':>11} | {'speedup':>9}")
    print("-" * 50)
    rng = np.random.default_rng(1)
    for dim in (8, 64, 512, 4096, 65536):
        data_a, data_b = rng.normal(size=dim), rng.normal(size=dim)
        va, vb = Vector(data_a), Vector(data_b)
        na, nb = np.asarray(data_a), np.asarray(data_b)
        reps = max(3, min(2000, 2_000_000 // dim))

        slow = _time(lambda: va.cosine_similarity(vb), reps)
        fast = _time(lambda: cosine_similarity(na, nb), reps)
        print(f"{dim:>7} | {slow*1e6:>10.1f} us | {fast*1e6:>8.1f} us | {slow/fast:>8.1f}x")

    print("\nRead the speedup column top to bottom before you say anything.")
    print("It is not constant. Why not? That is the question for this step.\n")


if __name__ == "__main__":
    if check_agreement():
        benchmark()