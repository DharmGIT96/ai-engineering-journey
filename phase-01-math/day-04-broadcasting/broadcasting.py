"""
broadcasting.py  --  Phase 01, Day 04: shapes, axes, and silent bugs

    python broadcasting.py

Put it in  phase-01-math/day-04-broadcasting/

THE TWO RULES, and everything here follows from them:

    1. The axis you name is the one that DISAPPEARS.
    2. Shapes align from the RIGHT. Missing dims count as 1.
       A dim of size 1 stretches. Anything else is an error.

PART 1 is a prediction quiz -- you write the answer BEFORE running.
PART 2 and 3 are code.
"""

import numpy as np


# ===========================================================================
# PART 1 -- predict the shape. Fill in each `None` with a tuple, e.g. (3, 4).
#           Write "ERROR" (the string) if it does not broadcast.
#           Do this WITHOUT running anything. That is the whole exercise.
# ===========================================================================

PREDICTIONS = {
    # (shape_A, shape_B) : your answer
    ((3,),        (3,)):        None,
    ((3, 1),      (1, 4)):      None,
    ((5,),        (5, 1)):      None,
    ((2, 3, 4),   (4,)):        None,
    ((2, 3, 4),   (3, 1)):      None,
    ((2, 3, 4),   (2, 1, 1)):   None,
    ((3,),        (4,)):        None,
    ((32, 768),   (768,)):      None,
    ((32, 768),   (32,)):       None,
    ((32, 768),   (32, 1)):     None,
    ((10, 1, 8),  (1, 6, 8)):   None,
    ((7, 1),      (7,)):        None,
}

# And these are reductions. Give the OUTPUT shape.
REDUCTIONS = {
    # (input_shape, axis, keepdims) : your answer
    ((3, 4),      0, False): None,
    ((3, 4),      1, False): None,
    ((3, 4),      1, True):  None,
    ((32, 768),   0, False): None,
    ((32, 768),   1, True):  None,
    ((2, 3, 4),   1, False): None,
    ((2, 3, 4),   None, False): None,   # axis=None means "collapse everything"
}


# ===========================================================================
# PART 2 -- fix the bugs. Each function is WRONG. The tests say how.
# ===========================================================================

def mse(pred, labels):
    """Mean squared error over N samples.

    pred and labels may arrive with shapes (N,) or (N,1) -- you do not
    control who calls you. The answer must be the same either way, and it
    must be a scalar.

    BUG: this returns a different number depending on the input shapes.
    """
    return (((pred - labels).mean()) ** 2)


def center_rows(X):
    """Subtract the mean embedding from every row of X, shape (N, D).

    'Mean embedding' = the average over all N rows, one value per dimension.
    Result has shape (N, D), and each COLUMN should then have mean ~0.

    BUG: it centers along the wrong axis.
    """
    return X - X.mean(axis=0, keepdims=True)


def normalize_rows(X):
    """Scale every row of X, shape (N, D), to unit length.

    Result shape (N, D); every row's norm should be 1.0.

    BUG: shapes do not line up. Fix it WITHOUT a loop.
    Hint: you need the norms to have shape (N, 1), not (N,). Why?
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms


# ===========================================================================
# PART 3 -- use broadcasting on purpose.
# ===========================================================================

def pairwise_sq_distances(A, B):
    """Squared euclidean distance between every row of A and every row of B.

    A is (N, D), B is (M, D)  ->  result is (N, M)
    result[i, j] == sum((A[i] - B[j]) ** 2)

    Do it with NO Python loop. Broadcasting only.

    The move: make A into (N, 1, D) and B into (1, M, D). Then A - B
    broadcasts to (N, M, D) -- every pair, in one array -- and you reduce
    the last axis away.

    Use `[:, None, :]` and `[None, :, :]` to insert the size-1 axes.

    This is the core of k-NN, of k-means, and of every vector database's
    brute-force search path. It is also a memory trap: think about how big
    (N, M, D) is before you run it on real data.
    """
    A_expanded = A[:, np.newaxis, :]  # Shape: (N, 1, D)
    B_expanded = B[np.newaxis, :, :]  # Shape: (1, M, D)

    # 2. Subtract, square, and sum over the last axis (D)
    return np.sum((A_expanded - B_expanded) ** 2, axis=-1)

# ===========================================================================
# HARNESS -- do not edit.
# ===========================================================================

def _fmt(x):
    return "ERROR" if x == "ERROR" else str(tuple(x))


def check_predictions():
    print("PART 1a -- broadcast shapes")
    score = 0
    for (a, b), guess in PREDICTIONS.items():
        try:
            truth = np.broadcast_shapes(a, b)
        except ValueError:
            truth = "ERROR"
        if guess is None:
            print(f"  ....  {str(a):>12} {str(b):>12}  -> not answered (truth: {_fmt(truth)})")
            continue
        g = "ERROR" if guess == "ERROR" else tuple(guess)
        ok = (g == truth) if truth == "ERROR" else (g == tuple(truth))
        score += ok
        print(f"  {'pass' if ok else 'FAIL'}  {str(a):>12} {str(b):>12}  ->  "
              f"you {_fmt(g):<14} truth {_fmt(truth)}")
    print(f"  {score}/{len(PREDICTIONS)}\n")

    print("PART 1b -- reduction shapes")
    score = 0
    for (shape, axis, keep), guess in REDUCTIONS.items():
        truth = np.zeros(shape).mean(axis=axis, keepdims=keep).shape
        if guess is None:
            print(f"  ....  {str(shape):>12} axis={str(axis):<4} keepdims={str(keep):<5}"
                  f"  -> not answered (truth: {truth})")
            continue
        ok = tuple(guess) == truth
        score += ok
        print(f"  {'pass' if ok else 'FAIL'}  {str(shape):>12} axis={str(axis):<4} "
              f"keepdims={str(keep):<5}  ->  you {str(tuple(guess)):<10} truth {truth}")
    print(f"  {score}/{len(REDUCTIONS)}\n")


def check_fixes():
    rng = np.random.default_rng(0)
    print("PART 2 -- bug fixes")

    # mse: shape-independent and scalar
    p, l = rng.normal(size=100), rng.normal(size=100)
    a = mse(p, l)
    b = mse(p.reshape(-1, 1), l)
    c = mse(p, l.reshape(-1, 1))
    d = mse(p.reshape(-1, 1), l.reshape(-1, 1))
    ok = all(np.isscalar(x) or np.ndim(x) == 0 for x in (a, b, c, d)) and \
         np.allclose([a, b, c, d], a)
    print(f"  {'pass' if ok else 'FAIL'}  mse is shape-independent   "
          f"(N,)(N,)={a:.6f}  (N,1)(N,)={b:.6f}  (N,)(N,1)={c:.6f}  (N,1)(N,1)={d:.6f}")

    # center_rows: columns must have mean ~0
    X = rng.normal(5.0, 2.0, size=(32, 768))
    C = center_rows(X)
    col = abs(C.mean(axis=0)).max()
    row = abs(C.mean(axis=1)).max()
    ok = C.shape == X.shape and col < 1e-12
    print(f"  {'pass' if ok else 'FAIL'}  center_rows                shape {C.shape}  "
          f"max |column mean| {col:.3e}  (max |row mean| {row:.3e})")
    if not ok and row < 1e-12:
        print("        ^ you centered the ROWS. Every row averages to zero,")
        print("          but the columns still do not. Wrong axis.")

    # normalize_rows
    try:
        U = normalize_rows(X)
        norms = np.linalg.norm(U, axis=1)
        ok = U.shape == X.shape and np.allclose(norms, 1.0)
        print(f"  {'pass' if ok else 'FAIL'}  normalize_rows             shape {U.shape}  "
              f"row norms in [{norms.min():.6f}, {norms.max():.6f}]")
    except ValueError as e:
        print(f"  FAIL  normalize_rows             {e}")
    print()


def check_pairwise():
    rng = np.random.default_rng(1)
    print("PART 3 -- pairwise distances")
    A, B = rng.normal(size=(40, 16)), rng.normal(size=(25, 16))
    D = pairwise_sq_distances(A, B)
    print(f"  shape {D.shape}\n{D}")
    brute = np.empty((40, 25))
    for i in range(40):
        for j in range(25):
            brute[i, j] = ((A[i] - B[j]) ** 2).sum()
    ok = D.shape == (40, 25) and np.allclose(D, brute)
    print(f"  {'pass' if ok else 'FAIL'}  shape {D.shape}  max diff vs loops "
          f"{abs(D - brute).max():.3e}")

    print("\n  the memory trap, for a realistic retrieval workload:")
    for n, m, d in [(1_000, 1_000, 768), (10_000, 10_000, 768)]:
        gb = n * m * d * 8 / 1e9
        out = n * m * 8 / 1e9
        print(f"    N={n:,} M={m:,} D={d}:  intermediate (N,M,D) = {gb:>8.1f} GB"
              f"   result (N,M) = {out:.3f} GB")
    print("""
  That intermediate is why nobody computes distances this way at scale.
  The identity used instead:

      ||a - b||^2  ==  ||a||^2 - 2*a.b + ||b||^2

  Every term on the right is either a matmul (N,D)@(D,M) -> (N,M) or a
  per-row norm. The (N,M,D) array is never built. Same answer, and the
  memory drops by a factor of D.

  Try it as a stretch goal: write pairwise_sq_distances_fast(A, B) using
  that identity, check it agrees with your version to 1e-9, and time both.
  Watch out -- the identity is numerically worse. You now know why.
""")


if __name__ == "__main__":
    check_predictions()
    check_fixes()
    check_pairwise()
