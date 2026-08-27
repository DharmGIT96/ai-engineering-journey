"""
svd_lab.py  --  Phase 01, Day 05: eigenvalues, SVD, and low-rank approximation

    python svd_lab.py

Put it in  phase-01-math/day-05-eigen-svd/  next to camera.png

Four functions. The first two are the ideas; the third is LoRA; the fourth
you will actually look at with your eyes.

You may use np.linalg.svd and np.linalg.eig. You may NOT use
np.linalg.matrix_rank -- that is the one you are reimplementing.
"""

import numpy as np


# ===========================================================================
# 1. EFFECTIVE RANK -- reimplement np.linalg.matrix_rank
# ===========================================================================

def effective_rank(A, tol=None):
    """Count the singular values that are meaningfully nonzero.

    tol=None means use numpy's own default rule:

        tol = max(singular_values) * max(A.shape) * finfo(float).eps

    Note it scales with the LARGEST singular value -- that is the whole point.
    A fixed constant would be wrong on data of a different magnitude, which is
    exactly the mistake you avoided on Day 02.

    SHAPE: (M, N) -> scalar int
    """
    s = np.linalg.svd(A, compute_uv=False)
    if tol is None:
        tol = s.max() * max(A.shape) * np.finfo(A.dtype).eps
    return np.count_nonzero(s > tol)


def spectral_gap(A):
    """Find where the singular value spectrum falls off a cliff.

    Return (k, ratio) where k is the number of singular values ABOVE the
    largest gap, and ratio is s[k-1] / s[k].

    This is the "look at the spectrum and cut at the gap" move, automated.
    Ignore gaps involving exact zeros. If there are fewer than 2 singular
    values, return (len(s), inf).

    SHAPE: (M, N) -> (int, float)
    """
    s = np.linalg.svd(A, compute_uv=False)

    # Rule 1: If there are fewer than 2 singular values, return (len(s), inf)
    if len(s) < 2:
        return len(s), float('inf')

    # Rule 2: Ignore gaps involving exact zeros (or practically zero noise)
    # We only look at indices where the *next* value is safely above zero
    tol = np.finfo(s.dtype).eps * s.max()
    valid_len = np.count_nonzero(s > tol)

    if valid_len < 2:
        return len(s), float('inf')

    # Fix the math: calculate gaps as (current - next) so they are positive numbers
    gaps = s[:-1] / s[1:]

    # Only look at the gaps within our non-zero/valid singular values
    valid_gaps = gaps[:valid_len - 1]

    if len(valid_gaps) == 0:
        return len(s), float('inf')

    # Find the index of the LARGEST drop-off gap
    # k_gap is the index in the gap array; k is the number of values ABOVE it
    k_gap = np.argmax(valid_gaps)
    k = k_gap + 1
    ratio = s[k - 1] / s[k]
    return k, ratio


# ===========================================================================
# 2. LOW-RANK APPROXIMATION -- this is LoRA
# ===========================================================================

def rank_k_approx(A, k):
    """The best possible rank-k approximation of A. Provably best -- no other
    rank-k matrix is closer (Eckart-Young theorem).

    Take the SVD, keep the top k singular values and their vectors, rebuild.

    Do NOT build diag(S) and matmul it -- that materializes a (k, k) matrix
    for no reason. Scale the columns of U directly: `U[:, :k] * S[:k]`
    broadcasts (M, k) against (k,) and costs nothing. Day 04 in anger.

    SHAPE: (M, N), int -> (M, N)
    """
    U, S, V = np.linalg.svd(A, full_matrices=False)
    return U[:, :k] @ np.diag(S[:k]) @ V[:k]


def compression_ratio(shape, k):
    """How many numbers does a rank-k factorization store, vs the full matrix?

    Full:   M*N
    Rank-k: M*k + k*N     (you store the two skinny factors, never the product)

    Return (stored, full, fraction).

    SHAPE: (M, N), int -> (int, int, float)
    """
    M, N = shape
    stored = M * k + k * N
    full = M * N
    fraction = stored / full
    return stored, full, fraction


# ===========================================================================
# HARNESS -- do not edit.
# ===========================================================================

def check_rank():
    print("PART 1 -- effective_rank")
    rng = np.random.default_rng(0)
    cases = [
        ("[[1,2],[2,4]] (rank 1)", np.array([[1., 2.], [2., 4.]]), 1),
        ("[[1,2],[3,1]] (rank 2)", np.array([[1., 2.], [3., 1.]]), 2),
        ("zeros(4,4)", np.zeros((4, 4)), 0),
        ("identity(5)", np.eye(5), 5),
        ("(8,3)@(3,8) -> rank 3", rng.normal(size=(8, 3)) @ rng.normal(size=(3, 8)), 3),
        ("(20,50) random -> rank 20", rng.normal(size=(20, 50)), 20),
    ]
    for name, M, want in cases:
        got = effective_rank(M)
        print(f"  {'pass' if got == want else 'FAIL'}  {name:<28} got {got}  want {want}")

    print("\n  the tolerance is a decision -- rank-3 signal + 1e-9 noise:")
    B = rng.normal(size=(6, 3)) @ rng.normal(size=(3, 6))
    Bn = B + rng.normal(scale=1e-9, size=B.shape)
    for tol in (None, 1e-10, 1e-8, 1e-6):
        label = "default" if tol is None else f"{tol:.0e}"
        print(f"    tol={label:<9} -> rank {effective_rank(Bn, tol)}")
    k, ratio = spectral_gap(Bn)
    print(f"    spectral_gap says {k} (gap of {ratio:.2e}x)   <-- finds it without being told\n")


def check_approx():
    print("PART 2 -- rank_k_approx")
    rng = np.random.default_rng(1)
    A = rng.normal(size=(60, 40))

    r = min(A.shape)
    full = rank_k_approx(A, r)
    print(f"  {'pass' if np.allclose(full, A) else 'FAIL'}  k=min(shape) rebuilds A exactly "
          f"(max diff {np.abs(full - A).max():.2e})")

    A1 = rank_k_approx(A, 1)
    print(f"  {'pass' if effective_rank(A1) == 1 else 'FAIL'}  k=1 really has rank 1")

    # Eckart-Young: no random rank-k matrix should beat it
    k = 5
    best = np.linalg.norm(A - rank_k_approx(A, k))
    beaten = 0
    for _ in range(200):
        R = rng.normal(size=(60, k)) @ rng.normal(size=(k, 40))
        R *= np.linalg.norm(A) / np.linalg.norm(R)
        if np.linalg.norm(A - R) < best - 1e-9:
            beaten += 1
    print(f"  {'pass' if beaten == 0 else 'FAIL'}  Eckart-Young: 0 of 200 random rank-{k} "
          f"matrices beat it (beaten={beaten})")

    print(f"\n  {'k':>4} | {'error':>8} | {'stored':>9} | {'of full':>8}")
    print("  " + "-" * 38)
    for k in (1, 2, 5, 10, 20, 40):
        err = np.linalg.norm(A - rank_k_approx(A, k)) / np.linalg.norm(A)
        stored, fullsz, frac = compression_ratio(A.shape, k)
        print(f"  {k:>4} | {err:>7.2%} | {stored:>9,} | {frac:>7.1%}")
    print()


def check_image():
    from PIL import Image
    print("PART 3 -- compress a real image")
    try:
        img = np.array(Image.open("camera.png").convert("L"), dtype=float)
    except FileNotFoundError:
        print("  camera.png not found -- put it next to this file.")
        return

    print(f"  image {img.shape}, full rank {effective_rank(img)}\n")
    print(f"  {'k':>4} | {'error':>8} | {'stored':>10} | {'of full':>8}")
    print("  " + "-" * 40)
    panels = [img]
    ks = [1, 5, 10, 25, 50, 100]
    for k in ks:
        Ak = rank_k_approx(img, k)
        err = np.linalg.norm(img - Ak) / np.linalg.norm(img)
        stored, fullsz, frac = compression_ratio(img.shape, k)
        print(f"  {k:>4} | {err:>7.2%} | {stored:>10,} | {frac:>7.1%}")
        panels.append(np.clip(Ak, 0, 255))

    strip = np.hstack([np.pad(p, ((0, 0), (0, 4)), constant_values=255) for p in panels])
    Image.fromarray(strip.astype(np.uint8)).save("compressed_strip.png")
    print("\n  wrote compressed_strip.png  (original, then k =", ", ".join(map(str, ks)), ")")
    print("""
  LOOK AT IT. Find the smallest k where you would not notice.
  Then read that row's 'of full' column -- that is your honest compression
  ratio, decided by your eyes rather than by a norm.

  Then ask the real question: the error at that k is probably 5-10%.
  Why does a 10% error look like nothing? What is the 10% made of?
""")


if __name__ == "__main__":
    check_rank()
    check_approx()
    check_image()
