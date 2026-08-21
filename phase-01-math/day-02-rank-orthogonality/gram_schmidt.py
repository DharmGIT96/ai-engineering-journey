"""
gram_schmidt.py  --  Phase 01, Day 02: orthogonalization

Same folder as linalg_scratch.py. Pure Python, your own Vector class.

    python gram_schmidt.py

THE RULE, in one line:
    take each vector, subtract everything already covered, keep the leftover.

You already own both halves:  v.project_onto(u)  and  v - p.

There are TWO functions to write. They are mathematically identical and
numerically very different. Write classical first, run it, read what the
harness prints, THEN write modified. Do not peek ahead by writing both.
"""

import numpy as np

from ailib.vectors import Vector


# ===========================================================================
# PART 1 -- classical Gram-Schmidt
# ===========================================================================

def gram_schmidt_classical(vectors, tol=1e-10):
    """Return a list of orthonormal Vectors spanning the same space.

    For each incoming vector v:
      1. subtract v's projection onto EVERY basis vector found so far.
         All projections are taken against the ORIGINAL v. That is what
         makes this the "classical" variant -- remember it, it matters.
      2. if what's left is shorter than tol, v was redundant -> skip it.
      3. otherwise normalize the leftover and append it to the basis.

    SHAPE: list of k Vectors of dim N -> list of r Vectors of dim N, r <= k
           (r is the rank)
    """
    basis = []
    for v in vectors:
        v_original = np.array(v, dtype=float)
        v_copy = v_original.copy()
        for u in basis:
            v_projection = u.dot(v_original) / u.dot(u)
            v_copy -= v_projection * u
        v_norm = np.linalg.norm(v_copy)
        if v_norm < tol:
          continue
        v_copy /= v_norm
        basis.append(Vector(v_copy))
    return basis


# ===========================================================================
# PART 2 -- modified Gram-Schmidt.  Write this only AFTER running Part 1.
# ===========================================================================

def gram_schmidt_modified(vectors, tol=1e-10):
    """Same output, one word different in the algorithm.

    Classical: every projection is taken against the original v.
    Modified:  subtract each projection from a RUNNING remainder, and take
               the next projection against that updated remainder.

    So instead of
        leftover = v - proj(v, u1) - proj(v, u2) - proj(v, u3)
    you do
        r = v
        r = r - proj(r, u1)
        r = r - proj(r, u2)
        r = r - proj(r, u3)

    In exact arithmetic these are the same. In floating point they are not,
    and the harness will show you by how much.
    """
    basis = []

    for v in vectors:
        r = np.array(v, dtype=float)
        for u in basis:
            projection_scalar = np.dot(r, u)
            r -= projection_scalar * u

        r_norm = np.linalg.norm(r)
        if r_norm > tol:
            basis.append(r / r_norm)

    return basis



# ===========================================================================
# HARNESS -- do not edit.
# ===========================================================================

def _orthogonality_error(basis):
    """Largest |u_i . u_j| for i != j. Perfect basis -> 0."""
    worst = 0.0
    for i in range(len(basis)):
        for j in range(i + 1, len(basis)):
            worst = max(worst, abs(basis[i].dot(basis[j])))
    return worst


def _unit_error(basis):
    """Largest | ||u|| - 1 |. Perfect basis -> 0."""
    return max((abs(u.norm() - 1.0) for u in basis), default=0.0)


def _check(name, fn):
    print(f"\n{'='*62}\n{name}\n{'='*62}")

    print("\n1. two independent vectors in R^2")
    b = fn([Vector([3, 0]), Vector([2, 2])])
    print(f"   rank {len(b)} (expect 2)   basis {[[round(x,4) for x in u] for u in b]}")
    assert len(b) == 2, "expected rank 2"

    print("\n2. a redundant third vector: v3 = v1 + v2")
    b = fn([Vector([3, 0]), Vector([2, 2]), Vector([5, 2])])
    print(f"   rank {len(b)} (expect 2)   third vector should be discarded")
    assert len(b) == 2, f"expected rank 2, got {len(b)} -- is your tol check working?"

    print("\n3. the floating-point trap: v3 = 2*v1 + 3*v2 with awkward decimals")
    v1, v2 = Vector([0.1, 0.2, 0.3]), Vector([0.4, 0.5, 0.6])
    v3 = Vector([2*a + 3*c for a, c in zip(v1, v2)])
    b = fn([v1, v2, v3])
    print(f"   rank {len(b)} (expect 2)")
    assert len(b) == 2, (f"got rank {len(b)} -- you kept a noise vector. "
                         "The leftover is ~1e-15, not 0.0.")

    print("\n4. all-zero vector in the input")
    b = fn([Vector([1, 0, 0]), Vector([0, 0, 0]), Vector([0, 1, 0])])
    print(f"   rank {len(b)} (expect 2)")
    assert len(b) == 2, "the zero vector must be discarded, not normalized"

    print("\n5. quality on a well-behaved input")
    basis = fn([Vector([1, 1, 0]), Vector([1, 0, 1]), Vector([0, 1, 1])])
    print(f"   rank {len(basis)}   orthogonality error {_orthogonality_error(basis):.3e}"
          f"   unit error {_unit_error(basis):.3e}")

    print(f"\n   {name} passes all structural checks.")
    return True


def _ill_conditioned(n=8, eps=1e-7):
    """Nearly-dependent vectors: each is e_1 nudged by a tiny amount in one
    other axis. Mathematically independent, numerically almost not."""
    out = []
    for i in range(n):
        c = [1.0] * n
        c[i] += eps
        out.append(Vector(c))
    return out

def _stability_showdown():
    print(f"\n{'='*62}\nSTABILITY: classical vs modified on nearly-dependent input\n{'='*62}")
    vecs = _ill_conditioned()
    print(f"\n{'':>10} | {'orthogonality error':>21} | {'unit error':>12}")
    print("-" * 52)
    results = {}
    for label, fn in (("classical", gram_schmidt_classical),
                      ("modified", gram_schmidt_modified)):
        b = fn(vecs)
        oe, ue = _orthogonality_error(b), _unit_error(b)
        results[label] = oe
        print(f"{label:>10} | {oe:>21.3e} | {ue:>12.3e}")
    if results["classical"] > 0 and results["modified"] > 0:
        print(f"\n   modified is {results['classical']/results['modified']:.1f}x more orthogonal")
    print("\n   Both are 'correct'. Only one of them is usable.")
    print("   The vectors here are independent, but only barely -- measured")
    print("   pairwise cosine similarity is 0.9999999999999989. That is the")
    print("   regime real embedding matrices live in.\n")


if __name__ == "__main__":
    _check("CLASSICAL Gram-Schmidt", gram_schmidt_classical)
    try:
        _check("MODIFIED Gram-Schmidt", gram_schmidt_modified)
    except NotImplementedError:
        print("\n\n>>> Part 1 done. Now read the classical results above,")
        print(">>> then write gram_schmidt_modified and run again.\n")
        raise SystemExit(0)
    _stability_showdown()