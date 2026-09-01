"""
triangular.py  --  Phase 01, Day 06: solving Ax = b without inverting anything

    python triangular.py

Put it in  phase-01-math/day-06-solving-systems/

You planned this one before writing it. The plan:

    INPUT   T (n,n) triangular, b (n,), lower (bool)
    OUTPUT  x (n,)  with T @ x == b

    1. n = len(b); x = zeros(n)
    2. row order:  lower -> 0 … n-1        upper -> n-1 … 0
    3. for each row i in that order:
         a. if T[i,i] == 0 -> raise ValueError (numpy will NOT raise for you)
         b. known = already-solved part of row i  ·  already-solved part of x
              upper: T[i, i+1:] · x[i+1:]
              lower: T[i, :i]   · x[:i]
         c. x[i] = (b[i] - known) / T[i,i]
    4. return x
"""

import numpy as np


def solve_triangular(T, b, lower=False):
    """Solve T @ x == b by substitution. No np.linalg.solve, no inv.

    SHAPE: (n,n), (n,) -> (n,)
    """
    n = len(b)
    x = np.zeros(n)
    if lower:
        for i in range(n):
            if T[i,i] == 0:
                raise ValueError("Cannot divide by zero, singular matrix")
            known = np.dot(T[i, :i], x[:i])
            x[i] = (b[i] - known) / T[i, i]
    else:
        for i in range(n-1, -1, -1):
            if T[i, i] == 0:  # <--- Added the check here too!
                raise ValueError("Cannot divide by zero, singular matrix")
            known = np.dot(T[i, i+1:], x[i+1:])
            x[i] = (b[i] - known) / T[i, i]
    return x


def lu_decompose(A):
    """Gaussian elimination WITHOUT pivoting. Return (L, U) with L @ U == A.

    This is the elimination you watched on the canteen problem:
      for each column c, for each row r below it,
        multiplier m = A[r,c] / A[c,c]
        row r -= m * row c
      L[r,c] = m, and L has 1.0 on its diagonal.

    Raise ValueError if a pivot A[c,c] is zero -- without pivoting this
    algorithm genuinely cannot continue. (Real LAPACK swaps rows instead;
    that is what the P in PA=LU means. Out of scope here, but know it exists.)

    SHAPE: (n,n) -> ((n,n), (n,n))
    """
    n = len(A)
    L = np.eye(n)
    U = np.array(A, dtype=float)
    for i in range(n):
        if U[i, i] == 0:
            raise ValueError("Zero pivot encountered. LU decomposition without pivoting cannot continue.")
        for j in range(i+1, n):
            multiplier = U[j, i] / U[i, i]
            L[j, i] = multiplier
            U[j, i:] -= multiplier * U[i, i:]
    return L, U


def solve_via_lu(A, b):
    """Solve A @ x == b using your own LU and your own substitutions.

    Two lines once the pieces exist: forward-solve L y = b, back-solve U x = y.

    SHAPE: (n,n), (n,) -> (n,)
    """
    L, U = lu_decompose(A)
    y = solve_triangular(L, b, lower=True)
    return solve_triangular(U, y)


# ===========================================================================
# HARNESS -- do not edit.
# ===========================================================================

def check_triangular():
    print("PART 1 -- solve_triangular")
    U = np.array([[2., 1., 3.], [0., 5., 1.], [0., 0., 4.]])
    b = np.array([13., 12., 8.])
    x = solve_triangular(U, b)
    ok = np.allclose(x, [2.5, 2., 2.])
    print(f"  {'pass' if ok else 'FAIL'}  the worked example        x = {x}  want [2.5 2. 2.]")

    L = np.array([[2., 0., 0.], [1., 5., 0.], [3., 1., 4.]])
    bl = L @ np.array([1., 2., 3.])
    xl = solve_triangular(L, bl, lower=True)
    print(f"  {'pass' if np.allclose(xl, [1., 2., 3.]) else 'FAIL'}  lower triangular          x = {xl}  want [1. 2. 3.]")

    rng = np.random.default_rng(0)
    for n in (5, 50, 200):
        M = np.triu(rng.normal(size=(n, n))) + np.eye(n) * 5
        xt = rng.normal(size=n)
        got = solve_triangular(M, M @ xt)
        err = np.abs(got - xt).max()
        print(f"  {'pass' if err < 1e-9 else 'FAIL'}  random upper n={n:<4}       max err {err:.2e}")

    S = np.array([[1., 2.], [0., 0.]])
    try:
        solve_triangular(S, np.array([1., 1.]))
        print("  FAIL  singular matrix        returned instead of raising")
    except ValueError as e:
        print(f"  pass  singular matrix        ValueError: {e}")
    except ZeroDivisionError:
        print("  FAIL  singular matrix        got ZeroDivisionError -- numpy does not "
              "raise that; you must check T[i,i] yourself")
    print()


def check_lu():
    print("PART 2 -- lu_decompose and solve_via_lu")
    A = np.array([[2., 1., 3.], [1., 3., 1.], [4., 1., 2.]])
    b = np.array([95., 80., 135.])

    L, U = lu_decompose(A)
    print(f"  {'pass' if np.allclose(L @ U, A) else 'FAIL'}  L @ U rebuilds A          "
          f"max diff {np.abs(L @ U - A).max():.2e}")
    print(f"  {'pass' if np.allclose(L, np.tril(L)) else 'FAIL'}  L is lower triangular")
    print(f"  {'pass' if np.allclose(np.diag(L), 1) else 'FAIL'}  L has unit diagonal")
    print(f"  {'pass' if np.allclose(U, np.triu(U)) else 'FAIL'}  U is upper triangular")

    x = solve_via_lu(A, b)
    print(f"  {'pass' if np.allclose(x, [25., 15., 10.]) else 'FAIL'}  canteen prices            "
          f"x = {x}  want [25. 15. 10.]")

    rng = np.random.default_rng(3)
    for n in (5, 40):
        M = rng.normal(size=(n, n)) + np.eye(n) * n     # diagonally dominant, no pivoting needed
        xt = rng.normal(size=n)
        got = solve_via_lu(M, M @ xt)
        ref = np.linalg.solve(M, M @ xt)
        print(f"  {'pass' if np.allclose(got, xt) else 'FAIL'}  random n={n:<4}              "
              f"err vs truth {np.abs(got-xt).max():.2e}   vs numpy {np.abs(got-ref).max():.2e}")
    print()


def check_reuse():
    print("PART 3 -- the payoff: one factorization, many right-hand sides")
    import time
    rng = np.random.default_rng(7)
    n, k = 300, 50
    A = rng.normal(size=(n, n)) + np.eye(n) * n
    B = rng.normal(size=(n, k))

    t0 = time.perf_counter()
    X1 = np.column_stack([np.linalg.solve(A, B[:, j]) for j in range(k)])
    t1 = time.perf_counter()
    X2 = np.linalg.solve(A, B)
    t2 = time.perf_counter()
    X3 = np.linalg.inv(A) @ B
    t3 = time.perf_counter()

    print(f"  {k} solves in a loop      {(t1-t0)*1e3:8.1f} ms")
    print(f"  solve(A, B) all at once  {(t2-t1)*1e3:8.1f} ms   {(t1-t0)/(t2-t1):.1f}x faster")
    print(f"  inv(A) @ B               {(t3-t2)*1e3:8.1f} ms")
    print(f"  same answers: {np.allclose(X1, X2) and np.allclose(X2, X3)}")
    print(f"\n  cond(A) = {np.linalg.cond(A):.2e}  -> about {np.log10(np.linalg.cond(A)):.1f} digits lost\n")


if __name__ == "__main__":
    check_triangular()
    check_lu()
    check_reuse()
