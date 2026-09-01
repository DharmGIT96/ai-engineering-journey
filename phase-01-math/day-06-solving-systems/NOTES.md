# Day 06 — Solving `Ax = b` Without Inverting Anything

Phase 01 · Math Foundations
Artifact: `triangular.py` — `solve_triangular`, `lu_decompose`, `solve_via_lu`

---

## Where `Ax = b` comes from

Three people buy snacks. You know the quantities and the totals, not the prices.

```
Person 1:  2 samosa + 1 tea + 3 biscuit  =  ₹95
Person 2:  1 samosa + 3 tea + 1 biscuit  =  ₹80
Person 3:  4 samosa + 1 tea + 2 biscuit  =  ₹135
```

Call the unknown prices `x1, x2, x3`:

```
2·x1 + 1·x2 + 3·x3 =  95
1·x1 + 3·x2 + 1·x3 =  80
4·x1 + 1·x2 + 2·x3 = 135
```

Stack the coefficients, the unknowns, the totals:

```
[ 2  1  3 ] [x1]   [  95 ]
[ 1  3  1 ] [x2] = [  80 ]
[ 4  1  2 ] [x3]   [ 135 ]
     A         x        b        A (3,3)   x (3,)   b (3,)
```

**`Ax = b` is just a stack of ordinary equations.** Each row of `A` dotted with `x` gives
one entry of `b` — matrix-vector multiplication from Day 01, written out.

That shape — *"I know the inputs and the outputs, what were the weights?"* — is linear
regression, curve fitting, sensor calibration, circuit solving, every Newton step.

## Nobody computes `x = A⁻¹b`

```
A) speed, 1500x1500
   inv(A) @ b      315.2 ms
   solve(A, b)      50.8 ms      6.2x faster

B) accuracy, Hilbert matrix n=12, true answer all ones
   inv(H) @ b   err = 7.125e+00      <-- garbage
   solve(H, b)  err = 2.762e-01      25.8x more accurate
```

Slower *and* worse. Not a tradeoff — strictly dominated.

**Why slower:** computing `A⁻¹` means solving `AX = I`, which is `n` systems, one per
column of the identity. You had one right-hand side and it answered `n` of them, then did
one more matmul.

```
solve(A, b)     factor once, substitute once            ~n³/3
inv(A) @ b      factor once, substitute n times, matmul  ~n³ + n²
```

## Condition number = `σ_max / σ_min`

```
Hilbert n=12:
  largest  singular value = 1.7954e+00
  smallest singular value = 1.0930e-16
  ratio                   = 1.6426e+16   == np.linalg.cond(H)
```

It's a ratio of singular values — Day 05's tool, and Day 05's *ratios not differences*
reasoning. It means **error amplification**: perturb the input by relative `ε` and the
output can move by `cond × ε`.

```
  n |      cond(H) | digits lost | digits left of 16 | actual max err
  3 |     5.24e+02 |         2.7 |              13.3 |       1.03e-14
  6 |     1.50e+07 |         7.2 |               8.8 |       6.65e-11
  9 |     4.93e+11 |        11.7 |               4.3 |       7.06e-06
 12 |     1.64e+16 |        16.2 |               0.0 |       2.76e-01
 15 |     3.37e+17 |        17.5 |               0.0 |       5.75e+01
```

**Prediction and measurement track each other down the whole table.** You lose
`log10(cond)` significant digits. At n=12 you're out of digits; at n=15 the answer isn't
the right order of magnitude. NumPy never warns — the matrix isn't singular and every
operation is performed correctly.

The budget is **your data's** precision, not float64's:

> `cond = 1e8` and data good to 6 digits → `6 − 8 = −2` → **zero trustworthy digits.**

And the damage is done by the problem, not the algorithm:

```
perturb b by relative 1e-15  ->  x moves by relative 2.36e-01  (24%)
```

An ill-conditioned system is one whose answer genuinely doesn't know what it is. No
algorithm recovers information the problem never encoded; the best you can do is not make
it worse — which is exactly what `inv` does and `solve` doesn't.

**Practical rule: check `np.linalg.cond` before trusting a solve.** Above `1e10` worry;
above `1e15` don't believe it.

### Rank and condition number are the same question

```
σ_min = 0  ⟺  rank < n  ⟺  not invertible  ⟺  cond = ∞  ⟺  no unique solution
```

> **Rank is the binary question: is any singular value exactly zero?
> Condition number is the continuous one: how close to zero is the smallest, relative to
> the largest?**

Ill-conditioned = *almost* rank-deficient. Day 02's tolerance problem and Day 06's
condition number are the same problem at different resolutions.

## What `solve` actually does

Factor `A` into pieces that are trivial to solve.

```
A = L U          L lower-triangular, U upper-triangular

then  Ax = b  becomes  LUx = b:
   solve  L y = b   for y     (forward substitution, top to bottom)
   solve  U x = y   for x     (back substitution, bottom to top)
```

### Why triangular is easy

```
[ 2  1  3 ] [x1]   [ 13 ]        2·x1 + 1·x2 + 3·x3 = 13
[ 0  5  1 ] [x2] = [ 12 ]        0·x1 + 5·x2 + 1·x3 = 12
[ 0  0  4 ] [x3]   [  8 ]        0·x1 + 0·x2 + 4·x3 =  8
```

The zeros do the work. Read bottom to top:

```
row 3:  4·x3 = 8                     ->  x3 = 2      one unknown
row 2:  5·x2 + 1·(2) = 12            ->  x2 = 2      one unknown again
row 1:  2·x1 + 1·(2) + 3·(2) = 13    ->  x1 = 2.5
```

**Every step has exactly one unknown, so every step is one division.** `n²` operations,
no linear algebra. Generalized:

```
known = T[i, i+1:] @ x[i+1:]        # upper: everything below is already solved
x[i]  = (b[i] - known) / T[i, i]
```

At `i = n-1` the slice is empty, the dot product is `0.0`, and the base case handles
itself. For lower-triangular, walk `0 … n-1` and use `T[i, :i] @ x[:i]`.

### Where the triangle comes from

Gaussian elimination — the schoolbook move. **Subtracting a multiple of one equation from
another doesn't change the answer**, and you pick the multiple to knock out a coefficient.

```
start:                    row2 -= 0.50×row1      row3 -= 2.00×row1      row3 -= −0.40×row2
[ 2   1   3 ] =  95       [ 2   1    3  ]= 95    [ 2  1    3  ]= 95     [ 2  1    3   ]= 95
[ 1   3   1 ] =  80  ->   [ 0  2.5 −0.5 ]= 32.5  [ 0 2.5 −0.5 ]= 32.5   [ 0 2.5 −0.5  ]= 32.5
[ 4   1   2 ] = 135       [ 4   1    2  ]= 135   [ 0 −1   −4  ]= −55    [ 0  0   −4.2 ]= −42

back-substitute:  x3 = 10,  x2 = 15,  x1 = 25
samosa ₹25, tea ₹15, biscuit ₹10        check against ORIGINAL: [95. 80. 135.] ✓
```

**And the multipliers `0.50, 2.00, −0.40` ARE `L`.** Put them in a unit lower-triangular
matrix, keep the triangle as `U`, and `L @ U == A`.

> **LU factorization isn't a separate algorithm — it's a receipt for the elimination you
> were already doing.** `U` is where you ended up; `L` records how you got there. Store
> both and never redo the elimination, however many `b` vectors arrive.

## The payoff: one factorization, many right-hand sides

```
50 solves in a loop         88.5 ms
solve(A, B) all at once      7.3 ms      12.2x faster
inv(A) @ B                  14.0 ms
```

Stack the right-hand sides into a matrix `B` of shape `(n, k)`. One factorization, `k`
substitutions. `scipy.linalg.lu_factor` / `lu_solve` exposes the split explicitly, which
is what makes iterative solvers and Newton methods affordable.

## Pivoting — why real LU returns three matrices

`if U[i,i] == 0` is an **exact** comparison, and exact zero almost never happens. A pivot
of `1e-17` passes the check, and then `multiplier = U[j,i] / 1e-17 ≈ 1e17` swamps every
other number in the row — catastrophic cancellation, same as classical Gram-Schmidt.

**A tiny pivot is nearly as bad as a zero one.** So real LAPACK does *partial pivoting*:
before eliminating column `i`, swap in whichever remaining row has the **largest** entry
in that column, keeping the divisor as big as possible.

That's the `P` in `PA = LU`, and it's why `scipy.linalg.lu` returns three matrices.

---

## Bugs found in review

| Bug | Why it mattered |
|---|---|
| `solve_via_lu` returned `np.linalg.solve(A, b)` | The task was to compose your own pieces. The harness passed it. |
| `solve_triangular(L, b)` without `lower=True` | Walked a lower-triangular matrix bottom-to-top. Residual `1.58e+02`. |
| Pivot check inside the inner `j` loop | For `i = n-1` the inner loop is empty, so the last pivot was never checked. `[[2,1],[4,2]]` returned a `U` with a zero on the diagonal. |
| `U[i,i] == 0` exact comparison | A `1e-17` pivot is not zero and is nearly as destructive. Needs a scaled tolerance, or pivoting. |

### The tell that a green test is lying

```
random n=5     err vs truth 4.44e-16   vs numpy 0.00e+00
random n=40    err vs truth 1.11e-15   vs numpy 0.00e+00
```

`vs numpy 0.00e+00`. **Exactly zero.** Two independent implementations of the same
algorithm never agree to exactly zero — they agree to ~`1e-15`, because they perform the
same additions in different orders and floating point notices (Day 03). A hard zero means
the same code ran twice.

The reference run, for contrast: `2.22e-16` and `1.55e-15`.

> **A test result that's too perfect is a signal, not a success.**

### The habit that would have caught everything

`Ax = b` is a claim you can check. Substitute the answer back:

```python
residual = np.abs(A @ x - b).max()
```

```
lower=False -> residual 1.580e+02      wrong
lower=True  -> residual 0.000e+00      right
```

**No reference implementation, no library, no known-correct answer required** — just the
question you were asked. This is the antidote to the recurring theme: every green-but-wrong
test this week compared code to *itself*. A residual compares it to *the problem statement*.

Whenever a function claims to solve something, make it prove it.

### And one good instinct, unprompted

`U[j, :] -=` was changed to `U[j, i:] -=`. Correct, and a real optimization — the columns
left of `i` are already zero, so touching them is wasted work. Same instinct as hoisting
`norm()` out of the loop on Day 01, arrived at without being asked.
