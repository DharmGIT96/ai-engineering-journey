# Day 02 — Rank, Linear Independence, Orthogonalization

Phase 01 · Math Foundations
Artifact: `gram_schmidt.py` — classical and modified Gram-Schmidt, pure Python over `ailib.vectors.Vector`

---

## Rank, without the jargon

Picture a spreadsheet. **Rank asks: how many columns carry real information, and how
many are other columns in disguise?**

```
  cents | dollars              cents | dollars | euros
  ------+--------              ------+---------+------
   100  |   1.00                100  |   1.00  |  0.92
   250  |   2.50                250  |   2.50  |  2.30
   500  |   5.00                500  |   5.00  |  4.60

  dollars = cents / 100         euros and dollars are both cents rescaled
  2 columns, 1 real → RANK 1    3 columns, 1 real → RANK 1
```

**Rank = how many columns survive after you delete every one you could rebuild from
the others.**

A vector is *redundant* if some combination of the others reproduces it exactly:

```
 v1     v2     v3            3 × [1,0]  =  [3,0]
[1]    [0]    [3]            5 × [0,1]  =  [0,5]
[0]    [1]    [5]                       +  ------
                                           [3,5]  = v3  ✓ rebuilt
```

So `{[1,0], [0,1], [3,5]}` has rank **2**, not 3. In `R²` you never get more than two
independent columns — a third is always some mix of the first two.

### The bound

```
rank ≤ min(rows, cols)
```

A `1000 × 768` embedding matrix can have rank at most **768** — you can't have more
independent columns than you have columns, and each row lives in 768-dimensional space.
Nothing *forces* a column to be redundant, so 768 is achievable; rank only drops when a
relation actually exists.

**This is what "low rank" means in LoRA.** A weight update that could be a full-rank
`768×768` matrix is constrained to rank 8 — only 8 independent directions of change are
permitted. That's why the adapter is tiny.

## Rank and invertibility are the same fact

Day 01 framing: a matrix is a **compiled function** `Rⁿ → Rᵐ`. Rank is the dimension of
its output.

```
A = [1, 2]     row2 = 2 × row1, rank 1
    [2, 4]

before:  the full plane          after A:  a single line
   ↑ ↑ ↑ ↑                                   ╱
   ↑ ↑ ↑ ↑        ────────►                ╱
   ↑ ↑ ↑ ↑                                ╱
```

Both rows point the same way, so every input lands on one line. Two different inputs
produce the same output; the information is destroyed and no inverse can exist.
**"Rank-deficient", "not invertible", and "destroys information" are three names for
one thing.**

This is why collinear features wreck linear regression: `XᵀX` becomes rank-deficient,
the inverse doesn't exist, and the solver returns garbage that depends on rounding noise.

## Gram-Schmidt: the redundancy test, made constructive

> **Take each vector. Subtract everything already covered. Whatever's left is new.**

"Subtract what's covered" is `v - v.project_onto(u)` — built on Day 01.

```
STEP 1  keep the first vector          u1 = [3, 0]

STEP 2  v2 = [2,2]
        projection onto u1  = [2, 0]   ← already covered
        leftover            = [0, 2]   ← NEW. keep it.
        check: u1 · u2 = 0.0           ← perpendicular, guaranteed

STEP 3  v3 = [5,5]
        projection onto u1  = [5, 0]
        projection onto u2  = [0, 5]
        leftover            = [0, 0]   ← nothing left. REDUNDANT, discard.

RANK = number of survivors = 2
```

Three consequences:

1. **The leftover is always perpendicular to everything before it.** You removed exactly
   the part pointing along `u1`, so what remains cannot point along `u1`.
2. **Zero leftover ⇒ redundant.** Counting survivors *is* a rank algorithm.
3. Normalizing the survivors gives an **orthonormal** set — perpendicular *and* unit
   length. The nicest basis to compute in, and the reason `Q` in QR is orthonormal.

## You cannot test `leftover == 0`

Build `v3` to be *exactly* `2·v1 + 3·v2` with awkward decimals and the true leftover is
`[0,0,0]`. What you actually get:

```
computed leftover : [-1.221e-15, -3.886e-16, +8.882e-16]
its length        : 1.559e-15                (should be 0.0)
```

Normalize that and it becomes a **full-length unit vector made entirely of rounding
noise**:

```
unit(leftover)    : [-0.783221, -0.249207, +0.569615]
length            : 1.000000
```

It passes every structural check — unit length, perpendicular to the others — and it is
pure garbage. Your code reports rank 3 for a rank-2 set. Nudge one input by a single
float step (`0.3` → `0.30000000000000004`) and the "direction" changes completely:

```
before : [-0.783221, -0.249207, +0.569615]
after  : [+0.957020, +0.205076, -0.205076]
```

**A basis that depends on the last bit of your input is not a basis.** And this is worse
than a `nan`, because a `nan` is loud — it poisons everything downstream until someone
notices. A unit-length noise vector is silent.

### The fix, and why it's a judgement call

```python
if leftover.norm() < tol:
    continue        # redundant
```

Choosing `tol` is real engineering:

- too small → keep noise vectors, overcount the rank
- too large → discard real directions, undercount
- **it cannot be a fixed constant** — a leftover of `1e-9` is noise if your vectors have
  magnitude `1e6`, and signal if they have magnitude `1e-8`. Tolerances scale with data.

`np.linalg.matrix_rank` exposes exactly this knob; its default derives from the largest
singular value × machine epsilon × dimension.

## Classical vs modified — one character apart

```python
# CLASSICAL                              # MODIFIED
leftover = v                             r = v
for u in basis:                          for u in basis:
    leftover = leftover - v.project_onto(u)   r = r - r.project_onto(u)
                          ^                                ^
              always the ORIGINAL v          the RUNNING remainder
```

**Analogy.** You're removing two contaminants, red and blue.
*Classical:* measure the red in the original sample, remove it; measure the blue in the
**original** sample, remove that. *Modified:* remove the red, then **look again** at
what's left and measure the blue in *that*.

If red and blue overlap, classical removes the overlap twice — it's working from a stale
measurement. `u1` and `u2` overlapping is what happens when the inputs are nearly
parallel, which for real embedding matrices is always.

### The measurement

Eight vectors, independent but barely — pairwise cosine similarity `0.9999999999999989`:

```
           | orthogonality error |   unit error
 classical |           8.258e-02 |    2.220e-16
  modified |           2.975e-10 |    0.000e+00

 modified is 277,557,930x more orthogonal
```

`8.3e-02` means two supposedly-perpendicular basis vectors are about 5° apart. Classical
didn't lose a few digits; it returned a wrong answer while every line did what it was told.

**Look at the unit-error column.** Classical normalizes *perfectly* — every returned
vector has length exactly 1.0. It passes every structural test. It just points the wrong
way. A reasonable test suite cannot tell these two implementations apart.

### Why classical fails

After removing `u1`, the remainder is ~`3e-9` — the large components cancelled. Classical
then computes the `u2` projection from the *original* `v3`, whose components are ~`1.0`.
It subtracts a quantity derived from numbers a billion times larger than what's actually
left. That's **catastrophic cancellation**: subtracting two nearly-equal floats and
keeping a result made mostly of the bits that survived.

Modified never lets the measurement go stale — each projection comes from a remainder
already at the right order of magnitude.

---

## The lesson

> **Textbook-correct is not implementation-correct.**

A derivation valid over the reals can be worthless over floats, and the gap only appears
on *nearly* degenerate inputs — which is what real data looks like. Embeddings are nearly
collinear; feature matrices are nearly rank-deficient. The well-conditioned test case you
would naturally write is precisely the case that cannot distinguish the two algorithms.

This is why you don't hand-roll QR in production. `numpy.linalg.qr` uses Householder
reflections — stabler than both variants here — and offers no algorithm choice, because
there isn't one worth making.

## Bugs found in review

| Bug | Why it mattered |
|---|---|
| `basis.append(r / r_norm)` — raw ndarray, not `Vector` | The two functions had the same contract and different return types. Crashed the harness. |
| `np.dot(r, u)` missing `/ np.dot(u, u)` | Correct **only because** basis vectors are unit length — an unstated precondition. Reuse it on a non-normalized basis and it silently returns nonsense (9× off in the test case). |

The second is a category worth naming: **an implicit precondition.** Not a typo, not a
performance mistake. The fix is either keep the divide (one multiply, works
unconditionally) or write the invariant in a comment. Silence is not an option.
