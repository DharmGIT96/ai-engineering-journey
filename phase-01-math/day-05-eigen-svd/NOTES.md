# Day 05 — Eigenvalues, SVD, and Low-Rank Approximation

Phase 01 · Math Foundations
Artifact: `svd_lab.py` — `effective_rank`, `spectral_gap`, `rank_k_approx`, `compression_ratio`,
plus image compression on `camera.png`

---

## Eigenvectors: the directions a matrix doesn't rotate

A matrix is a function that moves vectors. Feed `A = [[2,1],[1,2]]` some vectors and check
whether the output points the same way as the input — using cosine similarity, where
`cos(v, Av) == 1` means the direction survived:

```
         v          A@v        cos   stretch
    [1, 0]       [2, 1]   0.894427    2.2361
    [0, 1]       [1, 2]   0.894427    2.2361
    [1, 2]       [4, 5]   0.977802    2.8636
    [3, 1]       [7, 5]   0.955779    2.7203
    [1, 1]       [3, 3]   1.000000    3.0000   <-- UNCHANGED
   [1, -1]      [1, -1]   1.000000    1.0000   <-- UNCHANGED
```

Most vectors get rotated. Two come out pointing exactly where they went in. **Those are
the eigenvectors**; the stretch factor is the eigenvalue.

```
A @ v  =  λ v      "applying the matrix is the same as multiplying by a number"
```

A matrix rotates, stretches and shears all at once. **Along its eigenvectors it does none
of that.** Describe a vector in terms of the eigenvectors and the matrix stops being a
matrix and becomes a list of scalars.

### Why that's useful: `Aᵏ → λᵏ`

```
apply A ten times to [3,1]:
  brute force, 10 matmuls : [118099. 118097.]
  via eigenvalues         : [118099. 118097.]   agree to 0.0e+00

the eigen route raises two NUMBERS to the 10th power: 3¹⁰ = 59049, 1¹⁰ = 1
direction of the result: [0.7071, 0.7071]   <-- the λ=3 eigenvector
```

Because `3¹⁰` dwarfs `1¹⁰`, everything collapses onto the largest eigenvector. **Applying
a matrix repeatedly drags any vector toward its dominant eigenvector.**

That's an algorithm, not a curiosity — it's **power iteration**, and it's how PageRank
ranks the web. It's also why RNNs explode or vanish: the same weight matrix applied at
every timestep.

```
λ = 0.9   ->  0.9¹⁰⁰ = 2.66e-05      signal 38,000x smaller   (vanishing gradients)
λ = 1.1   ->  1.1¹⁰⁰ = 13,781        signal 13,781x bigger    (exploding gradients)
```

The entire stability problem of deep recurrent networks is one number, compounded.

### Symmetric matrices give orthogonal eigenvectors

```
A is symmetric -> eigenvectors [1,1] and [1,-1] -> dot = 2.2e-17
```

Guaranteed. And **covariance matrices are always symmetric**, so their eigenvectors form
an orthonormal basis — the perpendicular, unit-length set built by hand in Gram-Schmidt.
Those eigenvectors are the **principal components**; the eigenvalue is how much variance
lives along that direction. PCA is nothing more than that.

### Reading eigenvalues off a diagonal matrix

`[[3,0],[0,5]]` → eigenvalues `3, 5`, eigenvectors `[1,0]` and `[0,1]`. A diagonal matrix
already has the axes as its special directions; the diagonal entries are the stretch
factors. No calculation required.

---

## SVD: every matrix, no exceptions

Eigenvectors only exist for square matrices, and even then can be complex or
non-orthogonal. Real data is `1000 documents × 768 dimensions` — not square.

```
A  =  U Σ Vᵀ
```

> **Every matrix — any shape, any contents — does exactly three things:
> rotate, stretch along axes, rotate again.**

```
A.shape (2, 3)   U (2, 2)   S (2,)   Vt (2, 3)
singular values: [3.4641 3.1623]

U.T @ U = [[1, 0],       <-- orthonormal: exactly what Gram-Schmidt produced
           [0, 1]]

rebuild: max|U @ diag(S) @ Vt - A| = 8.88e-16
```

Singular values are always real, always ≥ 0, always sorted descending.

### Rank = the count of nonzero singular values

```
[[1,2],[2,4]]   singular values [5, 0]        rank 1
[[1,2],[3,1]]   singular values [3.6, 1.4]    rank 2
```

The rigorous version of Day 02's "count the survivors". But *nonzero* is a judgement call
— a genuinely rank-3 matrix plus `1e-9` noise:

```
s[0] = 5.8528e+00  ########################################
s[1] = 3.0045e+00  #######################################
s[2] = 4.1063e-01  ####################################
s[3] = 2.6398e-09  ###########
s[4] = 1.7953e-09  ###########
s[5] = 4.4326e-11  ######

gap between s[2] and s[3]: 1.6e+08x

matrix_rank(tol=1e-12) -> 6
matrix_rank(tol=1e-08) -> 3     <-- cutting inside the gap
matrix_rank(default)   -> 6
```

NumPy's default is `max(s) × max(shape) × eps = 7.8e-15`. The noise sits *above* that, so
NumPy honestly answers 6. **It isn't wrong — it cannot know that `1e-9` was noise.**

**The spectrum tells you where its own signal ends.** That eight-order-of-magnitude gap is
visible without knowing anything about the matrix beforehand — which is a far better tool
than guessing a tolerance.

Deciding rank from a spectrum uses **ratios, never differences** (see the bug section).

---

## Low-rank approximation: keep the biggest few

Truncating after `k` gives the **best possible rank-k approximation** — provably, no rank-k
matrix is closer (Eckart-Young). Verified in the harness: 0 out of 200 random rank-5
matrices beat it.

```
768 x 768 matrix:
    k |  error |  numbers stored |  vs full
    1 | 42.09% |           1,536 |    0.3%
    4 | 14.56% |           6,144 |    1.0%
    8 |  7.52% |          12,288 |    2.1%
   16 |  3.76% |          24,576 |    4.2%
   32 |  1.61% |          49,152 |    8.3%
  768 |  0.00% |         589,824 |  100.0%
```

**Rank 8 captures 92.5% of the matrix in 2.1% of the numbers. This is LoRA.** The `r=8` in
a LoRA config is exactly this `k`: instead of updating a full `d×d` weight matrix, learn
`W + BA` with `B` of shape `(d,8)` and `A` of shape `(8,d)`.

And you compute `(x@B)@A`, never `B@A` — the `d×d` product is never materialized. That is
Day 01's associativity lesson (`m·n·p`, the 500× gap) cashing out.

If a LoRA run barely moves the model, `r` may be below the rank the update actually needs
— but check `lora_alpha` (effective scale is `alpha/r`) and the learning rate first, since
raising `r` costs memory.

### Storage break-even

Rank-k stores `M*k + k*N` against a full `M*N`. On a `60×40` matrix, `k=40` stores
**166.7%** of the original — the "compressed" form is bigger. Low-rank only pays when
`k << min(M,N)`.

---

## Image compression: where the error actually goes

`camera.png`, 512×512:

```
    k |  error |     stored |  of full
    1 | 36.04% |      1,024 |    0.4%
    5 | 17.20% |      5,120 |    2.0%
   10 | 13.50% |     10,240 |    3.9%
   25 |  9.06% |     25,600 |    9.8%
   50 |  6.36% |     51,200 |   19.5%
  100 |  3.93% |    102,400 |   39.1%
```

Rank 25 — a tenth of the data — is clearly a photograph. So **what is the missing 9%?**

```
mean |error| in FLAT regions (sky, coat)  :   4.60
mean |error| at EDGES (sharpest 10%)      :  19.35
                                             4.2x more error at edges

energy captured by the first k components:
  k=  1   87.01%      <-- ONE component holds 87% of the image
  k=  5   97.04%
  k= 25   99.18%
```

The error is **concentrated in edges and texture**. One component holds the coarse
light-and-dark layout; large-scale structure is cheap. The expensive part is the long tail
of fine detail spread across hundreds of tiny singular values.

And it's not that the eye is insensitive to error. The eye is *extremely* sensitive to
edges — but rank-25 doesn't **move** the edges, it **softens** them. Structure intact,
texture blurred, and human vision reconstructs from structure.

> **The Frobenius norm treats every pixel as equally important. Your eye does not.
> "9% error" by the metric and "9% worse" to a human are unrelated numbers.**

That gap between what's measurable and what's wanted is central to ML. It's why JPEG
discards high frequencies deliberately (same trick, different basis), why image models
train with perceptual losses like LPIPS rather than plain MSE, and why generative models
are judged by FID instead of pixel error. **A loss function is a proxy for the goal, never
the goal** — and knowing where the proxy diverges is most of the job.

Same reason low-rank works on embeddings: semantic content is a few strong directions plus
a long tail of noise. LoRA bets a fine-tuning update is structure, not texture.

---

## Bugs found in review

| Bug | Why it mattered |
|---|---|
| `np.diff(s)` to find the gap | Singular values descend, so every difference is negative and the `> 0` test never fired. |
| `np.argmax(bool_array)` with nothing True | Returns **0** — indistinguishable from "found at index 0". The function confidently reported `k=0`. |
| Differences instead of ratios | Even sign-corrected, the largest *subtraction* is at the top of the spectrum. Found `k=1` instead of `k=3`. |
| `full_matrices=True` | On a `4000×80` matrix: **82× slower, 49× the memory**, identical answer. |
| `np.diag(S[:k])` | Materializes a `(k,k)` matrix that is 99% zeros; `U[:, :k] * S[:k]` broadcasts instead. |

### `argmax` on booleans — worth memorising

```python
np.argmax([False, False, False])  ->  0
np.argmax([True,  False, False])  ->  0
```

`argmax` returns the first maximum. On booleans that's the first `True` — a useful idiom —
but when nothing is `True` the max is `False`, found at index 0, and you get **0** anyway.
**"Not found" and "found at position 0" return the same value.** Always check `.any()`
first.

Same family as the zero-vector guard (Day 02), the `or` fallback (Day 03) and the
self-consistent `mse` (Day 04): *a function that cannot express "I found nothing", and so
reports something plausible instead.*

### Differences vs ratios — third appearance

```
     between |   difference |        ratio
s[0] -> s[1] |   2.8482e+00 |   1.9480e+00
s[1] -> s[2] |   2.5939e+00 |   7.3169e+00
s[2] -> s[3] |   4.1063e-01 |   1.5555e+08     <-- the real cliff
s[3] -> s[4] |   8.4453e-10 |   1.4704e+00

biggest DIFFERENCE -> k=1   WRONG
biggest RATIO      -> k=3   CORRECT
```

Differences are dominated by wherever the numbers happen to be biggest. **Ratios are
scale-free.** Same lesson as relative-vs-absolute error on Day 03, and the same reasoning
used correctly to read the spectral gap by eye — it just came out as `np.diff` in code.

### `full_matrices` — the economy SVD

```
full_matrices=True    U.shape (4000, 4000)   128.0 MB   2222.3 ms
full_matrices=False   U.shape (4000, 80)       2.6 MB     27.0 ms
```

`True` builds the complete orthonormal basis of the output space so you can use 80 of its
columns. Use `full_matrices=False` (the *thin* or *economy* SVD) on any non-square matrix.

### And a debugging habit worth naming

The first fix attempt changed the `ratio` line — the value *reported* — rather than the
`gaps` line that `argmax` actually *searches*. Fixing where a symptom is printed instead of
where the decision is made is what makes debugging feel like flailing.

> Ask: **"which line chose the wrong answer?"** — not "which line shows it?"
