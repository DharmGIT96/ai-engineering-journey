# Day 01 — Linear Algebra Intuition

Phase 01 · Math Foundations · Lesson 01
Artifacts: `linalg_scratch.py` (pure Python), `vector_np.py` (NumPy + agreement + benchmark)

---

## What a vector is

A fixed-length array of floats naming a point in an invented space — **a dict with the
keys stripped**. Position in the array *is* the semantics: slot *i* means the same thing
in every vector produced by the same model, and nothing at all across different models.

Corollary that matters in production: embeddings from two different models live in
different coordinate systems. Dot products between them are meaningless-but-plausible
numbers. Change your embedding model, re-embed the entire corpus.

## The three op families

Every tensor operation is one of these. Naming the family names the output shape.

| Family | Examples | Shape effect |
|---|---|---|
| Element-wise | `add`, `sub`, `hadamard`, `relu` | unchanged: `[N] → [N]` |
| Reduction | `dot`, `sum`, `mean`, `norm` | an axis disappears: `[N] → scalar` |
| Contraction | `matmul` | inner dims cancel: `(m×n)@(n×p) → (m×p)` |

`hadamard` and `dot` do *identical multiplications*. The only difference is the `+`
between the terms, which collapses N numbers to one. That `+` is the reduction.

```
a = [1,2,3]   b = [4,5,6]

hadamard  [1*4, 2*5, 3*6]  = [4, 10, 18]    3 -> 3    element-wise
dot        1*4 + 2*5 + 3*6 =     32         3 -> 1    reduction
```

`32` and `[32]` are different types — shape `()` vs shape `(1,)`. A stray `(1,)`
broadcasts against a `(768,)` and silently produces 768 numbers where one was expected.

## Dot product, three readings

1. **Agreement score** — how much two vectors point the same way (weighted vote).
2. **One neuron** — `sum(w*x)` is a single unit's pre-activation.
3. **The retrieval primitive** — what RAG search and attention scoring are built from.

Raw dot product is length-biased: long documents win regardless of relevance.

## Normalization is the recurring move

- `‖a‖ = sqrt(a · a)` — the norm reuses the dot product; never write a second loop.
- **Cosine similarity** divides out both lengths → direction-only agreement, range [-1, 1].
- **Projection** of `a` onto `b` divides out `b`'s length *twice*:

```
proj = b * (a·b / b·b)
              ↑      ↑
        how far   turns b into a direction
```

`b·b` is `‖b‖²` — both normalizations fused into one division, so no `sqrt` is needed.
Proof that `b`'s length cancels completely, with `a = [3,4]`:

```
b=[1,0]     factor=3.00     -> [3, 0]
b=[10,0]    factor=0.30     -> [3, 0]
b=[100,0]   factor=0.03     -> [3, 0]
```

The factor shrinks exactly as fast as the vector grows. Same instinct as `√d_k` in
attention, LayerNorm, RMSNorm: strip the magnitude, keep the direction.

## Matrix = a compiled function

A matrix is a function `Rⁿ → Rᵐ`; the weights are the function body. Matrix-vector
multiply is calling it — each row is one neuron, one dot product. Matmul is **function
composition**. The shape rule `(m×n)@(n×p)→(m×p)` is a type signature; cost is `m·n·p`.

Associativity gives the same answer at wildly different cost — which is exactly why
**LoRA** is cheap: `(x@B)@A` never materializes the `d×d` matrix.

---

## Step 3 — vectorization results

Correctness first: all eight NumPy functions agree with the pure-Python reference to
**5.8e-11** over 200 random trials. Then the benchmark (cosine similarity, end to end):

```
    dim |   pure python |       numpy |   speedup
      8 |        5.5 us |      5.4 us |      1.0x
     64 |       14.2 us |      5.5 us |      2.6x
    512 |       87.8 us |      6.7 us |     13.1x
   4096 |      689.6 us |     10.1 us |     68.4x
  65536 |    11431.3 us |     76.6 us |    149.1x
```

### The cost model behind that column

Measured per-element cost, isolating fixed overhead from marginal cost:

```
        np.dot          Vector.dot (pure python)
dim=1   0.93 us         0.99 us
dim=8   1.13 us         1.34 us
dim=512 1.27 us        26.01 us
dim=4096 2.13 us      201.17 us

marginal cost:  ~0.25 ns/element     ~50 ns/element      (200x)
fixed cost:     ~0.9 us/call         ~0.9 us/call        (identical)
```

Both pay the same ~0.9 µs to make a Python function call. They differ only in the
per-element slope — 200×. So the speedup is `(0.9 + 50N) / (0.9 + 0.25N)`, which is
**1× at N=1 and approaches 200× as N grows**. NumPy doesn't make calls cheap; it makes
elements cheap. Below ~20 elements there aren't enough elements for that to matter.

**Rule: vectorize the inner dimension, not the outer loop.** One call over 10,000
elements beats 10,000 calls over one element, even though the arithmetic is identical.

---

## Bugs found in review (the actually useful part)

| Bug | Why it mattered |
|---|---|
| `unit()` called `norm()` inside the loop | O(N²) instead of O(N) — **1242× slower at dim=4096**. Loop-invariant computation left inside a loop. |
| `sum(a*b)` instead of `np.dot(a,b)` | Python builtin on an ndarray = interpreter loop. **2200× slower**, correct answer, invisible in review. |
| `np.linalg.norm(a) or ...` | `or` is short-circuit fallback, not a choice operator. Produced **silent nan** on the zero vector. |
| `zip` without a length check | Silently truncates to the shorter vector — a 3-dim and a 2-dim multiply "successfully". |
| Missing zero-vector guards | Tests passed *by accident* via `ZeroDivisionError`. Green tests are a floor, not a spec. |
| `self._check(self)` in `__mul__` | A check-shaped object that checks nothing. Pattern-matched from neighbouring code. |

### Two rules earned the hard way

1. **Never call a Python builtin on an ndarray.** `sum`/`min`/`max`/`abs`/`round` all
   drop you back into the interpreter. Use the array method or the `np.` function.
2. **Prefer a raised exception to a nan.** Every comparison against nan returns `False`,
   so guards silently don't fire and the poison spreads until the loss is nan and the
   origin is unrecoverable.

### Memory bandwidth, first sighting

`np.dot(a,b)` is **8× faster** than `(a*b).sum()` at N=100k, though both are compiled C:

```
np.dot(a,b)    0.027 ms   one fused pass, no temporary
(a*b).sum()    0.221 ms   allocate 800KB temp, write it, read it back, reduce
sum(a*b)      59.565 ms   interpreter loop
```

Past a certain size, performance is about **how many times you cross memory**, not how
many multiplications you do. This is the whole premise of fused kernels and of
FlashAttention, which computes attention without ever materializing the N×N score matrix.

---

## Open threads

- Floating-point associativity — why the agreement errors are `1e-11` and not exactly `0`.
- Broadcasting rules, properly (met informally via scalar multiplication).
- Re-test element-wise vs dot at self-attention (`QKᵀ` is contraction, masking is element-wise).
