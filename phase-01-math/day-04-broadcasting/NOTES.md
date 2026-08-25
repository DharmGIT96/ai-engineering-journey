# Day 04 — Broadcasting, Axes, and Silent Shape Bugs

Phase 01 · Math Foundations
Artifact: `broadcasting.py` — a shape-prediction quiz, three bug fixes, and pairwise distances

---

## Two rules. Everything below follows from them.

> **1. The axis you name is the one that DISAPPEARS.**
> **2. Shapes align from the RIGHT. Missing dims count as 1. A dim of size 1
>    stretches. Anything else is an error.**

## Rule 1 — `axis=` names the victim, not the survivor

```
X = [[ 1,  2,  3,  4],        shape (3, 4)
     [ 5,  6,  7,  8],
     [ 9, 10, 11, 12]]

X.mean(axis=0) -> [5. 6. 7. 8.]      shape (4,)   the 3 disappeared
                  (1+5+9)/3 = 5.0 ...             one number per COLUMN

X.mean(axis=1) -> [2.5 6.5 10.5]     shape (3,)   the 4 disappeared
                  (1+2+3+4)/4 = 2.5 ...           one number per ROW
```

The mistake to avoid: picking the axis by pointing at the number you want to **keep**.
`axis=` is `del`, not `select`. You name what dies.

```
X.shape = (32, 768)      32 sentences, 768 dims

want 768 numbers out (one per dimension) -> kill the 32  -> axis=0 -> (768,)
want  32 numbers out (one per sentence)  -> kill the 768 -> axis=1 -> (32,)
```

**Same array, opposite axis, because the question changed.** The axis is not a property of
the array — it's a property of what you're asking for.

`keepdims=True` keeps the collapsed axis at size 1 instead of deleting it:

```
(32, 768).mean(axis=1)                 -> (32,)
(32, 768).mean(axis=1, keepdims=True)  -> (32, 1)
(2, 3, 4).mean(axis=None)              -> ()        0-d. no axes at all.
```

`()` is not `(1,)`. A 0-d array is a scalar with nothing to index; a `(1,)` array has an
axis and will broadcast against your data.

## Rule 2 — align from the right

```
         A           B  ->  result
      (3,)        (3,)  ->  (3,)            identical, nothing happens
    (3, 1)      (1, 4)  ->  (3, 4)          BOTH stretch
      (5,)      (5, 1)  ->  (5, 5)          5 elements + 5 elements = 25
 (2, 3, 4)        (4,)  ->  (2, 3, 4)       B padded to (1,1,4)
 (2, 3, 4)      (3, 1)  ->  (2, 3, 4)       B padded to (1,3,1)
(10, 1, 8)   (1, 6, 8)  ->  (10, 6, 8)
    (7, 1)        (7,)  ->  (7, 7)          <-- looks like it should match. it doesn't.
      (3,)        (4,)  ->  ERROR           3 vs 4, neither is 1
 (32, 768)        (32,) ->  ERROR           768 vs 32, neither is 1
 (32, 768)      (32, 1) ->  (32, 768)       silent, and probably wrong
```

Worked by hand, the dangerous one:

```
  (100,)    ->  pad left  ->  (  1, 100)
  (100, 1)                    (100,   1)
                              ────────────
  right column: 100 vs 1  -> the 1 stretches -> 100
  left  column:   1 vs 100 -> the 1 stretches -> 100
                              ->  (100, 100)
```

**Padding happens on the left, which is why a trailing `1` is dangerous** — it lines up
against your real data dimension instead of disappearing.

## The bug you will write

100 predictions from a model, shape `(100,)`. 100 labels from a dataframe column, shape
`(100, 1)`.

```
(pred - labels).shape = (100, 100)      you wanted (100,)

mse         = 0.018564      averaged over 10,000 numbers, not 100
correct mse = 0.017544
```

No error. No warning. Every prediction was subtracted from every label, including 9,900
pairs that have nothing to do with each other. The loss is 6% off — small enough to look
plausible, large enough to poison training.

Note the pattern across the whole topic: **a trailing size-1 axis turns a crash into a
wrong answer.**

```
(100,)   vs (100, 1)  ->  (100, 100)   silently 100x too much work
(32,768) vs (32, 1)   ->  (32, 768)    silently the wrong subtraction
(32,768) vs (32,)     ->  ERROR        loud, harmless
```

The loud failure is the lucky one. `keepdims=True` is exactly what converts case 3 into
case 2 — so only reach for it when you can say out loud why you need the axis back.

### The fixes

```
x.ravel()      (3,1) -> (3,)     flatten to 1-D
x.squeeze()    (3,1) -> (3,)     drop EVERY size-1 axis
x[:, 0]        (3,1) -> (3,)     take the column explicitly
y[:, None]     (3,)  -> (3,1)    add an axis -- the reverse
```

`squeeze()` is convenient and dangerous: it drops *all* size-1 axes, so a batch of size 1
loses its batch dimension and the code breaks only in production, only on the last batch.
Prefer `ravel()` or explicit indexing when you know what you want.

## Broadcasting does not copy — it lies about strides

```
a.shape (5,)          a.strides (8,)     a.nbytes 40
b.shape (1000000, 5)  b.strides (0, 8)
b claims 40,000,000 bytes  ·  actual memory used: 40 bytes
```

A **stride** is how many bytes to step to reach the next element along an axis.
Broadcasting sets it to **0** — "don't move the pointer." Walking a million rows, the CPU
re-reads the same 40 bytes. Nothing is duplicated.

Same idea as `(x@B)@A` in LoRA never materializing the big matrix: **the cheapest data is
data you never write down.**

But it's lazy only until you do arithmetic:

```
intended result (20000,)           :     0.16 MB
what you actually get (20000,20000):     3.20 GB
```

**A shape bug is also a memory bug.** The classic symptom is a CUDA OOM on a line that
looks like it cannot allocate anything — a subtraction. Check your shapes before your
batch size.

## Broadcasting used on purpose: pairwise distances

```python
A[:, None, :]      # (N, D) -> (N, 1, D)
B[None, :, :]      # (M, D) -> (1, M, D)
((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=-1)   # -> (N, M)
```

Every pair of rows, no Python loop. This is the brute-force path in every vector
database, and the core of k-NN and k-means.

It is also a memory trap:

```
N=1,000  M=1,000  D=768:  intermediate (N,M,D) =    6.1 GB   result = 0.008 GB
N=10,000 M=10,000 D=768:  intermediate (N,M,D) =  614.4 GB   result = 0.800 GB
```

614 GB of intermediate to produce 0.8 GB of answers. The identity real libraries use:

```
||a - b||²  ==  ||a||² - 2·a·b + ||b||²
```

Every term is either a matmul `(N,D)@(D,M) -> (N,M)` or a per-row norm. The `(N,M,D)`
array is never built; memory drops by a factor of D. **It is also numerically worse** —
it subtracts large nearly-equal quantities, which is the catastrophic cancellation from
Day 03. That trade (D× less memory for a few lost digits) is made deliberately by every
vector search library.

---

## Bugs found in review

| Bug | Why it mattered |
|---|---|
| `mse` computed `(mean of errors)²` instead of `mean of (errors²)` | **101× too small**, and it passed the test suite. |
| Prediction quiz left unanswered | Predicting shapes on paper is the actual skill; writing the function is the easy half. |

### The `mse` bug is worth its own section

```python
((pred - labels).mean()) ** 2        # wrong: square of the mean
((pred - labels) ** 2).mean()        # right: mean of the squares
```

Why the order matters, in two numbers:

```
pred = [0, 2]    labels = [1, 1]    errors = [-1, +1]

square-of-mean = 0.0      <-- "perfect model"
mean-of-squares = 1.0
```

The `+1` and `−1` cancel *before* squaring. **A model wrong on every sample scores zero
loss.** Squaring first is what makes errors unable to cancel. In a real training loop the
gradient would be near zero, the model would never learn, and the loss curve would look
excellent.

### And the test suite did not catch it

The harness checked that four different input shapes agreed *with each other*, and never
checked any of them against a known-correct value. So a function that was
shape-independently wrong sailed through.

This is the third appearance of the same theme:

- **Day 02** — classical Gram-Schmidt returned perfectly unit-length, perfectly wrong
  vectors, and passed every structural check.
- **Day 03** — naive summation was *exact* on one case, by luck, and broke the moment the
  input magnitude crossed a threshold.
- **Day 04** — `mse` agreed with itself across four shapes while being 101× off.

> **A test that only checks self-consistency proves nothing about correctness.**
> Ask of your own tests what you would ask of someone else's: *what would this fail to
> catch?*
