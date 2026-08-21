# Day 03 — Floating Point: why your zeros aren't zero

Phase 01 · Math Foundations
Artifact: `summation.py` — naive, pairwise, and Kahan summation, benchmarked for accuracy

---

## The fact everything else follows from

```
a, b, c = 1e16, -1e16, 1.0

(a + b) + c  =  1.0
a + (b + c)  =  0.0
```

Same three numbers, same operator, different grouping, different answer. **Floating-point
addition is not associative.**

## Why

A `float64` holds about 16 **significant digits** — not 16 decimal places. So the gap
between neighbouring representable values grows with magnitude:

```
near 1e+00    next float is 2.220e-16 away
near 1e+03    next float is 1.137e-13 away
near 1e+09    next float is 1.192e-07 away
near 1e+16    next float is 2.000e+00 away     ← no integers live here
```

Past 2⁵³ = `9,007,199,254,740,992`, consecutive integers stop existing:

```
9007199254740992.0 + 1 == 9007199254740992.0   →  True
```

That's the mechanism above. `b + c` is `-1e16 + 1`; the `1` is smaller than the gap
between floats at that magnitude, so it cannot be represented and simply vanishes. Then
`a + (-1e16) = 0`. Group the other way and the giants cancel first, leaving `1`.

> **Adding a small number to a large one loses the small one.**

Like a bank balance kept to two decimals: add ₹0.001 to ₹10,000,000 and nothing happens.
Floats do this at every scale; only the threshold moves.

## What it does to a real sum

One large value plus a million tiny ones. `math.fsum` is exactly rounded, so it's truth:

```
method                     |                 result |    abs error
python loop (naive)        |     100001000.00202656 | 2.026558e-03
builtin sum()              |     100001000.00202656 | 2.026558e-03
np.sum  (pairwise)         |     100001000.00000003 | 2.980232e-08
math.fsum (exact)          |            100001000.0 | 0.000000e+00
```

The naive loop is off by `0.002`: every `1e-3` added to a total near `1e8` is partly
rounded away, a million times, all in the same direction.

**`np.sum` is 68,000× more accurate — and faster.** That should be suspicious, because
faster-*and*-more-accurate isn't normally on the menu.

The reason is **pairwise summation**: split the array in half, sum each half recursively,
add the two results. Every addition then happens between partial sums of *similar
magnitude*, instead of one accumulator repeatedly swallowing values far below its own
scale. Error grows like `log N` instead of `N`, at **the same number of additions**.

Sorting ascending first also helps, for the same reason:

```
sorted ascending           |     100000999.99999999 | 1.490116e-08
```

## Precision matters more than you'd guess

Averaging 10,000 loss values near `0.5`:

```
n=10,000, float32
  exact mean            0.5006311887599528
  naive loop            0.5006322265625      err 1.038e-06
  numpy .sum()          0.500631201171875    err 1.241e-08
  upcast to f64 first   0.5006311887599528   err 0.000e+00

n=10,000,000, float32          ← 1000x more values
  naive loop            err 2.033e-05        ← 20x worse
  numpy .sum()          err 2.265e-08        ← essentially unchanged

n=10,000, float16
  exact mean            0.49850963134765625
  naive loop            0.2048               ← not slightly wrong. wrong.
  numpy .sum()          0.4984
```

`0.2048` means the sum froze at exactly **2048**. In `float16` the gap between
neighbouring values at 2048 is `2.0`, so adding `0.5` rounds straight back to `2048`.
**The accumulator stopped accepting input halfway through and silently discarded 8,000
values.** This is called **stagnation**.

### The one-line fix

```python
losses.astype(np.float64).sum()      # err 0.000e+00
```

> **Accumulate in higher precision than you store.**

This is what the hardware does: NVIDIA tensor cores multiply in `fp16` and accumulate in
`fp32` by design; `torch.mean()` on a half tensor upcasts internally; autocast keeps
reductions in `fp32` while matmuls run in `fp16`. **Storage precision and accumulation
precision are separate decisions.** Cheap storage is fine. Cheap accumulation gives you
`0.2048`.

### Related: why mixed precision needs loss scaling

`float16`'s smallest normal value is about `6e-5`. Gradients are routinely smaller, so
they round to **zero** and the weight never updates. The fix is to multiply the loss by a
large constant before backprop — pushing gradients into representable range — then divide
it out before the optimizer step.

## Three summation algorithms

```
naive     1 flop/element   total += v
pairwise  1 flop/element   same additions, different tree
kahan     4 flops/element  y=v-c; t=total+y; c=(t-total)-y; total=t
```

**Kahan** keeps the rounding error in a compensation variable and adds it back next
iteration. `(t - total)` is how far the total really moved; `y` is how far it was asked
to move; the difference is what was lost.

> Do **not** simplify the algebra. In exact arithmetic `c` is always zero and the whole
> thing collapses to the naive loop — it works *because* floating point is inexact, so
> the parenthesisation is load-bearing. This is why `-ffast-math` is dangerous: the
> compiler "optimizes" the correction away and hands you back the naive sum.

### Measured

```
1e8 plus a million 1e-3   (n=1,000,001)
  naive        |   2.0266e-03 |  10.7 digits correct
  pairwise     |   1.1921e-07 |  14.9
  kahan        |   0.0000e+00 |  exact

a million losses near 0.5  (n=1,000,000)
  naive        |   3.1374e-08 |  13.2
  pairwise     |   5.8208e-11 |  15.9
  kahan        |   0.0000e+00 |  exact

wildly mixed magnitudes    (n=200,000)
  naive        |   7.3438e-01 |  13.9
  pairwise     |   7.8125e-03 |  15.8
  kahan        |   9.3750e-02 |  14.8      ← Kahan LOSES to pairwise here

alternating +1 / -1        (n=300,000)
  naive        |   0.0000e+00 |  exact     ← naive is exact, and fastest
```

## Four things the table teaches

**1. "Worst" depends on which error you measure.**

```
case                     |   abs error |   sum size |   relative
1e8 + a million 1e-3     |   2.027e-03 |   1.00e+08 |   2.03e-11
losses near 0.5          |   3.137e-08 |   5.00e+05 |   6.27e-14
mixed magnitudes         |   7.344e-01 |   4.70e+11 |   1.56e-12
```

By absolute error, "mixed magnitudes" is worst. By **relative** error, the `1e8` case is.
Relative is almost always the one that matters.

**2. Naive is fine when all values are of similar magnitude to the running total.**
Losses near `0.5` gave naive 13.2 correct digits — nothing falls off the end.

**3. Passing because of your input's magnitude is not passing.** Naive was *exact* on the
alternating case. Is that robust? Vary the large value:

```
big=1e+14   exact=100,000   naive=100,000   OK
big=1e+15   exact=100,000   naive=100,000   OK
big=1e+16   exact=100,000   naive=      0   BROKEN
big=1e+17   exact=100,000   naive=      0   BROKEN
```

At `1e15` the float gap is `0.125`, so `1e15 + 1` is representable. At `1e16` the gap
becomes `2.0`, the `+1` vanishes, the `-1e16` cancels back to zero, and the answer is
**0 instead of 100,000**. Not degraded — gone.

**4. Why NumPy uses pairwise and not Kahan.** Kahan is **serially dependent**: `c` from
iteration *i* is required before *i+1* can begin, so the loop cannot be split across SIMD
lanes or threads. Pairwise has no such chain — the two halves are independent, so it
vectorizes and parallelizes freely.

```
np.sum (pairwise, SIMD)     0.320 ms   err 0.000e+00
kahan  (python loop)       47.995 ms   err 0.000e+00

150x slower, for zero additional accuracy on this input
```

Pairwise buys 15–16 correct digits at naive's flop count *and* vectorizes. Kahan buys the
17th digit at 4× the flops with no vectorization. **NumPy chose the option that's free.**

---

## The technique worth keeping

> **To find out whether float error matters: recompute in higher precision and subtract.**

`math.fsum` is exactly rounded. `float64` has 16 digits where `float32` has 7. Run your
computation in the precision you care about, run it again in one you trust, take the
difference. That difference *is* your error. No intuition required — every error column
on this page was produced that way.

## Where this shows up again

- **Gram-Schmidt's error floor.** Day 02's modified variant bottomed out at `2.975e-10`,
  not `0.0`. Every dot product is a sum; every sum accumulates rounding. Good algorithms
  don't eliminate error, they stop *amplifying* it.
- **GPU training is not reproducible.** Thousands of threads produce partial sums in
  whatever order they finish. Different order → different rounding → different loss in
  the last digits → divergent runs from the same seed. That's what
  `torch.use_deterministic_algorithms(True)` buys, and why it costs speed.
