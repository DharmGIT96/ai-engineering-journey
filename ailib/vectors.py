"""
linalg_scratch.py  --  Phase 01, Day 01: Linear Algebra Intuition
Pure Python only. NO NumPy in this file. Correctness first; speed comes later.

RULES OF THE HOUSE
  1. Before you write a single line inside a method, fill in the SHAPE: line
     in its docstring. Vector or scalar. This is the habit from this morning's
     Q1 miss (you wrote [2*0, 0*5] -- element-wise -- when dot() was asked for).
     If you can name the output shape, you cannot make that mistake.
  2. Do not call NumPy. Loops and comprehensions only.
  3. Reuse your own methods. If you find yourself writing a loop that a method
     you already wrote does, stop and call that method instead.

HOW TO WORK
  Run it:  python linalg_scratch.py
  It will fail loudly and tell you the first thing that is wrong.
  Fix ONE method, re-run, repeat, until it prints:  all green ✓

  Do NOT try to write all eight at once. Work in the three groups marked below
  and check in with me at each ---- GATE ---- line.
"""

import math


class Vector:
    """A fixed-length sequence of floats naming a point in an invented space.

    A dict with the keys stripped: position in the array *is* the semantics.
    Slot i means the same thing in every Vector produced by the same model.
    """

    def __init__(self, components):
        # Stored as a tuple so a Vector cannot be mutated out from under you.
        self.c = tuple(float(x) for x in components)

    # --- plumbing: already written for you, do not change -------------------
    def __len__(self):
        return len(self.c)

    def __iter__(self):
        return iter(self.c)

    def __getitem__(self, i):
        return self.c[i]

    def __repr__(self):
        return f"Vector({list(self.c)})"

    def __eq__(self, other):
        return len(self) == len(other) and all(
            math.isclose(a, b, abs_tol=1e-9) for a, b in zip(self, other)
        )

    def _check(self, other):
        """Raise if dimensions disagree. Call this at the top of any 2-vector op."""
        if len(self) != len(other):
            raise ValueError(f"dim mismatch: {len(self)} vs {len(other)}")

    # =======================================================================
    # GROUP A -- element-wise ops.  Every one of these returns a VECTOR.
    # =======================================================================

    def __add__(self, other):
        """a + b, slot by slot.

        SHAPE: if one is [N,] and the other is [N,], return [N,].
        """
        self._check(other)
        return Vector([a + b for a, b in zip(self, other)])

    def __sub__(self, other):
        """a - b, slot by slot.

        SHAPE: same as for addition
        """
        self._check(other)
        return Vector([a - b for a, b in zip(self, other)])

    def __mul__(self, scalar):
        """Scale every component by a plain number. 3 * [1,2] -> [3,6].

        NOTE: this is scalar multiplication, NOT vector * vector.
        SHAPE: if one is scalar and the other is [N], return [N].
        """
        self._check(self)
        return Vector([a * scalar for a in self])

    def __rmul__(self, scalar):
        # Free: makes `3 * v` work as well as `v * 3`. Needs __mul__ to exist.
        return self.__mul__(scalar)

    def hadamard(self, other):
        """Element-wise vector*vector -- the thing you accidentally computed
        this morning. It is a real, useful op (gates in LSTMs, masking in
        attention). Implement it so the difference from dot() is in your hands,
        not just in your notes.

        SHAPE: if one is [N] and other is [N] then returns [N], where N is the length of the vectors.
        """
        return Vector(a * b for a,b in zip(self, other))

    # ---- GATE 1: run the file. When Group A passes, come back to me. -------

    # =======================================================================
    # GROUP B -- the reduction, and what is built on it.
    # =======================================================================

    def dot(self, other):
        """Agreement score / weighted vote / one neuron: sum(a_i * b_i).

        This REDUCES. It collapses n numbers into one.
        SHAPE: [N] and [N] → scalar
        """
        self._check(other)
        return sum(a * b for a, b in zip(self, other))

    def norm(self):
        """Length of the vector: ‖a‖ = sqrt(a · a).

        Read that formula again, then look at what you just wrote above.
        Do NOT write a second loop in here.
        SHAPE: if one is [N] then returns scaler number
        """
        return math.sqrt(self.dot(self))

    def unit(self):
        """The same direction, length 1. a / ‖a‖.

        Think about the input that breaks this. Handle it deliberately --
        raise, do not silently return garbage.
        SHAPE: if one is [N] returns [N]
        """
        n = self.norm()
        if n == 0:
            raise ValueError("cannot normalize the zero vector: it has no direction")
        return Vector(self * (1.0/n))

    # ---- GATE 2: run the file. When Group B passes, come back to me. -------

    # =======================================================================
    # GROUP C -- composition. Write these using ONLY the methods above.
    #            If a raw loop appears in this section, you took a wrong turn.
    # =======================================================================

    def cosine_similarity(self, other):
        """Direction-only agreement, length bias removed. Range [-1, 1].

        The fix for "long documents always win" from this morning's Block 3.
        SHAPE: [N] and [N] → scalar
        """
        n = self.norm()
        n1 = other.norm()
        return self.dot(other) / (n * n1)

    def project_onto(self, other):
        """The shadow self casts on other: the part of self that lies along
        other. This is the engine of Gram-Schmidt (tomorrow) and the reason
        "remove this direction from the embedding" is a thing people do.

        SHAPE: if [N] and [N] then -> [N]
        """
        n1 = self.dot(other)
        n2 = other.dot(other)
        if n2 == 0:
            raise ValueError("Cannot project onto a zero vector.")
        factor = n1 / n2
        return other * factor


# ===========================================================================
# SELF-TESTS -- do not edit. Your job is to make these pass.
# ===========================================================================

def _t(name, got, want, tol=1e-9):
    if isinstance(want, bool):
        ok = got is want
    elif isinstance(want, Vector):
        ok = isinstance(got, Vector) and got == want
    elif isinstance(want, (int, float)):
        ok = isinstance(got, (int, float)) and not isinstance(got, bool) \
             and math.isclose(got, want, abs_tol=tol)
    else:
        ok = got == want
    if not ok:
        kind = "scalar" if isinstance(want, (int, float)) else type(want).__name__
        raise AssertionError(
            f"\n  FAIL  {name}\n        got  {got!r}  ({type(got).__name__})"
            f"\n        want {want!r}  (expected a {kind})"
        )
    print(f"  pass  {name}")


def _raises(name, fn, exc=Exception):
    try:
        fn()
    except NotImplementedError:
        raise
    except exc:
        print(f"  pass  {name}")
        return
    raise AssertionError(f"\n  FAIL  {name}\n        expected it to raise, it did not")


def main():
    a = Vector([3, 4])
    b = Vector([1, 2])
    x = Vector([2, 0])
    y = Vector([0, 5])

    print("\nGROUP A -- element-wise (every result is a Vector)")
    _t("add",            a + b,              Vector([4, 6]))
    _t("sub",            a - b,              Vector([2, 2]))
    _t("mul (v*s)",      a * 2,              Vector([6, 8]))
    _t("mul (s*v)",      2 * a,              Vector([6, 8]))
    _t("hadamard",       x.hadamard(y),      Vector([0, 0]))
    _raises("dim mismatch raises", lambda: a + Vector([1, 2, 3]), ValueError)

    print("\nGROUP B -- reduction and length (dot and norm are SCALARS)")
    _t("dot",            a.dot(b),           11.0)
    _t("dot orthogonal", x.dot(y),           0.0)   # the Q1 vector, done right
    _t("norm",           a.norm(),           5.0)
    _t("norm via dot",   a.norm(),           math.sqrt(a.dot(a)))
    _t("unit direction", a.unit(),           Vector([0.6, 0.8]))
    _t("unit is len 1",  a.unit().norm(),    1.0)
    _raises("unit of zero vector raises", lambda: Vector([0, 0]).unit())

    print("\nGROUP C -- composition")
    _t("cos self",       a.cosine_similarity(a),          1.0)
    _t("cos opposite",   a.cosine_similarity(a * -1),    -1.0)
    _t("cos orthogonal", x.cosine_similarity(y),          0.0)
    # length-bias check: b scaled 100x is a different vector, same direction.
    _t("cos ignores length",
       a.cosine_similarity(b), a.cosine_similarity(b * 100))
    _t("dot does NOT ignore length",
       math.isclose(a.dot(b), a.dot(b * 100)), False)
    _t("project onto axis", a.project_onto(Vector([1, 0])), Vector([3, 0]))
    _t("project onto self", a.project_onto(a),              Vector([3, 4]))
    _t("project onto orthogonal", x.project_onto(y),        Vector([0, 0]))

    print("\nall green ✓\n")


if __name__ == "__main__":
    main()
