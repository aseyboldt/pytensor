# What the performance commits of this PR do, in plain terms

A plain-language companion to the PR description: one section per commit, with the idea and a small example.

**Background: what makes code fast.** A model's log-probability is a graph of operations: add these arrays, take a log, sum, multiply matrices. A backend turns that graph into machine code. numba does it one operation at a time with general-purpose code. The LLVM backend generates one specialised piece of code for the whole graph. Two things decide most of the speed:

- **Vectorization (SIMD).** Modern CPUs can do the same arithmetic on 4 numbers at once, if the code is simple and uniform enough. A loop the compiler vectorizes can be up to 4× faster. Things that block it: branches ("if this, do that"), calls into outside functions it can't see inside, and irregular memory access.
- **Memory.** Arithmetic is cheap; fetching data is not. The CPU keeps recently used data in small fast caches (on this machine, 512 KB per core and 4 MB shared). Code that reads memory in order, and doesn't keep needless copies, stays in the cache. Code that jumps around, or whose data doesn't fit, waits on main memory.

Almost every item below is one of those two ideas. The compile-time fixes from the same work are upstream: pymc-devs/pytensor#2679 and pymc-devs/pytensor#2680.

---

## A. Fewer operations handed back to numba

### 1. Mixed number types and true/false masks (`1feb00365`)
When the LLVM backend can't handle an operation, it hands just that one to numba, which costs time at every call. Two common cases are now handled directly:
- **Mixed-type matrix products:** a float32 matrix times a float64 one. Both are now converted to the output type first.
- **True/false masks:** `x[mask]` with a mask like `[True, False, True]` is turned into the positions it selects, `x[[0, 2]]`, which the backend handles natively.

---

## B. Fused loops (the core of the LLVM backend)

"Fusion" means combining many elementwise operations into one loop. Instead of computing `a = log(x)`, then `b = a * c`, then `sum(b)` as three passes, with two temporary arrays, it does one pass: `for each i: total += log(x[i]) * c[i]`.

### 2. Lookups from the same table (`c037260e6`)
Hierarchical models constantly look up effects by group index:

```python
mu = eff[group_a] + eff[group_b]
```

Before, each lookup became a separate step that built its own temporary array. Now both lookups happen inside the one fused loop, reading straight from `eff`. The same applies when one lookup result feeds several different calculations.

### 3. Several index arrays, of any shape (`6200d350d`)
This extends the same idea to loops that read through several index arrays at once, including 2-D index arrays. More of the graph fuses into fewer loops: mogp_lcm went from 64 separate steps to 48.

### 4. Bounds checks without branching (`16f5799c4`)
Every indexed read must check the index is valid (`0 ≤ i < n`). The obvious way is `if bad: raise error` inside the loop. That branch stops the compiler from vectorizing the loop. Now the loop just records "something was out of range" in a flag, and the error is raised after the loop:

```text
before:  for i: if idx[i] out of range → error;  y += x[idx[i]]
after:   for i: bad |= idx[i] out of range;       y += x[idx[i]]
         if bad → error
```

On a test loop: 1963 µs before, 660 µs after, against 1254 µs for numba.

### 5. Indexed writes in chunks (`de2ced579`)
Gradients often add values into a table by index: `grad[group] += values`. Writes to scattered positions can't be vectorized, which used to drag the computation of `values` down with them. Now the loop computes 128 values at a time into a small buffer, which vectorizes. Then a second short loop adds them into place. Two models: item_response_nba 1.7 → 1.04 ms, election88 1.31 → 0.24 ms.

### 6. One write loop per output (`bac40e130`)
This builds on 7. When a loop produces several outputs, each gets its own simple write loop. Outputs that write the same values share one buffer. Index work that's the same for a whole row is done once per row, not once per element.

### 7. One loop for all writes of a value (`6ff53f35c`)
In the SALK candidates model, one gradient gets added into 28 different factor tables. That used to be 28 loops, each recomputing the same values. Now it's one loop that computes each value once and writes it into all 28 places. Time spent on these writes: 113 ms → 4 ms.

---

## C. Runtime routines (hand-written Rust)

### 8. Copying in blocks (`993e58352`)
When copying an array where the last dimensions are laid out contiguously in both source and destination, copy that whole block with one `memcpy` instead of element by element.

### 9. Transposing in tiles (`44c73d165`)
Transposing a matrix reads rows and writes columns. Writing column by column jumps through memory, and for big matrices every write misses the cache. Now the copy goes 16×16 tile at a time, small enough that the reads and writes of each tile stay in the cache.

### 10. Cholesky and transposed products (`16c191f8f`)
- **Cholesky** (the matrix "square root" used for multivariate normals) now works in a layout where its inner loops walk memory in order.
- **`Aᵀ·x`** (a matrix-vector product with the transpose) got a vectorized routine that processes 4 columns at once.

### 11. Matrix-vector products for any row length (`b8d7bbdeb`)
For `A·x`, the routine keeps several partial results in CPU registers at once, the fastest storage there is. It now picks a layout by shape: 4 rows × 4 values at a time for short rows, 2 rows × 8 values for long ones. Short rows don't waste the registers, and long rows still get full speed.

---

## D. Constants and generated math

### 12. Constants keep their memory layout, and equal ones are stored once (`1f71dfaed`)
hsgp models contain a large constant matrix `Φ`, and the gradient also uses its transpose `Φᵀ`. The backend stored the constant in a fixed layout, so `Φᵀ` became a second full copy in memory. The two copies together no longer fit in the 4 MB cache, so every call fetched them from main memory.

Now a constant keeps its own layout, and two constants with identical bytes share one buffer. `Φᵀ` is just a different view of the same memory, half as much data, and it fits in the cache again. hsgp_basic_1d: 512 → 154 µs.

### 13. Making the compiler vectorize loops it declined (`483e7cbaf`)
LLVM estimates whether vectorizing a loop is worth it. For some fused loops, especially ones mixing float32, float64 and true/false arrays, its estimate said no, wrongly. Those loops now carry a marker saying "vectorize this". The frailty_coxph models became 1.3–1.5× faster. On the other ~100 catalogue models the generated code didn't change.

### 14. `gammaln` and `psi` written into the loop (`dbeb48d0b`)
`gammaln` (log of the gamma function) and `psi` (its derivative) appear in negative-binomial, beta and Dirichlet likelihoods. They used to be calls to precompiled Rust functions. The compiler can't see inside those, so any loop calling them ran one element at a time.

Now they're generated directly as plain arithmetic, written so there are no branches. Every case is computed and the right answer picked at the end (computing both sides is cheap when 4 lanes run at once anyway):

```text
gammaln(x) = pick(x < 0:  reflection formula,
                  else:   Lanczos approximation)
```

The loop vectorizes and they agree with scipy to within 2.3e-15. GLM_negative_binomial_regression went from 0.78× to 1.94× numba's speed.

### 15. Big matrix-vector products read from several places at once (`adc973766`)
For a matrix too big for the cache, `A·x` is limited by how fast memory can deliver `A`. Reading rows 0, 1, 2, 3 one after another is one stream of reads. Reading rows 0, ¼n, ½n and ¾n together is four separate streams, and the CPU's prefetcher fetches all four ahead in parallel:

```text
before:  row 0, row 1, row 2, row 3, …            (one stream)
after:   row 0, row n/4, row n/2, row 3n/4, …     (four streams)
```

The same trick applies to `Aᵀ·x` with columns. These products are now faster than single-threaded OpenBLAS at every shape tested.

### 16. Powers like `x**-2.5` as multiplications (`d8374a17c`)
`pow(x, c)` is a slow general-purpose function. When `c` is a constant that's a multiple of ½ (2, 3, 0.5, 1.5, −2.5, …), it can be built from multiplications, one square root and maybe a division:

```text
x**-2.5  =  1 / (x · x · sqrt(x))
```

The special values (0, −∞) are handled to match what `pow` gives. In malaria_hsgp, the loop with `(x + 3)**-2.5` went from 23.1 to 8.9 µs (numba: 9.1).

### 17. `erf`, `erfc` and `erfcx` vectorize (`a883b0f2b`)
These come up in censored and truncated normal distributions (normal CDFs).
- **`erf`, `erfc`:** glibc (the system's C library) has vectorized versions that do 4 values at once. The backend wasn't allowed to use them, with no reason recorded. Measured, they're as accurate as the one-at-a-time versions (`erfc` within 6.7e-16 of glibc's own scalar result), so now they're used.
- **`erfcx`:** this was a precompiled Rust call. It's now generated inline without branches, like `gammaln`.

censored_data_unimputed: 76 → 20 µs, now 3.16× numba's speed instead of 0.91×.

