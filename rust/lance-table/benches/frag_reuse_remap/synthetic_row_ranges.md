# Synthetic row ranges

A payload generator for benchmarking the two fragment-reuse remap forms against
each other. This document describes the idea from scratch; no familiarity with
the fragment reuse index is assumed.

- [What is being compared](#what-is-being-compared-and-why-it-needs-a-generator)
- [The idea](#the-idea)
- [Constants and knobs](#constants-and-knobs)
- [The algorithm](#the-algorithm)
- [Worked example: N=8192, H=4](#worked-example-n8192-h4)
- [Inside the structures](#what-this-looks-like-inside-the-structures)
- [Deletions](#deletions)
- [The oracle](#the-oracle)
- [Why the payloads are valid](#why-the-generated-payloads-are-valid)
- [The M region](#what-the-m-region-is-for)
- [Not yet modelled](#not-yet-modelled)

## What is being compared, and why it needs a generator

When compaction rewrites rows into new fragments, indices that store physical
row addresses go stale. Rather than rewrite every index immediately, Lance
records the old-address to new-address mapping in the **fragment reuse index**
(FRI) and translates addresses as index state is read. `FragReuseIndex` holds one
remap per compaction round, oldest first, in `row_addr_maps`; a lookup walks the
whole chain, feeding each version's output into the next.

Each link is a `RowAddrRemap`, which comes in two forms:

| form | representation | memory grows with |
|---|---|---|
| `Direct` | `HashMap<u64, Option<u64>>`, one entry per rewritten or deleted row | **rows** compaction has touched |
| `Compact` | per-fragment bitmaps of rewritten offsets, plus new-fragment row ranges | **fragments** |

`Compact` trades memory for work per lookup: it must `rank()` into a roaring
bitmap and binary-search a range list, where `Direct` does a single hash probe.
Build cost and footprint are already well characterised — `Direct` is minutes and
tens of gigabytes on a large production payload where `Compact` is sub-second and
under a gigabyte. **Lookup cost is the unmeasured axis, and it is what this
benchmark exists to measure.**

That needs payloads that can be dialled along several cost axes at once:

- **rowcount** — what `Direct` scales with
- **fragment count** — what `Compact` scales with
- **chain depth** — how many versions a lookup traverses
- **bitmap shape** — which roaring container type `rank()` runs against

and it needs them without the benchmark harness itself doing a lookup to decide
where a probe *should* land. An oracle backed by a `HashMap` would compete with
the structure under test for cache and cycles, and would muddy every result. So
the generator is built so that the correct answer for any probe is **closed-form
arithmetic over a handful of scalars** — no arrays, no maps, no reference to the
payload being measured.

## The idea

Take a synthetic row range `0..N` and cut it into fragments **geometrically**:
the first half of what remains becomes `H` equal fragments, the second half
recurses, stopping when a fragment would fall below the floor size `FLOOR_ROWS`.

Then record compactions from the small end upwards, `k` fragments per round,
until everything is one fragment. Each round is one FRI version, and each round
deletes a fraction of the rows it touches.

This models how compaction actually behaves in practice: small fragments get
merged into bigger ones, which later get merged into yet bigger ones, leaving one
large settled fragment holding most of the rows and a tail of small recent ones.
Because fragment size decays geometrically, most rows end up in fragments that
were rewritten only once or twice, while a small number of rows in the tail carry
the deepest chains. That skew is the point — it is what production looks like.

## Constants and knobs

Three quantities are **fixed**, not swept:

```rust
/// Floor fragment size, in rows. The geometric halving stops here.
const FLOOR_ROWS: u32 = 128;

/// Deletion period, in rows. Within each period the first `PERIOD - b` rows are
/// kept and the last `b` deleted.
const DELETION_PERIOD: u32 = 128;

/// Fragments per level. Splits each level horizontally without changing the row
/// weight of any level, so it moves fragment count without moving hop counts.
const FRAGS_PER_LEVEL: u32 = 4;
```

`FLOOR_ROWS` and `DELETION_PERIOD` are deliberately **equal**, which is what makes
the deletion model exact rather than approximate — see [Deletions](#deletions).
Every fragment size in the layout is a power of two at least `FLOOR_ROWS`, so with
`FLOOR_ROWS == DELETION_PERIOD` every fragment is an exact multiple of the period.
Consequences:

- every fragment loses exactly `size / 128 * b` rows, so the nominal rate holds at
  every level with no dependence on alignment
- `b < FLOOR_ROWS` for any `b` up to 127, so a deletion cluster can never swallow a
  whole fragment

`FRAGS_PER_LEVEL = 4` because it must be a power of two for `fsize = lsize/H` to
stay an integer multiple of 128; because `H = 1` would make `rows_before` always
zero and so never exercise the telescoping accumulation at all; and because it puts
fragment count in the range a real compacted dataset occupies (72 fragments at
`N = 2^26`, against roughly 67 at Lance's default `target_rows_per_fragment`). It
also gives each rewrite group 5 old fragments, so positional accumulation is
non-trivial.

The remaining knobs:

| knob | controls | independent of |
|---|---|---|
| `N` | rowcount — `Direct`'s cost axis | — |
| `k` | fragments consumed per compaction round, so chain depth | the layout |
| `b` | deleted rows per period, so the per-round deletion rate `b/128` | the bitmap run count |
| `M` | extends the probe space past `N`; rows in `N..M` are in no group at all | everything |

Derived quantities, with `H` and `FLOOR_ROWS` inlined:

```
levels          = log2(N / 512) + 1
fragment count  = 4 * levels
depth (k = H)   = levels

      N        levels   fragments   Direct entries      (k=H / k=1)
  2^13  (8k)       4*       20        16k  / 51k
  2^19 (524k)     10*       44         1M  / 3.3M
  2^26  (67M)     17*       72       134M  / 436M

  * geometric levels; total levels = depth(k=H) = geometric + 1 (the floor).
    Entry counts converge on 2N at k=H and 6.5N at k=1.
```

`k` exists because with `FLOOR_ROWS` and `H` both fixed, `k = H` pins depth to
`log2(N/512) + 1` — 18 at `N = 2^26`, below the 20–30 range seen in production, and
worse, *coupled to `N`*, so an `N` sweep could not separate table size from chain depth.
`k = 1` merges one fragment per round instead of a whole level, giving depth `F - 1`
(71 at `N = 2^26`) over the same rows.

**The `k = 1` consumption rule is load-bearing and verified:** consume initial fragments
in **descending id** order, with round 1 taking **two** of them and every later round
taking one plus the blob. Descending order is what keeps the blob holding both the
highest id and the highest rows at every round, so `old_frag_ids` stays ascending and
original row order still survives the cascade. Ascending order would put the blob last
while it held *lower* rows than the fresh fragment, destroying the survivor invariant and
invalidating the oracle. Round 1 taking two is what makes depth `F - 1` rather than `F`
with a degenerate one-fragment rename.

Verified at `N ∈ {2^12, 2^13, 2^14}` and `b ∈ {0, 3, 12}`: blob-last-and-highest,
ascending ids, the survivor invariant, and payload validity all hold.

Note `Direct`'s entry count is **not** `2N` at `k = 1`. Average hops rise from 2 to
`Σ 2^-(j+1)(4j + 2.5) = 6.5`, so it converges on `6.5N` — measured 5.97N / 6.23N / 6.37N
at `2^12` / `2^13` / `2^14`, which is why `Direct` is skipped at large `N` when `k = 1`.

## The algorithm

### Step 1 — layout

Cut `0..N` into fragments. Level `L` covers the first half of what remains after
`L` halvings, split into `H` equal pieces:

```
lstart(L) = N - (N >> L)          first row of level L
lsize(L)  = N >> (L + 1)          rows in level L
fsize(L)  = lsize(L) / H          rows per fragment in level L
```

Fragment `H*L + i` (for `i` in `0..H`) covers
`[lstart(L) + i*fsize(L), lstart(L) + (i+1)*fsize(L))`.

Halving stops at the first `L` where `fsize(L) < FLOOR_ROWS`; call it `Lmax`. The
remaining range `[lstart(Lmax), N)` becomes the **floor level**, cut into
`FLOOR_ROWS`-row pieces. Fragment ids continue from `H*Lmax`.

### Step 2 — compaction cascade

Compact from the small end upwards, one level per round when `k = H`:

**`k = H`** — one level per round:
- round 1 merges the floor level's fragments
- round `r >= 2` merges level `Lmax - r + 1`'s fragments **plus the blob** produced
  by round `r - 1`

**`k = 1`** — one fragment per round:
- consume initial fragments in **descending id** order
- round 1 takes **two**; every later round takes one plus the blob

In general `depth = ceil((F - 1) / k)`, which gives `levels` at `k = H` and `F - 1` at
`k = 1`. See [Constants and knobs](#constants-and-knobs) for why descending order and the
two-fragment first round are load-bearing.

Each round produces exactly one new fragment, whose id is `fid_max + r - 1` where
`fid_max` is one past the highest initial fragment id.

The geometric sizing makes this telescope: the sum of all fragments smaller than
`Fj` is always just under `Fj` itself, so the accumulated blob merges with the
next-larger level at every round. And because the blob always holds the highest
ids *and* the highest rows, listing a round's `old_frag_ids` in ascending id order
puts it last — so **original row order is preserved through the whole cascade.**

### Step 3 — deletions

Each round deletes rows from every fragment it touches, using a block-periodic
pattern over the **offset within that fragment**: within each period of 128 offsets,
the first `128 - b` are kept and the last `b` are deleted.

```
f(M)             = (M / 128) * b + max(0, (M % 128) - (128 - b))   deleted rows below offset M
is_deleted(o)    = (o % 128) >= (128 - b)
survivors(a, n)  = n - (f(a + n) - f(a))                           survivors in [a, a+n)
```

Deleted rows are simply not in the round's rewritten set, so `Compact` reports
them deleted and the chain terminates there.

**Offset-phased, and why that is not a limitation.** Every fragment's start row is a
multiple of 128: `lstart(L) = N - (N >> L)` is a multiple of 512 for every reachable
`L`, and `fsize(L)` is a power of two at least `FLOOR_ROWS`. So for a fresh fragment,
offset phase and global-row phase are **identical** — verified to the last digit at
every `N` and `b`. It is the 128-alignment that is load-bearing here, not the choice of
domain.

Stating it in offset terms is what makes the rule total. The accumulated blob's rows are
survivors of a non-contiguous set of original rows, so "global row index" has no meaning
for it; its deletion test is applied to its own offset, phase 0. Reading the rule as
"test the row's original global index" instead would be **idempotent** — a row that
survived round 1 has the same original index in round 2 and so survives again — which
collapses the aggregate rate to exactly `b/128`, leaves the blob's bitmap a single full
run in every round, and destroys the entire bitmap-shape axis the `b` knob exists to
drive. Do not implement it that way.

Because `/128` and `%128` are a shift and a mask, `f` is a handful of instructions.

### Step 4 — positional pairing

For a probe at `(fragment, offset)` in a round that covers that fragment:

```
a           = fragment's global start row
position    = survivors(a, offset)                    rank within the fragment
rows_before = survivors(lstart, i * fsize)            survivors in preceding fragments of the level
k           = rows_before + position
new address = (new_fragment_id, k)
```

`rows_before` for the blob is `survivors(lstart, lsize)` — the whole level, since
the blob sorts after all of it.

## Worked example: N=8192, H=4

Shown with `b = 0` so the layout is legible; deletions are layered on in
[Deletions](#deletions). Five levels, four fragments each, twenty fragments. Bar
width is proportional to fragment size (one block = 128 rows):

```
frag   rows           size
──────────────────────────────────────
f0        0..1023   ████████  1024   ┐
f1     1024..2047   ████████  1024   │ L0
f2     2048..3071   ████████  1024   │ [0,4096)
f3     3072..4095   ████████  1024   ┘ 4 x 1024
f4     4096..4607   ████       512   ┐
f5     4608..5119   ████       512   │ L1
f6     5120..5631   ████       512   │ [4096,6144)
f7     5632..6143   ████       512   ┘ 4 x 512
f8     6144..6399   ██         256   ┐
f9     6400..6655   ██         256   │ L2
f10    6656..6911   ██         256   │ [6144,7168)
f11    6912..7167   ██         256   ┘ 4 x 256
f12    7168..7295   █          128   ┐
f13    7296..7423   █          128   │ L3
f14    7424..7551   █          128   │ [7168,7680)
f15    7552..7679   █          128   ┘ 4 x 128  <- hits FLOOR_ROWS
f16    7680..7807   █          128   ┐
f17    7808..7935   █          128   │ floor
f18    7936..8063   █          128   │ [7680,8192)
f19    8064..8191   █          128   ┘ 4 x 128
```

The next level would have split `[7680,7936)` into four 64-row fragments, below
`FLOOR_ROWS`, so halving stops and the remaining `[7680,8192)` becomes the floor
level.

### Compaction schedule, k = H = 4

```
v1:  f16 f17 f18 f19          ->  f20    rows 7680..8191   (512 rows)
v2:  f12 f13 f14 f15  + f20   ->  f21    rows 7168..8191   (1024 rows)
v3:  f8  f9  f10 f11  + f21   ->  f22    rows 6144..8191   (2048 rows)
v4:  f4  f5  f6  f7   + f22   ->  f23    rows 4096..8191   (4096 rows)
v5:  f0  f1  f2  f3   + f23   ->  f24    rows    0..8191   (8192 rows)
```

### Where each row band lives

```
row band     │ init      │ v1 │ v2 │ v3 │ v4 │ v5 │ hops
─────────────┼───────────┼────┼────┼────┼────┼────┼──────
   0..4095   │ f0..f3    │ ·  │ ·  │ ·  │ ·  │f24 │  1
4096..6143   │ f4..f7    │ ·  │ ·  │ ·  │f23 │f24 │  2
6144..7167   │ f8..f11   │ ·  │ ·  │f22 │f23 │f24 │  3
7168..7679   │ f12..f15  │ ·  │f21 │f22 │f23 │f24 │  4
7680..8191   │ f16..f19  │f20 │f21 │f22 │f23 │f24 │  5
─────────────┼───────────┼────┼────┼────┼────┼────┼──────
8192..M      │ --        │ ·  │ ·  │ ·  │ ·  │ ·  │  0
```

`·` means the address is unchanged by that version. For `Compact` that is the
cheap early-out — `frag_to_group.get(&frag)` misses and returns immediately,
before any bitmap or range work.

**Every probe walks all 5 versions**; `remap_row_id` has no early exit. `hops` is
how many of those versions do real work; `depth - hops` are cheap misses.

Size-weighted average: `(4096·1 + 2048·2 + 1024·3 + 512·4 + 512·5) / 8192` =
**1.9375 hops**. Half the rows hop once. In the limit the geometric layout converges
on 2 hops per uniform probe regardless of depth, which is why deep chains are
cheap on average and expensive only for the tail.

## What this looks like inside the structures

Taking version `v2` — old fragments `f12 f13 f14 f15` plus the blob `f20`,
rewritten into `f21`, with `b = 0`.

### Compact

Positions are assigned by walking `old_frag_ids` in order and accumulating row
counts, so each old fragment records how many rewritten rows preceded it:

```
RowAddrRemap::Compact(CompactRowAddrRemap {
    frag_to_group: { 12 -> 0, 13 -> 0, 14 -> 0, 15 -> 0, 20 -> 0 },
    groups: [
        GroupRemap {
            frags: {
                12 -> (bitmap{0..127},   old_rows_before:   0),
                13 -> (bitmap{0..127},   old_rows_before: 128),
                14 -> (bitmap{0..127},   old_rows_before: 256),
                15 -> (bitmap{0..127},   old_rows_before: 384),
                20 -> (bitmap{0..511},   old_rows_before: 512),
            },
            new_frag_row_ranges: [ (frag 21, rows_before 0, physical_rows 1024) ],
        },
    ],
})
```

5 map entries, 5 bitmaps, 1 range — regardless of how many rows those fragments
hold.

A lookup of address `14:2` (fragment 14, offset 2):

```
frag_to_group[14]                -> group 0
frags[14]                        -> (bitmap{0..127}, before = 256)
bitmap.rank(2)                   -> 3        bits {0,1,2} are <= 2
position within fragment         =  3 - 1 = 2
k = before + position            =  256 + 2 = 258
range with 0 <= 258 < 0+1024     -> frag 21
new address                      =  21:258
```

Cross-check against the layout: `f14` starts at row 7424, so `14:2` is row 7426;
`f21` covers `[7168, 8192)`, so offset `7426 - 7168 = 258`.

### Direct

The same version as a materialized map — **1024 entries**, one per rewritten row:

```
12:0   -> 21:0        13:0   -> 21:128     14:0   -> 21:256     15:0   -> 21:384
12:1   -> 21:1        13:1   -> 21:129     14:1   -> 21:257     15:1   -> 21:385
12:2   -> 21:2        13:2   -> 21:130     14:2   -> 21:258     15:2   -> 21:386
  ...                   ...                  ...                  ...
12:127 -> 21:127      13:127 -> 21:255     14:127 -> 21:383     15:127 -> 21:511

20:0   -> 21:512      20:128 -> 21:640     20:256 -> 21:768     20:384 -> 21:896
  ...                   ...                  ...                  ...
20:127 -> 21:639      20:255 -> 21:767     20:383 -> 21:895     20:511 -> 21:1023
```

`14:2 -> 21:258`, agreeing with the compact lookup above. One hash probe, no
arithmetic — but every mapping is stored.

### The whole chain, and its cost

For the full N=8192, `b = 0` example:

```
version │ rewritten rows │ Direct entries │ Compact: map/bitmaps/ranges
────────┼────────────────┼────────────────┼────────────────────────────
   v1   │       512      │       512      │      4 /  4 / 1
   v2   │      1024      │      1024      │      5 /  5 / 1
   v3   │      2048      │      2048      │      5 /  5 / 1
   v4   │      4096      │      4096      │      5 /  5 / 1
   v5   │      8192      │      8192      │      5 /  5 / 1
────────┼────────────────┼────────────────┼────────────────────────────
 total  │     15872      │     15872      │     24 / 24 / 5
```

Note the identity: total `Direct` entries equals the sum of `hops` over all rows,
i.e. `avg_hops * N`. **At `k = H`** a geometric layout converges on `2N`, so `Direct`'s
whole-chain footprint is about `2N` entries at that `k`. This is *not* `k`-independent:
at `k = 1` average hops rise to 6.5 and the footprint with them. Meanwhile
`Compact`'s is about `fragment_count + depth`. That is the trade the benchmark is
measuring, and both sides are computable in advance from the knobs, which makes
the memory arm predictable rather than exploratory. It also caps the sweep:
`Direct` stops being constructible somewhere around `N = 2^26`–`2^27` (about `2N`
entries at 40–48 bytes each, so 5–13 GB).

### A full chain walk

Row 7681 lives in `f16` at offset 1, and after all five rounds must land at offset
7681 of the final fragment:

```
start           16:1
  v1  position    1,  before    0  ->  k =    1   ->  20:1
  v2  position    1,  before  512  ->  k =  513   ->  21:513
  v3  position  513,  before 1024  ->  k = 1537   ->  22:1537
  v4  position 1537,  before 2048  ->  k = 3585   ->  23:3585
  v5  position 3585,  before 4096  ->  k = 7681   ->  24:7681
```

`24:7681` — offset equals the original row number, as the oracle predicts. This is
the 5-hop worst case in this configuration; a row in `f0..f3` cheap-misses v1
through v4 and does real work only at v5.

## Deletions

### Why they are needed at all

With `b = 0` the cascade rewrites every row, so each group's address set is one
contiguous run, which `RoaringTreemap::optimize()` collapses to a single run
container. That is the **cheapest possible input to `rank()`**, and benchmarking
`Compact` only there would flatter it badly. Roaring's three `rank`
implementations have very different complexities, and `optimize()` picks a
container on **serialized size alone, never on lookup cost**
(`roaring::bitmap::container::Container::optimize`):

There are **two** layers, and the outer one dominates.

`RoaringBitmap::rank` (`bitmap/inherent.rs:735-749`) binary-searches to the container
holding `x`, ranks within it, and then **sums `len()` over every preceding container**.
That per-container `len()` is where the cost lives, because it is cached for two store
types and not for the third:

| store | `len()` | inner `rank(x)` |
|---|---|---|
| Array | `vec.len()` — O(1) | `vec.binary_search(&x)`, O(log n), <=12 probes |
| Bitmap | cached `u64` field — **O(1)** | popcount loop, <=1024 words over 8 KiB |
| Run | `self.0.iter().map(run_len).sum()` — **O(runs)** | linear scan over intervals |

So whole-bitmap `rank` costs `O(#containers_below)` for an array- or bitmap-encoded
bitmap, but `O(#containers_below * runs_each)` for a run-encoded one.

**Run is therefore the worst encoding on this path, and Bitmap the best** — the reverse
of what the per-container column alone suggests. `Run::rank` being a linear scan rather
than a binary search matters, but the uncached `IntervalStore::len()` in the outer sum
matters more, because it is paid for every container below the target rather than once.

Concretely, for a 1M-row old fragment (16 containers):

```
  b = 0  (whole fragment rewritten)     1 run/container    ~16 interval iterations
  b > 0  (deletion holes)            <=2048 runs/container  ~32k interval iterations
```

per hop, multiplied by chain depth. Two consequences the rest of this document depends
on:

- the deletion knob's job is to control **run count**
- if the resident bitmaps inherit run encoding, `Compact`'s *lookup* cost scales with
  **fragment width in rows** (via container count), not with fragment count

That second point was true of this code and is now fixed: `GroupRemap::new` chooses the
resident encoding by run density (`should_strip_runs`) instead of inheriting the payload's,
so lookup cost is flat in row count again while the serialized payload stays run-optimized.
The generator still produces run-encoded payloads, because that is what compaction writes;
what changed is what the remap keeps in memory. Do not read the run counts below as a
lookup cost -- they are the input to that decision.

### Why per-round

Deletions are applied at **every round a row participates in**, not once up front.
The alternative — deleting only on a row's first rewrite — leaves the blob dense,
and the blob is the majority of every group's rows from round 2 onward, so the
biggest bitmap in every late round would stay a single trivial run. Per-round
application is also what actually happens: rows get deleted from a compacted
fragment, then the next compaction drops them.

### Why the period is fixed

Two quantities matter, and `(b, DELETION_PERIOD)` separates them:

```
per-round rate     = b / 128
runs per container = min(fragment_rows, 65536) / 128
```

Run count depends **only on the period**. So fixing it at 128 and sweeping `b`
varies the deletion rate at *constant bitmap shape* — one variable at a time. 128
puts a full-width container at 512 runs, which is real `Run::rank` scan depth while
staying well below the ~2048-run point where `optimize()` stops preferring runs.

### The sweep

All four verified at `FLOOR_ROWS = DELETION_PERIOD = 128` against a ground-truth
model: payload validity, the survivors-land-consecutively invariant, and agreement
with the arithmetic oracle.

```
   b   rate/rnd   aggD    alive@10  alive@20  alive@30  kept-run  wiped
   0     0.000%    0.00%    100.0%    100.0%    100.0%     128       0
   3     2.344%    4.47%     78.9%     62.2%     49.1%     125       0
   6     4.688%    8.75%     61.9%     38.3%     23.7%     122       0
  12     9.375%   16.82%     37.4%     14.0%      5.2%     116       0
```

`aggD` measured at `N = 16384, H = 4` (depth 6); `alive@d` is `(1 - b/128)^d`, the
fraction of the *deepest* chain still live at depth `d`. `wiped` is fragments left
with no surviving rows — zero everywhere, which is the `FLOOR_ROWS == 128`
guarantee in action. No configuration produced a zero-row new fragment either, so
all four load.

### Per-round application compounds, so the rate is not the aggregate

A row is exposed once per round it survives, so its deletion probability is
`1 - (1 - b/128)^hops`. Measured at `b = 3, N = 16384`, the staircase tracks the
prediction at every level:

```
 level   actual / predicted
 L0       2.3% / 2.3%
 L1       4.6% / 4.6%
 L2       6.7% / 6.9%
 L3       8.9% / 9.1%
 L4      10.9% / 11.2%
 floor   12.9% / 13.3%
```

Consequences worth knowing:

- **`D` is an output, not an input.** The knob is the per-round rate `b/128`; the
  aggregate falls out of the hop distribution (roughly `1-(1-b/128)^2` for the
  geometric layout, since average hops is about 2).
- **The skew leans the wrong way.** Exposure scales with hop count, and in this
  layout the deepest chains are the smallest, highest-numbered fragments — the
  *newest* data. Reality is the opposite: older data has had longer to accumulate
  deletions. Fixable with per-round rates that vary by round; not currently done.

At `FLOOR_ROWS = 128` the staircase is clean at every level. At smaller floors it
is not: where the period exceeds a fragment's size, whether that fragment sees any
deletion is decided by alignment, so small levels become alignment lotteries rather
than rates. That was measured at `FLOOR_ROWS = 4` — levels reporting 0.0% against a
predicted 58.8% — and is the main reason the floor is fixed at the period.

### Run count needs large fragments to saturate

`runs = min(fragment_rows, 65536) / 128`, so the 512-run figure requires fragments
at least one roaring container wide. At small `N` the blob is far narrower than a
container and you get tens of runs rather than hundreds; benchmark-scale `N` gives
blobs spanning many containers, each carrying the full 512.

Either way a period of 128 stays in **Run** containers throughout and never
exercises `Bitmap::rank`'s popcount loop — reaching that needs a period of 32 or
less on full-width containers, which would be a separate arm and would force a
higher minimum rate (`b >= 1` gives 3.1%/round at a period of 32).

### Choose probes by survival

A deleted row is **cheaper** to look up, not more expensive: once the chain hits a
deletion, every remaining version does no work, because `remap_row_id`'s
`if mapped_value.is_some()` guard skips the body for the rest of the loop.

So at high `b` the deep tail is mostly deletions, and probing it would measure
*early termination* — how fast a dead row is discovered — rather than the deep live
traversal it looks like. The skew anti-correlates depth with expense. At `b = 12`
only 5.2% of the deepest chain survives to depth 30, so that arm is a bitmap-shape
arm, not a deep-traversal arm.

No generator change is needed: the pattern is deterministic, so the oracle already
knows which rows survive. Split it into two arms and measure both:

- **deep live traversal** — probes drawn from survivors at each hop depth; the
  expensive path
- **early termination** — probes drawn from deleted rows; a real production case
  with no coverage today

## The oracle

Given a probe row `r`, its **starting** address is arithmetic:

```
L      = log2(N) - ceil_log2(N - r)     // level index
lstart = N - (N >> L)                   // first row of the level
lsize  = N >> (L + 1)                   // rows in the level
fsize  = lsize / H                      // rows per fragment in the level
frag   = H*L + (r - lstart) / fsize
offset = (r - lstart) % fsize
```

with `ceil_log2(d) = 64 - (d - 1).leading_zeros()` **on `u64`** — on `u32` the formula is
off by 32 — and `log2(N) = N.trailing_zeros()`.

**The floor-level guard is mandatory, not cosmetic.** The geometric formula does not
degrade gracefully past `lstart(Lmax)`: at `N = 8192`, `r = 7680` yields `L = 4 = Lmax`,
but `r = 8191` yields `L = 13`, well out of range. So test the floor **first**:

```
if row >= N - (N >> Lmax) {                  // floor level
    let off_in_level = row - (N - (N >> Lmax));
    frag   = H * Lmax + off_in_level / FLOOR_ROWS;
    offset = off_in_level % FLOOR_ROWS;
} else {                                     // geometric level, as above
    ...
}
```

Note the floor branch divides by `FLOOR_ROWS`, not by `fsize(L)`. `row == N` must never be
passed, since `d = 0` underflows `(d - 1)`; `debug_assert!(row < N)`. `row = 0` and exact
powers of two are fine. When `N < 1024` there are no geometric levels (`Lmax = 0`) and
every row takes the floor branch.

Its **ending** address at `b = 0` is `(final_fragment, r)` — because every round
lists its old fragments in ascending id order and the blob always sorts last while
holding the highest rows, so original row order survives the whole cascade. With
deletions, the surviving rows still land in order, giving the invariant used to
validate the generator:

> the k-th surviving row lands at offset `k` of the final fragment

That is structure-independent — it asserts order preservation and density rather
than reimplementing the positional pairing — so each form can be checked against
arithmetic instead of merely against the other form.

### No data-structure lookups, one loop

A full chain walk needs **8 scalars** (`N, H, k, b` plus derived
`Lmax, cnt_floor, fid_max`, with the two constants inlined) and nothing else. Two
things that look like they need tables do not:

- **`rows_before` telescopes.** A level's `H` fragments are contiguous in global
  row space, so "survivors of all fragments preceding the i-th" is
  `survivors(lstart, i*fsize)` — one closed-form expression, not a sum over
  fragments. This only works because the deletion pattern is phased on the global
  row index; with a per-fragment phase it would not collapse and a table would be
  required.
- **The blob's row count is never needed on the probe path.** There is exactly one
  new fragment per round, so the new address is `(new_id, rows_before + position)`
  with no range list to search. Blob sizes are needed only at construction time to
  fill in `new_frags`' `physical_rows`, which is untimed setup.

The only loop is over rounds — inherent, since the answer is the composition of
`depth` links, and `remap_row_id` loops the same way. The loop body is O(1).

### Keep the oracle out of the timed loop

Everything reduces to `f(M) = (M >> 7) * b + max(0, (M & 127) - (128 - b))`, called
a few times per round. Even at that cost, computing *expected answers* while the
clock runs is unnecessary: validate once over the whole range up front, and in the
timed loop only issue probe addresses, which needs `start_addr` alone — no chain
walk at all.

## Why the generated payloads are valid

`RowAddrRemap::compact` rejects malformed groups, so a generator that produced
them would be measuring error paths. This construction satisfies every check by
design, verified for `b` in `{0, 3, 6, 12}`:

- **one new fragment per round**, so `new_frags` is trivially ascending — required
  because `compute_new_addr` accumulates row counts in that order
- **rewritten count equals the new fragment's `physical_rows`** by construction,
  including under deletions, since the new fragment is sized from the survivor
  count
- **`old_frag_ids` in ascending id order**, which for addresses in ascending order
  (how a serialized `RoaringTreemap` iterates) makes the positional and
  address-ordered pairings agree — the case a real writer always produces, since
  manifest fragment lists are stored in id order
- **no wiped fragments and no zero-row new fragments**, guaranteed by
  `b < FLOOR_ROWS`

## What the M region is for

Probes drawn from `N..M` land in fragments that appear in no group, so every
version cheap-misses. That is the dominant production case: most rows live in
large settled fragments that compaction has not touched, and the interesting
question is what walking a deep chain costs for them.

`M` must map to fragment ids above the highest the cascade mints, or the probe
becomes a hit and measures the wrong path.

Mixing the ranges gives a tunable hit/miss ratio: probes uniform over `0..N` give
the size-weighted hop distribution above, and adding `N..M` dilutes it toward
all-miss.

## Not yet modelled

Deliberate gaps, listed so nobody mistakes them for oversights:

- **`k != H`.** See the note under [Constants and knobs](#constants-and-knobs) —
  this is a prerequisite for reaching production depth, not an optional extra.
- **Array and Bitmap payloads.** A period of 128 keeps every *serialized* container
  Run-encoded, so the generator only ever hands the remap a run-encoded payload. Since
  `should_strip_runs` decides the resident encoding from run density, the strip branch is
  well covered and the keep branch is reached only at `b = 0`. A period of 32 or less would
  produce Bitmap payloads directly, but breaks the fixed period.
- **Mixed-density bitmaps.** The deletion pattern is uniform, so every container within a
  bitmap has the same density. `remove_run_compression` is all-or-nothing per bitmap, so a
  real payload whose containers differ in density gets one decision driven by the average.
  Untested; the failure mode is a suboptimal encoding, never a wrong answer.
- **Irregular run lengths.** Block-periodic deletion produces perfectly uniform
  runs; real deletion patterns do not. A seeded-PRNG arm would cover this, at the
  cost of the closed-form end-of-chain oracle (survivor positions would become rank
  queries). Worth one spot check, not the whole grid.
- **Deletion rates that vary by round**, which is what would correct the
  newest-data-culled-hardest skew.
- **Small fragments.** With the floor fixed at 128 the generator cannot produce the
  4-row fragments a dataset built from many tiny write operations accumulates. The
  floor is fixed at the period because that is what makes deletion rates exact; the
  cost is that this particular slice of layout realism is out of reach.
- **Multiple new fragments per group.** Every round produces exactly one new
  fragment, so `new_frag_row_ranges` always holds a single entry and
  `compute_new_addr`'s binary search over it is trivial. Real rewrite groups emit
  several new fragments — the upstream tests deliberately use 5 so the search lands
  strictly inside the list. This is a `Compact`-only cost the generator currently
  zeroes out, and the most significant of the gaps here.
- **Multiple rewrite groups per version.** Every round here is one group, so group
  indirection is trivially predictable.
- **Non-ascending `old_frag_ids`.** Not reachable from any real writer, and the two
  pairings disagree there, so it is out of scope for a performance comparison.
- **Probe distribution.** Uniform over rows gives the realistic ~2-hop average.
  Uniform over *fragments* would stress depth instead, since the long chains live
  in the small tail fragments. Both are worth having — see
  [Choose probes by survival](#choose-probes-by-survival) for the interaction with
  deletions.
