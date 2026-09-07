# `frag_reuse_remap` benchmark

Measures per-lookup cost of the two `RowAddrRemap` forms. The reader-side default is
`Direct` because that cost was never quantified: build time and footprint were
characterised on a production payload, latency was not.

- [`synthetic_row_ranges.md`](./synthetic_row_ranges.md) — the payload generator: layout,
  compaction cascade, deletion model, the closed-form oracle, and why each fixed constant
  has the value it does. **Read it before changing `generator.rs` or `oracle.rs`.**
- `generator.rs` / `oracle.rs` / `probes.rs` / `frag_reuse_remap.rs` — construction,
  expected answers, probe lists, harness.

```
cargo bench -p lance-table --bench frag_reuse_remap
FRAG_REUSE_CSV=/tmp/run.csv cargo bench -p lance-table --bench frag_reuse_remap
```

Writes one CSV row per cell — `target/frag_reuse_remap.csv` unless `FRAG_REUSE_CSV` says
otherwise — and a progress line per build. Run records are deliberately **not** checked in:
they are machine-specific, nothing asserts against them, and a stale one in the tree misleads
more than it helps. Quote figures with the hardware and date beside them.

## Axes

| axis | values | what it controls |
|---|---|---|
| `N` | 2^13 .. 2^24 | rows. Every cell fits `Direct`'s 8 GiB budget at this ceiling, so both forms run at every size |
| `b` | 0, 3, 6, 12 | deleted rows per 128-row period, so bitmap fragmentation |
| `k` | 4, 1 | fragments merged per round, so chain depth (5–16 vs 19–63) |
| `m` | 1, 2, 4, 128 | probe range is `m*N`, so the reuse-index hit rate is `1/m` |
| `c` | 1, 512, 65536 | run length: consecutive rows per jump |

Fixed, with the reasoning on each constant: `FLOOR_ROWS`, `DELETION_PERIOD`,
`FRAGS_PER_LEVEL`, `PROBES`.

## Correctness

Every cell asserts **all** its probe results against the oracle before it is timed, so no
reported figure can come from a configuration that computed a wrong address. On top of that,
startup validation checks payload validity, oracle agreement and the survivor invariant over
every row at the six smallest sizes, for all four `b` and both `k`.

The survivor invariant — the k-th surviving row lands at offset k of the final fragment — is
structure-independent, so it cannot pass against a stubbed lookup. The small sizes are kept
deliberately: `N < 1024` yields zero geometric levels, a branch in both the layout and the
oracle that no larger size reaches. Validation runs at startup rather than under `#[test]`
because a `harness = false` bench target is not run by `cargo test`.

## Limitations

Read any result with these in view.

- **No criterion, so no confidence intervals.** Eight hand-rolled passes per cell. Treat
  differences under ~10% as noise; one cell moved 1.33x to 0.69x between runs before `PROBES`
  was raised.
- **Fragment count is pinned** at 20–88 by `FRAGS_PER_LEVEL`, so `Compact`'s own size axis is
  untested. That flatters it in exactly the miss-dominated regime where it looks strongest.
- **`Direct` is absent from the deep-chain corner.** At `k=1` it needs ~6.5 entries per row, so
  above 2^24 it exceeds the budget — the region where `Compact` looks worst, and so the gap most
  likely to qualify any conclusion.
- **`Survival` is wired through `probes.rs` but never swept**, so the early-termination path that
  deleted rows take is not separated from live traversal.
- **One rewrite group per round**, so `compute_new_addr`'s binary search over new-fragment ranges
  stays trivial. Real groups emit several.
- **Uniform deletion holes**, so every container in a bitmap has the same density. Real deletions
  cluster, and `remove_run_compression` is all-or-nothing per bitmap.
- **`c=65536` holds only 4 runs per batch**, so those cells are seed-dominated. Use `c=512` for a
  trustworthy clustered point.
