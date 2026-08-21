// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark: `RowAddrRemap::Direct` vs `RowAddrRemap::Compact` lookup cost.
//!
//! Design: `README.md` for the axes and limitations, `synthetic_row_ranges.md` for the
//! generator.
//!
//! Emits one CSV row per cell so results can be plotted, with a progress line and a
//! summary on stdout. Timing is hand-rolled rather than criterion, so treat differences
//! under ~10% as noise; `README.md` lists that and the other limitations.

// Both print_stdout and print_stderr are denied workspace-wide, and a function-level
// allow would not cover the sibling modules.
#![allow(clippy::print_stdout)]

mod generator;
mod oracle;
mod probes;

use generator::{BuildParams, FRAGS_PER_LEVEL, Form, build, layout};
use oracle::Oracle;
use probes::{PROBES, ProbeParams, Survival, build_probes};
use std::io::Write as _;

const SWEEP_N: &[u64] = &[
    1 << 13,
    1 << 16,
    1 << 19,
    1 << 22,
    1 << 24,
    1 << 26,
    1 << 28,
    1 << 30,
];
const SWEEP_B: &[u64] = &[0, 3, 6, 12];
const SWEEP_K: &[u64] = &[FRAGS_PER_LEVEL, 1];
const SWEEP_M: &[u64] = &[1, 2, 4, 128];
const SWEEP_C: &[usize] = &[1, 512, 65536];

/// Skip `Direct` above this. Measured at ~44 bytes per entry over ~2N entries at `k=H`
/// and ~6.5N at `k=1`, so `k=1` reaches the ceiling two powers of two sooner. `Compact`
/// is four orders of magnitude smaller and runs at every size.
const DIRECT_BUDGET_BYTES: u64 = 8 << 30;
const DIRECT_BYTES_PER_ENTRY: u64 = 44;

/// Validation sizes: cache-resident, and the only ones that reach the
/// zero-geometric-levels branch (`N < 1024`).
const VALIDATE_N: &[u64] = &[256, 512, 1024, 2048, 4096, 8192];

fn direct_fits(n: u64, k: u64) -> bool {
    let entries_per_row = if k == FRAGS_PER_LEVEL { 2 } else { 7 };
    n * entries_per_row * DIRECT_BYTES_PER_ENTRY <= DIRECT_BUDGET_BYTES
}

type Index = lance_table::system_index::frag_reuse::FragReuseIndex;

/// Time the lookups, returning ns per lookup.
fn time_lookups(idx: &Index, addrs: &[u64]) -> f64 {
    for &a in addrs {
        std::hint::black_box(idx.remap_row_id(a));
    }
    let reps = 8;
    let t = std::time::Instant::now();
    for _ in 0..reps {
        for &a in addrs {
            std::hint::black_box(idx.remap_row_id(a));
        }
    }
    t.elapsed().as_nanos() as f64 / (reps * addrs.len()) as f64
}

/// The stream-and-loop floor with no lookup. It is present in every figure, so it is
/// measured and reported rather than silently subtracted.
fn floor_ns(addrs: &[u64]) -> f64 {
    for &a in addrs {
        std::hint::black_box(a);
    }
    let reps = 8;
    let t = std::time::Instant::now();
    for _ in 0..reps {
        for &a in addrs {
            std::hint::black_box(a);
        }
    }
    t.elapsed().as_nanos() as f64 / (reps * addrs.len()) as f64
}

/// Payload validity, oracle agreement over every real row, the survivor invariant, and
/// no degenerate payloads -- for both forms. See "Correctness" in `README.md`.
fn validate(p: BuildParams) -> Result<(), String> {
    let o = Oracle::new(p);
    assert!(!layout(p).frags.is_empty());
    for form in [Form::Compact, Form::Direct] {
        let (index, stats) = build(p, form);
        if stats.rewritten_per_round.contains(&0) {
            return Err(format!("{}: zero-row new fragment", form.label()));
        }
        let mut survivors = 0u64;
        for row in 0..p.n {
            let got = index.remap_row_id(o.start_addr(row));
            let want = o.walk(row);
            if got != want {
                return Err(format!(
                    "{}: row {row} -> {got:?}, oracle says {want:?}",
                    form.label()
                ));
            }
            if let Some(addr) = got {
                let final_frag = o.fid_max() + o.depth() - 1;
                if (addr >> 32) as u32 != final_frag || addr & 0xFFFF_FFFF != survivors {
                    return Err(format!(
                        "{}: survivor #{survivors} (row {row}) landed at {addr:#x}",
                        form.label()
                    ));
                }
                survivors += 1;
            }
        }
    }
    Ok(())
}

fn main() {
    // `cargo bench` runs with the cwd at the package root, not the workspace root, so a
    // bare `target/...` would resolve somewhere that does not exist.
    let csv_path = std::env::var("FRAG_REUSE_CSV").unwrap_or_else(|_| {
        format!(
            "{}/../../target/frag_reuse_remap.csv",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    if let Some(dir) = std::path::Path::new(&csv_path).parent() {
        std::fs::create_dir_all(dir).expect("create csv directory");
    }

    println!("frag_reuse_remap");
    println!(
        "consts  FLOOR_ROWS=128  DELETION_PERIOD=128  FRAGS_PER_LEVEL={FRAGS_PER_LEVEL}  \
         PROBES={PROBES}"
    );
    println!(
        "axes    N=2^13..2^30 ({})  b={SWEEP_B:?}  k={SWEEP_K:?}  m={SWEEP_M:?}  c={SWEEP_C:?}",
        SWEEP_N.len()
    );
    println!(
        "Direct skipped above {} GiB resident\n",
        DIRECT_BUDGET_BYTES >> 30
    );

    print!(
        "validating {} sizes x {} b x {} k ... ",
        VALIDATE_N.len(),
        SWEEP_B.len(),
        SWEEP_K.len()
    );
    let _ = std::io::stdout().flush();
    for &n in VALIDATE_N {
        for &k in SWEEP_K {
            for &b in SWEEP_B {
                if let Err(e) = validate(BuildParams::new(n, b, k)) {
                    println!("FAILED\n  N={n} k={k} b={b}: {e}");
                    std::process::exit(1);
                }
            }
        }
    }
    println!("ok\n");

    let mut csv = std::fs::File::create(&csv_path).expect("create csv");
    writeln!(
        csv,
        "n,log2n,b,k,form,depth,frags,containers,size_bytes,build_ms,m,c,c_eff,reps,\
         hit_pct,floor_ns,raw_ns,ns_per_lookup,us_per_batch"
    )
    .unwrap();

    let mut cells = 0usize;
    let mut skipped = 0usize;
    let started = std::time::Instant::now();

    for &n in SWEEP_N {
        for &b in SWEEP_B {
            for &k in SWEEP_K {
                for form in [Form::Direct, Form::Compact] {
                    if form == Form::Direct && !direct_fits(n, k) {
                        skipped += 1;
                        continue;
                    }
                    let p = BuildParams::new(n, b, k);
                    let o = Oracle::new(p);
                    let (idx, st) = build(p, form);
                    print!(
                        "\r  N=2^{:<2} b={:<3} k={} {:<8} depth={:<3} {:>13} B{:<12}",
                        n.trailing_zeros(),
                        b,
                        k,
                        form.label(),
                        o.depth(),
                        st.deep_size,
                        ""
                    );
                    let _ = std::io::stdout().flush();

                    for &m in SWEEP_M {
                        for &c in SWEEP_C {
                            let pr = build_probes(
                                p,
                                ProbeParams {
                                    m,
                                    c,
                                    survival: Survival::Any,
                                },
                                &o,
                            );
                            // No timing is reported for a configuration whose results
                            // were not first checked against the oracle.
                            for (&a, &want) in pr.addrs.iter().zip(&pr.expected) {
                                assert_eq!(
                                    idx.remap_row_id(a),
                                    want,
                                    "wrong result: N={n} b={b} k={k} {} m={m} c={c} \
                                     addr={a:#x}",
                                    form.label()
                                );
                            }
                            let floor = floor_ns(&pr.addrs);
                            let raw = time_lookups(&idx, &pr.addrs);
                            let net = raw - floor;
                            let hits = pr.addrs.iter().filter(|&&a| (a >> 32) < (1 << 20)).count();
                            writeln!(
                                csv,
                                "{},{},{},{},{},{},{},{},{},{},{},{},{},{:.4},{:.2},{:.2},\
                                 {:.2},{:.2},{:.1}",
                                n,
                                n.trailing_zeros(),
                                b,
                                k,
                                form.label(),
                                o.depth(),
                                st.fragments,
                                st.widest_containers,
                                st.deep_size,
                                st.build_millis,
                                m,
                                c,
                                pr.c_eff,
                                pr.reps,
                                100.0 * hits as f64 / pr.addrs.len() as f64,
                                floor,
                                raw,
                                net,
                                net * pr.addrs.len() as f64 / 1000.0
                            )
                            .unwrap();
                            cells += 1;
                        }
                    }
                }
            }
        }
    }

    println!("\r{:<90}", "");
    println!(
        "{cells} cells in {:.0}s -> {csv_path}",
        started.elapsed().as_secs_f64()
    );
    println!("{skipped} Direct configurations skipped, over the resident budget");
    println!();
    println!("ns_per_lookup = raw_ns - floor_ns; us_per_batch is over PROBES={PROBES}.");
    println!("c_eff is c clamped to the probe range m*n. At c=65536 a batch holds only");
    println!(
        "{} runs, so those cells are seed-dominated -- read them as indicative.",
        PROBES / 65536
    );
    println!("hit_pct is measured, not assumed: it should track 100/m.");
}
