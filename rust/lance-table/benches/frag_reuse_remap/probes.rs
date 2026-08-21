// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Probe-list construction.
//!
//! Built entirely in setup: the expectation for each probe is derived arithmetically in
//! the same pass, so the oracle is never invoked during the assertion pass or the timed
//! loop, and the checking machinery cannot perturb what it checks.

use crate::generator::BuildParams;
use crate::oracle::Oracle;

/// Lookups per timed iteration.
///
/// Each probe pulls one 64-byte line of `Direct`'s table, so this sets how much of that
/// table a pass touches -- and therefore whether repeated passes find it warm. Measured on
/// this machine, `Direct`'s per-lookup cost does not plateau until ~2^18 probes (16 MiB of
/// lines): at 2^16 it reads 17% low at a 1% hit rate, i.e. the table was still cached
/// between passes and `Direct` was being flattered. 2^18 lands within 1.3% of plateau.
/// `Compact` is flat at every count, its structures being small enough to stay resident.
pub const PROBES: usize = 1 << 18;

/// Fragment id base for rows beyond `N`. Must exceed any id the cascade mints, which
/// across the whole sweep tops out in the low hundreds.
const M_BASE: u64 = 1 << 20;

// `SurvivorsOnly` and the `c_eff`/`reps` fields are consumed by criterion cells not yet
// swept yet (see the limitations in `README.md`); they are part of the intended interface,
// so they stay rather than being reintroduced later.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Survival {
    /// Draw from all rows: what a real index holds.
    Any,
    /// Rows that survive the whole chain -- the deep live traversal.
    SurvivorsOnly,
    /// Rows the chain deletes -- the early-termination path.
    DeletedOnly,
}

#[derive(Clone, Copy, Debug)]
pub struct ProbeParams {
    /// Probe range is `m * N`, so the fragment-reuse hit rate is `1/m`.
    pub m: u64,
    /// Consecutive rows per jump.
    pub c: usize,
    pub survival: Survival,
}

#[allow(dead_code)]
pub struct Probes {
    /// Fed to `remap_row_id`.
    pub addrs: Vec<u64>,
    /// `oracle.walk(row)` for real rows; `Some(addr)` unchanged beyond `N`.
    pub expected: Vec<Option<u64>>,
    /// `c` after clamping to the probe range; differs from `c` at small `N`.
    pub c_eff: usize,
    /// `PROBES / M`, the repetition factor. Heavy repetition makes `rank`'s trip count
    /// branch-predictable and so understates it.
    pub reps: f64,
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub fn build_probes(bp: BuildParams, pp: ProbeParams, o: &Oracle) -> Probes {
    build_probes_n(bp, pp, o, PROBES)
}

/// As [`build_probes`], with an explicit probe count. Used to check whether `PROBES` is
/// large enough that `Direct`'s touched cache lines are not retained across repetitions.
pub fn build_probes_n(bp: BuildParams, pp: ProbeParams, o: &Oracle, count: usize) -> Probes {
    let m_rows = pp.m * bp.n;
    // A run of `c` consecutive rows cannot fit in a shorter range: N=2^8 with m=1 gives
    // M=256, so c=512 and c=65536 are both impossible.
    let c_eff = pp.c.min(m_rows as usize).max(1);
    assert!(
        pp.survival != Survival::DeletedOnly || bp.b > 0,
        "DeletedOnly has no candidates at b=0"
    );
    assert!(
        pp.survival == Survival::Any || c_eff == 1,
        "survival filtering rejects per row, so it only composes with c=1"
    );
    assert!(
        pp.survival != Survival::DeletedOnly || pp.m == 1,
        "rows beyond N are never deleted, so DeletedOnly implies m=1"
    );

    // Seeded from the cell's parameters, so a cell is byte-identical across runs and
    // machines. Never a clock.
    let mut seed = bp.n.wrapping_mul(0x9E37)
        ^ bp.b.wrapping_mul(0x85EB)
        ^ bp.k.wrapping_mul(0xC2B2)
        ^ pp.m.wrapping_mul(0x27D4)
        ^ (pp.c as u64).wrapping_mul(0x165667B1);

    let mut addrs = Vec::with_capacity(count);
    let mut expected = Vec::with_capacity(count);

    let map = |row: u64| -> (u64, Option<u64>) {
        if row < bp.n {
            let a = o.start_addr(row);
            (a, o.walk(row))
        } else {
            // Fragment ids no group covers, so every version cheap-misses.
            let off = row - bp.n;
            let a = (M_BASE + off / 128) << 32 | (off % 128);
            (a, Some(a))
        }
    };

    let acceptable = |row: u64| -> bool {
        match pp.survival {
            Survival::Any => true,
            Survival::SurvivorsOnly => row >= bp.n || o.survives(row),
            Survival::DeletedOnly => row < bp.n && !o.survives(row),
        }
    };

    while addrs.len() < count {
        let start = if m_rows > c_eff as u64 {
            splitmix64(&mut seed) % (m_rows - c_eff as u64 + 1)
        } else {
            0
        };
        for j in 0..c_eff {
            if addrs.len() == count {
                break;
            }
            let row = start + j as u64;
            if !acceptable(row) {
                continue;
            }
            let (a, want) = map(row);
            addrs.push(a);
            expected.push(want);
        }
    }

    Probes {
        addrs,
        expected,
        c_eff,
        reps: count as f64 / m_rows as f64,
    }
}
