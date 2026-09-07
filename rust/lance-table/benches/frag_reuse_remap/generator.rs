// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Synthetic fragment-reuse payload generator.
//!
//! See `synthetic_row_ranges.md` in this directory for the design. In brief: cut `0..N`
//! into geometrically shrinking fragments, then record compactions from the small end
//! upwards, deleting a fixed fraction of each round's rows. Every derived quantity is
//! closed-form arithmetic, so the oracle in `oracle.rs` needs no data structures.

use lance_core::utils::row_addr_remap::{GroupInput, RowAddrRemap};
use lance_table::system_index::frag_reuse::{
    FragDigest, FragReuseGroup, FragReuseIndex, FragReuseIndexDetails, FragReuseVersion,
};
use roaring::RoaringTreemap;
use std::collections::HashMap;
use uuid::Uuid;

/// Floor fragment size, in rows. The geometric halving stops here.
pub const FLOOR_ROWS: u64 = 128;

/// Deletion period, in rows. Equal to `FLOOR_ROWS` deliberately: that makes every
/// fragment an exact multiple of the period, so deletion rates are exact at every level
/// and no cluster can wipe a fragment.
pub const DELETION_PERIOD: u64 = 128;

/// Fragments per geometric level. Splits levels horizontally, moving fragment count
/// without moving hop counts.
pub const FRAGS_PER_LEVEL: u64 = 4;

/// Which remap form to materialise.
///
/// Defined locally rather than reusing `IndexRemapMode`: that type lives in the `lance`
/// crate, which depends on `lance-table`, so it cannot be reached from here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Form {
    Direct,
    Compact,
}

impl Form {
    pub fn label(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Compact => "compact",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BuildParams {
    /// Total rows. Power of two, at least 256.
    pub n: u64,
    /// Deleted rows per `DELETION_PERIOD`. Must be below `FLOOR_ROWS`.
    pub b: u64,
    /// Fragments consumed per compaction round: `FRAGS_PER_LEVEL` or 1.
    pub k: u64,
}

impl BuildParams {
    pub fn new(n: u64, b: u64, k: u64) -> Self {
        assert!(
            n.is_power_of_two() && n >= 256,
            "n must be a power of two >= 256"
        );
        assert!(
            b < FLOOR_ROWS,
            "b must be below FLOOR_ROWS or a cluster can wipe a fragment"
        );
        assert!(
            k == 1 || k == FRAGS_PER_LEVEL,
            "k must be 1 or FRAGS_PER_LEVEL"
        );
        Self { n, b, k }
    }
}

/// Deleted rows strictly below `offset`, within a 128-aligned fragment.
///
/// Every fragment start is a multiple of `DELETION_PERIOD`, so offset phase and
/// global-row phase coincide and no phase argument is needed. `/128` and `%128` are a
/// shift and a mask.
#[inline]
pub fn deleted_below(offset: u64, b: u64) -> u64 {
    (offset >> 7) * b + (offset & 127).saturating_sub(DELETION_PERIOD - b)
}

#[inline]
pub fn is_deleted(offset: u64, b: u64) -> bool {
    b > 0 && (offset & 127) >= DELETION_PERIOD - b
}

/// Survivors in `[0, rows)` of a 128-aligned fragment.
#[inline]
pub fn survivors(rows: u64, b: u64) -> u64 {
    rows - deleted_below(rows, b)
}

/// Rank of `offset` among the survivors below it.
#[inline]
pub fn position(offset: u64, b: u64) -> u64 {
    offset - deleted_below(offset, b)
}

#[derive(Clone, Copy, Debug)]
pub struct Frag {
    pub id: u32,
    /// Kept for reporting and debugging; the cascade works off ids.
    #[allow(dead_code)]
    pub start_row: u64,
    pub rows: u64,
    /// Geometric level index, or `lmax` for the floor level.
    pub level: u32,
}

#[derive(Clone, Debug)]
pub struct Layout {
    /// Ascending id, which is also ascending `start_row`.
    pub frags: Vec<Frag>,
    /// Number of geometric levels. The floor level sits at index `lmax`.
    pub lmax: u32,
    pub floor_count: u64,
    /// One past the highest initial fragment id.
    pub fid_max: u32,
}

impl Layout {
    /// First row of geometric level `l`.
    #[inline]
    pub fn lstart(n: u64, l: u32) -> u64 {
        n - (n >> l)
    }
    /// Rows per fragment in geometric level `l`.
    #[inline]
    pub fn fsize(n: u64, l: u32) -> u64 {
        (n >> (l + 1)) / FRAGS_PER_LEVEL
    }
    pub fn depth(&self, k: u64) -> u32 {
        let f = self.frags.len() as u64;
        (f - 1).div_ceil(k) as u32
    }
}

pub fn layout(p: BuildParams) -> Layout {
    let n = p.n;
    let mut lmax = 0u32;
    while Layout::fsize(n, lmax) >= FLOOR_ROWS {
        lmax += 1;
    }
    let mut frags = Vec::new();
    for l in 0..lmax {
        let (ls, fs) = (Layout::lstart(n, l), Layout::fsize(n, l));
        for i in 0..FRAGS_PER_LEVEL {
            frags.push(Frag {
                id: (FRAGS_PER_LEVEL * l as u64 + i) as u32,
                start_row: ls + i * fs,
                rows: fs,
                level: l,
            });
        }
    }
    // Floor level: whatever remains, cut into FLOOR_ROWS pieces.
    let floor_start = Layout::lstart(n, lmax);
    let floor_count = (n - floor_start) / FLOOR_ROWS;
    for i in 0..floor_count {
        frags.push(Frag {
            id: (FRAGS_PER_LEVEL * lmax as u64 + i) as u32,
            start_row: floor_start + i * FLOOR_ROWS,
            rows: FLOOR_ROWS,
            level: lmax,
        });
    }
    let fid_max = FRAGS_PER_LEVEL * lmax as u64 + floor_count;
    Layout {
        frags,
        lmax,
        floor_count,
        fid_max: fid_max as u32,
    }
}

/// One compaction round: the initial fragments it consumes, plus the previous round's
/// output. `old` is ascending id; `blob` (when present) always has the highest id *and*
/// the highest rows, so `old ++ [blob]` is ascending in both.
#[derive(Clone, Debug)]
pub struct Round {
    pub fresh: Vec<u32>,
    pub blob: Option<u32>,
    pub new_id: u32,
}

pub fn schedule(l: &Layout, k: u64) -> Vec<Round> {
    let batches: Vec<Vec<u32>> = if k == FRAGS_PER_LEVEL {
        // Floor level first, then geometric levels from deepest to shallowest.
        let mut b = vec![
            l.frags
                .iter()
                .filter(|f| f.level == l.lmax)
                .map(|f| f.id)
                .collect::<Vec<_>>(),
        ];
        for lv in (0..l.lmax).rev() {
            b.push(
                l.frags
                    .iter()
                    .filter(|f| f.level == lv)
                    .map(|f| f.id)
                    .collect(),
            );
        }
        b
    } else {
        // Descending id, two in the first round. Descending is what keeps the blob
        // holding the highest rows as well as the highest id; ascending would break
        // the survivor invariant.
        let mut desc: Vec<u32> = l.frags.iter().map(|f| f.id).collect();
        desc.reverse();
        let mut b = vec![vec![desc[0], desc[1]]];
        for &id in &desc[2..] {
            b.push(vec![id]);
        }
        b
    };

    let mut rounds = Vec::with_capacity(batches.len());
    let mut blob = None;
    for (i, mut fresh) in batches.into_iter().enumerate() {
        fresh.sort_unstable();
        let new_id = l.fid_max + i as u32;
        rounds.push(Round {
            fresh,
            blob,
            new_id,
        });
        blob = Some(new_id);
    }
    rounds
}

#[derive(Clone, Debug, Default)]
pub struct BuildStats {
    #[allow(dead_code)]
    pub fragments: usize,
    #[allow(dead_code)]
    pub depth: u32,
    /// Rewritten (surviving) rows per round, i.e. each new fragment's `physical_rows`.
    pub rewritten_per_round: Vec<u64>,
    /// Total `Direct` entries: one per rewritten *or deleted* row, summed over rounds.
    pub direct_entries: u64,
    /// Container breakdown of the widest bitmap in the final round. This is what the
    /// `run_encoding` differential turns on: `RoaringBitmap::rank` sums `len()` over
    /// preceding containers, and that is O(runs) for a run container but O(1) for a
    /// bitset one.
    pub widest_containers: u64,
    pub widest_run_containers: u64,
    pub widest_bitset_containers: u64,
    /// Bytes in run containers; runs is this over 4 (an `Interval` is two `u16`).
    pub widest_run_bytes: u64,
    /// Total serialized `changed_row_addrs` bytes across all rounds -- the on-disk cost
    /// that `optimize()` exists to reduce.
    pub payload_bytes: u64,
    pub build_millis: u128,
    pub deep_size: usize,
}

/// Build the fragment reuse index in the requested form.
///
/// `Compact` is always constructed first and `Direct` derived from it by enumeration.
/// That is a deliberate benchmark choice, not the production path — production's
/// `Direct` arm goes through `transpose_row_ids_from_digest` — but the two are provably
/// identical on the well-formed payloads this generator emits, which is what makes the
/// comparison fair.
pub fn build(p: BuildParams, form: Form) -> (FragReuseIndex, BuildStats) {
    let started = std::time::Instant::now();
    let l = layout(p);
    let rounds = schedule(&l, p.k);

    let mut stats = BuildStats {
        fragments: l.frags.len(),
        depth: rounds.len() as u32,
        ..Default::default()
    };
    let mut maps: Vec<RowAddrRemap> = Vec::with_capacity(rounds.len());
    let mut versions: Vec<FragReuseVersion> = Vec::with_capacity(rounds.len());
    // Row count of each fragment as it enters its round: initial sizes, plus each
    // round's output once produced.
    let mut rows_at: HashMap<u32, u64> = l.frags.iter().map(|f| (f.id, f.rows)).collect();

    for (round_idx, round) in rounds.iter().enumerate() {
        let mut olds: Vec<u32> = round.fresh.clone();
        if let Some(b) = round.blob {
            olds.push(b); // highest id, so still ascending
        }

        let mut addrs = RoaringTreemap::new();
        let mut old_digests = Vec::with_capacity(olds.len());
        let mut total_survivors = 0u64;
        for &fid in &olds {
            let rows = rows_at[&fid];
            let surv = survivors(rows, p.b);
            for off in 0..rows {
                if !is_deleted(off, p.b) {
                    addrs.insert((fid as u64) << 32 | off);
                }
            }
            old_digests.push(FragDigest {
                id: fid as u64,
                physical_rows: rows as usize,
                num_deleted_rows: (rows - surv) as usize,
            });
            total_survivors += surv;
            // Direct holds an entry per rewritten *or* deleted row.
            stats.direct_entries += rows;
        }

        // Run-optimize, as compaction does before writing. What the *resident* bitmaps
        // end up encoded as is `GroupRemap::new`'s decision, not ours.
        addrs.optimize();
        // Round-trip through the serialized form, as the reader does, so the container
        // encoding reaching `rank()` is the encoding the payload actually carries.
        let mut bytes = Vec::with_capacity(addrs.serialized_size());
        addrs.serialize_into(&mut bytes).expect("serialize");
        let addrs = RoaringTreemap::deserialize_from(&bytes[..]).expect("deserialize");
        stats.payload_bytes += bytes.len() as u64;

        if round_idx == rounds.len() - 1
            && let Some((_, widest)) = addrs.bitmaps().max_by_key(|(_, bm)| bm.len())
        {
            let st = widest.statistics();
            stats.widest_containers = st.n_containers as u64;
            stats.widest_run_containers = st.n_run_containers as u64;
            stats.widest_bitset_containers = st.n_bitset_containers as u64;
            stats.widest_run_bytes = st.n_bytes_run_containers;
        }

        let group = GroupInput {
            rewritten_old_row_addrs: addrs.clone(),
            old_frag_ids: olds.clone(),
            new_frags: vec![(round.new_id, total_survivors as u32)],
        };
        let compact = RowAddrRemap::compact([group]).expect("payload must be well-formed");

        maps.push(match form {
            Form::Compact => compact,
            Form::Direct => {
                // Enumerate every offset of every old fragment, not just the rewritten
                // set: a deleted row must map to `Some(None)` so the chain terminates,
                // matching `transpose_row_ids_from_digest`'s `MissingAddrs` inserts.
                let cap: u64 = olds.iter().map(|f| rows_at[f]).sum();
                let mut map = HashMap::with_capacity(cap as usize);
                for &fid in &olds {
                    for off in 0..rows_at[&fid] {
                        let addr = (fid as u64) << 32 | off;
                        map.insert(addr, compact.get(addr).expect("covered fragment"));
                    }
                }
                RowAddrRemap::direct(map)
            }
        });

        versions.push(FragReuseVersion {
            dataset_version: round_idx as u64 + 1,
            groups: vec![FragReuseGroup {
                changed_row_addrs: bytes,
                old_frags: old_digests,
                new_frags: vec![FragDigest {
                    id: round.new_id as u64,
                    physical_rows: total_survivors as usize,
                    num_deleted_rows: 0,
                }],
            }],
        });

        rows_at.insert(round.new_id, total_survivors);
        stats.rewritten_per_round.push(total_survivors);
    }

    let index =
        FragReuseIndex::new_from_remaps(Uuid::new_v4(), maps, FragReuseIndexDetails { versions });
    stats.build_millis = started.elapsed().as_millis();
    stats.deep_size = lance_core::deepsize::DeepSizeOf::deep_size_of(&index);
    (index, stats)
}
