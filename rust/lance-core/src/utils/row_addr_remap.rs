// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compact row-address remapping for compaction.
//!
//! Compaction rewrites rows into new fragments, so indices that store physical
//! row addresses need an old-address to new-address mapping without building an
//! O(total rows) `HashMap<u64, Option<u64>>`.
//!
//! Layout:
//!
//! * Old rows: `old_fragment_id -> (old_offsets, old_rows_before)`
//!     * `old_offsets`: rewritten old row offsets in this old fragment.
//!     * `old_rows_before`: rewritten row count before this old fragment.
//! * New rows: ordered new-fragment ranges
//!   `(fragment_id, new_rows_before, physical_rows)`
//!     * `new_rows_before`: rewritten row count before this new fragment.
//!
//! Lookup:
//!
//! * An address whose fragment was not rewritten returns `None`.
//! * For an address whose fragment was rewritten:
//!     * Read `(old_offsets, old_rows_before)` from the old-row layout.
//!     * If `offset` is not in `old_offsets`, return `Some(None)` because the
//!       row was deleted.
//!     * Otherwise, `old_offsets.rank(offset) - 1` is this row's 0-based
//!       position among rewritten old rows in this old fragment. Add
//!       `old_rows_before` to get `k`, the row's 0-based position among all
//!       rewritten old rows.
//!     * In the new-row layout, find the range
//!       `(fragment_id, new_rows_before, physical_rows)` where
//!       `new_rows_before <= k < new_rows_before + physical_rows`.
//!     * The new address is `(fragment_id, k - new_rows_before)`.
//!
//! Ordering:
//!
//! Compact remap does not store each old-to-new row mapping. It computes `k`
//! from the old-row layout, then maps it to the k-th row written to the new
//! fragments. This requires the reader-to-writer pipeline to preserve row order.
//!
//! * `old_frag_ids` must match the order old fragments are read. Within each
//!   old fragment, rewritten rows are interpreted by ascending old row offset.
//! * `new_frags` must match the order new rows are written.
//! * Current compaction satisfies this because it scans selected fragments in
//!   order and writes the resulting stream without reordering rows.

use crate::utils::address::RowAddress;
use crate::{Error, Result};
use roaring::{RoaringBitmap, RoaringTreemap};
use std::collections::HashMap;

/// A queryable row-address remapping with the exact semantics of
/// `HashMap<u64, Option<u64>>::get(&addr).copied()`:
///
/// * `None` — the address is not affected by this remap (keep it unchanged)
/// * `Some(None)` — the row was deleted
/// * `Some(Some(addr))` — the row moved to `addr`
#[derive(Clone)]
pub enum RowAddrRemap {
    /// Compact, `O(#fragments)` remap built from per-group rewritten-row
    /// bitmaps and new-fragment layouts.
    Compact(CompactRowAddrRemap),
    /// Full materialized old-to-new address map. Uses `O(#rows)` memory.
    Direct(HashMap<u64, Option<u64>>),
}

impl RowAddrRemap {
    pub fn compact(groups: impl IntoIterator<Item = GroupInput>) -> Result<Self> {
        Ok(Self::Compact(CompactRowAddrRemap::new(groups)?))
    }

    /// Build a remap from a fully materialized old-to-new address map.
    pub fn direct(map: HashMap<u64, Option<u64>>) -> Self {
        Self::Direct(map)
    }

    /// An empty remap that leaves every address unchanged.
    pub fn empty() -> Self {
        Self::Direct(HashMap::new())
    }

    /// Look up `addr`. See [`RowAddrRemap`] for the tri-state return semantics.
    #[inline]
    pub fn get(&self, addr: u64) -> Option<Option<u64>> {
        match self {
            Self::Compact(c) => c.get(addr),
            Self::Direct(m) => m.get(&addr).copied(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Compact(c) => c.is_empty(),
            Self::Direct(m) => m.is_empty(),
        }
    }

    /// Every fragment whose addresses this remap can answer non-`None` for.
    ///
    /// Must never under-report: callers skip a remap entirely for fragments absent from
    /// this set, so a missing fragment silently drops that fragment's remaps. Over-reporting
    /// is fine -- it only costs a `get` that answers `None`. Any new [`RowAddrRemap`] variant
    /// must uphold that direction.
    pub fn affected_fragments(&self) -> RoaringBitmap {
        match self {
            // Exact: `get` returns `None` precisely when the fragment is absent here.
            Self::Compact(c) => RoaringBitmap::from_iter(c.frag_to_group.keys().copied()),
            // Superset: `get` answers non-`None` only for listed addresses, whose fragments
            // are all covered.
            Self::Direct(m) => RoaringBitmap::from_iter(m.keys().map(|addr| (addr >> 32) as u32)),
        }
    }

    pub fn fully_deleted_fragments(&self) -> Option<RoaringBitmap> {
        match self {
            Self::Compact(c) => c.fully_deleted_fragments(),
            Self::Direct(m) => {
                if m.values().all(|v| v.is_none()) {
                    Some(RoaringBitmap::from_iter(
                        m.keys().map(|addr| (addr >> 32) as u32),
                    ))
                } else {
                    None
                }
            }
        }
    }
}

/// Input describing one rewrite group: the old row addresses that were
/// rewritten plus the fragment layout before/after the rewrite.
pub struct GroupInput {
    /// Old row addresses that were read and re-written into the new fragments.
    pub rewritten_old_row_addrs: RoaringTreemap,
    /// Old fragment ids covered by this group.
    pub old_frag_ids: Vec<u32>,
    /// New fragments produced by this group, as `(fragment_id, physical_rows)`,
    pub new_frags: Vec<(u32, u32)>,
}

#[derive(Clone, crate::deepsize::DeepSizeOf)]
struct GroupRemap {
    /// Old fragment id -> (rewritten old row offsets in that fragment,
    /// rewritten row count before this fragment in the group).
    frags: HashMap<u32, (RoaringBitmap, u64)>,
    /// New fragment ranges as `(fragment_id, rewritten_rows_before, physical_rows)`,
    /// used to map a rewritten row's group-local index to its new address via binary search.
    new_frag_row_ranges: Vec<(u32, u64, u32)>,
}

/// Average runs per run container above which run encoding costs more to `rank` than it
/// saves. A container holding a single run ranks as cheaply as a cached `len()` read and
/// is far smaller, so it is worth keeping; beyond a handful the interval rescan dominates.
///
/// Deliberately at the low end: the error is asymmetric. Keeping runs that should be
/// stripped costs a term that grows with fragment width, while stripping runs that should
/// be kept costs a small constant.
const MAX_RUNS_PER_CONTAINER: u64 = 4;

/// Bytes roaring spends per run when serializing a run container: an `Interval` is two
/// `u16`. Needed because `statistics()` reports run bytes but not a run count.
const RUN_BYTES: u64 = 4;

/// Bytes of run count roaring writes per run container.
const RUN_COUNT_BYTES: u64 = 2;

/// Whether stripping run encoding from `bitmap` would make `get` cheaper.
///
/// `RoaringBitmap::rank` sums `len()` over every container below the target. That is a
/// cached field for bitmap containers and an interval rescan for run containers, so the
/// encoding chosen for the *serialized* payload (which `optimize` picks by size) can be
/// the wrong one to keep resident.
fn should_strip_runs(bitmap: &RoaringBitmap) -> bool {
    let stats = bitmap.statistics();
    // `n_bytes_run_containers` is `sum(RUN_COUNT_BYTES + RUN_BYTES * runs)` over the run
    // containers, so recover the run count from it. With no run containers both terms are
    // zero and this answers false, which is the right answer -- there is nothing to strip.
    let run_containers = stats.n_run_containers as u64;
    let runs = (stats.n_bytes_run_containers - RUN_COUNT_BYTES * run_containers) / RUN_BYTES;
    runs > MAX_RUNS_PER_CONTAINER * run_containers
}

impl GroupRemap {
    fn new(input: GroupInput) -> Result<Self> {
        // Note the asymmetry between the two fragment lists. `old_frag_ids` carries no
        // ordering requirement beyond being the order the fragments were read: positions are
        // assigned by walking it, so any order is handled correctly and none is rejected.
        // `new_frags` is different, and is checked below.
        //
        // `compute_new_addr` maps a rewritten row's group-local index to a new
        // address by accumulating `physical_rows` in `new_frags` order, so that
        // order must be the order rows were written. New fragment ids are
        // reserved monotonically in write order (see `reserve_fragment_ids` in
        // compaction), so ascending id is a proxy for write order; reject any
        // input that violates it before it can silently misplace addresses.
        let mut new_frag_row_ranges = Vec::with_capacity(input.new_frags.len());
        let mut rewritten_rows_before = 0u64;
        let mut prev_frag_id: Option<u32> = None;
        for (frag_id, physical_rows) in input.new_frags {
            if physical_rows == 0 {
                continue;
            }
            if let Some(prev) = prev_frag_id
                && frag_id <= prev
            {
                return Err(Error::invalid_input(format!(
                    "compaction new fragments must be in ascending id (write) order, but fragment {frag_id} follows {prev}",
                )));
            }
            prev_frag_id = Some(frag_id);
            new_frag_row_ranges.push((frag_id, rewritten_rows_before, physical_rows));
            rewritten_rows_before += physical_rows as u64;
        }
        let total_new_rows = rewritten_rows_before;

        // Choose the encoding we keep resident, rather than inheriting the payload's.
        //
        // Compaction run-optimizes the address set before serializing it, which makes the
        // payload dramatically smaller on disk and is worth keeping. But `get` reaches
        // `RoaringBitmap::rank`, which sums `len()` over every container below the target,
        // and that is O(1) for bitmap and array containers while for run containers it
        // rescans the interval list. So a payload of many short runs -- what deletion
        // holes produce -- would make each lookup O(#containers x #runs), scaling with
        // fragment width in rows, when this structure exists to be bounded by fragment
        // count.
        //
        // Stripping unconditionally is not right either: a fragment rewritten whole is one
        // contiguous run per container, which ranks as cheaply as a cached read and costs
        // six bytes against a bitmap container's fixed 8 KiB. Hence the per-bitmap choice.
        //
        // Serialization is unaffected either way; this only changes what stays resident.
        let mut per_frag: HashMap<u32, RoaringBitmap> = input
            .rewritten_old_row_addrs
            .bitmaps()
            .map(|(frag_id, bitmap)| {
                let mut bitmap = bitmap.clone();
                if should_strip_runs(&bitmap) {
                    bitmap.remove_run_compression();
                }
                (frag_id, bitmap)
            })
            .collect();
        let mut frags = HashMap::new();
        let mut rewritten_rows_before = 0u64;
        for &frag_id in &input.old_frag_ids {
            // A fragment with no rewritten rows (fully deleted) contributes
            // nothing to the rewritten row sequence.
            if let Some(bitmap) = per_frag.remove(&frag_id) {
                let num_rewritten_rows = bitmap.len();
                frags.insert(frag_id, (bitmap, rewritten_rows_before));
                rewritten_rows_before += num_rewritten_rows;
            }
        }
        // Rewritten old row addresses must reference only fragments listed in `old_frag_ids`.
        if !per_frag.is_empty() {
            return Err(Error::invalid_input(format!(
                "compaction rewritten old row addresses reference fragments {:?} not in the rewrite group's old fragments {:?}",
                per_frag.keys().collect::<Vec<_>>(),
                input.old_frag_ids,
            )));
        }

        // Rewritten old rows are mapped positionally onto the new rows, so the
        // two counts must match exactly
        let total_rewritten_old_rows = input.rewritten_old_row_addrs.len();
        if total_new_rows != total_rewritten_old_rows {
            return Err(Error::invalid_input(format!(
                "compaction rewrote {total_rewritten_old_rows} old rows from fragments {:?} but the new fragments hold {total_new_rows} rows",
                input.old_frag_ids,
            )));
        }

        Ok(Self {
            frags,
            new_frag_row_ranges,
        })
    }

    fn compute_new_addr(&self, rewritten_row_index: u64) -> u64 {
        let idx =
            match self
                .new_frag_row_ranges
                .binary_search_by(|(_, rewritten_rows_before, _)| {
                    rewritten_rows_before.cmp(&rewritten_row_index)
                }) {
                Ok(i) => i,
                Err(i) => i - 1,
            };
        let (frag_id, rewritten_rows_before, _rows) = self.new_frag_row_ranges[idx];
        let offset = (rewritten_row_index - rewritten_rows_before) as u32;
        u64::from(RowAddress::new_from_parts(frag_id, offset))
    }

    /// Compute the new address for an old row in this group.
    /// Returns `None` if the old row was not rewritten.
    #[inline]
    fn get(&self, frag: u32, offset: u32) -> Option<u64> {
        match self.frags.get(&frag) {
            Some((bitmap, rewritten_rows_before)) if bitmap.contains(offset) => {
                let rewritten_row_index = rewritten_rows_before + bitmap.rank(offset) - 1;
                Some(self.compute_new_addr(rewritten_row_index))
            }
            _ => None,
        }
    }
}

/// Compact remap backed by per-group rewritten row bitmaps + new-fragment layouts.
#[derive(Clone, crate::deepsize::DeepSizeOf)]
pub struct CompactRowAddrRemap {
    groups: Vec<GroupRemap>,
    /// Old fragment id -> index into `groups`. Size is O(#fragments), not rows.
    frag_to_group: HashMap<u32, usize>,
}

impl CompactRowAddrRemap {
    fn new(groups: impl IntoIterator<Item = GroupInput>) -> Result<Self> {
        let mut frag_to_group = HashMap::new();
        let mut group_remaps = Vec::new();
        for input in groups {
            let gi = group_remaps.len();
            for &frag_id in &input.old_frag_ids {
                frag_to_group.insert(frag_id, gi);
            }
            group_remaps.push(GroupRemap::new(input)?);
        }
        Ok(Self {
            groups: group_remaps,
            frag_to_group,
        })
    }

    #[inline]
    pub fn get(&self, addr: u64) -> Option<Option<u64>> {
        let frag = (addr >> 32) as u32;
        // Not in any rewrite group -> unaffected by this remap.
        let gi = *self.frag_to_group.get(&frag)?;
        Some(self.groups[gi].get(frag, addr as u32))
    }

    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// Number of rewrite groups held.
    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }

    /// Number of old fragments covered. This is what the structure scales with.
    pub fn num_fragments(&self) -> usize {
        self.frag_to_group.len()
    }

    fn fully_deleted_fragments(&self) -> Option<RoaringBitmap> {
        // A group with any rewritten row moved at least one row.
        if self.groups.iter().any(|g| !g.frags.is_empty()) {
            return None;
        }
        Some(RoaringBitmap::from_iter(self.frag_to_group.keys().copied()))
    }
}

// `Debug` is written by hand rather than derived because the payload is bitmaps and
// hash maps of unbounded size: a derived impl would print the whole remap. The shape
// and scale are what a caller logging a remap actually wants.
impl std::fmt::Debug for RowAddrRemap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compact(compact) => f
                .debug_struct("RowAddrRemap::Compact")
                .field("groups", &compact.num_groups())
                .field("fragments", &compact.num_fragments())
                .finish(),
            Self::Direct(map) => f
                .debug_struct("RowAddrRemap::Direct")
                .field("entries", &map.len())
                .finish(),
        }
    }
}

impl crate::deepsize::DeepSizeOf for RowAddrRemap {
    fn deep_size_of_children(&self, cx: &mut crate::deepsize::Context) -> usize {
        match self {
            Self::Compact(compact) => compact.deep_size_of_children(cx),
            Self::Direct(map) => map.deep_size_of_children(cx),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(frag: u32, offset: u32) -> u64 {
        u64::from(RowAddress::new_from_parts(frag, offset))
    }

    #[test]
    fn test_compact_lookup() {
        // Group A: out-of-order old frags [4, 3], split new frags (11 empty),
        // some deletions. frag 4 (5 rows) keeps 0,2,4; frag 3 keeps 0,1, so the
        // rewritten rows (4,0)(4,2)(4,4)(3,0)(3,1) go to new frags 10(2), 12(3).
        // Group B is a fully-deleted fragment.
        let group_a = GroupInput {
            rewritten_old_row_addrs: RoaringTreemap::from_iter([
                addr(4, 0),
                addr(4, 2),
                addr(4, 4),
                addr(3, 0),
                addr(3, 1),
            ]),
            old_frag_ids: vec![4, 3],
            new_frags: vec![(10, 2), (11, 0), (12, 3)],
        };
        let group_b = GroupInput {
            rewritten_old_row_addrs: RoaringTreemap::new(),
            old_frag_ids: vec![7],
            new_frags: vec![],
        };
        let remap = RowAddrRemap::compact([group_a, group_b]).unwrap();

        // Moves, in rewrite order; frag 4 comes first despite the larger id.
        assert_eq!(remap.get(addr(4, 0)), Some(Some(addr(10, 0))));
        assert_eq!(remap.get(addr(4, 2)), Some(Some(addr(10, 1))));
        // Rank 2 skips the zero-row new fragment 11 and lands in fragment 12.
        assert_eq!(remap.get(addr(4, 4)), Some(Some(addr(12, 0))));
        assert_eq!(remap.get(addr(3, 0)), Some(Some(addr(12, 1))));
        assert_eq!(remap.get(addr(3, 1)), Some(Some(addr(12, 2))));
        // Deleted offsets inside a rewritten fragment.
        assert_eq!(remap.get(addr(4, 1)), Some(None));
        assert_eq!(remap.get(addr(4, 3)), Some(None));
        // Covered but fully-deleted fragment -> Some(None), not None.
        assert_eq!(remap.get(addr(7, 0)), Some(None));
        // Fragment in no group -> unaffected.
        assert_eq!(remap.get(addr(9, 0)), None);
        assert!(!remap.is_empty());
    }

    #[test]
    fn test_fragment_sets() {
        // No rewritten rows at all: every covered fragment is fully deleted.
        let dead = RowAddrRemap::compact([GroupInput {
            rewritten_old_row_addrs: RoaringTreemap::new(),
            old_frag_ids: vec![3, 7],
            new_frags: vec![],
        }])
        .unwrap();
        assert_eq!(
            dead.fully_deleted_fragments(),
            Some(RoaringBitmap::from_iter([3u32, 7u32]))
        );
        assert_eq!(
            dead.affected_fragments(),
            RoaringBitmap::from_iter([3u32, 7u32])
        );

        // At least one rewritten row -> not fully deleted, but both covered
        // fragments (including the fully-deleted frag 1) are still affected.
        let alive = RowAddrRemap::compact([GroupInput {
            rewritten_old_row_addrs: RoaringTreemap::from_iter([addr(0, 0)]),
            old_frag_ids: vec![0, 1],
            new_frags: vec![(10, 1)],
        }])
        .unwrap();
        assert!(alive.fully_deleted_fragments().is_none());
        assert_eq!(
            alive.affected_fragments(),
            RoaringBitmap::from_iter([0u32, 1u32])
        );
    }

    #[test]
    fn test_compact_rejects_rewritten_addrs_outside_old_frags() {
        // Rewritten addresses reference frag 5, not in old_frag_ids. The count
        // still matches (2 == 2), so only the per-fragment split catches it.
        let input = GroupInput {
            rewritten_old_row_addrs: RoaringTreemap::from_iter([addr(0, 0), addr(5, 0)]),
            old_frag_ids: vec![0],
            new_frags: vec![(10, 2)],
        };
        assert!(RowAddrRemap::compact([input]).is_err());
    }

    #[test]
    fn test_compact_rejects_new_frags_out_of_write_order() {
        // New fragments out of ascending id (write) order would make
        // `compute_new_addr` accumulate rows in the wrong order, silently
        // misplacing addresses. A zero-row fragment between them is ignored.
        let input = GroupInput {
            rewritten_old_row_addrs: RoaringTreemap::from_iter([addr(0, 0), addr(0, 1)]),
            old_frag_ids: vec![0],
            new_frags: vec![(12, 1), (11, 1)],
        };
        assert!(RowAddrRemap::compact([input]).is_err());
    }

    #[test]
    fn test_resident_bitmaps_are_not_run_encoded() {
        // Compaction run-optimizes the address set before writing it, so what arrives
        // here is run-encoded. Keeping that encoding resident makes `get` O(#containers x
        // #runs), because `RoaringBitmap::rank` sums `len()` over preceding containers and
        // `len()` rescans the interval list for run containers. Two fragments wide enough
        // to span several containers, with periodic holes so run-encoding is what
        // `optimize` picks.
        let mut rewritten = RoaringTreemap::new();
        let mut kept = 0u32;
        for frag in 0..2u32 {
            for offset in 0..200_000u32 {
                if offset % 128 < 125 {
                    rewritten.insert(addr(frag, offset));
                    kept += 1;
                }
            }
        }
        assert!(rewritten.optimize(), "payload should be run-encodable");
        for (frag_id, bitmap) in rewritten.bitmaps() {
            assert!(
                bitmap.statistics().n_run_containers > 0,
                "fragment {frag_id} should arrive run-encoded, or this test proves nothing"
            );
        }

        let remap = CompactRowAddrRemap::new([GroupInput {
            rewritten_old_row_addrs: rewritten,
            old_frag_ids: vec![0, 1],
            new_frags: vec![(10, kept)],
        }])
        .unwrap();

        for group in &remap.groups {
            for (frag_id, (bitmap, _)) in &group.frags {
                let stats = bitmap.statistics();
                assert_eq!(
                    stats.n_run_containers, 0,
                    "fragment {frag_id} kept {} run containers; `rank` would pay \
                     O(containers x runs) per lookup",
                    stats.n_run_containers
                );
                assert!(
                    stats.n_containers > 1,
                    "test needs a multi-container bitmap"
                );
            }
        }

        // Stripping the encoding must not move any address.
        assert_eq!(remap.get(addr(0, 0)), Some(Some(addr(10, 0))));
        assert_eq!(remap.get(addr(0, 124)), Some(Some(addr(10, 124))));
        assert_eq!(remap.get(addr(0, 125)), Some(None), "hole in the pattern");
        assert_eq!(remap.get(addr(0, 128)), Some(Some(addr(10, 125))));
        // First surviving offset of fragment 1 follows all of fragment 0's survivors.
        assert_eq!(remap.get(addr(1, 0)), Some(Some(addr(10, kept / 2))));
    }

    #[test]
    fn test_no_run_containers_needs_no_special_case() {
        // `should_strip_runs` has no zero guard: with no run containers both terms of the
        // comparison are zero. Pins that, since without it the byte arithmetic would be a
        // subtraction waiting to underflow.
        let scattered: Vec<u64> = (0..500u32).map(|i| addr(0, i * 977)).collect();
        let mut rewritten = RoaringTreemap::from_iter(scattered.iter().copied());
        rewritten.optimize();
        for (_, bitmap) in rewritten.bitmaps() {
            assert_eq!(
                bitmap.statistics().n_run_containers,
                0,
                "a scattered set should not be run-encoded, or this test proves nothing"
            );
            assert!(!should_strip_runs(bitmap), "nothing to strip");
        }

        let remap = CompactRowAddrRemap::new([GroupInput {
            rewritten_old_row_addrs: rewritten,
            old_frag_ids: vec![0],
            new_frags: vec![(10, scattered.len() as u32)],
        }])
        .unwrap();
        assert_eq!(remap.get(scattered[0]), Some(Some(addr(10, 0))));
        assert_eq!(remap.get(scattered[499]), Some(Some(addr(10, 499))));
        assert_eq!(
            remap.get(addr(0, 1)),
            Some(None),
            "not in the scattered set"
        );
    }

    #[test]
    fn test_run_encoding_kept_when_runs_are_few() {
        // The other side of `should_strip_runs`. A fragment rewritten whole is one
        // contiguous run per container, which ranks as cheaply as a cached `len()` read
        // and costs 6 bytes against a bitmap container's fixed 8 KiB. Stripping that
        // would be a regression on both axes, so it must be left alone.
        let rows = 200_000u32;
        let mut rewritten = RoaringTreemap::new();
        for offset in 0..rows {
            rewritten.insert(addr(0, offset));
        }
        assert!(rewritten.optimize());

        let remap = CompactRowAddrRemap::new([GroupInput {
            rewritten_old_row_addrs: rewritten,
            old_frag_ids: vec![0],
            new_frags: vec![(10, rows)],
        }])
        .unwrap();

        for group in &remap.groups {
            for (frag_id, (bitmap, _)) in &group.frags {
                let stats = bitmap.statistics();
                assert!(
                    stats.n_containers > 1,
                    "test needs a multi-container bitmap"
                );
                assert_eq!(
                    stats.n_run_containers, stats.n_containers,
                    "fragment {frag_id} should stay run-encoded: one run per container is \
                     both cheaper to rank and far smaller"
                );
            }
        }

        assert_eq!(remap.get(addr(0, 0)), Some(Some(addr(10, 0))));
        assert_eq!(remap.get(addr(0, rows - 1)), Some(Some(addr(10, rows - 1))));
    }

    #[test]
    fn test_deep_size_of_reaches_the_bitmaps() {
        use crate::deepsize::DeepSizeOf;

        // The derived impl has to walk Vec -> HashMap -> tuple -> RoaringBitmap. If any
        // link stopped short the two remaps below would report the same size, since they
        // differ only in how many rows their bitmaps hold.
        let remap_over = |rows: u32| {
            RowAddrRemap::compact([GroupInput {
                rewritten_old_row_addrs: RoaringTreemap::from_iter(
                    (0..rows).map(|offset| addr(0, offset)),
                ),
                old_frag_ids: vec![0],
                new_frags: vec![(10, rows)],
            }])
            .unwrap()
        };

        // 8 values fit an array container; 40k promote to a bitmap container.
        let small = remap_over(8).deep_size_of();
        let large = remap_over(40_000).deep_size_of();
        assert!(small > 0, "a non-empty remap must report a non-zero size");
        assert!(
            large > small,
            "size must grow with bitmap contents, got {large} vs {small}"
        );
    }

    #[test]
    fn test_direct_and_empty() {
        // Direct covers arbitrary maps the compact form can't express.
        let mut map = HashMap::new();
        map.insert(addr(2, 0), Some(addr(9, 9)));
        map.insert(addr(5, 1), None);
        let remap = RowAddrRemap::direct(map);
        assert_eq!(remap.get(addr(2, 0)), Some(Some(addr(9, 9))));
        assert_eq!(remap.get(addr(5, 1)), Some(None));
        assert_eq!(remap.get(addr(2, 1)), None);
        // affected_fragments over an explicit map: the fragment of every key.
        assert_eq!(
            remap.affected_fragments(),
            RoaringBitmap::from_iter([2u32, 5u32])
        );

        let empty = RowAddrRemap::empty();
        assert!(empty.is_empty());
        assert_eq!(empty.get(addr(0, 0)), None);
    }
}
