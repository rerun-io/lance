// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, sync::Arc};

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{Array, ArrayRef, PrimitiveArray, RecordBatch, UInt64Array};
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_core::{Error, Result};
use lance_select::RowAddrTreeMap;
use roaring::{RoaringBitmap, RoaringTreemap};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::format::pb::fragment_reuse_index_details::InlineContent;
use crate::format::{ExternalFile, Fragment, pb};

pub const FRAG_REUSE_INDEX_NAME: &str = "__lance_frag_reuse";
pub const FRAG_REUSE_DETAILS_FILE_NAME: &str = "details.binpb";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct FragDigest {
    pub id: u64,
    pub physical_rows: usize,
    pub num_deleted_rows: usize,
}

impl From<&FragDigest> for pb::fragment_reuse_index_details::FragmentDigest {
    fn from(digest: &FragDigest) -> Self {
        Self {
            id: digest.id,
            physical_rows: digest.physical_rows as u64,
            num_deleted_rows: digest.num_deleted_rows as u64,
        }
    }
}

impl From<&Fragment> for FragDigest {
    fn from(fragment: &Fragment) -> Self {
        Self {
            id: fragment.id,
            physical_rows: fragment
                .physical_rows
                .expect("Fragment doesn't have physical rows recorded"),
            num_deleted_rows: fragment
                .deletion_file
                .as_ref()
                .and_then(|d| d.num_deleted_rows)
                .unwrap_or(0),
        }
    }
}

impl TryFrom<pb::fragment_reuse_index_details::FragmentDigest> for FragDigest {
    type Error = Error;

    fn try_from(digest: pb::fragment_reuse_index_details::FragmentDigest) -> Result<Self> {
        Ok(Self {
            id: digest.id,
            physical_rows: digest.physical_rows as usize,
            num_deleted_rows: digest.num_deleted_rows as usize,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct FragReuseGroup {
    pub changed_row_addrs: Vec<u8>,
    pub old_frags: Vec<FragDigest>,
    pub new_frags: Vec<FragDigest>,
}

impl From<&FragReuseGroup> for pb::fragment_reuse_index_details::Group {
    fn from(group: &FragReuseGroup) -> Self {
        Self {
            changed_row_addrs: group.changed_row_addrs.clone(),
            old_fragments: group.old_frags.iter().map(|f| f.into()).collect(),
            new_fragments: group.new_frags.iter().map(|f| f.into()).collect(),
        }
    }
}

impl TryFrom<pb::fragment_reuse_index_details::Group> for FragReuseGroup {
    type Error = Error;

    fn try_from(group: pb::fragment_reuse_index_details::Group) -> Result<Self> {
        Ok(Self {
            changed_row_addrs: group.changed_row_addrs,
            old_frags: group
                .old_fragments
                .into_iter()
                .map(FragDigest::try_from)
                .collect::<Result<_>>()?,
            new_frags: group
                .new_fragments
                .into_iter()
                .map(FragDigest::try_from)
                .collect::<Result<_>>()?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct FragReuseVersion {
    pub dataset_version: u64,
    pub groups: Vec<FragReuseGroup>,
}

impl From<&FragReuseVersion> for pb::fragment_reuse_index_details::Version {
    fn from(version: &FragReuseVersion) -> Self {
        Self {
            dataset_version: version.dataset_version,
            groups: version.groups.iter().map(|g| g.into()).collect(),
        }
    }
}

impl TryFrom<pb::fragment_reuse_index_details::Version> for FragReuseVersion {
    type Error = Error;

    fn try_from(version: pb::fragment_reuse_index_details::Version) -> Result<Self> {
        Ok(Self {
            dataset_version: version.dataset_version,
            groups: version
                .groups
                .into_iter()
                .map(FragReuseGroup::try_from)
                .collect::<Result<_>>()?,
        })
    }
}

impl FragReuseVersion {
    pub fn old_frag_ids(&self) -> Vec<u64> {
        self.groups
            .iter()
            .flat_map(|g| g.old_frags.iter().map(|f| f.id))
            .collect::<Vec<_>>()
    }

    pub fn new_frag_ids(&self) -> Vec<u64> {
        self.groups
            .iter()
            .flat_map(|g| g.new_frags.iter().map(|f| f.id))
            .collect::<Vec<_>>()
    }

    pub fn new_frag_bitmap(&self) -> RoaringBitmap {
        RoaringBitmap::from_iter(self.new_frag_ids().iter().map(|&id| id as u32))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub enum FragReuseIndexDetailsContentType {
    Inline(FragReuseIndexDetails),
    External(ExternalFile),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct FragReuseIndexDetails {
    pub versions: Vec<FragReuseVersion>,
}

impl From<&FragReuseIndexDetails> for InlineContent {
    fn from(details: &FragReuseIndexDetails) -> Self {
        let mut versions: Vec<pb::fragment_reuse_index_details::Version> =
            details.versions.iter().map(|m| m.into()).collect();
        // sort from oldest to latest version
        versions.sort_by_key(|v| v.dataset_version);
        Self { versions }
    }
}

impl TryFrom<InlineContent> for FragReuseIndexDetails {
    type Error = Error;

    fn try_from(content: InlineContent) -> Result<Self> {
        Ok(Self {
            versions: content
                .versions
                .into_iter()
                .map(|m| m.try_into())
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

impl FragReuseIndexDetails {
    pub fn new_frag_bitmap(&self) -> RoaringBitmap {
        RoaringBitmap::from_iter(
            self.versions
                .iter()
                .flat_map(|v| v.new_frag_ids().into_iter().map(|id| id as u32)),
        )
    }
}

/// Which reuse versions can affect each fragment, so [`FragReuseIndex::remap_row_id`] can
/// visit only the versions that might move an address instead of all of them.
///
/// CSR layout: `slots` maps a fragment id to a `(start, len)` window into `version_indices`,
/// which holds that fragment's version indices in ascending order. One allocation for the
/// windows plus one for the indices, rather than a `Vec` per fragment -- most fragments are
/// touched by exactly one version.
///
/// Sized by (versions x affected fragments), not by rows.
// `PartialEq`/`Eq` so `FragReuseIndex` can keep deriving them. Redundant in substance --
// this is derived from `row_addr_maps`, so equal remaps always yield an equal index -- but
// the derive on the outer struct needs every field to participate.
#[derive(Clone, PartialEq, Eq, DeepSizeOf)]
struct VersionsByFragment {
    slots: HashMap<u32, (u32, u32)>,
    version_indices: Vec<u32>,
}

impl VersionsByFragment {
    fn new(row_addr_maps: &[RowAddrRemap]) -> Self {
        // Version indices and window bounds are stored as `u32`; a silent wrap here would
        // misdirect lookups. Both bounds are unreachable in practice -- production carries
        // tens of versions -- so panicking is the right answer if one is ever hit.
        assert!(
            u32::try_from(row_addr_maps.len()).is_ok(),
            "fragment reuse index has {} versions, more than a u32 index can address",
            row_addr_maps.len(),
        );

        // Materialize once: `affected_fragments` builds a fresh bitmap per call and both
        // passes below need it.
        let affected = row_addr_maps
            .iter()
            .map(|m| m.affected_fragments())
            .collect::<Vec<_>>();

        // Pass 1: count the versions touching each fragment.
        let mut slots: HashMap<u32, (u32, u32)> = HashMap::new();
        let mut total = 0usize;
        for bitmap in &affected {
            for frag in bitmap {
                slots.entry(frag).or_insert((0, 0)).1 += 1;
                total += 1;
            }
        }

        assert!(
            u32::try_from(total).is_ok(),
            "fragment reuse index has {total} (version, fragment) pairs, more than a u32 \
             offset can address",
        );

        // Hand out a contiguous window per fragment, resetting `len` to a fill cursor.
        let mut next_start = 0u32;
        for (start, len) in slots.values_mut() {
            *start = next_start;
            next_start += *len;
            *len = 0;
        }

        // Pass 2: fill. Versions are walked oldest-first, so each window ends up ascending,
        // and each fragment's cursor ends at its true length.
        let mut version_indices = vec![0u32; total];
        for (vi, bitmap) in affected.iter().enumerate() {
            for frag in bitmap {
                let (start, filled) = slots.get_mut(&frag).expect("counted in pass 1");
                version_indices[(*start + *filled) as usize] = vi as u32;
                *filled += 1;
            }
        }

        Self {
            slots,
            version_indices,
        }
    }

    /// Index of the oldest version at or after `from` whose remap can affect `frag`.
    #[inline]
    fn first_affecting(&self, frag: u32, from: u32) -> Option<u32> {
        let &(start, len) = self.slots.get(&frag)?;
        let window = &self.version_indices[start as usize..(start + len) as usize];
        window.get(window.partition_point(|&vi| vi < from)).copied()
    }
}

// Printed by shape, like `RowAddrRemap`: the payload is unbounded in fragment count.
impl std::fmt::Debug for VersionsByFragment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VersionsByFragment")
            .field("fragments", &self.slots.len())
            .field("entries", &self.version_indices.len())
            .finish()
    }
}

/// An index that stores row ID maps.
/// A row ID map describes the mapping from old row address to new address after compactions.
/// Each version contains the mapping for one round of compaction.
///
/// Equality compares the stored remaps as represented, not the addresses they resolve to:
/// see [`RowAddrRemap`]. Two indices built from the same payload in different forms are
/// unequal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FragReuseIndex {
    pub uuid: Uuid,
    /// One remap per reuse version, oldest first. Order is load-bearing: each version is
    /// applied to the previous version's output.
    ///
    /// Built by the open path in whichever form the reader asked for, defaulting to
    /// [`RowAddrRemap::Direct`]. A materialized map holds one entry per rewritten or deleted
    /// row, so it scales with the number of rows compaction has touched;
    /// [`RowAddrRemap::Compact`] scales with fragment count instead. The index is opened on
    /// the read path and the result cached, so readers pay whichever cost.
    ///
    /// Treat as read-only after construction: `versions_by_fragment` is derived from it and
    /// is not rebuilt, so mutating this in place would make `remap_row_id` skip versions.
    pub row_addr_maps: Vec<RowAddrRemap>,
    pub details: FragReuseIndexDetails,
    /// Derived from `row_addr_maps` at construction. See [`VersionsByFragment`].
    versions_by_fragment: VersionsByFragment,
}

impl DeepSizeOf for FragReuseIndex {
    fn deep_size_of_children(&self, cx: &mut Context) -> usize {
        self.row_addr_maps.deep_size_of_children(cx)
            + self.details.deep_size_of_children(cx)
            + self.versions_by_fragment.deep_size_of_children(cx)
    }
}

impl FragReuseIndex {
    /// Build from already-materialized maps, one per version.
    ///
    /// Kept for callers that already hold maps; it stores them as
    /// [`RowAddrRemap::Direct`], whose memory scales with the number of rows touched.
    /// Prefer [`Self::new_from_remaps`].
    pub fn new(
        uuid: Uuid,
        row_id_maps: Vec<HashMap<u64, Option<u64>>>,
        details: FragReuseIndexDetails,
    ) -> Self {
        Self::new_from_remaps(
            uuid,
            row_id_maps.into_iter().map(RowAddrRemap::direct).collect(),
            details,
        )
    }

    pub fn new_from_remaps(
        uuid: Uuid,
        row_addr_maps: Vec<RowAddrRemap>,
        details: FragReuseIndexDetails,
    ) -> Self {
        let versions_by_fragment = VersionsByFragment::new(&row_addr_maps);
        Self {
            uuid,
            row_addr_maps,
            details,
            versions_by_fragment,
        }
    }

    /// Walk `row_id` through every reuse version, oldest first, and return where it lands:
    /// `None` if some version deleted it, otherwise its current address (unchanged if no
    /// version moved it).
    ///
    /// Only the versions that can affect the address' current fragment are visited, which
    /// is typically one -- a row is compacted into a large fragment that later rounds leave
    /// alone. Skipping a version is sound because a version absent from
    /// [`RowAddrRemap::affected_fragments`] answers `None` for every address in that
    /// fragment, which the loop below treats as a no-op anyway.
    pub fn remap_row_id(&self, row_id: u64) -> Option<u64> {
        let mut addr = row_id;
        // Versions are applied in ascending order and never revisited: a remap can land an
        // address in a fragment an *earlier* version also rewrote, and replaying that
        // version would be both wrong and non-terminating.
        let mut next_version = 0u32;
        loop {
            let frag = (addr >> 32) as u32;
            let Some(vi) = self
                .versions_by_fragment
                .first_affecting(frag, next_version)
            else {
                return Some(addr);
            };
            match self.row_addr_maps[vi as usize].get(addr) {
                // The fragment is affected but this address is not.
                None => {}
                Some(None) => return None,
                Some(Some(new_addr)) => addr = new_addr,
            }
            // `vi >= next_version`, so this strictly advances and the loop is bounded by
            // the version count.
            next_version = vi + 1;
        }
    }

    pub fn remap_row_addrs_tree_map(&self, row_addrs: &RowAddrTreeMap) -> RowAddrTreeMap {
        RowAddrTreeMap::from_iter(row_addrs.row_addrs().unwrap().filter_map(|addr| {
            let addr_as_u64 = u64::from(addr);
            self.remap_row_id(addr_as_u64)
        }))
    }

    pub fn remap_row_ids_roaring_tree_map(&self, row_ids: &RoaringTreemap) -> RoaringTreemap {
        RoaringTreemap::from_iter(row_ids.iter().filter_map(|addr| self.remap_row_id(addr)))
    }

    /// Remap a record batch that contains a row_id column at index `row_id_idx`
    /// Currently this assumes there are only 2 columns in the schema,
    /// which is the case for all indexes.
    /// For example, for btree, the schema is (value, row_id).
    /// For vector index storage, the schema is (row_id, vector).
    pub fn remap_row_ids_record_batch(
        &self,
        batch: RecordBatch,
        row_id_idx: usize,
    ) -> Result<RecordBatch> {
        assert_eq!(batch.schema().fields().len(), 2);
        let other_column_idx = 1 - row_id_idx;
        let row_ids = batch.column(row_id_idx).as_primitive::<UInt64Type>();
        let (val_indices, new_row_ids): (Vec<u64>, Vec<u64>) = row_ids
            .values()
            .iter()
            .enumerate()
            .filter_map(|(idx, old_id)| {
                self.remap_row_id(*old_id)
                    .map(|new_id| (idx as u64, new_id))
            })
            .unzip();
        let new_val_indices = UInt64Array::from_iter_values(val_indices);
        let new_vals =
            arrow::compute::take(batch.column(other_column_idx), &new_val_indices, None)?;

        let mut batch_data: Vec<(usize, ArrayRef)> = vec![
            (
                row_id_idx,
                Arc::new(UInt64Array::from_iter_values(new_row_ids)) as ArrayRef,
            ),
            (other_column_idx, Arc::new(new_vals)),
        ];
        batch_data.sort_by_key(|(i, _)| *i);
        Ok(RecordBatch::try_new(
            batch.schema(),
            batch_data.into_iter().map(|(_, item)| item).collect(),
        )?)
    }

    pub fn remap_row_ids_array(&self, array: ArrayRef) -> PrimitiveArray<UInt64Type> {
        let primitive_array = array
            .as_any()
            .downcast_ref::<PrimitiveArray<UInt64Type>>()
            .expect("expected row IDs to be uint64 array");
        (0..primitive_array.len())
            .map(|i| {
                if primitive_array.is_null(i) {
                    None
                } else {
                    self.remap_row_id(primitive_array.value(i))
                }
            })
            .collect()
    }

    pub fn remap_fragment_bitmap(&self, fragment_bitmap: &mut RoaringBitmap) -> Result<()> {
        for version in self.details.versions.iter() {
            for group in version.groups.iter() {
                let mut removed = 0;
                for old_frag in group.old_frags.iter() {
                    if fragment_bitmap.remove(old_frag.id as u32) {
                        removed += 1;
                    }
                }

                if removed > 0 {
                    if removed != group.old_frags.len() {
                        // Straddle: the index covered only part of this rewrite
                        // group. Caused by the bug fixed in
                        // <https://github.com/lance-format/lance/pull/6610>.
                        // We've already removed the indexed old_frags from the
                        // bitmap above; deliberately do NOT insert new_frags,
                        // since the merged fragment also contains rows that
                        // were never indexed. Affected rows fall through to
                        // flat scan until the next optimize_indices. The fix
                        // is persisted on the next write via build_manifest.
                        tracing::warn!(
                            "Healing straddling fragment-reuse rewrite group in index bitmap: \
                             group {:?} was only partially indexed ({} of {} old fragments). \
                             Affected rows will use flat scan until the next optimize_indices.",
                            group.old_frags,
                            removed,
                            group.old_frags.len(),
                        );
                        continue;
                    }

                    for new_frag in group.new_frags.iter() {
                        fragment_bitmap.insert(new_frag.id as u32);
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use arrow_array::StringArray;
    use arrow_schema::{DataType, Field, Schema};
    use lance_core::utils::address::RowAddress;
    use lance_core::utils::row_addr_remap::GroupInput;
    use rand::{Rng, SeedableRng, rngs::SmallRng};
    use rstest::rstest;

    fn addr(frag: u32, offset: u32) -> u64 {
        u64::from(RowAddress::new_from_parts(frag, offset))
    }

    /// One compaction round: `rewritten` old addresses moving into `new_frags`, in order.
    fn compact_round(
        rewritten: &[(u32, u32)],
        old_frag_ids: Vec<u32>,
        new_frags: Vec<(u32, u32)>,
    ) -> RowAddrRemap {
        RowAddrRemap::compact([GroupInput {
            rewritten_old_row_addrs: RoaringTreemap::from_iter(
                rewritten.iter().map(|&(frag, offset)| addr(frag, offset)),
            ),
            old_frag_ids,
            new_frags,
        }])
        .unwrap()
    }

    /// `details` is left empty: nothing on these paths reads it, and keeping it in step
    /// with `maps` would add noise to every test below.
    fn index_from(maps: Vec<RowAddrRemap>) -> FragReuseIndex {
        FragReuseIndex::new_from_remaps(
            Uuid::new_v4(),
            maps,
            FragReuseIndexDetails { versions: vec![] },
        )
    }

    /// frag 0 offsets {0,2} -> frag 10; then frag 10 offset {0} -> frag 20.
    fn two_round_index() -> FragReuseIndex {
        index_from(vec![
            compact_round(&[(0, 0), (0, 2)], vec![0], vec![(10, 2)]),
            compact_round(&[(10, 0)], vec![10], vec![(20, 1)]),
        ])
    }

    #[test]
    fn test_remap_row_id_over_compact_rounds() {
        let index = two_round_index();
        // Moved twice.
        assert_eq!(index.remap_row_id(addr(0, 0)), Some(addr(20, 0)));
        // Moved by round 1, dropped by round 2.
        assert_eq!(index.remap_row_id(addr(0, 2)), None);
        // Deleted before round 1 ever ran.
        assert_eq!(index.remap_row_id(addr(0, 1)), None);
        // Untouched by both rounds.
        assert_eq!(index.remap_row_id(addr(5, 5)), Some(addr(5, 5)));
    }

    /// The pre-`VersionsByFragment` loop, kept verbatim as the oracle for the fuzz below.
    fn remap_row_id_reference(index: &FragReuseIndex, row_id: u64) -> Option<u64> {
        let mut mapped_value = Some(row_id);
        for row_addr_map in index.row_addr_maps.iter() {
            if mapped_value.is_some() {
                mapped_value = row_addr_map
                    .get(mapped_value.unwrap())
                    .unwrap_or(mapped_value);
            }
        }

        mapped_value
    }

    /// How many versions actually answered for `row_id` along the reference walk. Used only
    /// to prove the fuzz reached multi-hop chains rather than only trivial addresses.
    fn reference_applied_versions(index: &FragReuseIndex, row_id: u64) -> usize {
        let mut mapped_value = Some(row_id);
        let mut applied = 0;
        for row_addr_map in index.row_addr_maps.iter() {
            let Some(addr) = mapped_value else { break };
            if let Some(answer) = row_addr_map.get(addr) {
                applied += 1;
                mapped_value = answer;
            }
        }
        applied
    }

    #[test]
    fn test_multi_hop_chain_applies_every_later_version() {
        // A row moved by version 1 lands in a fragment version 3 rewrites again. Versions 0
        // and 2 touch unrelated fragments, so the fragment index has to skip them without
        // losing the second hop -- the case a naive "stop at the first match" index breaks.
        let index = index_from(vec![
            compact_round(&[(90, 0)], vec![90], vec![(91, 1)]),
            compact_round(&[(0, 0)], vec![0], vec![(10, 1)]),
            compact_round(&[(92, 0)], vec![92], vec![(93, 1)]),
            compact_round(&[(10, 0)], vec![10], vec![(20, 1)]),
        ]);

        assert_eq!(index.remap_row_id(addr(0, 0)), Some(addr(20, 0)));
        assert_eq!(
            remap_row_id_reference(&index, addr(0, 0)),
            Some(addr(20, 0))
        );
        assert_eq!(reference_applied_versions(&index, addr(0, 0)), 2);
    }

    #[test]
    fn test_remap_into_an_earlier_versions_fragment_does_not_replay_it() {
        // Version 1 moves a row into fragment 5, which version 0 rewrote. Version 0 must not
        // be applied to it: versions are ordered, and replaying one would both change the
        // answer and let the walk cycle.
        let index = index_from(vec![
            compact_round(&[(5, 0)], vec![5], vec![(7, 1)]),
            compact_round(&[(3, 0)], vec![3], vec![(5, 1)]),
        ]);

        assert_eq!(index.remap_row_id(addr(3, 0)), Some(addr(5, 0)));
        assert_eq!(remap_row_id_reference(&index, addr(3, 0)), Some(addr(5, 0)));
    }

    #[test]
    fn test_deletion_short_circuits_later_versions() {
        // Offset 1 is inside a rewritten fragment but not rewritten, so version 0 reports it
        // deleted. Version 1 would map that same address somewhere real; it must never run.
        let index = index_from(vec![
            compact_round(&[(0, 0)], vec![0], vec![(10, 1)]),
            RowAddrRemap::direct(HashMap::from_iter([(addr(0, 1), Some(addr(30, 0)))])),
        ]);

        assert_eq!(index.remap_row_id(addr(0, 1)), None);
        assert_eq!(remap_row_id_reference(&index, addr(0, 1)), None);
        // The version that would have resurrected it is reachable for other addresses, or
        // this test would pass against an index that simply lost version 1.
        assert_eq!(index.remap_row_id(addr(0, 3)), None);
        assert_eq!(
            index.versions_by_fragment.first_affecting(0, 1),
            Some(1),
            "version 1 does cover fragment 0"
        );
    }

    #[test]
    fn test_untouched_fragment_costs_no_version_probes() {
        let index = two_round_index();
        assert_eq!(index.remap_row_id(addr(5, 5)), Some(addr(5, 5)));
        // The observable half: no version covers fragment 5, so the walk returns without
        // consulting a single `RowAddrRemap`.
        assert_eq!(index.versions_by_fragment.first_affecting(5, 0), None);
        // Fragments that are covered still resolve to the one version that covers them.
        assert_eq!(index.versions_by_fragment.first_affecting(0, 0), Some(0));
        assert_eq!(index.versions_by_fragment.first_affecting(10, 0), Some(1));
        // ...and asking past it finds nothing.
        assert_eq!(index.versions_by_fragment.first_affecting(0, 1), None);
    }

    /// Fragment ids and row offsets the fuzz draws from. Small enough that versions collide,
    /// chain into each other, and revisit fragments earlier versions already rewrote.
    const FUZZ_FRAGS: u32 = 10;
    const FUZZ_ROWS: u32 = 5;

    /// `n` fragment ids not already in `used`, appended to it.
    fn sample_frags(rng: &mut SmallRng, n: usize, used: &mut Vec<u32>) -> Vec<u32> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..(n * 8) {
            if out.len() == n {
                break;
            }
            let frag = rng.random_range(0..FUZZ_FRAGS);
            if !used.contains(&frag) {
                used.push(frag);
                out.push(frag);
            }
        }
        out
    }

    fn random_compact(rng: &mut SmallRng) -> RowAddrRemap {
        // Old fragments are drawn without replacement across groups: a fragment in two
        // groups of one remap is malformed input, not a case worth fuzzing.
        let mut used_old = Vec::new();
        let mut groups = Vec::new();
        for _ in 0..rng.random_range(1..=2) {
            let wanted_old = rng.random_range(1..=2);
            let old_frag_ids = sample_frags(rng, wanted_old, &mut used_old);
            if old_frag_ids.is_empty() {
                continue;
            }
            // Offsets left out are the rows the rewrite deleted.
            let mut rewritten = RoaringTreemap::new();
            for &frag in &old_frag_ids {
                for offset in 0..FUZZ_ROWS {
                    if rng.random_bool(0.6) {
                        rewritten.insert(addr(frag, offset));
                    }
                }
            }
            // New fragments come from the same id space, so later versions rewrite them.
            let total = rewritten.len() as u32;
            let mut new_frags = Vec::new();
            if total > 0 {
                let wanted = (rng.random_range(1..=2) as u32).min(total) as usize;
                let mut ids = sample_frags(rng, wanted, &mut Vec::new());
                ids.sort_unstable();
                let parts = ids.len() as u32;
                for (i, id) in ids.into_iter().enumerate() {
                    let rows = total / parts + u32::from((i as u32) < total % parts);
                    new_frags.push((id, rows));
                }
            }
            groups.push(GroupInput {
                rewritten_old_row_addrs: rewritten,
                old_frag_ids,
                new_frags,
            });
        }
        RowAddrRemap::compact(groups).unwrap()
    }

    fn random_direct(rng: &mut SmallRng) -> RowAddrRemap {
        let mut map = HashMap::new();
        for _ in 0..rng.random_range(0..8) {
            let key = addr(
                rng.random_range(0..FUZZ_FRAGS),
                rng.random_range(0..FUZZ_ROWS),
            );
            let value = (!rng.random_bool(0.3)).then(|| {
                addr(
                    rng.random_range(0..FUZZ_FRAGS),
                    rng.random_range(0..FUZZ_ROWS),
                )
            });
            map.insert(key, value);
        }
        RowAddrRemap::direct(map)
    }

    #[test]
    fn test_remap_row_id_matches_reference_over_random_indices() {
        const TRIALS: u64 = 3000;

        let mut probes = 0u64;
        let mut moved = 0u64;
        let mut deleted = 0u64;
        let mut chained = 0u64;

        for seed in 0..TRIALS {
            let mut rng = SmallRng::seed_from_u64(seed);
            let maps = (0..rng.random_range(0..8))
                .map(|_| {
                    if rng.random_bool(0.5) {
                        random_direct(&mut rng)
                    } else {
                        random_compact(&mut rng)
                    }
                })
                .collect::<Vec<_>>();
            let index = index_from(maps);

            // The index's own shape: windows must be non-empty and strictly ascending, or
            // `first_affecting`'s `partition_point` would return the wrong version.
            for &(start, len) in index.versions_by_fragment.slots.values() {
                let window =
                    &index.versions_by_fragment.version_indices[start as usize..][..len as usize];
                assert!(len > 0, "seed {seed}: empty window");
                assert!(
                    window.windows(2).all(|pair| pair[0] < pair[1]),
                    "seed {seed}: window {window:?} is not strictly ascending"
                );
            }

            // Probe past both bounds so unaffected fragments and unaffected offsets inside
            // affected fragments are covered, plus one address in no fragment at all.
            let addrs = (0..=FUZZ_FRAGS + 1)
                .flat_map(|frag| (0..=FUZZ_ROWS + 1).map(move |offset| addr(frag, offset)))
                .chain([addr(1_000_000, 7)]);
            for probe in addrs {
                let expected = remap_row_id_reference(&index, probe);
                assert_eq!(
                    index.remap_row_id(probe),
                    expected,
                    "seed {seed}, addr {probe:#x}, index {index:?}"
                );

                // Requirement the whole optimization rests on: any version that answers for
                // an address must be listed against that address' fragment. Checked against
                // the addresses actually reached, not just the starting one.
                let mut cur = Some(probe);
                for (vi, map) in index.row_addr_maps.iter().enumerate() {
                    let Some(a) = cur else { break };
                    if let Some(answer) = map.get(a) {
                        let frag = (a >> 32) as u32;
                        let listed = index
                            .versions_by_fragment
                            .slots
                            .get(&frag)
                            .map(|&(start, len)| {
                                index.versions_by_fragment.version_indices[start as usize..]
                                    [..len as usize]
                                    .contains(&(vi as u32))
                            })
                            .unwrap_or(false);
                        assert!(
                            listed,
                            "seed {seed}: version {vi} answers for fragment {frag} but is not \
                             indexed against it"
                        );
                        cur = answer;
                    }
                }

                probes += 1;
                match expected {
                    None => deleted += 1,
                    Some(landed) if landed != probe => moved += 1,
                    Some(_) => {}
                }
                if reference_applied_versions(&index, probe) >= 2 {
                    chained += 1;
                }
            }
        }

        // Without these the run could be thousands of no-op indices proving nothing.
        assert!(probes > 100_000, "only {probes} probes");
        assert!(moved > 1_000, "only {moved} addresses moved");
        assert!(deleted > 1_000, "only {deleted} addresses deleted");
        assert!(chained > 1_000, "only {chained} multi-version chains");
    }

    #[test]
    fn test_chaining_works_with_a_direct_link() {
        // Nothing in-tree builds a mixed chain -- `new` produces all `Direct`, the open
        // path all `Compact` -- but `new_from_remaps` and the public field permit one, and
        // the walk must not care which form a link takes. Asserted against absolute
        // expectations: a differential check against an all-compact chain would also pass
        // against a stubbed `remap_row_id`.
        let mixed = index_from(vec![
            compact_round(&[(0, 0), (0, 2)], vec![0], vec![(10, 2)]),
            RowAddrRemap::direct(HashMap::from_iter([
                (addr(10, 0), Some(addr(20, 0))),
                (addr(10, 1), None),
            ])),
        ]);

        assert_eq!(mixed.remap_row_id(addr(0, 0)), Some(addr(20, 0)));
        assert_eq!(mixed.remap_row_id(addr(0, 2)), None);
        assert_eq!(mixed.remap_row_id(addr(0, 1)), None);
        assert_eq!(mixed.remap_row_id(addr(5, 5)), Some(addr(5, 5)));
        // The two forms are not interchangeable, only chainable: a `Direct` link knows only
        // the addresses it lists, so an unlisted offset in a covered fragment is untouched,
        // where the compact form would report it deleted.
        assert_eq!(mixed.remap_row_id(addr(10, 2)), Some(addr(10, 2)));
    }

    /// `row_id_idx` is 1 for scalar indices (`btree.rs`) and 0 for vector storage
    /// (`vector/flat/storage.rs`, `vector/sq/storage.rs`). Both layouts are in use, and the
    /// method's only non-trivial logic is the `1 - row_id_idx` swap plus the `take`.
    #[rstest]
    #[case::row_id_last(1)]
    #[case::row_id_first(0)]
    fn test_remap_row_ids_record_batch_keeps_values_paired(#[case] row_id_idx: usize) {
        let index = two_round_index();
        // (0,0) survives to (20,0); (0,2) is dropped by round 2; (5,5) passes through.
        let values: Arc<dyn Array> = Arc::new(StringArray::from(vec!["keep", "drop", "pass"]));
        let row_ids: Arc<dyn Array> =
            Arc::new(UInt64Array::from(vec![addr(0, 0), addr(0, 2), addr(5, 5)]));
        let (value_field, columns) = if row_id_idx == 0 {
            (1, vec![row_ids, values])
        } else {
            (0, vec![values, row_ids])
        };
        let mut fields = vec![
            Field::new("value", DataType::Utf8, false),
            Field::new("row_id", DataType::UInt64, false),
        ];
        if row_id_idx == 0 {
            fields.swap(0, 1);
        }
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();

        let remapped = index.remap_row_ids_record_batch(batch, row_id_idx).unwrap();
        assert_eq!(remapped.num_rows(), 2);
        let out_values = remapped
            .column(value_field)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let out_row_ids = remapped
            .column(row_id_idx)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        // Each surviving value must still sit beside its own remapped address.
        assert_eq!(out_values.value(0), "keep");
        assert_eq!(out_row_ids.value(0), addr(20, 0));
        assert_eq!(out_values.value(1), "pass");
        assert_eq!(out_row_ids.value(1), addr(5, 5));
    }

    #[test]
    fn test_remap_row_addrs_tree_map_drops_deleted_rows() {
        // The wrapper with the most callers in-tree (bitmap, rtree and label_list indices).
        let index = two_round_index();
        let input = RowAddrTreeMap::from_iter([addr(0, 0), addr(0, 2), addr(5, 5)]);
        assert_eq!(
            index.remap_row_addrs_tree_map(&input),
            RowAddrTreeMap::from_iter([addr(20, 0), addr(5, 5)])
        );
    }

    #[test]
    fn test_remap_roaring_tree_map_drops_deleted_rows() {
        let index = two_round_index();
        let input = RoaringTreemap::from_iter([addr(0, 0), addr(0, 2), addr(5, 5)]);
        assert_eq!(
            index.remap_row_ids_roaring_tree_map(&input),
            RoaringTreemap::from_iter([addr(20, 0), addr(5, 5)])
        );
    }

    #[tokio::test]
    async fn test_serialize_deserialize_index_details() {
        // Create sample FragReuseVersions with different dataset versions
        let version1 = FragReuseVersion {
            dataset_version: 2,
            groups: vec![FragReuseGroup {
                changed_row_addrs: vec![1, 2, 3],
                old_frags: vec![FragDigest {
                    id: 1,
                    physical_rows: 1,
                    num_deleted_rows: 0,
                }],
                new_frags: vec![
                    FragDigest {
                        id: 2,
                        physical_rows: 1,
                        num_deleted_rows: 0,
                    },
                    FragDigest {
                        id: 3,
                        physical_rows: 1,
                        num_deleted_rows: 0,
                    },
                ],
            }],
        };

        let version2 = FragReuseVersion {
            dataset_version: 1,
            groups: vec![FragReuseGroup {
                changed_row_addrs: vec![4, 5, 6],
                old_frags: vec![FragDigest {
                    id: 2,
                    physical_rows: 1,
                    num_deleted_rows: 0,
                }],
                new_frags: vec![
                    FragDigest {
                        id: 4,
                        physical_rows: 1,
                        num_deleted_rows: 0,
                    },
                    FragDigest {
                        id: 5,
                        physical_rows: 1,
                        num_deleted_rows: 0,
                    },
                ],
            }],
        };

        // Create FragReuseIndexDetails with versions in reverse order
        let details = FragReuseIndexDetails {
            versions: vec![version1, version2],
        };

        // Convert to protobuf format
        let inline_content: InlineContent = (&details).into();

        // Convert back to FragReuseIndexDetails
        let roundtrip_details = FragReuseIndexDetails::try_from(inline_content).unwrap();

        // Verify the roundtrip
        assert_eq!(roundtrip_details.versions.len(), 2);

        // Verify versions are sorted by dataset_version (oldest to latest)
        assert_eq!(roundtrip_details.versions[0].dataset_version, 1);
        assert_eq!(
            roundtrip_details.versions[0].groups[0].changed_row_addrs,
            vec![4, 5, 6]
        );
        assert_eq!(
            roundtrip_details.versions[0].groups[0].new_frags,
            vec![
                FragDigest {
                    id: 4,
                    physical_rows: 1,
                    num_deleted_rows: 0,
                },
                FragDigest {
                    id: 5,
                    physical_rows: 1,
                    num_deleted_rows: 0,
                }
            ]
        );
        assert_eq!(
            roundtrip_details.versions[0].groups[0].old_frags,
            vec![FragDigest {
                id: 2,
                physical_rows: 1,
                num_deleted_rows: 0,
            }]
        );

        assert_eq!(roundtrip_details.versions[1].dataset_version, 2);
        assert_eq!(
            roundtrip_details.versions[1].groups[0].changed_row_addrs,
            vec![1, 2, 3]
        );
        assert_eq!(
            roundtrip_details.versions[1].groups[0].new_frags,
            vec![
                FragDigest {
                    id: 2,
                    physical_rows: 1,
                    num_deleted_rows: 0,
                },
                FragDigest {
                    id: 3,
                    physical_rows: 1,
                    num_deleted_rows: 0,
                }
            ]
        );
        assert_eq!(
            roundtrip_details.versions[1].groups[0].old_frags,
            vec![FragDigest {
                id: 1,
                physical_rows: 1,
                num_deleted_rows: 0,
            }]
        );
    }
}
