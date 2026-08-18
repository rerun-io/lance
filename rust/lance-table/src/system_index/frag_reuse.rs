// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, sync::Arc};

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{Array, ArrayRef, PrimitiveArray, RecordBatch, UInt64Array};
use lance_core::deepsize::{Context, DeepSizeOf};
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

/// A maximal span over which consecutive old row addresses map to consecutive new ones.
///
/// Compaction copies surviving rows out of the old fragments and into the new ones in order,
/// so the address mapping is piecewise affine. A span ends at a deleted row, at an old
/// fragment boundary, or at a new fragment boundary, and nowhere else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct RemapRun {
    /// First old row address in the span.
    pub old_start: u64,
    /// Address the first row of the span moved to.
    pub new_start: u64,
    /// How many consecutive addresses the span covers.
    ///
    /// A span cannot cross a fragment boundary on either side, and a fragment holds fewer
    /// than `2^32` rows, so this always fits.
    pub len: u32,
}

/// One reuse version's old-to-new row address mapping, stored as runs.
///
/// This replaces a `HashMap<u64, Option<u64>>` holding one entry per remapped row. The
/// hashmap form costs about 40 bytes per row resident, so a single compaction of a large
/// table can cost tens of gibibytes, paid by every reader that opens the index. The runs
/// form holds the same information in a few entries per fragment.
///
/// [`Self::get`] answers with the same three outcomes the hashmap did, and callers must keep
/// distinguishing them:
///
/// * `Some(Some(addr))`, the row moved,
/// * `Some(None)`, the row was deleted by the compaction,
/// * `None`, this version never saw the address, so the caller keeps it unchanged.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct RowAddrRunMap {
    /// Sorted by `old_start`, disjoint.
    runs: Vec<RemapRun>,
    /// Inclusive address ranges `[start, last]` this version deleted. Sorted, disjoint.
    ///
    /// Inclusive rather than half-open so that `u64::MAX`, which is a real address
    /// ([`lance_core::utils::address::RowAddress::TOMBSTONE_ROW`]), is representable.
    deleted: Vec<(u64, u64)>,
}

impl RowAddrRunMap {
    /// Equivalent of `HashMap::get(&addr).copied()`; see the type docs for the three outcomes.
    #[inline]
    pub fn get(&self, addr: u64) -> Option<Option<u64>> {
        let i = self.runs.partition_point(|run| run.old_start <= addr);
        if i > 0 {
            let run = &self.runs[i - 1];
            if addr - run.old_start < run.len as u64 {
                return Some(Some(run.new_start + (addr - run.old_start)));
            }
        }
        let j = self.deleted.partition_point(|(start, _)| *start <= addr);
        if j > 0 && addr <= self.deleted[j - 1].1 {
            return Some(None);
        }
        None
    }

    /// True when this version maps nothing at all.
    ///
    /// Note this is not the same question as "are there no reuse versions". A version that
    /// deleted every row it touched has no runs but is not empty.
    pub fn is_empty(&self) -> bool {
        self.runs.is_empty() && self.deleted.is_empty()
    }

    /// Every old address this version has an answer for, ascending.
    ///
    /// Replaces `HashMap::keys()`. Note this is O(rows), so it is only for callers that
    /// genuinely need to enumerate; prefer [`Self::get`].
    pub fn iter_keys(&self) -> impl Iterator<Item = u64> + '_ {
        let mapped = self
            .runs
            .iter()
            .flat_map(|run| run.old_start..run.old_start + run.len as u64);
        let deleted = self.deleted.iter().flat_map(|(start, last)| *start..=*last);
        mapped.chain(deleted)
    }

    /// Number of addresses this version answers for. Equivalent to `HashMap::len()`.
    pub fn len(&self) -> u64 {
        let mapped: u64 = self.runs.iter().map(|run| run.len as u64).sum();
        let deleted: u64 = self
            .deleted
            .iter()
            .map(|(start, last)| last - start + 1)
            .sum();
        mapped + deleted
    }

    /// Number of runs held, for tests and diagnostics.
    pub fn num_runs(&self) -> usize {
        self.runs.len()
    }
}

impl From<HashMap<u64, Option<u64>>> for RowAddrRunMap {
    /// Run-encode an already-materialized map.
    ///
    /// Only for callers that already hold one; the point of this type is to avoid building
    /// it at all. Prefer [`RowAddrRunMapBuilder`].
    fn from(map: HashMap<u64, Option<u64>>) -> Self {
        let mut mapped: Vec<(u64, u64)> = Vec::new();
        let mut deleted: Vec<u64> = Vec::new();
        for (old, new) in map {
            match new {
                Some(new) => mapped.push((old, new)),
                None => deleted.push(old),
            }
        }
        mapped.sort_unstable();
        deleted.sort_unstable();

        let mut builder = RowAddrRunMapBuilder::default();
        for (old, new) in mapped {
            builder.push_mapped(old, new);
        }
        for old in deleted {
            builder.push_deleted(old);
        }
        // Sorted input cannot overlap, so this cannot fail.
        builder
            .finish()
            .expect("run map from a HashMap cannot overlap")
    }
}

/// Accumulates a [`RowAddrRunMap`] from the same two streams that used to be inserted into
/// the hashmap: the mapped pairs, and the deleted addresses.
///
/// Feeding the identical streams is what makes the result bit-for-bit equivalent, including
/// any quirk in how the caller decides which addresses count as deleted.
#[derive(Debug, Default)]
pub struct RowAddrRunMapBuilder {
    runs: Vec<RemapRun>,
    deleted: Vec<(u64, u64)>,
}

impl RowAddrRunMapBuilder {
    /// Record that `old` moved to `new`. Ascending within a group; groups may arrive in any
    /// order.
    pub fn push_mapped(&mut self, old: u64, new: u64) {
        if let Some(run) = self.runs.last_mut()
            && old == run.old_start + run.len as u64
            && new == run.new_start + run.len as u64
            && run.len < u32::MAX
        {
            run.len += 1;
            return;
        }
        self.runs.push(RemapRun {
            old_start: old,
            new_start: new,
            len: 1,
        });
    }

    /// Record that `old` was deleted.
    pub fn push_deleted(&mut self, old: u64) {
        if let Some(range) = self.deleted.last_mut()
            && range.1 < u64::MAX
            && old == range.1 + 1
        {
            range.1 = old;
            return;
        }
        self.deleted.push((old, old));
    }

    /// Sort and validate. Errors if two spans claim the same address, which would make the
    /// lookup order-dependent where the hashmap was last-write-wins.
    pub fn finish(mut self) -> Result<RowAddrRunMap> {
        self.runs.sort_unstable_by_key(|run| run.old_start);
        self.deleted.sort_unstable();
        // Half-open ends for the overlap check, saturating so the tombstone address at
        // `u64::MAX` cannot overflow. Saturation only collapses a span that already runs to
        // the end of the address space, and nothing can start above it.
        let run_ends =
            |run: &RemapRun| (run.old_start, run.old_start.saturating_add(run.len as u64));
        let deleted_ends = |(start, last): &(u64, u64)| (*start, last.saturating_add(1));
        check_disjoint(self.runs.iter().map(run_ends), "mapped")?;
        check_disjoint(self.deleted.iter().map(deleted_ends), "deleted")?;
        // A run and a deletion overlapping would make `get` order-dependent too.
        let mut merged: Vec<(u64, u64)> = self
            .runs
            .iter()
            .map(run_ends)
            .chain(self.deleted.iter().map(deleted_ends))
            .collect();
        merged.sort_unstable();
        check_disjoint(merged.into_iter(), "mapped and deleted")?;
        self.runs.shrink_to_fit();
        self.deleted.shrink_to_fit();
        Ok(RowAddrRunMap {
            runs: self.runs,
            deleted: self.deleted,
        })
    }
}

fn check_disjoint(ranges: impl Iterator<Item = (u64, u64)>, what: &str) -> Result<()> {
    let mut prev_end = 0u64;
    let mut first = true;
    for (start, end) in ranges {
        if !first && start < prev_end {
            return Err(Error::invalid_input(format!(
                "fragment reuse index: overlapping {what} row address spans \
                 (span starting at {start} overlaps a span ending at {prev_end})"
            )));
        }
        first = false;
        prev_end = end;
    }
    Ok(())
}

/// An index that stores row ID maps.
/// A row ID map describes the mapping from old row address to new address after compactions.
/// Each version contains the mapping for one round of compaction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FragReuseIndex {
    pub uuid: Uuid,
    /// One entry per reuse version, oldest first. Order is load-bearing: each version is
    /// applied to the previous version's output.
    pub row_addr_maps: Vec<RowAddrRunMap>,
    pub details: FragReuseIndexDetails,
}

impl DeepSizeOf for FragReuseIndex {
    fn deep_size_of_children(&self, cx: &mut Context) -> usize {
        self.row_addr_maps.deep_size_of_children(cx) + self.details.deep_size_of_children(cx)
    }
}

impl FragReuseIndex {
    pub fn new(
        uuid: Uuid,
        row_id_maps: Vec<HashMap<u64, Option<u64>>>,
        details: FragReuseIndexDetails,
    ) -> Self {
        Self::new_from_run_maps(
            uuid,
            row_id_maps.into_iter().map(RowAddrRunMap::from).collect(),
            details,
        )
    }

    pub fn new_from_run_maps(
        uuid: Uuid,
        row_addr_maps: Vec<RowAddrRunMap>,
        details: FragReuseIndexDetails,
    ) -> Self {
        Self {
            uuid,
            row_addr_maps,
            details,
        }
    }

    pub fn remap_row_id(&self, row_id: u64) -> Option<u64> {
        let mut mapped_value = Some(row_id);
        for row_addr_map in self.row_addr_maps.iter() {
            if mapped_value.is_some() {
                mapped_value = row_addr_map
                    .get(mapped_value.unwrap())
                    .unwrap_or(mapped_value);
            }
        }

        mapped_value
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

#[cfg(test)]
mod run_map_tests {
    use super::*;

    /// Build both representations from the same pairs and assert they answer identically
    /// for every address in `probes`.
    fn assert_equivalent(pairs: &[(u64, Option<u64>)], probes: &[u64]) {
        let hash: HashMap<u64, Option<u64>> = pairs.iter().copied().collect();
        let runs = RowAddrRunMap::from(hash.clone());
        for &addr in probes {
            assert_eq!(
                hash.get(&addr).copied(),
                runs.get(addr),
                "addr {addr} diverged"
            );
        }
        assert_eq!(hash.len() as u64, runs.len());
        let mut from_runs: Vec<u64> = runs.iter_keys().collect();
        from_runs.sort_unstable();
        let mut from_hash: Vec<u64> = hash.keys().copied().collect();
        from_hash.sort_unstable();
        assert_eq!(from_hash, from_runs, "iter_keys must match HashMap::keys");
    }

    fn addr(frag: u64, offset: u32) -> u64 {
        (frag << 32) | offset as u64
    }

    #[test]
    fn contiguous_span_collapses_to_one_run() {
        let pairs: Vec<_> = (0..1000).map(|i| (addr(7, i), Some(addr(9, i)))).collect();
        let runs = RowAddrRunMap::from(pairs.iter().copied().collect::<HashMap<_, _>>());
        assert_eq!(runs.num_runs(), 1, "a contiguous span is one run");
        let probes: Vec<u64> = (0..1002).map(|i| addr(7, i)).collect();
        assert_equivalent(&pairs, &probes);
    }

    #[test]
    fn deletion_splits_a_run_and_reports_deleted() {
        let mut pairs = vec![];
        for i in 0..10u32 {
            if i == 4 || i == 5 {
                pairs.push((addr(1, i), None));
            } else {
                pairs.push((addr(1, i), Some(addr(2, i))));
            }
        }
        let runs = RowAddrRunMap::from(pairs.iter().copied().collect::<HashMap<_, _>>());
        assert_eq!(runs.num_runs(), 2, "one gap splits the span in two");
        assert_eq!(runs.get(addr(1, 4)), Some(None), "deleted, not absent");
        assert_eq!(runs.get(addr(1, 20)), None, "absent, not deleted");
        let probes: Vec<u64> = (0..24).map(|i| addr(1, i)).collect();
        assert_equivalent(&pairs, &probes);
    }

    /// The three outcomes are distinct and callers depend on it: absent means "keep the
    /// address", deleted means "the row is gone".
    #[test]
    fn absent_is_not_deleted() {
        let map = RowAddrRunMap::from(HashMap::from([(addr(1, 0), None)]));
        assert_eq!(map.get(addr(1, 0)), Some(None));
        assert_eq!(map.get(addr(1, 1)), None);
        assert_eq!(map.get(addr(2, 0)), None);
    }

    /// New fragments are visited in slice order, not id order, so the new side can go
    /// backwards. A run must break there.
    #[test]
    fn non_monotone_new_side_breaks_runs() {
        let pairs = vec![
            (addr(1, 0), Some(addr(9, 0))),
            (addr(1, 1), Some(addr(9, 1))),
            (addr(1, 2), Some(addr(3, 0))),
            (addr(1, 3), Some(addr(3, 1))),
        ];
        let runs = RowAddrRunMap::from(pairs.iter().copied().collect::<HashMap<_, _>>());
        assert_eq!(runs.num_runs(), 2);
        let probes: Vec<u64> = (0..6).map(|i| addr(1, i)).collect();
        assert_equivalent(&pairs, &probes);
    }

    #[test]
    fn old_fragment_boundary_breaks_runs() {
        let pairs = [
            (addr(1, 0), Some(addr(9, 0))),
            (addr(2, 0), Some(addr(9, 1))),
        ];
        let runs = RowAddrRunMap::from(pairs.iter().copied().collect::<HashMap<_, _>>());
        assert_eq!(
            runs.num_runs(),
            2,
            "addresses are not consecutive across fragments"
        );
    }

    /// A version that deleted everything it touched has no runs, but it is emphatically not
    /// empty. Treating it as empty would skip the remap and leave stale addresses live.
    #[test]
    fn all_deleted_version_is_not_empty() {
        let map = RowAddrRunMap::from(HashMap::from([(addr(1, 0), None), (addr(1, 1), None)]));
        assert_eq!(map.num_runs(), 0);
        assert!(!map.is_empty(), "no runs is not the same as nothing mapped");
        assert_eq!(map.get(addr(1, 0)), Some(None));
    }

    #[test]
    fn empty_map_is_pure_passthrough() {
        let map = RowAddrRunMap::default();
        assert!(map.is_empty());
        assert_eq!(map.get(0), None);
        assert_eq!(map.get(u64::MAX), None);
    }

    /// Address 0 and the tombstone address are ordinary values as far as the map is
    /// concerned, and must not be special-cased by the range search.
    #[test]
    fn boundary_addresses_round_trip() {
        let pairs = [(0u64, Some(addr(1, 0))), (u64::MAX, None)];
        assert_equivalent(&pairs, &[0, 1, u64::MAX, u64::MAX - 1, addr(1, 0)]);
    }

    /// Chaining is sticky: once a version deletes a row, later versions must not resurrect
    /// it, and a version that never saw the address must leave it alone.
    #[test]
    fn versions_chain_with_sticky_deletion() {
        let v0 = RowAddrRunMap::from(HashMap::from([
            (addr(1, 0), Some(addr(2, 0))),
            (addr(1, 1), None),
        ]));
        let v1 = RowAddrRunMap::from(HashMap::from([(addr(2, 0), Some(addr(3, 0)))]));
        let index = FragReuseIndex::new_from_run_maps(
            Uuid::nil(),
            vec![v0, v1],
            FragReuseIndexDetails { versions: vec![] },
        );
        assert_eq!(index.remap_row_id(addr(1, 0)), Some(addr(3, 0)), "chained");
        assert_eq!(
            index.remap_row_id(addr(1, 1)),
            None,
            "deleted stays deleted"
        );
        assert_eq!(
            index.remap_row_id(addr(8, 0)),
            Some(addr(8, 0)),
            "untouched"
        );
    }

    /// `FragReuseIndex::new` is public and callers build one from maps with no matching
    /// details, so coverage cannot be inferred from `details`.
    #[test]
    fn constructible_from_hash_maps_without_details() {
        let index = FragReuseIndex::new(
            Uuid::nil(),
            vec![HashMap::from([(0u64, Some(5000u64))])],
            FragReuseIndexDetails { versions: vec![] },
        );
        assert_eq!(index.remap_row_id(0), Some(5000));
        assert_eq!(index.remap_row_id(1), Some(1));
    }

    /// Two spans claiming the same address would make lookups depend on sort order, where
    /// the hashmap was last-write-wins. Refuse to build rather than answer arbitrarily.
    #[test]
    fn overlapping_spans_are_rejected() {
        let mut builder = RowAddrRunMapBuilder::default();
        builder.push_mapped(100, 200);
        builder.push_mapped(500, 600);
        // Same address again, from a notional second group.
        builder.push_mapped(100, 900);
        assert!(builder.finish().is_err(), "overlap must not build silently");

        let mut builder = RowAddrRunMapBuilder::default();
        builder.push_mapped(100, 200);
        builder.push_deleted(100);
        assert!(
            builder.finish().is_err(),
            "an address cannot be both mapped and deleted"
        );
    }

    /// Groups arrive one after another and need not be in ascending address order.
    #[test]
    fn out_of_order_groups_merge() {
        let mut builder = RowAddrRunMapBuilder::default();
        builder.push_mapped(addr(5, 0), addr(9, 0));
        builder.push_mapped(addr(5, 1), addr(9, 1));
        builder.push_mapped(addr(1, 0), addr(8, 0));
        builder.push_deleted(addr(1, 1));
        let map = builder.finish().unwrap();
        assert_eq!(map.get(addr(1, 0)), Some(Some(addr(8, 0))));
        assert_eq!(map.get(addr(1, 1)), Some(None));
        assert_eq!(map.get(addr(5, 1)), Some(Some(addr(9, 1))));
        assert_eq!(map.get(addr(3, 0)), None);
    }

    /// The point of the change: a large contiguous remap must not scale with rows.
    #[test]
    fn memory_is_bounded_by_runs_not_rows() {
        let mut builder = RowAddrRunMapBuilder::default();
        for offset in 0..1_000_000u32 {
            builder.push_mapped(addr(1, offset), addr(2, offset));
        }
        let map = builder.finish().unwrap();
        assert_eq!(map.num_runs(), 1);
        assert_eq!(map.len(), 1_000_000);
        assert!(
            map.deep_size_of() < 4096,
            "a million contiguous rows must not cost more than a handful of bytes, got {}",
            map.deep_size_of()
        );
    }
}
