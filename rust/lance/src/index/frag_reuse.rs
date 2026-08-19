// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::Dataset;
use crate::dataset::optimize::IndexRemapMode;
use crate::dataset::optimize::remapping::transpose_row_ids_from_digest;
use crate::index::DatasetIndexExt;
use lance_core::Error;
use lance_core::utils::row_addr_remap::{GroupInput, RowAddrRemap};
use lance_index::frag_reuse::{
    FRAG_REUSE_DETAILS_FILE_NAME, FRAG_REUSE_INDEX_NAME, FragReuseGroup, FragReuseIndex,
    FragReuseIndexDetails, FragReuseVersion,
};
use lance_table::format::IndexMetadata;
use lance_table::format::pb::fragment_reuse_index_details::{Content, InlineContent};
use lance_table::format::pb::{ExternalFile, FragmentReuseIndexDetails};
use prost::Message;
use roaring::{RoaringBitmap, RoaringTreemap};
use std::io::Cursor;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

/// Load fragment reuse index details from index metadata
pub async fn load_frag_reuse_index_details(
    dataset: &Dataset,
    index: &IndexMetadata,
) -> lance_core::Result<Arc<FragReuseIndexDetails>> {
    let details_any = index.index_details.clone();
    if details_any.is_none()
        || !details_any
            .as_ref()
            .unwrap()
            .type_url
            .ends_with("FragmentReuseIndexDetails")
    {
        return Err(Error::index(
            "Index details is not for the fragment reuse index",
        ));
    }

    let proto = details_any.unwrap().to_msg::<FragmentReuseIndexDetails>()?;
    match &proto.content {
        None => Err(Error::index("Index details content is not found")),
        Some(Content::Inline(content)) => {
            Ok(Arc::new(FragReuseIndexDetails::try_from(content.clone())?))
        }
        Some(Content::External(external_file)) => {
            let file_path = dataset
                .indices_dir()
                .join(index.uuid.to_string())
                .join(external_file.path.clone());

            // the file content will be cached in the index cache later
            // so we do not put it to the file cache
            let range = external_file.offset as usize
                ..(external_file.offset as usize + external_file.size as usize);
            let data = dataset
                .object_store
                .open(&file_path)
                .await?
                .get_range(range)
                .await?;

            let pb_sequence = InlineContent::decode(data)?;
            Ok(Arc::new(FragReuseIndexDetails::try_from(pb_sequence)?))
        }
    }
}

/// Open a fragment reuse index from its metadata details, in the given form.
///
/// The two forms are not interchangeable: see [`ReadParams::frag_reuse_remap_mode`] for what
/// they cost and where they differ.
pub(crate) async fn open_frag_reuse_index_with_mode(
    uuid: Uuid,
    details: &FragReuseIndexDetails,
    mode: IndexRemapMode,
) -> lance_core::Result<FragReuseIndex> {
    if mode == IndexRemapMode::Direct {
        // The pre-compact behaviour, kept as a fallback. Materializes one entry per rewritten
        // or deleted row, so its memory grows with the rows compaction has touched -- which is
        // the cost the compact form exists to avoid. It cannot fail, because pairing rows by
        // address order silently truncates rather than rejecting a payload whose counts
        // disagree; see `transpose_row_ids_from_digest`.
        let mut row_id_maps = Vec::with_capacity(details.versions.len());
        for version in &details.versions {
            let mut row_id_map = std::collections::HashMap::new();
            for group in version.groups.iter() {
                let changed =
                    RoaringTreemap::deserialize_from(Cursor::new(&group.changed_row_addrs))?;
                row_id_map.extend(transpose_row_ids_from_digest(
                    changed,
                    &group.old_frags,
                    &group.new_frags,
                ));
            }
            row_id_maps.push(RowAddrRemap::direct(row_id_map));
        }
        return Ok(FragReuseIndex::new_from_remaps(
            uuid,
            row_id_maps,
            details.clone(),
        ));
    }

    // Build the compact form rather than a materialized per-row map. This runs on every
    // index open and the result is cached, so a per-row map would charge readers a cost
    // that grows with the number of rows compaction has touched, rather than with the
    // number of fragments. Expanding a large reuse history that way can exhaust memory.
    let mut row_addr_maps: Vec<RowAddrRemap> = Vec::with_capacity(details.versions.len());
    for version in &details.versions {
        let mut groups = Vec::with_capacity(version.groups.len());
        for group in version.groups.iter() {
            let cursor = Cursor::new(&group.changed_row_addrs);
            let rewritten_old_row_addrs = RoaringTreemap::deserialize_from(cursor)?;

            // Lance 0.30.0 through 4.0.0-beta.6 recorded an empty set of rewritten
            // addresses for a stable-row-id dataset whose index remap was deferred, while
            // still recording how many rows the new fragments received. Such payloads are
            // on disk and must keep opening.
            //
            // Positional mapping needs the two counts to agree, so the new fragments are
            // dropped for these groups, leaving every covered address resolving to
            // deleted. That is what the previous per-row map produced for the same input,
            // so the behaviour is unchanged; only the failure mode would be new.
            let is_legacy_empty_rewrite =
                rewritten_old_row_addrs.is_empty() && !group.new_frags.is_empty();
            if is_legacy_empty_rewrite {
                tracing::warn!(
                    old_frags = ?group.old_frags.iter().map(|frag| frag.id).collect::<Vec<_>>(),
                    "fragment reuse group records no rewritten rows but non-empty new \
                     fragments; treating its fragments as deleted"
                );
            }
            let new_frags = if is_legacy_empty_rewrite {
                Vec::new()
            } else {
                group
                    .new_frags
                    .iter()
                    .map(|frag| (frag.id as u32, frag.physical_rows as u32))
                    .collect()
            };

            groups.push(GroupInput {
                rewritten_old_row_addrs,
                old_frag_ids: group.old_frags.iter().map(|frag| frag.id as u32).collect(),
                new_frags,
            });
        }
        row_addr_maps.push(RowAddrRemap::compact(groups)?);
    }

    Ok(FragReuseIndex::new_from_remaps(
        uuid,
        row_addr_maps,
        details.clone(),
    ))
}

pub(crate) async fn build_new_frag_reuse_index(
    dataset: &mut Dataset,
    frag_reuse_groups: Vec<FragReuseGroup>,
    new_fragment_bitmap: RoaringBitmap,
) -> lance_core::Result<IndexMetadata> {
    let new_version = FragReuseVersion {
        dataset_version: dataset.manifest.version,
        groups: frag_reuse_groups,
    };

    let index_meta = dataset.load_indices().await.map(|indices| {
        indices
            .iter()
            .find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
            .cloned()
    })?;

    let new_index_details = match &index_meta {
        None => FragReuseIndexDetails {
            versions: Vec::from([new_version]),
        },
        Some(index_meta) => {
            let current_details = load_frag_reuse_index_details(dataset, index_meta).await?;
            let mut versions = current_details.versions.clone();
            versions.push(new_version);
            FragReuseIndexDetails { versions }
        }
    };

    build_frag_reuse_index_metadata(
        dataset,
        index_meta.as_ref(),
        new_index_details,
        new_fragment_bitmap,
    )
    .await
}

pub(crate) async fn build_frag_reuse_index_metadata(
    dataset: &Dataset,
    index_meta: Option<&IndexMetadata>,
    new_index_details: FragReuseIndexDetails,
    new_fragment_bitmap: RoaringBitmap,
) -> lance_core::Result<IndexMetadata> {
    let index_id = uuid::Uuid::new_v4();
    let new_index_details_proto = InlineContent::from(&new_index_details);
    let proto = if new_index_details_proto.encoded_len() > 204800 {
        let file_path = dataset
            .indices_dir()
            .join(index_id.to_string())
            .join(FRAG_REUSE_DETAILS_FILE_NAME);
        let mut writer = dataset.object_store.create(&file_path).await?;
        writer
            .write_all(new_index_details_proto.encode_to_vec().as_slice())
            .await?;
        writer.shutdown().await?;
        let external_file = ExternalFile {
            path: FRAG_REUSE_DETAILS_FILE_NAME.to_owned(),
            offset: 0,
            size: new_index_details_proto.encoded_len() as u64,
        };
        FragmentReuseIndexDetails {
            content: Some(Content::External(external_file)),
        }
    } else {
        FragmentReuseIndexDetails {
            content: Some(Content::Inline(new_index_details_proto)),
        }
    };

    Ok(IndexMetadata {
        uuid: index_id,
        name: FRAG_REUSE_INDEX_NAME.to_string(),
        fields: vec![],
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(new_fragment_bitmap),
        index_details: Some(Arc::new(prost_types::Any::from_msg(&proto)?)),
        index_version: index_meta.map_or(0, |index_meta| index_meta.index_version),
        created_at: Some(chrono::Utc::now()),
        base_id: None,
        // Fragment reuse index is inline (no files)
        files: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::optimize::remapping::transpose_row_ids_from_digest;
    use lance_core::utils::address::RowAddress;
    use lance_core::utils::row_addr_remap::RowAddrRemap;
    use lance_index::frag_reuse::{FragDigest, FragReuseVersion};
    use rstest::rstest;

    fn addr(frag: u32, offset: u32) -> u64 {
        u64::from(RowAddress::new_from_parts(frag, offset))
    }

    fn digest(id: u64, physical_rows: usize) -> FragDigest {
        FragDigest {
            id,
            physical_rows,
            // Never read on this path; `transpose_row_ids_from_digest` uses it only as a
            // capacity hint, so no assertion here can distinguish its value.
            num_deleted_rows: 0,
        }
    }

    /// One rewrite group. `rewritten` is `(old_frag, offset)` in any order: the treemap
    /// sorts and run-optimizes it, which is what `rewrite_files` writes.
    fn group(
        rewritten: impl IntoIterator<Item = (u32, u32)>,
        old_frags: Vec<FragDigest>,
        new_frags: Vec<FragDigest>,
    ) -> FragReuseGroup {
        let mut addrs = RoaringTreemap::from_iter(
            rewritten
                .into_iter()
                .map(|(frag, offset)| addr(frag, offset)),
        );
        addrs.optimize();
        let mut changed_row_addrs = Vec::with_capacity(addrs.serialized_size());
        addrs.serialize_into(&mut changed_row_addrs).unwrap();
        FragReuseGroup {
            changed_row_addrs,
            old_frags,
            new_frags,
        }
    }

    fn details(versions: Vec<Vec<FragReuseGroup>>) -> FragReuseIndexDetails {
        FragReuseIndexDetails {
            versions: versions
                .into_iter()
                .enumerate()
                .map(|(i, groups)| FragReuseVersion {
                    dataset_version: i as u64 + 1,
                    groups,
                })
                .collect(),
        }
    }

    /// Opens in the compact form, which is what most tests below are about. Tests that care
    /// about the other form call `open_frag_reuse_index_with_mode` directly.
    async fn open(details: &FragReuseIndexDetails) -> FragReuseIndex {
        open_frag_reuse_index_with_mode(Uuid::new_v4(), details, IndexRemapMode::Compact)
            .await
            .expect("index should open")
    }

    #[test]
    fn test_builder_default_uses_the_documented_default() {
        // Two default paths exist -- `ReadParams::default()` and the builder's own field --
        // and the builder's originally took `IndexRemapMode`'s derived default rather than the
        // documented one, so the environment variable silently did not apply through the
        // builder, which is the common way to open a dataset.
        //
        // Asserted against the builder's own field, not against `ReadParams`: comparing the
        // two documented defaults to each other passes whether or not the builder consults
        // either. This also catches the derived default and the documented one drifting apart,
        // which is what would happen if `IndexRemapMode`'s `#[default]` ever moved.
        let builder = crate::dataset::builder::DatasetBuilder::from_uri("memory://test");
        let expected = crate::dataset::default_frag_reuse_remap_mode();
        assert!(
            format!("{builder:?}").contains(&format!("frag_reuse_remap_mode: {expected:?}")),
            "builder default should be {expected:?}: {builder:?}"
        );
    }

    #[test]
    fn test_read_params_mode_survives_the_builder() {
        // `with_read_params` copies fields one by one, so a new one is easy to forget: this
        // caught exactly that.
        let mut params = crate::dataset::ReadParams::default();
        params.frag_reuse_remap_mode(IndexRemapMode::Compact);
        let builder = crate::dataset::builder::DatasetBuilder::from_uri("memory://test")
            .with_read_params(params);
        assert!(
            format!("{builder:?}").contains("frag_reuse_remap_mode: Compact"),
            "the builder should carry the mode from the read params: {builder:?}"
        );
    }

    #[test]
    fn test_default_mode_is_direct() {
        // Pinned because it is the compatibility contract: a reader that says nothing keeps
        // the behaviour it had before the compact form existed, so picking up a Lance that
        // carries this change is not itself a behaviour change. Callers that want the compact
        // form ask for it.
        //
        // Reads `ReadParams::default()`, so it also covers the env default being absent, which
        // is the state any normal test process is in.
        assert_eq!(
            crate::dataset::ReadParams::default().frag_reuse_remap_mode,
            IndexRemapMode::Direct
        );
    }

    #[rstest]
    #[case::compact(IndexRemapMode::Compact)]
    #[case::direct(IndexRemapMode::Direct)]
    #[tokio::test]
    async fn test_both_modes_agree_on_a_real_payload(#[case] mode: IndexRemapMode) {
        // The switch's whole purpose is that either mode can serve a reader, so for a payload
        // a writer can actually produce they must resolve every real address identically.
        // Probed against expectations written once, not against each other, so a shared bug
        // could not make both pass.
        let old = vec![digest(0, 5), digest(1, 4)];
        let new = vec![digest(10, 3), digest(11, 3)];
        let rewritten = [(0, 0), (0, 2), (0, 4), (1, 0), (1, 1), (1, 3)];
        let index = open_frag_reuse_index_with_mode(
            Uuid::new_v4(),
            &details(vec![vec![group(rewritten, old, new)]]),
            mode,
        )
        .await
        .unwrap();

        // Positions run in read order across both new fragments.
        assert_eq!(index.remap_row_id(addr(0, 0)), Some(addr(10, 0)));
        assert_eq!(index.remap_row_id(addr(0, 2)), Some(addr(10, 1)));
        assert_eq!(index.remap_row_id(addr(0, 4)), Some(addr(10, 2)));
        assert_eq!(index.remap_row_id(addr(1, 0)), Some(addr(11, 0)));
        assert_eq!(index.remap_row_id(addr(1, 1)), Some(addr(11, 1)));
        assert_eq!(index.remap_row_id(addr(1, 3)), Some(addr(11, 2)));
        // Deleted before compaction.
        assert_eq!(index.remap_row_id(addr(0, 1)), None);
        assert_eq!(index.remap_row_id(addr(0, 3)), None);
        assert_eq!(index.remap_row_id(addr(1, 2)), None);
        // Untouched by the group.
        assert_eq!(index.remap_row_id(addr(9, 0)), Some(addr(9, 0)));
    }

    #[tokio::test]
    async fn test_direct_mode_yields_the_materialized_form() {
        let index = open_frag_reuse_index_with_mode(
            Uuid::new_v4(),
            &details(vec![vec![group(
                [(0, 0), (0, 2)],
                vec![digest(0, 3)],
                vec![digest(10, 2)],
            )]]),
            IndexRemapMode::Direct,
        )
        .await
        .unwrap();
        assert!(matches!(index.row_addr_maps[0], RowAddrRemap::Direct(_)));
    }

    #[tokio::test]
    async fn test_direct_mode_accepts_a_payload_compact_rejects() {
        // The fallback has to be usable for the case it exists for. A payload whose row counts
        // disagree is rejected by the compact form and silently truncated by the map form, so
        // switching modes has to turn that failure into an open.
        let details = details(vec![vec![group(
            [(0, 0), (0, 1)],
            vec![digest(0, 2)],
            vec![digest(10, 1)],
        )]]);
        let uuid = Uuid::new_v4();
        assert!(
            open_frag_reuse_index_with_mode(uuid, &details, IndexRemapMode::Compact)
                .await
                .is_err()
        );
        assert!(
            open_frag_reuse_index_with_mode(uuid, &details, IndexRemapMode::Direct)
                .await
                .is_ok()
        );
    }

    #[tokio::test]
    async fn test_open_builds_the_compact_form() {
        // The reason this path exists. A materialized map answers every lookup below
        // identically, so without this assertion a revert to `transpose_row_ids_from_digest`
        // would reproduce the memory blowup with every other test still green.
        let index = open(&details(vec![vec![group(
            [(0, 0), (0, 2)],
            vec![digest(0, 3)],
            vec![digest(10, 2)],
        )]]))
        .await;
        assert!(matches!(index.row_addr_maps[0], RowAddrRemap::Compact(_)));
    }

    #[tokio::test]
    async fn test_open_maps_moved_deleted_and_untouched() {
        let index = open(&details(vec![vec![group(
            [(0, 0), (0, 2)],
            vec![digest(0, 3)],
            vec![digest(10, 2)],
        )]]))
        .await;

        // Moved, in read order.
        assert_eq!(index.remap_row_id(addr(0, 0)), Some(addr(10, 0)));
        assert_eq!(index.remap_row_id(addr(0, 2)), Some(addr(10, 1)));
        // Offset 1 was deleted before compaction: gone, not unchanged.
        assert_eq!(index.remap_row_id(addr(0, 1)), None);
        // A fragment no group covers is left alone.
        assert_eq!(index.remap_row_id(addr(9, 0)), Some(addr(9, 0)));
        // The output address is not remapped again.
        assert_eq!(index.remap_row_id(addr(10, 0)), Some(addr(10, 0)));
        // Past the fragment's end: the compact form reports these deleted where the map
        // form reported them absent, i.e. unchanged. The one intended semantic difference.
        assert_eq!(index.remap_row_id(addr(0, 3)), None);
        assert_eq!(index.remap_row_id(addr(0, 99)), None);
    }

    #[tokio::test]
    async fn test_open_agrees_with_transpose_on_ascending_old_frags() {
        // Ascending old fragments are compaction's scan order, the only case where the
        // positional and address-ordered pairings must agree. Probe every real address.
        let old = vec![digest(0, 5), digest(1, 4), digest(3, 3)];
        let new = vec![digest(10, 4), digest(11, 5)];
        let rewritten = [
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 0),
            (1, 1),
            (1, 3),
            (3, 0),
            (3, 1),
            (3, 2),
        ];
        let index = open(&details(vec![vec![group(
            rewritten,
            old.clone(),
            new.clone(),
        )]]))
        .await;

        let expected = transpose_row_ids_from_digest(
            RoaringTreemap::from_iter(rewritten.iter().map(|&(f, o)| addr(f, o))),
            &old,
            &new,
        );
        for frag in &old {
            for offset in 0..frag.physical_rows as u32 {
                let a = addr(frag.id as u32, offset);
                assert_eq!(
                    index.remap_row_id(a),
                    // `remap_row_id` passes untouched addresses through, the map omits them.
                    expected.get(&a).copied().unwrap_or(Some(a)),
                    "mismatch at ({}, {offset})",
                    frag.id
                );
            }
        }
    }

    #[tokio::test]
    async fn test_open_pairs_rows_in_read_order_not_address_order() {
        // `old_frags` records the order compaction read the fragments, and the compact form
        // pairs rows positionally in that order rather than by address.
        //
        // The two coincide in practice: `build_manifest` sorts the manifest's fragment list
        // by id after every operation, which `Manifest::fragments` documents as an
        // invariant, so a rewrite group's fragments arrive ascending. This input is
        // therefore not reachable from any current writer. It is pinned because it records
        // which pairing is the correct one, and would catch a regression if that invariant
        // ever stopped holding.
        let old = vec![digest(4, 2), digest(3, 2)];
        let new = vec![digest(10, 4)];
        let rewritten = [(4, 0), (4, 1), (3, 0), (3, 1)];
        let index = open(&details(vec![vec![group(
            rewritten,
            old.clone(),
            new.clone(),
        )]]))
        .await;

        assert_eq!(index.remap_row_id(addr(4, 0)), Some(addr(10, 0)));
        assert_eq!(index.remap_row_id(addr(4, 1)), Some(addr(10, 1)));
        assert_eq!(index.remap_row_id(addr(3, 0)), Some(addr(10, 2)));
        assert_eq!(index.remap_row_id(addr(3, 1)), Some(addr(10, 3)));

        // The map form pairs rewritten rows with new addresses in ascending-address order,
        // which for this input is not the order they were written. `MissingAddrs` then
        // treats frag 4's rows as gaps and overwrites their real mappings with `None`, so
        // two live rows are reported deleted. That is a latent bug in the form being
        // replaced rather than a live one, since the sorted-manifest invariant keeps this
        // input from arising; pinned so the difference is on record either way.
        let transposed = transpose_row_ids_from_digest(
            RoaringTreemap::from_iter(rewritten.iter().map(|&(f, o)| addr(f, o))),
            &old,
            &new,
        );
        assert_eq!(transposed.get(&addr(4, 0)).copied(), Some(None));
        assert_eq!(transposed.get(&addr(4, 1)).copied(), Some(None));
        assert_eq!(
            transposed.get(&addr(3, 0)).copied(),
            Some(Some(addr(10, 0)))
        );
    }

    #[rstest]
    #[case::ascending(vec![0, 1, 2])]
    #[case::descending(vec![2, 1, 0])]
    #[case::rotated(vec![1, 2, 0])]
    #[case::sparse_scrambled(vec![9, 0, 4])]
    #[tokio::test]
    async fn test_open_follows_old_frag_order_whatever_it_is(#[case] order: Vec<u64>) {
        // `old_frag_ids` imposes no ordering requirement: positions come from walking the
        // list, so every permutation has to map correctly and none is rejected. Manifest
        // fragments are id-sorted in practice, so this guards the relaxed assumption rather
        // than a reachable input -- and it would fail if the implementation ever started
        // sorting internally, which a hand-written two-element case would not catch.
        //
        // Each fragment keeps offsets 0 and 2 of three rows, so deletions interact with the
        // ordering, and the output spans two fragments so the range search does too.
        const KEPT: [u32; 2] = [0, 2];
        let rewritten: Vec<(u32, u32)> = order
            .iter()
            .flat_map(|&frag| KEPT.iter().map(move |&offset| (frag as u32, offset)))
            .collect();
        let total = rewritten.len();
        let split = total / 2;
        let index = open(&details(vec![vec![group(
            rewritten,
            order.iter().map(|&frag| digest(frag, 3)).collect(),
            vec![digest(20, split), digest(21, total - split)],
        )]]))
        .await;

        // Expectations are derived from the contract, not written out: the k-th kept row in
        // list order occupies the k-th slot across the new fragments in their order.
        let mut position = 0usize;
        for &frag in &order {
            for &offset in &KEPT {
                let expected = if position < split {
                    addr(20, position as u32)
                } else {
                    addr(21, (position - split) as u32)
                };
                assert_eq!(
                    index.remap_row_id(addr(frag as u32, offset)),
                    Some(expected),
                    "fragment {frag} offset {offset} should hold position {position}"
                );
                position += 1;
            }
            // The hole in each fragment is deleted wherever that fragment sits in the list.
            assert_eq!(
                index.remap_row_id(addr(frag as u32, 1)),
                None,
                "hole in {frag}"
            );
        }
    }

    #[tokio::test]
    async fn test_open_handles_large_run_optimized_bitmaps() {
        // Production payloads span fragments of up to `max_rows_per_file` and are
        // run-optimized before serializing, so the offsets land in run and bitmap
        // containers rather than the small array containers every other test here uses.
        // This is what exercises `RoaringBitmap::rank` on those container types.
        const ROWS: u32 = 200_000;
        let holes = [7u32, 65_535, 65_536, 131_072];
        let kept: Vec<u32> = (0..ROWS).filter(|o| !holes.contains(o)).collect();
        let index = open(&details(vec![vec![group(
            kept.iter().map(|&o| (0u32, o)),
            vec![digest(0, ROWS as usize)],
            vec![digest(10, kept.len())],
        )]]))
        .await;

        // Every kept offset shifts down by the number of holes below it.
        for probe in [0u32, 6, 8, 65_534, 65_537, 131_073, ROWS - 1] {
            let rank = probe - holes.iter().filter(|&&h| h < probe).count() as u32;
            assert_eq!(
                index.remap_row_id(addr(0, probe)),
                Some(addr(10, rank)),
                "offset {probe}"
            );
        }
        for hole in holes {
            assert_eq!(index.remap_row_id(addr(0, hole)), None, "hole {hole}");
        }
    }

    #[tokio::test]
    async fn test_open_resolves_across_many_new_fragments() {
        // A task whose output exceeds `max_rows_per_file` rolls into several fragments.
        // With only one or two ranges the binary search in `compute_new_addr` never lands
        // strictly inside the list, so a mid-list off-by-one would go unnoticed.
        let new: Vec<FragDigest> = (10..15).map(|id| digest(id, 3)).collect();
        let index = open(&details(vec![vec![group(
            (0..15u32).map(|o| (0u32, o)),
            vec![digest(0, 15)],
            new,
        )]]))
        .await;

        for offset in 0..15u32 {
            let expected = addr(10 + offset / 3, offset % 3);
            assert_eq!(
                index.remap_row_id(addr(0, offset)),
                Some(expected),
                "{offset}"
            );
        }
    }

    #[tokio::test]
    async fn test_open_skips_emptied_fragment_without_shifting_later_ones() {
        // The emptied fragment sits between two live ones. If its rows were charged to the
        // running position, frag 1's rows would land two slots late.
        let index = open(&details(vec![vec![group(
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            vec![digest(0, 2), digest(7, 2), digest(1, 2)],
            vec![digest(10, 4)],
        )]]))
        .await;

        assert_eq!(index.remap_row_id(addr(0, 0)), Some(addr(10, 0)));
        assert_eq!(index.remap_row_id(addr(0, 1)), Some(addr(10, 1)));
        assert_eq!(index.remap_row_id(addr(1, 0)), Some(addr(10, 2)));
        assert_eq!(index.remap_row_id(addr(1, 1)), Some(addr(10, 3)));
        // The emptied fragment is covered, so its addresses are deleted rather than kept.
        assert_eq!(index.remap_row_id(addr(7, 0)), None);
        assert_eq!(index.remap_row_id(addr(7, 1)), None);
        // A fragment outside the group is still untouched.
        assert_eq!(index.remap_row_id(addr(8, 0)), Some(addr(8, 0)));
    }

    #[tokio::test]
    async fn test_open_composes_a_deep_chain() {
        // Version i rewrites fragment i into fragment i+1, so a row entering at (0,0) must
        // arrive at (32,0) having passed through every link.
        const VERSIONS: u32 = 32;
        let chain = (0..VERSIONS)
            .map(|i| {
                vec![group(
                    [(i, 0)],
                    vec![digest(i as u64, 1)],
                    vec![digest(i as u64 + 1, 1)],
                )]
            })
            .collect();
        let index = open(&details(chain)).await;

        assert_eq!(index.row_addr_maps.len(), VERSIONS as usize);
        assert_eq!(index.remap_row_id(addr(0, 0)), Some(addr(VERSIONS, 0)));
        // Entering midway walks only the remaining links.
        assert_eq!(
            index.remap_row_id(addr(VERSIONS / 2, 0)),
            Some(addr(VERSIONS, 0))
        );
    }

    #[tokio::test]
    async fn test_deletion_mid_chain_is_terminal() {
        // v1 moves (0,0) -> (10,0); v2 covers frag 10 and keeps nothing, so the row dies.
        // v3 also covers frag 10 and would move it on, so a walk that failed to stop would
        // resurrect the row at (30,0) rather than merely passing a stale address along.
        let index = open(&details(vec![
            vec![group([(0, 0)], vec![digest(0, 1)], vec![digest(10, 1)])],
            vec![group([], vec![digest(10, 1)], vec![])],
            vec![group([(10, 0)], vec![digest(10, 1)], vec![digest(30, 1)])],
        ]))
        .await;

        assert_eq!(index.remap_row_id(addr(0, 0)), None);
        // Entering at the intermediate address still dies in v2, before v3 is consulted.
        assert_eq!(index.remap_row_id(addr(10, 0)), None);
    }

    #[tokio::test]
    async fn test_open_keeps_groups_in_one_version_independent() {
        // One compaction commits several rewrite groups. They share a version, so they
        // collapse into a single remap, and each group's positions restart.
        let index = open(&details(vec![vec![
            group([(0, 0), (0, 1)], vec![digest(0, 2)], vec![digest(10, 2)]),
            group([(1, 0), (1, 1)], vec![digest(1, 2)], vec![digest(11, 2)]),
        ]]))
        .await;

        assert_eq!(index.row_addr_maps.len(), 1);
        assert_eq!(index.remap_row_id(addr(0, 1)), Some(addr(10, 1)));
        // Group 2's first row starts at its own new fragment, not offset 2 of frag 10.
        assert_eq!(index.remap_row_id(addr(1, 0)), Some(addr(11, 0)));
        assert_eq!(index.remap_row_id(addr(1, 1)), Some(addr(11, 1)));
    }

    #[rstest]
    #[case::no_versions(vec![], 0)]
    #[case::version_without_groups(vec![vec![]], 1)]
    #[tokio::test]
    async fn test_open_with_nothing_to_remap(
        #[case] versions: Vec<Vec<FragReuseGroup>>,
        #[case] expected_links: usize,
    ) {
        let index = open(&details(versions)).await;
        assert_eq!(index.row_addr_maps.len(), expected_links);
        assert!(index.row_addr_maps.iter().all(|map| map.is_empty()));
        assert_eq!(index.remap_row_id(addr(0, 0)), Some(addr(0, 0)));
    }

    #[rstest]
    // Rewritten rows and new-fragment rows must match exactly, or positions are unsound.
    // These two are the only coverage of that check; the other two validations
    // `GroupRemap::new` performs are tested next to it in `row_addr_remap.rs`.
    #[case::too_few_new_rows(vec![(0, 0), (0, 1)], vec![digest(0, 2)], vec![digest(10, 1)])]
    #[case::too_many_new_rows(vec![(0, 0)], vec![digest(0, 1)], vec![digest(10, 2)])]
    #[tokio::test]
    async fn test_open_rejects_row_count_mismatch(
        #[case] rewritten: Vec<(u32, u32)>,
        #[case] old_frags: Vec<FragDigest>,
        #[case] new_frags: Vec<FragDigest>,
    ) {
        // The map form silently truncated these; opening now fails instead.
        let details = details(vec![vec![group(rewritten, old_frags, new_frags)]]);
        let err =
            open_frag_reuse_index_with_mode(Uuid::new_v4(), &details, IndexRemapMode::Compact)
                .await
                .expect_err("inconsistent details should be rejected");
        assert!(
            err.to_string().contains("old rows"),
            "expected the row-count validation, got: {err}"
        );
    }

    #[tokio::test]
    async fn test_legacy_empty_rewrite_payload_still_opens() {
        // Lance 0.30.0 through 4.0.0-beta.6 wrote this shape for a stable-row-id dataset
        // with deferred index remap: no rewritten addresses, but new fragments with real
        // row counts. Those two disagree, so a strict reading rejects the group -- and
        // `open_frag_reuse_index` runs inside `load_indices`, which would take a dataset
        // that used to open and make every scan, validate and commit fail.
        let index = open(&details(vec![vec![group(
            [],
            vec![digest(0, 400), digest(1, 400)],
            vec![digest(2, 800)],
        )]]))
        .await;

        // The old per-row map resolved every covered address to deleted for this input, so
        // that is what is preserved. The rows are unreachable either way until the index is
        // rebuilt; what matters is that opening succeeds.
        assert_eq!(index.remap_row_id(addr(0, 0)), None);
        assert_eq!(index.remap_row_id(addr(1, 399)), None);
        // Fragments the group never claimed are still untouched.
        assert_eq!(index.remap_row_id(addr(2, 0)), Some(addr(2, 0)));
        assert_eq!(index.remap_row_id(addr(9, 0)), Some(addr(9, 0)));
    }

    #[tokio::test]
    async fn test_genuinely_emptied_group_is_unaffected_by_the_legacy_path() {
        // The legacy shape is distinguished by new fragments being present. A group that
        // really did delete everything carries none, and must behave as before.
        let index = open(&details(vec![vec![group([], vec![digest(7, 4)], vec![])]])).await;
        assert_eq!(index.remap_row_id(addr(7, 0)), None);
        assert_eq!(index.remap_row_id(addr(8, 0)), Some(addr(8, 0)));
    }

    #[tokio::test]
    async fn test_corrupt_changed_row_addrs_is_an_error_not_a_panic() {
        // A plausible length prefix followed by a truncated body, so the failure happens
        // inside treemap parsing rather than on the first read.
        let mut group = group([(0, 0)], vec![digest(0, 1)], vec![digest(10, 1)]);
        let mut corrupt = 4u64.to_le_bytes().to_vec();
        corrupt.extend_from_slice(&[0xff; 6]);
        group.changed_row_addrs = corrupt;
        let details = details(vec![vec![group]]);
        assert!(
            open_frag_reuse_index_with_mode(Uuid::new_v4(), &details, IndexRemapMode::Compact)
                .await
                .is_err()
        );
    }
}
