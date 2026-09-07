// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Caches for Lance indices. They are organized in a hierarchical manner to
//! avoid collisions.
//!
//!  GlobalIndexCache
//!     │
//!     ├─► DSIndexCache (prefixed by dataset URI)
//!     │    │
//!     └────┴──► Index-specific cache (prefixed by index UUID and FRI UUID)

use std::{borrow::Cow, ops::Deref, sync::Arc};

use crate::dataset::optimize::IndexRemapMode;
use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder, LanceCache};
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_index::frag_reuse::FragReuseIndex;
use lance_table::format::IndexMetadata;
use uuid::Uuid;

/// A type-safe wrapper around a LanceCache that enforces namespaces for index data.
pub struct GlobalIndexCache(pub(super) LanceCache);

impl GlobalIndexCache {
    pub fn for_dataset(&self, uri: &str) -> DSIndexCache {
        // Create a sub-cache for the dataset by adding the URI as a key prefix.
        // This prevents collisions between different datasets.
        DSIndexCache(self.0.with_key_prefix(uri))
    }

    /// As [`Self::for_dataset`], keeping the two fragment-reuse remap forms apart.
    ///
    /// What an index caches is not the index as stored: with the remap deferred, its row
    /// addresses are translated through the fragment reuse index as it loads. So the cached
    /// state is the index *as translated*, and the two forms do not translate identically --
    /// an offset past a fragment's `physical_rows`, for instance, is deleted under
    /// [`IndexRemapMode::Compact`] and untouched under [`IndexRemapMode::Direct`].
    ///
    /// Datasets sharing a [`Session`](crate::session::Session) may disagree about the form,
    /// so without this a reader would silently be served state translated the other way, and
    /// whichever form opened first would decide for the rest.
    ///
    /// Costs nothing when a session sees one form, which is the ordinary case: one form means
    /// one prefix, and the cache behaves exactly as it did before. Two forms cost a second
    /// copy of the state for the datasets that differ.
    ///
    /// If `Direct` is eventually removed there is only one form left, and this collapses back
    /// into [`Self::for_dataset`].
    ///
    /// The form is added as its own hierarchy segment rather than interpolated into the URI's:
    /// [`LanceCache::with_key_prefix`] frames each call, so a dataset whose URI happens to end
    /// in the name of a form cannot collide with the same dataset under that form.
    pub fn for_dataset_with_remap_mode(&self, uri: &str, mode: IndexRemapMode) -> DSIndexCache {
        DSIndexCache(
            self.0
                .with_key_prefix(uri)
                .with_key_prefix(remap_mode_segment(mode)),
        )
    }
}

/// The cache namespace segment naming a fragment-reuse remap form.
///
/// Matched explicitly rather than taken from `Debug`: a new [`IndexRemapMode`] variant should
/// fail to compile here rather than silently mint a fresh cache namespace, and a cache
/// namespace should not depend on a `Debug` impl that anyone is free to reword.
pub(crate) fn remap_mode_segment(mode: IndexRemapMode) -> &'static str {
    match mode {
        IndexRemapMode::Compact => "compact",
        IndexRemapMode::Direct => "direct",
    }
}

impl Clone for GlobalIndexCache {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl Deref for GlobalIndexCache {
    type Target = LanceCache;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DeepSizeOf for GlobalIndexCache {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.0.deep_size_of_children(context)
    }
}

/// A type-safe wrapper around a LanceCache that enforces namespaces and keys
/// for dataset-specific index data.
pub struct DSIndexCache(pub(crate) LanceCache);

impl Deref for DSIndexCache {
    type Target = LanceCache;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DSIndexCache {
    /// Create an index-specific cache with the given UUID prefix.
    pub fn for_index(&self, uuid: &Uuid, fri_uuid: Option<&Uuid>) -> LanceCache {
        let mut uuid_buffer = Uuid::encode_buffer();
        let cache = self
            .0
            .with_key_prefix(uuid.as_hyphenated().encode_lower(&mut uuid_buffer));
        if let Some(fri_uuid) = fri_uuid {
            // If a FRI UUID is provided, use it to create a more specific cache key.
            let mut fri_uuid_buffer = Uuid::encode_buffer();
            cache.with_key_prefix(fri_uuid.as_hyphenated().encode_lower(&mut fri_uuid_buffer))
        } else {
            // Otherwise, just use the index UUID as the key prefix.
            cache
        }
    }
}

pub(crate) fn write_index_identity(builder: &mut KeyBuilder, uuid: &Uuid, fri_uuid: Option<&Uuid>) {
    builder.write_fixed_bytes(uuid.as_bytes());
    if let Some(fri_uuid) = fri_uuid {
        builder.write_some();
        builder.write_fixed_bytes(fri_uuid.as_bytes());
    } else {
        builder.write_none();
    }
}

// Cache key types for type-safe cache access

#[derive(Debug)]
pub struct FragReuseIndexKey<'a> {
    pub uuid: &'a Uuid,
}

impl CacheKey for FragReuseIndexKey<'_> {
    type ValueType = FragReuseIndex;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("frag_reuse/{}", self.uuid))
    }

    fn type_name() -> &'static str {
        "FragReuseIndex"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.fragment-reuse-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_fixed_bytes(self.uuid.as_bytes());
    }
}

#[derive(Clone, Copy, Debug)]
pub struct IndexMetadataKey<'a> {
    pub version: u64,
    pub store_identity: &'a str,
}

impl CacheKey for IndexMetadataKey<'_> {
    type ValueType = Vec<IndexMetadata>;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!(
            "{}:{}/{}",
            self.store_identity.len(),
            self.store_identity,
            self.version
        ))
    }

    fn type_name() -> &'static str {
        "Vec<IndexMetadata>"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.metadata-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_str(self.store_identity);
        builder.write_u64(self.version);
    }

    fn codec() -> Option<lance_core::cache::CacheCodec> {
        Some(lance_table::format::index_metadata_codec())
    }
}

pub struct ProstAny(pub Arc<prost_types::Any>);

impl DeepSizeOf for ProstAny {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.0.type_url.deep_size_of_children(context) + self.0.value.deep_size_of_children(context)
    }
}

/// Cache key for scalar index details
///
/// Typically we don't use the cache for scalar index details because they are stored
/// in the manifest and readily available.  However, old versions of Lance didn't store
/// details in the manifest, and we have to perform an expensive inference process to determine
/// what they are.  These we cache.
#[derive(Debug)]
pub struct ScalarIndexDetailsKey<'a> {
    pub uuid: &'a Uuid,
}

impl CacheKey for ScalarIndexDetailsKey<'_> {
    type ValueType = ProstAny;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("type/{}", self.uuid))
    }

    fn type_name() -> &'static str {
        "ScalarIndexDetails"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.scalar-details-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_fixed_bytes(self.uuid.as_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_metadata_key_isolates_object_store_identity() {
        let first = IndexMetadataKey {
            version: 7,
            store_identity: "s3$first-options",
        };
        let second = IndexMetadataKey {
            version: 7,
            store_identity: "s3$second-options",
        };

        assert_ne!(first.key(), second.key());
    }

    /// The remap form is its own hierarchy segment, so it cannot be confused with a URI
    /// segment that happens to spell a form.
    ///
    /// Interpolating `{uri}/{mode}` into one segment makes these two handles identical:
    /// the unpartitioned handle for a dataset stored under `.../direct`, and the
    /// partitioned handle for the dataset one level up read in `Direct`. Both would frame
    /// the single string `memory://ds/direct`. Framing them separately keeps them apart.
    ///
    /// Asserted behaviourally, by whether an entry written to one is visible through the
    /// other: there is no namespace-scoped key enumeration to inspect instead.
    #[tokio::test]
    async fn remap_mode_prefix_cannot_collide_with_a_uri_segment() {
        let global = GlobalIndexCache(LanceCache::with_capacity(64 * 1024 * 1024));
        let uuid = Uuid::new_v4();
        let key = FragReuseIndexKey { uuid: &uuid };

        let uri_named_like_a_form = global.for_dataset("memory://ds/direct");
        let partitioned = global.for_dataset_with_remap_mode("memory://ds", IndexRemapMode::Direct);

        uri_named_like_a_form
            .insert_with_key(
                &key,
                Arc::new(FragReuseIndex::new(
                    uuid,
                    vec![],
                    lance_index::frag_reuse::FragReuseIndexDetails { versions: vec![] },
                )),
            )
            .await;

        assert!(
            partitioned.get_with_key(&key).await.is_none(),
            "the partitioned handle read an entry written by a dataset whose URI ends in a \
             form name, so the form is not its own cache segment"
        );
    }
}
