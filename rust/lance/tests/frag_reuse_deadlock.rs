// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reproduction of a re-entrant index-cache deadlock in
//! `DatasetIndexInternalExt::open_frag_reuse_index`.
//!
//! `open_frag_reuse_index` first calls `load_index_by_name` (which runs
//! `load_indices` and does a `get_or_insert_with_key` on the
//! `frag_reuse/{uuid}` cache key), then calls
//! `index_cache.get_or_insert_with_key` on the SAME key. If the entry is not
//! cached at that second call (e.g. it was evicted, or the cache cannot hold
//! it), the loader runs `self.load_index(&uuid)` -> `load_indices()` ->
//! `get_or_insert_with_key` on the same key again. moka's
//! `optionally_get_with` holds the waiter's write lock across the init
//! future, so the inner call parks on `read().await` against its own outer
//! write guard: a permanent, zero-CPU deadlock.
//!
//! We force the "entry not cached" condition deterministically by opening
//! the dataset with a Session whose index cache capacity is 0 (moka treats
//! `max_capacity == 0` as a disabled map: every get misses, every insert is
//! a no-op), so the second `get_or_insert_with_key` always runs its loader.

use std::sync::Arc;
use std::time::Duration;

use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};

use lance::dataset::builder::DatasetBuilder;
use lance::dataset::optimize::{CompactionOptions, compact_files};
use lance::dataset::{Dataset, WriteParams};
use lance::index::{DatasetIndexExt, DatasetIndexInternalExt};
use lance::session::Session;
use lance_index::IndexType;
use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::ScalarIndexParams;
use lance_io::object_store::ObjectStoreRegistry;

#[tokio::test]
async fn test_open_frag_reuse_index_does_not_deadlock_with_tiny_index_cache() {
    let tmp = tempfile::tempdir().unwrap();
    let uri = tmp.path().join("ds").to_str().unwrap().to_string();

    // (a) Small local dataset: 600 rows across 6 fragments.
    let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from_iter_values(0..600))],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let mut dataset = Dataset::write(
        reader,
        &uri,
        Some(WriteParams {
            max_rows_per_file: 100,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    // Some deletions so compaction has real work.
    dataset.delete("i < 50").await.unwrap();

    // BTree scalar index on `i`.
    dataset
        .create_index(
            &["i"],
            IndexType::Scalar,
            Some("i_idx".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();

    // (b) Compaction with deferred index remap creates the frag-reuse index.
    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();

    let indices = dataset.load_indices().await.unwrap();
    assert!(
        indices.iter().any(|i| i.name == FRAG_REUSE_INDEX_NAME),
        "frag-reuse index was not created by deferred-remap compaction"
    );

    // (c) Re-open with a Session whose INDEX cache capacity is zero, so the
    // `frag_reuse/{uuid}` entry can never stay cached between the probe in
    // load_indices() and the get_or_insert_with_key in open_frag_reuse_index.
    let session = Arc::new(Session::new(
        /*index_cache_size=*/ 0,
        /*metadata_cache_size=*/ 4 * 1024 * 1024,
        Arc::new(ObjectStoreRegistry::default()),
    ));
    let dataset = DatasetBuilder::from_uri(&uri)
        .with_session(session)
        .load()
        .await
        .unwrap();

    // (d) Call the poisoned path with a 20s timeout.
    let result = tokio::time::timeout(
        Duration::from_secs(20),
        dataset.open_frag_reuse_index(&NoOpMetricsCollector),
    )
    .await;

    match result {
        // Success: open_frag_reuse_index returned the frag-reuse index within the timeout.
        Ok(Ok(Some(_fri))) => {}
        Ok(Ok(None)) => panic!("unexpected: open_frag_reuse_index returned None (FRI exists)"),
        Ok(Err(e)) => panic!("unexpected error from open_frag_reuse_index: {e}"),
        Err(_elapsed) => panic!(
            "DEADLOCK REPRODUCED: open_frag_reuse_index hung for 20s at 0% CPU. \
             Re-entrant index_cache.get_or_insert_with_key on the frag_reuse/{{uuid}} key: \
             the outer moka waiter holds its write lock across the init future while the \
             inner load_indices() call awaits read() on the same waiter."
        ),
    }
}
