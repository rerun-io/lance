// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

// The printed table is this binary's result.
#![allow(clippy::print_stdout)]

//! Probe for the cost of the training read behind a bitmap index build.
//!
//! Writes (once) a multi-fragment dataset with a Boolean column `b` (1%
//! nulls), an Int64 column `i` (uniform over 100 values), a Utf8 column `s`
//! (10 head strings over 90% of rows, a 1M-string tail) and a
//! FixedSizeBinary(32) column `h` (uniform over 50,000 keys, 1% nulls), then
//! times, per column, the unordered training scan of `(value, _rowid)` alone,
//! the same scan driving the bitmap builder, the value-sorted scan the old
//! build needed, and the whole `create_index`.
//!
//! ```text
//! cargo bench -p lance --bench bitmap_scan_probe -- \
//!     [--path DIR] [--rows 10000000] [--col b,i,s,h] [--mode scan,build,sorted,index] [--repeat 2]
//! ```

use std::sync::Arc;
use std::time::Instant;

use arrow_array::{
    BooleanArray, FixedSizeBinaryArray, Int64Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::TryStreamExt;
use lance::dataset::scanner::ColumnOrdering;
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance::index::DatasetIndexExt;
use lance_core::cache::LanceCache;
use lance_index::IndexType;
use lance_index::scalar::bitmap::{BitmapBuildInput, BitmapBuildOptions, BitmapIndexPlugin};
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use rand::{Rng, SeedableRng, rngs::SmallRng};

const BATCH_ROWS: u64 = 8192;
const NUM_FRAGMENTS: u64 = 10;

/// Width of the fixed-size binary column's keys.
const FSB_WIDTH: i32 = 32;

/// Distinct keys of the fixed-size binary column.
const FSB_KEYS: u32 = 50_000;

/// A 32-byte key for key number `k`: four words of a splitmix64 of `k`, so
/// keys differ in every byte rather than in a counter's low bytes.
fn fsb_key(k: u32) -> [u8; FSB_WIDTH as usize] {
    let mut out = [0u8; FSB_WIDTH as usize];
    let mut x = (k as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
    for word in out.chunks_exact_mut(8) {
        x ^= x >> 30;
        x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^= x >> 31;
        word.copy_from_slice(&x.to_le_bytes());
        x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    }
    out
}

struct Args {
    path: String,
    rows: u64,
    cols: Vec<String>,
    modes: Vec<String>,
    repeat: usize,
}

fn parse_args() -> Args {
    let mut args = Args {
        path: "/tmp/bitmap_scan_probe".to_string(),
        rows: 10_000_000,
        cols: vec!["b".into(), "i".into(), "s".into(), "h".into()],
        modes: vec![
            "scan".into(),
            "build".into(),
            "sorted".into(),
            "index".into(),
        ],
        repeat: 2,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--path" => args.path = it.next().unwrap(),
            "--rows" => args.rows = it.next().unwrap().parse().unwrap(),
            "--col" => args.cols = it.next().unwrap().split(',').map(String::from).collect(),
            "--mode" => args.modes = it.next().unwrap().split(',').map(String::from).collect(),
            "--repeat" => args.repeat = it.next().unwrap().parse().unwrap(),
            _ => {}
        }
    }
    args
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("b", DataType::Boolean, true),
        Field::new("i", DataType::Int64, true),
        Field::new("s", DataType::Utf8, true),
        Field::new("h", DataType::FixedSizeBinary(FSB_WIDTH), true),
    ]))
}

fn batch(batch_idx: u64, rows: u64) -> RecordBatch {
    let start = batch_idx * BATCH_ROWS;
    let end = (start + BATCH_ROWS).min(rows);
    let mut rng = SmallRng::seed_from_u64(batch_idx);
    let b = BooleanArray::from(
        (start..end)
            .map(|_| rng.random_bool(0.99).then(|| rng.random_bool(0.5)))
            .collect::<Vec<_>>(),
    );
    let i = Int64Array::from(
        (start..end)
            .map(|_| rng.random_range(0..100i64))
            .collect::<Vec<_>>(),
    );
    let s = StringArray::from(
        (start..end)
            .map(|_| {
                if rng.random_bool(0.9) {
                    format!("head_{:02}", rng.random_range(0..10u32))
                } else {
                    format!("tail_{:07}", rng.random_range(0..1_000_000u32))
                }
            })
            .collect::<Vec<_>>(),
    );
    let h = FixedSizeBinaryArray::try_from_sparse_iter_with_size(
        (start..end).map(|_| {
            rng.random_bool(0.99)
                .then(|| fsb_key(rng.random_range(0..FSB_KEYS)))
        }),
        FSB_WIDTH,
    )
    .unwrap();
    RecordBatch::try_new(
        schema(),
        vec![Arc::new(b), Arc::new(i), Arc::new(s), Arc::new(h)],
    )
    .unwrap()
}

/// Reuse the dataset at `path` when it has every column and `rows` rows,
/// else write it afresh.
async fn open_or_write(path: &str, rows: u64) -> Dataset {
    if let Ok(dataset) = Dataset::open(path).await
        && schema()
            .fields()
            .iter()
            .all(|field| dataset.schema().field(field.name()).is_some())
        && dataset.count_rows(None).await.unwrap() as u64 == rows
    {
        return dataset;
    }
    let start = Instant::now();
    let num_batches = rows.div_ceil(BATCH_ROWS);
    let reader = RecordBatchIterator::new(
        (0..num_batches).map(move |idx| Ok(batch(idx, rows))),
        schema(),
    );
    let params = WriteParams {
        max_rows_per_file: rows.div_ceil(NUM_FRAGMENTS) as usize,
        mode: WriteMode::Overwrite,
        ..Default::default()
    };
    let dataset = Dataset::write(reader, path, Some(params)).await.unwrap();
    println!(
        "wrote {} rows in {} fragments in {:.1}s",
        rows,
        dataset.get_fragments().len(),
        start.elapsed().as_secs_f64()
    );
    dataset
}

/// The scan `scan_training_data` issues for `TrainingOrdering::None`
/// (`project_with_transform` + `with_row_id`, no ordering), or with the
/// value ordering it issued for `TrainingOrdering::Values`.
async fn training_stream(dataset: &Dataset, col: &str, sorted: bool) -> SendableRecordBatchStream {
    let mut scan = dataset.scan();
    if sorted {
        scan.order_by(Some(vec![ColumnOrdering::asc_nulls_first(col.to_string())]))
            .unwrap();
    }
    scan.with_row_id();
    scan.project_with_transform(&[("value", col)]).unwrap();
    SendableRecordBatchStream::from(scan.try_into_stream().await.unwrap())
}

/// Rows, batches and decoded bytes (`get_array_memory_size`) of a drained stream.
async fn drain(mut stream: SendableRecordBatchStream) -> (u64, u64, u64) {
    let (mut rows, mut batches, mut bytes) = (0u64, 0u64, 0u64);
    while let Some(batch) = stream.try_next().await.unwrap() {
        rows += batch.num_rows() as u64;
        batches += 1;
        bytes += batch.column(0).get_array_memory_size() as u64;
    }
    (rows, batches, bytes)
}

#[tokio::main]
async fn main() {
    let args = parse_args();
    let mut dataset = open_or_write(&args.path, args.rows).await;

    println!(
        "{:<4} {:<8} {:>4} {:>8}  detail",
        "col", "mode", "run", "wall_s"
    );
    for col in &args.cols {
        for mode in &args.modes {
            for run in 0..args.repeat {
                let start = Instant::now();
                let detail = match mode.as_str() {
                    "scan" | "sorted" => {
                        let stream = training_stream(&dataset, col, mode == "sorted").await;
                        let (rows, batches, bytes) = drain(stream).await;
                        format!(
                            "rows={rows} batches={batches} value_MiB={:.1}",
                            bytes as f64 / (1024.0 * 1024.0)
                        )
                    }
                    "build" => {
                        let tempdir = tempfile::tempdir().unwrap();
                        let store = LanceIndexStore::new(
                            Arc::new(ObjectStore::local()),
                            Path::from_filesystem_path(tempdir.path()).unwrap(),
                            Arc::new(LanceCache::no_cache()),
                        );
                        let stream = training_stream(&dataset, col, false).await;
                        let (_, report) = BitmapIndexPlugin::train_bitmap_index_with_options(
                            BitmapBuildInput::Stream(stream),
                            &store,
                            &BitmapBuildOptions::default(),
                        )
                        .await
                        .unwrap();
                        format!(
                            "{:?} keys={} spill={} first_pass_s={:.2}",
                            report.strategy,
                            report.distinct_keys,
                            report.replayed_spill,
                            report.first_pass_time.as_secs_f64()
                        )
                    }
                    "index" => {
                        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap);
                        dataset
                            .create_index(
                                &[col],
                                IndexType::Bitmap,
                                Some(format!("{col}_bitmap")),
                                &params,
                                true,
                            )
                            .await
                            .unwrap();
                        "create_index + commit".to_string()
                    }
                    other => panic!("unknown mode {other}"),
                };
                println!(
                    "{:<4} {:<8} {:>4} {:>8.2}  {}",
                    col,
                    mode,
                    run,
                    start.elapsed().as_secs_f64(),
                    detail
                );
            }
        }
    }
}
