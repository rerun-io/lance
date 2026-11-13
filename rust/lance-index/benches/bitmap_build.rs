// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

// The printed table is this binary's result.
#![allow(clippy::print_stdout)]

//! Measures building a bitmap index from unsorted synthetic streams: wall
//! time, the peak heap allocated during the build (through a counting global
//! allocator, so it includes the sorter, the writer and Arrow batches, not
//! just the builder's state), and what the builder reports it did.
//!
//! It is a plain binary rather than a Criterion group because each build is
//! run once: at 10M rows a build takes seconds, and the figure of interest is
//! the memory profile of one run, not a distribution of timings.
//!
//! ```text
//! cargo bench -p lance-index --bench bitmap_build -- \
//!     [--rows 1000000,10000000] \
//!     [--dist bool_nulls,uniform_low,zipf,uniform_high,utf8_200,utf8_200_skew,fsb_50k,fsb_50k_skew] \
//!     [--variant stream,source,stream_passes,source_passes,partitioned,sorted] \
//!     [--pool-bytes 536870912]
//! ```
//!
//! Variants are [`BitmapBuildOptions`] presets over an input the build reads
//! once (`stream`: a reread replays a spill) or can reopen (`source`: the
//! generator runs again). `*_passes` caps a partition pass at 64 MiB of row
//! ids, so a 10M-row input takes two passes; `partitioned` turns the direct
//! pass off so every input is counted and partitioned; `sorted` forces the
//! fallback, which is the value-sorted build the partitioning replaces.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use arrow_array::{
    BooleanArray, FixedSizeBinaryArray, Int64Array, RecordBatch, StringArray, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SortOptions};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{ExecutionPlan, SendableRecordBatchStream, sorts::sort::SortExec};
use datafusion_physical_expr::{PhysicalSortExpr, expressions::Column};
use futures::stream;
use lance_core::cache::LanceCache;
use lance_datafusion::exec::{LanceExecutionOptions, OneShotExec, execute_plan};
use lance_index::scalar::bitmap::{
    BitmapBuildInput, BitmapBuildOptions, BitmapBuildReport, BitmapIndexPlugin,
    DEFAULT_PARTITION_MEMORY_LIMIT_BYTES,
};
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::registry::{TrainingCriteria, TrainingDataSource, TrainingOrdering};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use rand::{Rng, SeedableRng, rngs::SmallRng};

/// Counts live and peak heap bytes so a build's footprint can be read without
/// an allocator-specific API.
struct CountingAllocator;

static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let live = LIVE_BYTES.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn reset_peak() -> usize {
    let live = LIVE_BYTES.load(Ordering::Relaxed);
    PEAK_BYTES.store(live, Ordering::Relaxed);
    live
}

const BATCH_ROWS: u64 = 8192;

/// Width of the fixed-size binary keys.
const FSB_WIDTH: i32 = 32;

/// Distinct keys of the fixed-size binary distributions.
const FSB_KEYS: u32 = 50_000;

/// Distinct keys of the short-string distributions.
const UTF8_KEYS: u32 = 200;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Distribution {
    /// Boolean, 1% nulls, the rest a fair coin.
    BoolNulls,
    /// Int64 uniform over 100 keys.
    UniformLow,
    /// Int64: 90% of rows over 10 head keys, 10% over a 1M-key tail.
    Zipf,
    /// Int64 equal to the row number: every value distinct.
    UniformHigh,
    /// Utf8 uniform over 200 short strings, 1% nulls.
    Utf8_200,
    /// Utf8: 90% of rows over 10 head strings, 10% over the other 190, 1% nulls.
    Utf8_200Skew,
    /// FixedSizeBinary(32) uniform over 50,000 keys, 1% nulls.
    Fsb50k,
    /// FixedSizeBinary(32): 90% of rows over 10 head keys, 10% over the
    /// other 49,990, 1% nulls.
    Fsb50kSkew,
}

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

/// `None` 1% of the time, else a key in `0..keys`, uniform or with 90% of
/// the rows on the first 10 keys.
fn key_number(rng: &mut SmallRng, keys: u32, skewed: bool) -> Option<u32> {
    rng.random_bool(0.99).then(|| {
        if skewed && rng.random_bool(0.9) {
            rng.random_range(0..10)
        } else if skewed {
            rng.random_range(10..keys)
        } else {
            rng.random_range(0..keys)
        }
    })
}

impl Distribution {
    const ALL: [Self; 8] = [
        Self::BoolNulls,
        Self::UniformLow,
        Self::Zipf,
        Self::UniformHigh,
        Self::Utf8_200,
        Self::Utf8_200Skew,
        Self::Fsb50k,
        Self::Fsb50kSkew,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::BoolNulls => "bool_nulls",
            Self::UniformLow => "uniform_low",
            Self::Zipf => "zipf",
            Self::UniformHigh => "uniform_high",
            Self::Utf8_200 => "utf8_200",
            Self::Utf8_200Skew => "utf8_200_skew",
            Self::Fsb50k => "fsb_50k",
            Self::Fsb50kSkew => "fsb_50k_skew",
        }
    }

    fn value_type(self) -> DataType {
        match self {
            Self::BoolNulls => DataType::Boolean,
            Self::UniformLow | Self::Zipf | Self::UniformHigh => DataType::Int64,
            Self::Utf8_200 | Self::Utf8_200Skew => DataType::Utf8,
            Self::Fsb50k | Self::Fsb50kSkew => DataType::FixedSizeBinary(FSB_WIDTH),
        }
    }

    /// Batches are generated as they are pulled, from a per-batch seed, so the
    /// stream costs one batch of memory at a time and is reproducible.
    fn stream(self, rows: u64) -> SendableRecordBatchStream {
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", self.value_type(), true),
            Field::new("_rowid", DataType::UInt64, false),
        ]));
        let num_batches = rows.div_ceil(BATCH_ROWS);
        let batch_schema = schema.clone();
        let batches = (0..num_batches).map(move |batch_idx| {
            let start = batch_idx * BATCH_ROWS;
            let end = (start + BATCH_ROWS).min(rows);
            let mut rng = SmallRng::seed_from_u64(batch_idx);
            let row_ids = Arc::new(UInt64Array::from((start..end).collect::<Vec<_>>()));
            let values: Arc<dyn arrow_array::Array> = match self {
                Self::BoolNulls => Arc::new(BooleanArray::from(
                    (start..end)
                        .map(|_| rng.random_bool(0.99).then(|| rng.random_bool(0.5)))
                        .collect::<Vec<_>>(),
                )),
                Self::UniformLow => Arc::new(Int64Array::from(
                    (start..end)
                        .map(|_| rng.random_range(0..100i64))
                        .collect::<Vec<_>>(),
                )),
                Self::Zipf => Arc::new(Int64Array::from(
                    (start..end)
                        .map(|_| {
                            if rng.random_bool(0.9) {
                                rng.random_range(0..10i64)
                            } else {
                                rng.random_range(10..1_000_010i64)
                            }
                        })
                        .collect::<Vec<_>>(),
                )),
                Self::UniformHigh => Arc::new(Int64Array::from(
                    (start..end).map(|i| i as i64).collect::<Vec<_>>(),
                )),
                Self::Utf8_200 | Self::Utf8_200Skew => Arc::new(StringArray::from(
                    (start..end)
                        .map(|_| {
                            key_number(&mut rng, UTF8_KEYS, self == Self::Utf8_200Skew)
                                .map(|k| format!("key_{k:03}"))
                        })
                        .collect::<Vec<_>>(),
                )),
                Self::Fsb50k | Self::Fsb50kSkew => Arc::new(
                    FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                        (start..end).map(|_| {
                            key_number(&mut rng, FSB_KEYS, self == Self::Fsb50kSkew).map(fsb_key)
                        }),
                        FSB_WIDTH,
                    )
                    .unwrap(),
                ),
            };
            Ok(RecordBatch::try_new(batch_schema.clone(), vec![values, row_ids]).unwrap())
        });
        Box::pin(RecordBatchStreamAdapter::new(schema, stream::iter(batches)))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    Stream,
    Source,
    StreamPasses,
    SourcePasses,
    Partitioned,
    Sorted,
}

impl Variant {
    const ALL: [Self; 6] = [
        Self::Stream,
        Self::Source,
        Self::StreamPasses,
        Self::SourcePasses,
        Self::Partitioned,
        Self::Sorted,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Stream => "stream",
            Self::Source => "source",
            Self::StreamPasses => "stream_passes",
            Self::SourcePasses => "source_passes",
            Self::Partitioned => "partitioned",
            Self::Sorted => "sorted",
        }
    }

    fn is_source(self) -> bool {
        matches!(self, Self::Source | Self::SourcePasses | Self::Partitioned)
    }

    fn options(self, pool_bytes: Option<u64>) -> BitmapBuildOptions {
        let base = BitmapBuildOptions {
            sort_mem_pool_bytes: pool_bytes,
            ..BitmapBuildOptions::default()
        };
        match self {
            Self::Stream | Self::Source => base,
            Self::StreamPasses | Self::SourcePasses => BitmapBuildOptions {
                partition_memory_limit_bytes: DEFAULT_PARTITION_MEMORY_LIMIT_BYTES / 8,
                ..base
            },
            Self::Partitioned => BitmapBuildOptions {
                max_direct_keys: 0,
                ..base
            },
            Self::Sorted => BitmapBuildOptions {
                max_direct_keys: 0,
                max_distinct_keys: 1,
                ..base
            },
        }
    }
}

/// The generator as a reopenable source: every open runs it again from the
/// same seeds, in the same order. A value-sorted open (what the sorted
/// fallback asks a source for) runs the generated stream through the same
/// spilling DataFusion sort the scanner would use.
struct GeneratedSource {
    dist: Distribution,
    rows: u64,
    pool_bytes: Option<u64>,
}

#[async_trait::async_trait]
impl TrainingDataSource for GeneratedSource {
    async fn open(
        &self,
        criteria: &TrainingCriteria,
    ) -> lance_core::Result<SendableRecordBatchStream> {
        let stream = self.dist.stream(self.rows);
        match criteria.ordering {
            TrainingOrdering::None => Ok(stream),
            TrainingOrdering::Values => {
                let input = Arc::new(OneShotExec::new(stream));
                let value_idx = input.schema().index_of("value")?;
                let sort_expr = PhysicalSortExpr {
                    expr: Arc::new(Column::new("value", value_idx)),
                    options: SortOptions {
                        descending: false,
                        nulls_first: true,
                    },
                };
                let plan = Arc::new(SortExec::new([sort_expr].into(), input));
                execute_plan(
                    plan,
                    LanceExecutionOptions {
                        use_spilling: true,
                        mem_pool_size: self.pool_bytes,
                        ..Default::default()
                    },
                )
            }
            TrainingOrdering::Addresses => Ok(stream),
        }
    }
}

struct Args {
    rows: Vec<u64>,
    dists: Vec<Distribution>,
    variants: Vec<Variant>,
    pool_bytes: Option<u64>,
}

fn parse_args() -> Args {
    let mut args = Args {
        rows: vec![1_000_000, 10_000_000],
        dists: Distribution::ALL.to_vec(),
        variants: Variant::ALL.to_vec(),
        pool_bytes: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--rows" => {
                args.rows = it
                    .next()
                    .expect("--rows takes a list")
                    .split(',')
                    .map(|s| s.parse().expect("row count"))
                    .collect();
            }
            "--dist" => {
                args.dists = it
                    .next()
                    .expect("--dist takes a list")
                    .split(',')
                    .map(|name| {
                        *Distribution::ALL
                            .iter()
                            .find(|d| d.name() == name)
                            .unwrap_or_else(|| panic!("unknown distribution {name}"))
                    })
                    .collect();
            }
            "--variant" => {
                args.variants = it
                    .next()
                    .expect("--variant takes a list")
                    .split(',')
                    .map(|name| {
                        *Variant::ALL
                            .iter()
                            .find(|v| v.name() == name)
                            .unwrap_or_else(|| panic!("unknown variant {name}"))
                    })
                    .collect();
            }
            "--pool-bytes" => {
                args.pool_bytes = Some(
                    it.next()
                        .expect("--pool-bytes takes a size")
                        .parse()
                        .unwrap(),
                );
            }
            // `cargo bench` passes `--bench`; anything else is Criterion-style noise.
            _ => {}
        }
    }
    args
}

async fn build(
    dist: Distribution,
    rows: u64,
    variant: Variant,
    options: &BitmapBuildOptions,
) -> lance_core::Result<BitmapBuildReport> {
    let tempdir = tempfile::tempdir().unwrap();
    let store = LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        Path::from_filesystem_path(tempdir.path()).unwrap(),
        Arc::new(LanceCache::no_cache()),
    );
    let source = GeneratedSource {
        dist,
        rows,
        pool_bytes: options.sort_mem_pool_bytes,
    };
    let input = if variant.is_source() {
        BitmapBuildInput::Source(&source)
    } else {
        BitmapBuildInput::Stream(dist.stream(rows))
    };
    let (_, report) =
        BitmapIndexPlugin::train_bitmap_index_with_options(input, &store, options).await?;
    Ok(report)
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn main() {
    let args = parse_args();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    println!(
        "{:<13} {:>10} {:<13} {:>7} {:>9}  report",
        "distribution", "rows", "variant", "wall_s", "peak_MiB"
    );
    for &rows in &args.rows {
        for &dist in &args.dists {
            for &variant in &args.variants {
                let options = variant.options(args.pool_bytes);
                let live_before = reset_peak();
                let start = Instant::now();
                let outcome = runtime.block_on(build(dist, rows, variant, &options));
                let wall = start.elapsed().as_secs_f64();
                let peak = PEAK_BYTES.load(Ordering::Relaxed) - live_before;
                let report = match outcome {
                    Ok(report) => format!(
                        "{:?} keys={} admitted={} scattered={} passes={} direct={} spill={} first_pass_s={:.2} partition_s={:.2}",
                        report.strategy,
                        report.distinct_keys,
                        report.admitted_keys,
                        report.scattered_rows,
                        report.partition_passes,
                        report.direct_keys,
                        report.replayed_spill,
                        report.first_pass_time.as_secs_f64(),
                        report.partition_time.as_secs_f64()
                    ),
                    Err(err) => format!("FAILED: {}", err.to_string().lines().next().unwrap_or("")),
                };
                println!(
                    "{:<13} {:>10} {:<13} {:>7.2} {:>9.1}  {}",
                    dist.name(),
                    rows,
                    variant.name(),
                    wall,
                    mib(peak),
                    report
                );
            }
        }
    }
}
