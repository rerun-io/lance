// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! How a bitmap build turns an unsorted `(value, row_id)` input into row sets
//! in key order.
//!
//! ```text
//! input ──► KeyTable (value → dense id) ──► one row set per id, each batch's
//!           rows for a key appended as one ascending run (the scan is in
//!           address order)
//!                             │
//!           at most max_direct_keys keys seen: emitted in key order
//!                             │ more keys than that: the row sets are dropped
//!                             ▼ and the input is read again, value-sorted
//!           consecutive equal values form one run, null first
//! ```
//!
//! A type whose domain is small and fixed (Boolean, 8-bit integers) always
//! finishes in the one pass, whatever the limits: every possible key gets a
//! row set up front and nothing is spilled or reread
//! ([`BuildStrategy::Direct`]). Any other type finishes in that pass when it
//! has at most [`BitmapBuildOptions::max_direct_keys`] distinct keys; above
//! that the per-key accumulators (a row set per key, with a bitmap per
//! fragment the key touches) would cost more than the index is worth, and
//! the build falls back to a value-sorted read of the input -- the scanner's
//! sort for a [`TrainingDataSource`], a spilling DataFusion sort of a replay
//! for a caller that only has a stream -- and takes its runs from that
//! ([`BuildStrategy::Sorted`]). The replay keeps the batches the first pass
//! saw, in memory up to [`BitmapBuildOptions::replay_memory_limit_bytes`]
//! and in an Arrow IPC file beyond.

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use arrow::compute::partition;
use arrow_array::{Array, ArrayRef, RecordBatch, UInt64Array};
use arrow_schema::{DataType, SchemaRef};
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion_common::ScalarValue;
use futures::TryStreamExt;
use lance_core::{Error, ROW_ID, Result, error::LanceOptionExt};
use lance_datafusion::spill::{SpillReceiver, SpillSender, create_replay_spill};
use lance_select::RowAddrTreeMap;
use tracing::{info, warn};

use super::keys::{KeyTable, SortedKeys};
use crate::scalar::registry::{
    TrainingCriteria, TrainingDataSource, TrainingOrdering, VALUE_COLUMN_NAME, sort_stream_by_value,
};

/// Default for [`BitmapBuildOptions::max_direct_keys`]: 4096 keys.
///
/// A direct build keeps one row set per key, and a row set holds one bitmap
/// per fragment the key has rows in, so its resident overhead grows with
/// keys × fragments on top of the rows themselves; a few thousand keys keep
/// that in the tens of megabytes even on a dataset of hundreds of fragments.
pub const DEFAULT_MAX_DIRECT_KEYS: usize = 1 << 12;

/// Default for [`BitmapBuildOptions::replay_memory_limit_bytes`]: 512 MiB.
///
/// How much of a single-stream input stays in memory for a reread before the
/// replay spills to disk.
pub const DEFAULT_REPLAY_MEMORY_LIMIT_BYTES: usize = 512 * 1024 * 1024;

/// Default for [`BitmapBuildOptions::max_distinct_keys`]: 2^22 keys.
///
/// The key table holds every distinct value (bits or bytes, plus the map
/// slot), so a near-unique column of many millions of rows would cost
/// hundreds of megabytes before any row set exists; past this many keys the
/// first pass stops assigning ids and only keeps the input for the sorted
/// read. A bitmap index on such a column is also a poor fit regardless of
/// how it is built.
pub const DEFAULT_MAX_DISTINCT_KEYS: usize = 1 << 22;

/// Ids are `u32`, with the lowest value taken by null and the highest kept
/// free, so a table can never hold more keys than this whatever
/// `max_distinct_keys` says; past it the build sorts.
const MAX_TABLE_KEYS: usize = u32::MAX as usize - 2;

/// Tunables for a bitmap build.
///
/// The defaults are read from the environment once per process when set,
/// which lets an experiment change them without a code change, but every
/// field can also be set directly, which is the supported way to configure a
/// build:
///
/// | field                        | environment override                     |
/// |------------------------------|------------------------------------------|
/// | `max_direct_keys`            | `LANCE_BITMAP_MAX_DIRECT_KEYS`           |
/// | `replay_memory_limit_bytes`  | `LANCE_BITMAP_REPLAY_MEMORY_LIMIT_BYTES` |
/// | `max_distinct_keys`          | `LANCE_BITMAP_MAX_DISTINCT_KEYS`         |
///
/// `sort_mem_pool_bytes` has no override of its own: left `None`, the
/// fallback sort runs in the pool every Lance plan uses, which
/// `LANCE_MEM_POOL_SIZE` sets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitmapBuildOptions {
    /// Distinct non-null keys up to which the build finishes in its first
    /// pass with a row set per key. A type with a small fixed domain
    /// (Boolean, 8-bit integers) always does, whatever this says. Capped by
    /// `max_distinct_keys`.
    pub max_direct_keys: usize,
    /// Bytes of a single-stream input kept in memory for a reread before the
    /// replay spills to disk. A reopenable source is reread instead and never
    /// spills.
    pub replay_memory_limit_bytes: usize,
    /// Distinct non-null keys above which the first pass stops assigning
    /// ids: the key table is dropped and the input is only kept for the
    /// sorted read.
    pub max_distinct_keys: usize,
    /// Size of the DataFusion memory pool the fallback sort runs in. `None`
    /// uses the shared Lance default; a value of its own gives the sort a pool
    /// of its own.
    pub sort_mem_pool_bytes: Option<u64>,
}

impl Default for BitmapBuildOptions {
    /// The environment is read once per process, so every build sees the same
    /// defaults and a malformed override is reported once.
    fn default() -> Self {
        static FROM_ENV: OnceLock<BitmapBuildOptions> = OnceLock::new();
        FROM_ENV
            .get_or_init(|| Self {
                max_direct_keys: env_override(
                    "LANCE_BITMAP_MAX_DIRECT_KEYS",
                    DEFAULT_MAX_DIRECT_KEYS,
                ),
                replay_memory_limit_bytes: env_override(
                    "LANCE_BITMAP_REPLAY_MEMORY_LIMIT_BYTES",
                    DEFAULT_REPLAY_MEMORY_LIMIT_BYTES,
                ),
                max_distinct_keys: env_override(
                    "LANCE_BITMAP_MAX_DISTINCT_KEYS",
                    DEFAULT_MAX_DISTINCT_KEYS,
                ),
                sort_mem_pool_bytes: None,
            })
            .clone()
    }
}

impl BitmapBuildOptions {
    pub(super) fn validate(&self) -> Result<()> {
        if self.max_distinct_keys == 0 {
            return Err(Error::invalid_input(
                "bitmap index: max_distinct_keys must be at least 1".to_string(),
            ));
        }
        Ok(())
    }
}

fn env_override<T: std::str::FromStr + Copy>(name: &str, default: T) -> T {
    match std::env::var(name) {
        Ok(raw) => raw.parse().unwrap_or_else(|_| {
            warn!("ignoring {name}={raw:?}: not a valid value, using the default");
            default
        }),
        Err(_) => default,
    }
}

/// How a build produced its runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BuildStrategy {
    /// Every key got a row set in the one and only pass, with nothing
    /// spilled or reread: the value type fixes a small domain (Boolean,
    /// 8-bit integers), or the input had at most
    /// [`BitmapBuildOptions::max_direct_keys`] keys.
    #[default]
    Direct,
    /// Too many distinct keys for a row set each: the input was value-sorted.
    Sorted,
}

/// What a bitmap build did with its input, for logs, tests and benchmarks.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BitmapBuildReport {
    pub strategy: BuildStrategy,
    /// Rows of new input; 0 on the [`BuildStrategy::Sorted`] path, which
    /// does not count them.
    pub rows: u64,
    /// Distinct keys with at least one row, nulls counted as one; 0 on the
    /// sorted path.
    pub distinct_keys: usize,
    /// Whether the sorted read replayed a spill of the input rather than
    /// reopening it.
    pub replayed_spill: bool,
    /// Time in the first pass, including the spill tee: the whole build of a
    /// [`BuildStrategy::Direct`] one.
    pub first_pass_time: Duration,
}

/// The input of a build: a source the build can reopen, or a single stream.
pub enum BitmapBuildInput<'a> {
    /// Reopened for every pass, with the criteria each pass wants.
    Source(&'a dyn TrainingDataSource),
    /// Read once; a reread replays a spill of it.
    Stream(SendableRecordBatchStream),
}

/// Where a reread of the input gets its batches.
enum Replay<'a> {
    Source(&'a dyn TrainingDataSource),
    Spill {
        receiver: SpillReceiver,
        sender: SpillSender,
        /// Holds the spill file's directory until the build is done.
        _tempdir: tempfile::TempDir,
    },
}

impl<'a> Replay<'a> {
    fn new(
        source: Option<&'a dyn TrainingDataSource>,
        schema: SchemaRef,
        options: &BitmapBuildOptions,
    ) -> Result<Self> {
        if let Some(source) = source {
            return Ok(Self::Source(source));
        }
        let tempdir = tempfile::tempdir().map_err(|e| {
            Error::io(format!(
                "bitmap index: cannot create a directory for the input replay: {e}"
            ))
        })?;
        let (sender, receiver) = create_replay_spill(
            tempdir.path().join("bitmap_training_replay.arrow"),
            schema,
            options.replay_memory_limit_bytes,
        );
        Ok(Self::Spill {
            receiver,
            sender,
            _tempdir: tempdir,
        })
    }

    /// Keep a batch of the first pass for the rereads (a spill only; a
    /// source is reopened).
    async fn tee(&mut self, batch: &RecordBatch) -> Result<()> {
        if let Self::Spill { sender, .. } = self {
            sender.write(batch.clone()).await?;
        }
        Ok(())
    }

    async fn finish(&mut self) -> Result<()> {
        if let Self::Spill { sender, .. } = self {
            sender.finish().await?;
        }
        Ok(())
    }

    /// The input in value order, nulls first, for the sorted fallback.
    ///
    /// Nulls first is what `TrainingOrdering::Values` means (the Lance
    /// scanner sorts `asc_nulls_first`, as does [`sort_stream_by_value`]);
    /// the build relies on it to write the null entry at the head of the file.
    async fn open_sorted(&self, options: &BitmapBuildOptions) -> Result<SendableRecordBatchStream> {
        match self {
            Self::Source(source) => {
                source
                    .open(&TrainingCriteria::new(TrainingOrdering::Values).with_row_id())
                    .await
            }
            Self::Spill { receiver, .. } => {
                sort_stream_by_value(receiver.read(), options.sort_mem_pool_bytes)
            }
        }
    }

    fn is_spill(&self) -> bool {
        matches!(self, Self::Spill { .. })
    }
}

fn value_and_row_id_columns(batch: &RecordBatch) -> Result<(&dyn Array, &UInt64Array)> {
    let values = batch.column_by_name(VALUE_COLUMN_NAME).expect_ok()?;
    let row_ids = batch.column_by_name(ROW_ID).expect_ok()?;
    let row_ids = row_ids
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "bitmap index: expected a UInt64 {ROW_ID} column, got {}",
                row_ids.data_type()
            ))
        })?;
    if values.len() != row_ids.len() {
        return Err(Error::invalid_input(format!(
            "bitmap index: {} values but {} row ids in one batch",
            values.len(),
            row_ids.len()
        )));
    }
    Ok((values.as_ref(), row_ids))
}

/// Runs of `(key, row set)` in ascending key order, null first, from either
/// strategy.
pub(super) enum RunProducer<'a> {
    Direct(Box<DirectRuns>),
    Sorted(Box<SortedFallback<'a>>),
}

impl RunProducer<'_> {
    pub(super) async fn next_run(&mut self) -> Result<Option<(ScalarValue, RowAddrTreeMap)>> {
        match self {
            Self::Direct(runs) => Ok(runs.pending.pop_front()),
            Self::Sorted(fallback) => fallback.runs.next_run().await,
        }
    }

    pub(super) fn report(&self) -> BitmapBuildReport {
        match self {
            Self::Direct(runs) => runs.report.clone(),
            Self::Sorted(fallback) => fallback.report.clone(),
        }
    }

    pub(super) fn value_type(&self) -> &DataType {
        match self {
            Self::Direct(runs) => &runs.value_type,
            Self::Sorted(fallback) => &fallback.value_type,
        }
    }
}

/// The row sets of a direct build while the first pass is running: one per
/// dense id, grown as ids are handed out, with each batch's rows for a key
/// collected into a run and appended once.
#[derive(Default)]
struct DirectAccumulator {
    row_sets: Vec<RowAddrTreeMap>,
    /// This batch's rows per id, ascending in scan order.
    runs: Vec<Vec<u64>>,
    /// Ids with rows in this batch: the only runs the flush walks, so a
    /// batch costs its rows, not the whole domain.
    touched: Vec<u32>,
}

impl DirectAccumulator {
    fn push_batch(&mut self, num_ids: usize, ids: &[u32], row_ids: &[u64]) {
        self.row_sets.resize_with(num_ids, RowAddrTreeMap::new);
        self.runs.resize_with(num_ids, Vec::new);
        for (id, row_id) in ids.iter().zip(row_ids) {
            let run = &mut self.runs[*id as usize];
            if run.is_empty() {
                self.touched.push(*id);
            }
            run.push(*row_id);
        }
        for id in self.touched.drain(..) {
            let run = &mut self.runs[id as usize];
            self.row_sets[id as usize].insert_sorted(run);
            run.clear();
        }
    }

    /// The row sets in key order, keys with no rows dropped.
    fn into_runs(self, table: &KeyTable) -> Result<VecDeque<(ScalarValue, RowAddrTreeMap)>> {
        let SortedKeys { keys, remap } = table.sorted_keys()?;
        let mut by_position = (0..keys.len()).map(|_| None).collect::<Vec<_>>();
        for (id, row_set) in self.row_sets.into_iter().enumerate() {
            if !lance_select::RowSetOps::is_empty(&row_set) {
                by_position[remap[id] as usize] = Some(row_set);
            }
        }
        let mut pending = VecDeque::new();
        for (position, row_set) in by_position.into_iter().enumerate() {
            if let Some(row_set) = row_set {
                pending.push_back((keys[position].clone(), row_set));
            }
        }
        Ok(pending)
    }
}

/// The runs of a direct build, already in key order.
pub(super) struct DirectRuns {
    pending: VecDeque<(ScalarValue, RowAddrTreeMap)>,
    value_type: DataType,
    report: BitmapBuildReport,
}

/// Read `input` once, building every key's row set as long as the keys fit
/// the direct cap, then either finish or fall back to a sorted read.
pub(super) async fn plan_build<'a>(
    input: BitmapBuildInput<'a>,
    options: &BitmapBuildOptions,
) -> Result<RunProducer<'a>> {
    options.validate()?;
    let started = Instant::now();

    let (mut stream, source) = match input {
        BitmapBuildInput::Source(source) => (
            source
                .open(&TrainingCriteria::new(TrainingOrdering::None).with_row_id())
                .await?,
            Some(source),
        ),
        BitmapBuildInput::Stream(stream) => (stream, None),
    };
    let schema = stream.schema();
    let value_type = schema
        .field(schema.index_of(VALUE_COLUMN_NAME)?)
        .data_type()
        .clone();
    let mut table = KeyTable::new(&value_type);

    // A fixed domain can never outgrow the direct cap, so its input is
    // never reread and need not be kept; the caps only apply to a table
    // that grows with the values it sees.
    let (mut replay, direct_cap, key_cap) = if table.has_fixed_domain() {
        (None, usize::MAX, usize::MAX)
    } else {
        let key_cap = options.max_distinct_keys.min(MAX_TABLE_KEYS);
        (
            Some(Replay::new(source, schema.clone(), options)?),
            options.max_direct_keys.min(key_cap),
            key_cap,
        )
    };

    let mut accumulator = DirectAccumulator::default();
    let mut ids = Vec::new();
    let mut rows = 0u64;
    let mut too_many_keys = false;

    while let Some(batch) = stream.try_next().await? {
        if let Some(replay) = &mut replay {
            replay.tee(&batch).await?;
        }
        // Past the cap only the replay matters: the rest of the stream still
        // has to reach the spill for the sorted fallback to see it.
        if too_many_keys {
            continue;
        }
        let (values, row_ids) = value_and_row_id_columns(&batch)?;
        ids.clear();
        table.assign_ids(values, &mut ids)?;
        rows += row_ids.len() as u64;
        let distinct = table.num_ids() - 1;
        if distinct > direct_cap || distinct > key_cap {
            too_many_keys = true;
            accumulator = DirectAccumulator::default();
            if !replay.as_ref().is_some_and(Replay::is_spill) {
                break;
            }
            continue;
        }
        accumulator.push_batch(table.num_ids(), &ids, row_ids.values());
    }
    if let Some(replay) = &mut replay {
        replay.finish().await?;
    }
    let first_pass_time = started.elapsed();

    if too_many_keys {
        let replay = replay.expect_ok()?;
        info!(
            "bitmap index: more than {} distinct keys; sorting the input instead of holding a row set per key",
            direct_cap
        );
        drop(table);
        let stream = replay.open_sorted(options).await?;
        let report = BitmapBuildReport {
            strategy: BuildStrategy::Sorted,
            replayed_spill: replay.is_spill(),
            first_pass_time,
            ..Default::default()
        };
        return Ok(RunProducer::Sorted(Box::new(SortedFallback {
            runs: SortedRuns::new(stream),
            value_type,
            _replay: replay,
            report,
        })));
    }

    let pending = accumulator.into_runs(&table)?;
    let report = BitmapBuildReport {
        strategy: BuildStrategy::Direct,
        rows,
        distinct_keys: pending.len(),
        replayed_spill: false,
        first_pass_time,
    };
    Ok(RunProducer::Direct(Box::new(DirectRuns {
        pending,
        value_type,
        report,
    })))
}

/// The sorted fallback's runs, with what keeps its input alive and what it
/// reports.
pub(super) struct SortedFallback<'a> {
    runs: SortedRuns,
    value_type: DataType,
    /// Keeps a spilled input's file alive while the runs read it.
    _replay: Replay<'a>,
    report: BitmapBuildReport,
}

/// Runs read off a value-sorted stream: consecutive equal values form one
/// run, across batch boundaries, nulls wherever the sort put them.
pub(super) struct SortedRuns {
    stream: SendableRecordBatchStream,
    /// The batch being read, cut into its runs once, with the next run of
    /// it to take.
    batch: Option<PartitionedBatch>,
    /// The run being accumulated.
    current: Option<(ScalarValue, RowAddrTreeMap)>,
}

/// A batch with its two columns and its ranges of equal values, all taken
/// once when the batch arrives: a batch of many runs is resumed once per
/// run, and none of that is redone then.
struct PartitionedBatch {
    values: ArrayRef,
    row_ids: UInt64Array,
    ranges: Vec<Range<usize>>,
    next_range: usize,
}

impl PartitionedBatch {
    fn new(batch: &RecordBatch) -> Result<Self> {
        let (values, row_ids) = value_and_row_id_columns(batch)?;
        let values = values.slice(0, values.len());
        let ranges = partition(std::slice::from_ref(&values))?.ranges();
        Ok(Self {
            values,
            row_ids: row_ids.clone(),
            ranges,
            next_range: 0,
        })
    }

    fn is_spent(&self) -> bool {
        self.next_range == self.ranges.len()
    }
}

impl SortedRuns {
    pub(super) fn new(stream: SendableRecordBatchStream) -> Self {
        Self {
            stream,
            batch: None,
            current: None,
        }
    }

    /// The next complete run, or `None` once the stream is spent.
    pub(super) async fn next_run(&mut self) -> Result<Option<(ScalarValue, RowAddrTreeMap)>> {
        let Self {
            stream,
            batch,
            current,
        } = self;
        loop {
            if batch.as_ref().is_none_or(PartitionedBatch::is_spent) {
                *batch = match stream.try_next().await? {
                    Some(next) => Some(PartitionedBatch::new(&next)?),
                    None => return Ok(current.take()),
                };
            }
            let pending = batch.as_mut().expect_ok()?;
            while pending.next_range < pending.ranges.len() {
                let range = pending.ranges[pending.next_range].clone();
                pending.next_range += 1;
                let key = ScalarValue::try_from_array(pending.values.as_ref(), range.start)?;
                let rows = &pending.row_ids.values()[range];
                // A run of one row (every row of an all-distinct column) is a
                // plain insert; the append path pays off for a run of
                // several rows.
                let append = |row_set: &mut RowAddrTreeMap| match rows {
                    [row] => {
                        row_set.insert(*row);
                    }
                    _ => {
                        row_set.insert_sorted(rows);
                    }
                };
                match current {
                    Some((current_key, row_set)) if *current_key == key => append(row_set),
                    _ => {
                        let mut row_set = RowAddrTreeMap::new();
                        append(&mut row_set);
                        if let Some(run) = current.replace((key, row_set)) {
                            return Ok(Some(run));
                        }
                    }
                }
            }
        }
    }
}
