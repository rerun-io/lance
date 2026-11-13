// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! How a bitmap build turns an unsorted `(value, row_id)` input into row sets
//! in key order.
//!
//! ```text
//! pass 1  input ──► KeyTable (value → dense id) ──► one row set per admitted
//!                   id, each batch's rows for a key appended as one ascending
//!                   run (the scan is in address order); a key first seen once
//!                   max_direct_keys keys are admitted is only counted
//!                                   │
//!            every key admitted: done, emitted in key order
//!                                   │ some keys only counted:
//!                                   ▼
//!              keys sorted once; ids remapped to key order; passes planned
//!                                   │
//! pass 2  input ──► lookup id ──► admitted ids skipped; every other row id
//!                                 scattered to its key's cursor in one flat
//!                                 buffer (keys with many rows are appended
//!                                 straight into their row set instead)
//!                                   │
//!                 admitted row sets and buffer slices emitted in key order,
//!                 each slice becoming a row set by one sorted append
//! ```
//!
//! A type whose domain is small and fixed (Boolean, 8-bit integers) always
//! finishes in the first pass, whatever the limits: every possible key gets a
//! row set up front and nothing is spilled or reread
//! ([`BuildStrategy::Direct`]). Any other type finishes in the first pass
//! when every key it sees is admitted, which is the case for the first
//! [`BitmapBuildOptions::max_direct_keys`] distinct keys; keys first seen
//! after that are counted in the first pass and scattered in the second
//! ([`BuildStrategy::Partitioned`]). Nothing admitted is ever dropped. The
//! cap is a key count rather than a memory budget because a row set per key
//! costs more per row than the flat buffer once keys are many and their
//! rows few: an admitted key's rows arrive as short runs per batch, each an
//! append of its own, where the buffer takes a scatter per row and one
//! append per key.
//!
//! The flat buffer costs 8 bytes per row; when the input has more rows than
//! [`BitmapBuildOptions::partition_memory_limit_bytes`] allows, pass 2 is
//! split into several passes over consecutive key ranges, each re-reading the
//! input. The input is re-read either by reopening a [`TrainingDataSource`]
//! (a dataset scan, cheap and local) or, for a caller that only has a stream,
//! by replaying a spill of the batches pass 1 saw.
//!
//! If pass 1 finds more distinct keys than
//! [`BitmapBuildOptions::max_distinct_keys`], the key table is dropped and
//! the build falls back to a value-sorted read of the input -- the scanner's
//! sort for a source, a spilling DataFusion sort of the replay for a stream
//! -- and takes its runs from that ([`BuildStrategy::Sorted`]).

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

use super::keys::{KeyTable, SortedKeys, UNKNOWN_ID};
use crate::scalar::registry::{
    TrainingCriteria, TrainingDataSource, TrainingOrdering, VALUE_COLUMN_NAME, sort_stream_by_value,
};

/// Default for [`BitmapBuildOptions::max_direct_keys`]: 4096 keys.
///
/// Keys admitted to a row set of their own in the first pass. A row set holds
/// one bitmap per fragment its key has rows in, so its resident overhead
/// grows with keys × fragments on top of the rows; a few thousand keys keep
/// that in the tens of megabytes even on a dataset of hundreds of fragments,
/// and with that many keys or fewer each batch brings a key enough rows for
/// its append to beat a scatter. Measured on a column of 50,000 keys, a row
/// set per key ran 1.7-2x slower than counting and scattering at 1.7x the
/// memory, whatever the budget, so the cap stays small.
pub const DEFAULT_MAX_DIRECT_KEYS: usize = 1 << 12;

/// Default for [`BitmapBuildOptions::replay_memory_limit_bytes`]: 512 MiB.
///
/// How much of a single-stream input stays in memory for a reread before the
/// replay spills to disk; the same figure as the partition limit, so a stream
/// build holds at most twice it during a partition pass on top of the
/// admitted keys' row sets.
pub const DEFAULT_REPLAY_MEMORY_LIMIT_BYTES: usize = 512 * 1024 * 1024;

/// Default for [`BitmapBuildOptions::partition_memory_limit_bytes`]: 512 MiB,
/// or 64M rows per partition pass.
///
/// Each extra pass re-reads the input and looks every row's key up again,
/// so the limit is set high enough that a build of a few hundred million
/// rows needs a handful of passes.
pub const DEFAULT_PARTITION_MEMORY_LIMIT_BYTES: usize = 512 * 1024 * 1024;

/// Default for [`BitmapBuildOptions::max_distinct_keys`]: 2^22 keys.
///
/// The key table holds every distinct value (bits or bytes, plus the map
/// slot), so a near-unique column of many millions of rows would cost
/// hundreds of megabytes before any row set exists; past this many keys the
/// value-sorted fallback, which spills, is the safer build. A bitmap index on
/// such a column is also a poor fit regardless of how it is built.
pub const DEFAULT_MAX_DISTINCT_KEYS: usize = 1 << 22;

/// Ids are `u32`, with the lowest value taken by null and the highest kept
/// free, so a table can never hold more keys than this whatever
/// `max_distinct_keys` says; past it the build sorts.
const MAX_TABLE_KEYS: usize = u32::MAX as usize - 2;

/// Keys with at least this share of a partition pass' row budget are
/// appended straight into their row set during the pass instead of being
/// scattered: they would take a large slice of the flat buffer for rows
/// that arrive in long ascending runs anyway.
const DIRECT_KEY_BUDGET_SHARE: usize = 16;

/// ...but never a key with fewer rows than this: below it a dedicated
/// accumulator costs more than its slice of the buffer, and with a tiny
/// budget (tests, experiments) every key would otherwise go direct and the
/// budget would partition nothing.
const DIRECT_KEY_MIN_ROWS: u64 = 1 << 12;

/// Tunables for a bitmap build.
///
/// The defaults are read from the environment once per process when set,
/// which lets an experiment change them without a code change, but every
/// field can also be set directly, which is the supported way to configure a
/// build:
///
/// | field                           | environment override                        |
/// |---------------------------------|---------------------------------------------|
/// | `max_direct_keys`               | `LANCE_BITMAP_MAX_DIRECT_KEYS`              |
/// | `replay_memory_limit_bytes`     | `LANCE_BITMAP_REPLAY_MEMORY_LIMIT_BYTES`    |
/// | `partition_memory_limit_bytes`  | `LANCE_BITMAP_PARTITION_MEMORY_LIMIT_BYTES` |
/// | `max_distinct_keys`             | `LANCE_BITMAP_MAX_DISTINCT_KEYS`            |
///
/// `sort_mem_pool_bytes` has no override of its own: left `None`, the
/// fallback sort runs in the pool every Lance plan uses, which
/// `LANCE_MEM_POOL_SIZE` sets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitmapBuildOptions {
    /// Distinct non-null keys the first pass admits to a row set of their
    /// own, in first-seen order; a key seen after that many is only counted,
    /// for the second pass to scatter. A type with a small fixed domain
    /// (Boolean, 8-bit integers) admits every key whatever this says. Capped
    /// by `max_distinct_keys`.
    pub max_direct_keys: usize,
    /// Bytes of a single-stream input kept in memory for a reread before the
    /// replay spills to disk. A reopenable source is reread instead and never
    /// spills.
    pub replay_memory_limit_bytes: usize,
    /// Bytes of row ids (8 per row) one partition pass may hold; an input
    /// with more rows is partitioned in several passes over key ranges.
    pub partition_memory_limit_bytes: usize,
    /// Distinct non-null keys above which the build sorts its input instead
    /// of partitioning it.
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
                partition_memory_limit_bytes: env_override(
                    "LANCE_BITMAP_PARTITION_MEMORY_LIMIT_BYTES",
                    DEFAULT_PARTITION_MEMORY_LIMIT_BYTES,
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
    /// Rows one partition pass may scatter.
    fn partition_rows(&self) -> usize {
        (self.partition_memory_limit_bytes / std::mem::size_of::<u64>()).max(1)
    }

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
    /// [`BitmapBuildOptions::max_direct_keys`] non-null keys.
    #[default]
    Direct,
    /// Some keys were only counted in the first pass; their rows were
    /// partitioned in one or more further passes, the admitted keys' row
    /// sets kept from the first.
    Partitioned,
    /// Too many distinct keys for a key table: the input was value-sorted.
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
    /// Keys whose row sets were built in the first pass (all of them on the
    /// direct path).
    pub admitted_keys: usize,
    /// Rows of keys that were only counted in the first pass and scattered
    /// in a partition pass.
    pub scattered_rows: u64,
    /// Partition passes over the input after the first.
    pub partition_passes: usize,
    /// Keys appended straight into their row set during a partition pass
    /// instead of being scattered.
    pub direct_keys: usize,
    /// Whether the second and later passes replayed a spill of the input
    /// rather than reopening it.
    pub replayed_spill: bool,
    /// Time in the first pass, including the spill tee: the whole build of a
    /// [`BuildStrategy::Direct`] one, the counting pass of a partitioned one.
    pub first_pass_time: Duration,
    /// Time reading the input in the partition passes, which includes the
    /// row sets of the directly appended keys; a scattered key's row set is
    /// built when it is emitted, outside this.
    pub partition_time: Duration,
}

/// The input of a build: a source the build can reopen, or a single stream.
pub enum BitmapBuildInput<'a> {
    /// Reopened for every pass, with the criteria each pass wants.
    Source(&'a dyn TrainingDataSource),
    /// Read once; later passes replay a spill of it.
    Stream(SendableRecordBatchStream),
}

/// Where the passes after the first get their batches.
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

    async fn open(&self) -> Result<SendableRecordBatchStream> {
        match self {
            Self::Source(source) => {
                source
                    .open(&TrainingCriteria::new(TrainingOrdering::None).with_row_id())
                    .await
            }
            Self::Spill { receiver, .. } => Ok(receiver.read()),
        }
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
    Partitioned(Box<PartitionedRuns<'a>>),
    Sorted(Box<SortedFallback<'a>>),
}

impl RunProducer<'_> {
    pub(super) async fn next_run(&mut self) -> Result<Option<(ScalarValue, RowAddrTreeMap)>> {
        match self {
            Self::Direct(runs) => Ok(runs.pending.pop_front()),
            Self::Partitioned(runs) => runs.next_run().await,
            Self::Sorted(fallback) => fallback.runs.next_run().await,
        }
    }

    pub(super) fn report(&self) -> BitmapBuildReport {
        match self {
            Self::Direct(runs) => runs.report.clone(),
            Self::Partitioned(runs) => runs.report.clone(),
            Self::Sorted(fallback) => fallback.report.clone(),
        }
    }

    pub(super) fn value_type(&self) -> &DataType {
        match self {
            Self::Direct(runs) => &runs.value_type,
            Self::Partitioned(runs) => &runs.value_type,
            Self::Sorted(fallback) => &fallback.value_type,
        }
    }
}

/// The row sets of the first pass: one per admitted dense id, with each
/// batch's rows for a key collected into a run and appended once.
///
/// Ids are handed out in first-seen order, so the admitted ids are exactly
/// `0..=admitted_max`: null (id 0) is admitted from the start, and the first
/// `max_direct_keys` non-null keys after it.
struct DirectAccumulator {
    row_sets: Vec<RowAddrTreeMap>,
    /// This batch's rows per id, ascending in scan order.
    runs: Vec<Vec<u64>>,
    /// Ids with rows in this batch: the only runs the flush walks, so a
    /// batch costs its rows, not the whole domain.
    touched: Vec<u32>,
    admitted_max: u32,
    /// Rows pushed into admitted row sets.
    admitted_rows: u64,
}

impl DirectAccumulator {
    /// Null admitted, everything else subject to the cap.
    fn new() -> Self {
        Self::with_admitted(1)
    }

    /// Every id of a fixed domain admitted up front: such a table hands
    /// out ids in domain order, not first-seen order, and never grows.
    fn with_domain(num_ids: usize) -> Self {
        Self::with_admitted(num_ids)
    }

    fn with_admitted(num_ids: usize) -> Self {
        Self {
            row_sets: (0..num_ids).map(|_| RowAddrTreeMap::new()).collect(),
            runs: (0..num_ids).map(|_| Vec::new()).collect(),
            touched: Vec::new(),
            admitted_max: num_ids as u32 - 1,
            admitted_rows: 0,
        }
    }

    /// Push a batch's rows for admitted keys, admitting keys first seen in
    /// it while there are fewer than `cap` non-null ones; the rows of every
    /// other key are only counted in `counts`, by id.
    fn push_batch(&mut self, ids: &[u32], row_ids: &[u64], cap: usize, counts: &mut [u64]) {
        let mut pushed = 0u64;
        for (id, row_id) in ids.iter().zip(row_ids) {
            if *id > self.admitted_max {
                // A new key: ids are dense and first-seen, so until the cap
                // it is always the next one; past the cap it is only counted.
                if self.admitted_max as usize >= cap {
                    counts[*id as usize] += 1;
                    continue;
                }
                self.admitted_max += 1;
                self.row_sets.push(RowAddrTreeMap::new());
                self.runs.push(Vec::new());
            }
            let run = &mut self.runs[*id as usize];
            if run.is_empty() {
                self.touched.push(*id);
            }
            run.push(*row_id);
            pushed += 1;
        }
        for id in self.touched.drain(..) {
            let run = &mut self.runs[id as usize];
            self.row_sets[id as usize].insert_sorted(run);
            run.clear();
        }
        self.admitted_rows += pushed;
    }

    /// Admitted keys with at least one row.
    fn keys_with_rows(&self) -> usize {
        self.row_sets
            .iter()
            .filter(|set| !lance_select::RowSetOps::is_empty(*set))
            .count()
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

/// Read `input` once (pass 1), building every key's row set as long as the
/// keys fit the direct cap and counting the keys regardless, then either
/// finish, set up the partition passes, or fall back to a sorted read.
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

    // A fixed domain admits every key and so is never reread: its input
    // need not be kept, and the caps only apply to a table that grows with
    // the values it sees.
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

    let mut accumulator = if table.has_fixed_domain() {
        DirectAccumulator::with_domain(table.num_ids())
    } else {
        DirectAccumulator::new()
    };
    let mut counts: Vec<u64> = Vec::new();
    let mut ids = Vec::new();
    let mut rows = 0u64;
    let mut too_many_keys = false;

    while let Some(batch) = stream.try_next().await? {
        if let Some(replay) = &mut replay {
            replay.tee(&batch).await?;
        }
        // Past the key cap only the replay matters: the rest of the stream
        // still has to reach the spill for the sorted fallback to see it.
        if too_many_keys {
            continue;
        }
        let (values, row_ids) = value_and_row_id_columns(&batch)?;
        ids.clear();
        table.assign_ids(values, &mut ids)?;
        rows += row_ids.len() as u64;
        counts.resize(table.num_ids(), 0);
        accumulator.push_batch(&ids, row_ids.values(), direct_cap, &mut counts);
        if table.num_ids() - 1 > key_cap {
            too_many_keys = true;
            if !replay.as_ref().is_some_and(Replay::is_spill) {
                break;
            }
        }
    }
    if let Some(replay) = &mut replay {
        replay.finish().await?;
    }
    let first_pass_time = started.elapsed();

    if too_many_keys {
        let replay = replay.expect_ok()?;
        info!(
            "bitmap index: more than {} distinct keys; sorting the input instead of partitioning it",
            options.max_distinct_keys
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

    let scattered_rows = rows - accumulator.admitted_rows;
    if scattered_rows == 0 {
        let admitted_keys = accumulator.keys_with_rows();
        let pending = accumulator.into_runs(&table)?;
        let report = BitmapBuildReport {
            strategy: BuildStrategy::Direct,
            rows,
            distinct_keys: pending.len(),
            admitted_keys,
            first_pass_time,
            ..Default::default()
        };
        return Ok(RunProducer::Direct(Box::new(DirectRuns {
            pending,
            value_type,
            report,
        })));
    }

    let replay = replay.expect_ok()?;
    // Admitted keys have their row sets already and were never counted, so
    // only the rest goes into the passes.
    let admitted_max = accumulator.admitted_max;
    let admitted_keys = accumulator.keys_with_rows();
    let admitted = accumulator
        .row_sets
        .into_iter()
        .map(|set| (!lance_select::RowSetOps::is_empty(&set)).then_some(set))
        .collect::<Vec<_>>();
    let sorted = table_into_sorted(table, &counts)?;
    let mut id_by_position = vec![0u32; sorted.keys.len()];
    for (id, position) in sorted.remap.iter().enumerate() {
        id_by_position[*position as usize] = id as u32;
    }
    let distinct_keys = admitted_keys + sorted.counts.iter().filter(|c| **c > 0).count();
    let passes = plan_passes(&sorted.counts, options.partition_rows());
    let direct_keys = passes.iter().map(|pass| pass.direct_keys).sum();
    info!(
        "bitmap index: {rows} rows and {distinct_keys} keys, {admitted_keys} of them built in the first pass; partitioning {scattered_rows} rows in {} passes of at most {} rows each",
        passes.len(),
        options.partition_rows()
    );
    let report = BitmapBuildReport {
        strategy: BuildStrategy::Partitioned,
        rows,
        distinct_keys,
        admitted_keys,
        scattered_rows,
        partition_passes: passes.len(),
        direct_keys,
        replayed_spill: replay.is_spill(),
        first_pass_time,
        partition_time: Duration::ZERO,
    };
    Ok(RunProducer::Partitioned(Box::new(PartitionedRuns {
        keys: sorted.keys,
        counts: sorted.counts,
        remap: sorted.remap,
        id_by_position,
        admitted,
        admitted_max,
        table: sorted.table,
        passes: passes.into(),
        current: None,
        replay,
        ids: Vec::new(),
        value_type,
        report,
    })))
}

/// The key table after pass 1, with its keys and counts in key order.
struct SortedTable {
    keys: Vec<ScalarValue>,
    /// Rows per sorted position.
    counts: Vec<u64>,
    /// First-seen id to sorted position.
    remap: Vec<u32>,
    /// Kept for the lookups of pass 2.
    table: KeyTable,
}

fn table_into_sorted(table: KeyTable, counts_by_id: &[u64]) -> Result<SortedTable> {
    let SortedKeys { keys, remap } = table.sorted_keys()?;
    let mut counts = vec![0u64; keys.len()];
    for (id, count) in counts_by_id.iter().enumerate() {
        counts[remap[id] as usize] = *count;
    }
    Ok(SortedTable {
        keys,
        counts,
        remap,
        table,
    })
}

/// One partition pass: a range of sorted positions, which of them are
/// appended directly, and the flat buffer layout of the rest.
#[derive(Debug, PartialEq, Eq)]
struct PassPlan {
    /// Sorted positions `first..first + slots.len()`.
    first: usize,
    slots: Vec<Slot>,
    /// Rows the scatter buffer holds.
    scatter_rows: usize,
    direct_keys: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    /// A key with no rows in the new input; nothing to emit.
    Empty,
    /// Rows scattered to `offset..offset + count` of the buffer.
    Scatter { offset: usize },
    /// Rows appended into the key's own row set as they arrive; `ordinal`
    /// numbers the pass' direct slots, which is all the per-batch flush walks.
    Direct { ordinal: usize },
}

/// Cut the sorted key positions into passes whose scatter buffers hold at
/// most `partition_rows` rows each. Every key lands in exactly one pass.
fn plan_passes(counts: &[u64], partition_rows: usize) -> Vec<PassPlan> {
    let direct_min_rows =
        ((partition_rows / DIRECT_KEY_BUDGET_SHARE) as u64).max(DIRECT_KEY_MIN_ROWS);
    let mut passes = Vec::new();
    let mut current = PassPlan {
        first: 0,
        slots: Vec::new(),
        scatter_rows: 0,
        direct_keys: 0,
    };
    for (position, count) in counts.iter().enumerate() {
        let slot = if *count == 0 {
            Slot::Empty
        } else if *count >= direct_min_rows {
            let ordinal = current.direct_keys;
            current.direct_keys += 1;
            Slot::Direct { ordinal }
        } else {
            let count = *count as usize;
            if current.scatter_rows + count > partition_rows
                && (current.scatter_rows > 0 || current.direct_keys > 0)
            {
                passes.push(std::mem::replace(
                    &mut current,
                    PassPlan {
                        first: position,
                        slots: Vec::new(),
                        scatter_rows: 0,
                        direct_keys: 0,
                    },
                ));
            }
            let offset = current.scatter_rows;
            current.scatter_rows += count;
            Slot::Scatter { offset }
        };
        current.slots.push(slot);
    }
    if !current.slots.is_empty() {
        passes.push(current);
    }
    passes
}

/// Runs produced by the partition passes, one pass at a time: the next pass
/// runs when the previous one's keys have all been emitted, so one flat
/// buffer is live at a time, and a scattered key's row set is built from
/// the buffer only when it is emitted, so one such row set is live at a
/// time. The admitted keys' row sets from the first pass, and a pass'
/// directly appended ones, stay live until each is emitted.
pub(super) struct PartitionedRuns<'a> {
    keys: Vec<ScalarValue>,
    counts: Vec<u64>,
    remap: Vec<u32>,
    /// Sorted position to first-seen id.
    id_by_position: Vec<u32>,
    /// The first pass' row sets by id, taken as they are emitted; `None`
    /// for an admitted key with no rows and for every id past
    /// `admitted_max`.
    admitted: Vec<Option<RowAddrTreeMap>>,
    admitted_max: u32,
    table: KeyTable,
    passes: VecDeque<PassPlan>,
    /// The pass whose keys are being emitted.
    current: Option<FinishedPass>,
    replay: Replay<'a>,
    ids: Vec<u32>,
    value_type: DataType,
    report: BitmapBuildReport,
}

/// A partition pass after its read: the scattered rows, the directly built
/// row sets, and the next slot to emit.
struct FinishedPass {
    plan: PassPlan,
    buffer: Vec<u64>,
    direct: Vec<Option<RowAddrTreeMap>>,
    next_slot: usize,
}

impl PartitionedRuns<'_> {
    async fn next_run(&mut self) -> Result<Option<(ScalarValue, RowAddrTreeMap)>> {
        loop {
            if let Some(run) = self.emit_next()? {
                return Ok(Some(run));
            }
            let Some(pass) = self.passes.pop_front() else {
                return Ok(None);
            };
            let started = Instant::now();
            self.current = Some(self.run_pass(pass).await?);
            self.report.partition_time += started.elapsed();
        }
    }

    /// The next key of the current pass with its row set, or `None` when the
    /// pass is spent (and dropped, buffer included).
    fn emit_next(&mut self) -> Result<Option<(ScalarValue, RowAddrTreeMap)>> {
        let Some(current) = self.current.as_mut() else {
            return Ok(None);
        };
        while current.next_slot < current.plan.slots.len() {
            let slot = current.next_slot;
            current.next_slot += 1;
            let position = current.plan.first + slot;
            let row_set = match current.plan.slots[slot] {
                // An empty slot is an admitted key (its row set comes from
                // the first pass) or a key with no new rows.
                Slot::Empty => match self
                    .admitted
                    .get_mut(self.id_by_position[position] as usize)
                {
                    Some(admitted) => match admitted.take() {
                        Some(row_set) => row_set,
                        None => continue,
                    },
                    None => continue,
                },
                Slot::Direct { ordinal } => current.direct[ordinal].take().expect_ok()?,
                Slot::Scatter { offset } => {
                    let count = self.counts[position] as usize;
                    let mut row_set = RowAddrTreeMap::new();
                    row_set.insert_sorted(&current.buffer[offset..offset + count]);
                    row_set
                }
            };
            return Ok(Some((self.keys[position].clone(), row_set)));
        }
        self.current = None;
        Ok(None)
    }

    /// Re-read the input and scatter the pass' rows.
    async fn run_pass(&mut self, pass: PassPlan) -> Result<FinishedPass> {
        let mut buffer = vec![0u64; pass.scatter_rows];
        let mut cursors = pass
            .slots
            .iter()
            .map(|slot| match slot {
                Slot::Scatter { offset } => *offset,
                _ => 0,
            })
            .collect::<Vec<_>>();
        let mut direct: Vec<Option<RowAddrTreeMap>> = (0..pass.direct_keys)
            .map(|_| Some(RowAddrTreeMap::new()))
            .collect();
        let mut direct_runs: Vec<Vec<u64>> = (0..pass.direct_keys).map(|_| Vec::new()).collect();
        let last = pass.first + pass.slots.len();

        let mut stream = self.replay.open().await?;
        let mut ids = std::mem::take(&mut self.ids);
        while let Some(batch) = stream.try_next().await? {
            let (values, row_ids) = value_and_row_id_columns(&batch)?;
            ids.clear();
            self.table.lookup_ids(values, &mut ids)?;
            for (id, row_id) in ids.iter().zip(row_ids.values()) {
                if *id == UNKNOWN_ID {
                    return Err(Error::internal(
                        "bitmap index: the input changed between passes (unknown key)".to_string(),
                    ));
                }
                // Rows of admitted keys were taken in the first pass; a
                // reread that changed them is not detected, only the
                // scattered keys are checked against their counts below.
                if *id <= self.admitted_max {
                    continue;
                }
                let position = self.remap[*id as usize] as usize;
                if position < pass.first || position >= last {
                    continue;
                }
                let slot = position - pass.first;
                match pass.slots[slot] {
                    Slot::Scatter { .. } => {
                        let cursor = &mut cursors[slot];
                        let Some(cell) = buffer.get_mut(*cursor) else {
                            return Err(Error::internal(
                                "bitmap index: the input changed between passes (more rows)"
                                    .to_string(),
                            ));
                        };
                        *cell = *row_id;
                        *cursor += 1;
                    }
                    Slot::Direct { ordinal } => direct_runs[ordinal].push(*row_id),
                    Slot::Empty => {
                        return Err(Error::internal(
                            "bitmap index: the input changed between passes (new rows for a key)"
                                .to_string(),
                        ));
                    }
                }
            }
            // One flush per direct key per batch: the run is this batch's
            // rows for the key, ascending in scan order.
            for (run, row_set) in direct_runs.iter_mut().zip(direct.iter_mut()) {
                if let Some(row_set) = row_set
                    && !run.is_empty()
                {
                    row_set.insert_sorted(run);
                    run.clear();
                }
            }
        }
        self.ids = ids;

        // Every scatter slot must have been filled exactly as counted.
        for (slot, kind) in pass.slots.iter().enumerate() {
            if let Slot::Scatter { offset } = kind {
                let position = pass.first + slot;
                let count = self.counts[position] as usize;
                if cursors[slot] != offset + count {
                    return Err(Error::internal(format!(
                        "bitmap index: the input changed between passes ({} rows for key {} where {count} were counted)",
                        cursors[slot] - offset,
                        self.keys[position]
                    )));
                }
            }
        }
        Ok(FinishedPass {
            plan: pass,
            buffer,
            direct,
            next_slot: 0,
        })
    }
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
