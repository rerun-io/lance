// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! How a bitmap build turns value-sorted `(value, row_id)` input into runs
//! of `(key, row set)`.
//!
//! Each batch is cut into its runs of equal values by Arrow's `partition`
//! kernel, which compares adjacent values through the typed array (bits for
//! primitives, byte slices for strings and binaries, a comparator for the
//! rest) rather than through a `ScalarValue` per row; one `ScalarValue` is
//! built per run and batch, to tell a run that continues from the previous
//! batch from a new one. A run's row ids are appended to its row set in one
//! go: the scan emits them in address order within a value, and
//! [`RowAddrTreeMap::insert_sorted`] falls back to single inserts for any
//! that are not.

use std::ops::Range;

use arrow::compute::partition;
use arrow_array::{Array, ArrayRef, RecordBatch, UInt64Array};
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion_common::ScalarValue;
use futures::TryStreamExt;
use lance_core::{Error, ROW_ID, Result, error::LanceOptionExt};
use lance_select::RowAddrTreeMap;

use crate::scalar::registry::VALUE_COLUMN_NAME;

/// The value column and the `UInt64` row id column of a training batch.
fn value_and_row_id_columns(batch: &RecordBatch) -> Result<(ArrayRef, UInt64Array)> {
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
    Ok((values.clone(), row_ids.clone()))
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
        let ranges = partition(std::slice::from_ref(&values))?.ranges();
        Ok(Self {
            values,
            row_ids,
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
