// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Per-fragment-open schema machinery, on wide flat schemas.
//!
//! `FileFragment::open_reader` materialises the whole data-file schema and then
//! intersects it by name with the projection.  Both are `O(schema width)` even
//! when the projection is a single column, which is the shape an index scan or a
//! `take` against a wide segment/partition manifest uses.
//!
//! ```
//! cargo bench --bench fragment_open
//! ```

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;

use arrow_array::{
    BooleanArray, Int64Array, RecordBatch, RecordBatchIterator, StringArray, UInt64Array,
};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use lance::dataset::fragment::FragReadConfig;
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance_core::datatypes::Schema;
use lance_encoding::version::LanceFileVersion;
use lance_table::format::DataFile;

/// Schema widths to sweep. 4096 is the production cap on manifest columns.
const WIDTHS: [usize; 4] = [58, 250, 1000, 4096];

const FRAGS: usize = 32;
const ROWS_PER_FRAG: usize = 128;

/// Field names are ~38 chars, like the real component-descriptor columns.
fn field_name(i: usize) -> String {
    format!("log_tick_{i:04}:rerun_MyPoints:colors:start")
}

/// ~4.7 KV pairs per field on average, mode 7 -- the measured segment-metadata shape.
fn field_metadata(nkeys: usize) -> HashMap<String, String> {
    const KEYS: [&str; 8] = [
        "rerun:index",
        "rerun:component",
        "rerun:component_type",
        "rerun:archetype",
        "rerun:index_marker",
        "rerun:index_kind",
        "rerun:component_descriptor",
        "rerun:kind",
    ];
    const VALS: [&str; 8] = [
        "log_tick",
        "rerun.components.Position3D",
        "rerun.components.Position3D",
        "rerun.archetypes.Points3D",
        "start",
        "temporal",
        "Points3D:positions",
        "control",
    ];
    (0..nkeys.min(8))
        .map(|k| (KEYS[k].to_string(), VALS[k].to_string()))
        .collect()
}

/// Flat, no nesting, no schema-level metadata.
fn arrow_schema(width: usize) -> ArrowSchema {
    let fields = (0..width)
        .map(|i| {
            let nkeys = match i % 12 {
                0 => 0,
                1 => 1,
                2 | 3 => 3,
                4 | 5 => 4,
                6 => 5,
                _ => 7,
            };
            let dt = match i % 4 {
                0 => DataType::UInt64,
                1 => DataType::Int64,
                2 => DataType::Utf8,
                _ => DataType::Boolean,
            };
            ArrowField::new(field_name(i), dt, true).with_metadata(field_metadata(nkeys))
        })
        .collect::<Vec<_>>();
    ArrowSchema::new(fields)
}

struct Case {
    full: Schema,
    data_file: DataFile,
    /// One-column projection carved out of `full`.
    projection: Schema,
}

fn case(width: usize) -> Case {
    let full = Schema::try_from(&arrow_schema(width)).unwrap();
    let ids = full.field_ids();
    let data_file = DataFile::new(
        "0.lance",
        ids.clone(),
        (0..ids.len() as i32).collect(),
        2,
        1,
        None,
        None,
    );
    let projection = full.project_by_ids(&[ids[width / 2]], false);
    Case {
        full,
        data_file,
        projection,
    }
}

fn bench_schema_per_file(c: &mut Criterion) {
    let mut group = c.benchmark_group("frag_open_schema");
    for width in WIDTHS {
        let case = case(width);
        group.bench_with_input(BenchmarkId::new("baseline", width), &case, |b, case| {
            b.iter(|| {
                let data_file_schema = case.data_file.schema(&case.full);
                black_box(
                    case.projection
                        .intersection_ignore_types(&data_file_schema)
                        .unwrap(),
                )
            })
        });
    }
    group.finish();
}

async fn build_dataset(width: usize, dir: &std::path::Path) -> Dataset {
    let arrow = Arc::new(arrow_schema(width));
    let uri = dir.join("ds").to_str().unwrap().to_string();
    let mut ds = None;
    for _ in 0..FRAGS {
        let cols: Vec<Arc<dyn arrow_array::Array>> = arrow
            .fields()
            .iter()
            .enumerate()
            .map(|(i, f)| match f.data_type() {
                DataType::UInt64 => Arc::new(UInt64Array::from(vec![i as u64; ROWS_PER_FRAG])) as _,
                DataType::Int64 => Arc::new(Int64Array::from(vec![i as i64; ROWS_PER_FRAG])) as _,
                DataType::Utf8 => Arc::new(StringArray::from(vec!["x"; ROWS_PER_FRAG])) as _,
                _ => Arc::new(BooleanArray::from(vec![true; ROWS_PER_FRAG])) as _,
            })
            .collect();
        let batch = RecordBatch::try_new(arrow.clone(), cols).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], arrow.clone());
        let params = WriteParams {
            mode: if ds.is_none() {
                WriteMode::Create
            } else {
                WriteMode::Append
            },
            max_rows_per_file: ROWS_PER_FRAG,
            data_storage_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        };
        ds = Some(Dataset::write(reader, &uri, Some(params)).await.unwrap());
    }
    ds.unwrap()
}

fn bench_open(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("frag_open_e2e");
    // Reported time is for `FRAGS` opens; divide to get per-open cost.
    group.throughput(criterion::Throughput::Elements(FRAGS as u64));
    for width in WIDTHS {
        let dir = tempfile::tempdir().unwrap();
        let ds = rt.block_on(build_dataset(width, dir.path()));
        let name = field_name(width / 2);
        let projection = ds.schema().project(&[name.as_str()]).unwrap();
        let frags = ds.get_fragments();
        assert_eq!(frags.len(), FRAGS);
        // Warm the file-metadata cache so we measure schema work, not I/O.
        rt.block_on(async {
            for f in frags.iter() {
                f.open(&projection, FragReadConfig::default().with_row_id(true))
                    .await
                    .unwrap();
            }
        });
        group.bench_with_input(
            BenchmarkId::from_parameter(width),
            &(frags, projection),
            |b, (frags, projection)| {
                b.to_async(&rt).iter(|| async {
                    for f in frags.iter() {
                        black_box(
                            f.open(projection, FragReadConfig::default().with_row_id(true))
                                .await
                                .unwrap(),
                        );
                    }
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_schema_per_file, bench_open);
criterion_main!(benches);
