// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `ReaderProjection::from_field_ids` on wide flat schemas.
//!
//! Every `read_range_tasks` / `read_all_tasks` / `take_all_tasks` call builds one
//! of these, and the callers already hold an `Arc<Schema>`.
//!
//! ```
//! cargo bench --bench reader_projection
//! ```

use std::collections::{BTreeMap, HashMap};
use std::hint::black_box;
use std::sync::Arc;

use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use lance_core::datatypes::Schema;
use lance_encoding::version::LanceFileVersion;
use lance_file::reader::ReaderProjection;

/// Schema widths to sweep. 4096 is the production cap on manifest columns.
const WIDTHS: [usize; 4] = [58, 250, 1000, 4096];

fn field_name(i: usize) -> String {
    format!("log_tick_{i:04}:rerun_MyPoints:colors:start")
}

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

fn bench_from_field_ids(c: &mut Criterion) {
    let mut group = c.benchmark_group("reader_projection");
    for width in WIDTHS {
        let full = Schema::try_from(&arrow_schema(width)).unwrap();
        let ids = full.field_ids();
        let map: BTreeMap<u32, u32> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| (*id as u32, i as u32))
            .collect();
        let narrow = Arc::new(full.project_by_ids(&[ids[width / 2]], false));
        let full = Arc::new(full);

        for (shape, schema) in [("narrow", &narrow), ("full", &full)] {
            group.bench_with_input(
                BenchmarkId::new(format!("{shape}/from_field_ids"), width),
                schema,
                |b, schema| {
                    b.iter(|| {
                        black_box(
                            ReaderProjection::from_field_ids(
                                LanceFileVersion::V2_1,
                                schema.as_ref(),
                                &map,
                            )
                            .unwrap(),
                        )
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("{shape}/from_field_ids_arc"), width),
                schema,
                |b, schema| {
                    b.iter(|| {
                        black_box(
                            ReaderProjection::from_field_ids_arc(
                                LanceFileVersion::V2_1,
                                (*schema).clone(),
                                &map,
                            )
                            .unwrap(),
                        )
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_from_field_ids);
criterion_main!(benches);
