// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

// Expanding a dictionary-encoded block into a plain Arrow array.
use arrow_array::{Int64Array, StringArray};
use arrow_schema::DataType;
use criterion::{Criterion, criterion_group, criterion_main};
use lance_encoding::buffer::LanceBuffer;
use lance_encoding::data::{BlockInfo, DataBlock, DictionaryDataBlock, FixedWidthDataBlock};

fn indices(n: usize, card: u32) -> FixedWidthDataBlock {
    let mut state = 0x9E3779B97F4A7C15_u64;
    let v: Vec<u32> = (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % card as u64) as u32
        })
        .collect();
    FixedWidthDataBlock {
        data: LanceBuffer::reinterpret_vec(v),
        bits_per_value: 32,
        num_values: n as u64,
        block_info: BlockInfo::new(),
    }
}

fn run(c: &mut Criterion, name: &str, dict: DataBlock, n: usize, card: u32, dt: DataType) {
    let idx = indices(n, card);
    c.bench_function(name, |b| {
        b.iter(|| {
            let block = DataBlock::Dictionary(DictionaryDataBlock {
                indices: idx.clone(),
                dictionary: Box::new(dict.clone()),
            });
            let arr = block.into_arrow(dt.clone(), true).unwrap();
            assert_eq!(arr.len(), n);
            arr
        })
    });
}

fn bench(c: &mut Criterion) {
    let low: Vec<String> = (0..8).map(|i| format!("category_value_{i}")).collect();
    let low_dict = DataBlock::from_array(StringArray::from(low));
    run(
        c,
        "dict_utf8_low_card_524k",
        low_dict,
        524_288,
        8,
        DataType::Utf8,
    );

    let mut vals: Vec<Option<String>> = (0..999)
        .map(|i| Some(format!("value-{i:06}-xxxxxxxx")))
        .collect();
    vals.push(None);
    let nul_dict = DataBlock::from_array(StringArray::from(vals));
    assert!(matches!(nul_dict, DataBlock::Nullable(_)));
    run(
        c,
        "dict_utf8_nullable_1k_card_262k",
        nul_dict,
        262_144,
        1000,
        DataType::Utf8,
    );

    let ints = DataBlock::from_array(Int64Array::from_iter_values(
        (0..128).map(|i| i * 1_000_003),
    ));
    run(
        c,
        "dict_i64_128_card_524k",
        ints,
        524_288,
        128,
        DataType::Int64,
    );
}

criterion_group!(benches, bench);
criterion_main!(benches);
