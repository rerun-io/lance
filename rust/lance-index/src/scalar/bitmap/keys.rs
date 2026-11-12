// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Dense ids for the values of a bitmap build.
//!
//! A [`KeyTable`] maps every value of the training column to a small integer
//! id, one batch at a time, without building a `ScalarValue` per row: the
//! value array is matched on its type once per batch and the rows are read
//! through the typed array. Null is always [`NULL_ID`]; every other id is
//! handed out in first-seen order. When the build is done,
//! [`KeyTable::sorted_keys`] gives the keys in index
//! order plus a remap from first-seen ids to sorted ids, building one
//! `ScalarValue` per distinct key rather than per row.
//!
//! Hashing is the per-row cost that matters, so the maps use a small
//! multiplicative hasher rather than SipHash: the keys come from a column the
//! caller already trusts, not from an adversary.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;

use arrow::buffer::ScalarBuffer;
use arrow_array::cast::AsArray;
use arrow_array::types::{
    ArrowPrimitiveType, Date32Type, Date64Type, DurationMicrosecondType, DurationMillisecondType,
    DurationNanosecondType, DurationSecondType, Float32Type, Float64Type, Int8Type, Int16Type,
    Int32Type, Int64Type, Time32MillisecondType, Time32SecondType, Time64MicrosecondType,
    Time64NanosecondType, TimestampMicrosecondType, TimestampMillisecondType,
    TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::types::{GenericBinaryType, GenericStringType};
use arrow_array::{
    Array, ArrayRef, BooleanArray, FixedSizeBinaryArray, GenericByteArray, OffsetSizeTrait,
    PrimitiveArray, new_null_array,
};
use arrow_schema::{DataType, TimeUnit};
use datafusion_common::ScalarValue;
use lance_core::{Error, Result};

use crate::scalar::btree::OrderableScalarValue;

/// The id of the null key. Reserved whether or not the input has nulls, so
/// non-null ids start at 1.
pub(super) const NULL_ID: u32 = 0;

/// A multiplicative hasher for integer and byte-string keys; the FxHash
/// recipe, written here so no crate is added for it.
#[derive(Default)]
struct KeyHasher(u64);

impl KeyHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

impl Hasher for KeyHasher {
    /// The multiply only moves bits upward, and hashbrown picks the bucket
    /// from the low bits, so finish with splitmix64's finaliser, which mixes
    /// every bit into every other; a rotate alone (rustc-hash's choice) still
    /// leaves keys that differ only in their high bits -- integral floats,
    /// aligned strings differing in the last byte -- sharing probe chains.
    #[inline]
    fn finish(&self) -> u64 {
        let mut x = self.0;
        x ^= x >> 30;
        x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            let mut word = [0u8; 8];
            word.copy_from_slice(chunk);
            self.add(u64::from_le_bytes(word));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut word = [0u8; 8];
            word[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(word));
        }
    }

    #[inline]
    fn write_u64(&mut self, word: u64) {
        self.add(word);
    }

    #[inline]
    fn write_usize(&mut self, word: usize) {
        self.add(word as u64);
    }
}

type KeyMap<K> = HashMap<K, u32, BuildHasherDefault<KeyHasher>>;

/// A native value as the bits the map is keyed on.
///
/// Floats fold every NaN into one canonical NaN, both as the map key and as
/// the key that is written: `OrderableScalarValue` orders by `total_cmp` and
/// `ScalarValue` compares floats by their bits, so every NaN sign and
/// payload in the column would otherwise become a key of its own, of which
/// an `= NaN` query could match at most one. `-0.0` and `+0.0` are distinct
/// under `total_cmp` and stay distinct here.
trait KeyBits: Copy {
    /// The value with every NaN folded into one pattern.
    fn canonical(self) -> Self;

    /// The bits the map is keyed on, of the canonical value.
    fn key_bits(self) -> u64;
}

macro_rules! int_key_bits {
    ($($t:ty),*) => {$(
        impl KeyBits for $t {
            #[inline]
            fn canonical(self) -> Self {
                self
            }

            #[inline]
            fn key_bits(self) -> u64 {
                self as u64
            }
        }
    )*};
}
int_key_bits!(i8, i16, i32, i64, u8, u16, u32, u64);

impl KeyBits for f32 {
    #[inline]
    fn canonical(self) -> Self {
        if self.is_nan() { Self::NAN } else { self }
    }

    #[inline]
    fn key_bits(self) -> u64 {
        self.canonical().to_bits() as u64
    }
}

impl KeyBits for f64 {
    #[inline]
    fn canonical(self) -> Self {
        if self.is_nan() { Self::NAN } else { self }
    }

    #[inline]
    fn key_bits(self) -> u64 {
        self.canonical().to_bits()
    }
}

/// One value type's way of assigning ids and giving the keys back.
trait TypedKeys: Send {
    /// Append one id per row of `values` to `ids`, assigning ids to values
    /// seen for the first time.
    fn assign_ids(&mut self, values: &dyn Array, ids: &mut Vec<u32>) -> Result<()>;

    /// Number of distinct non-null keys assigned so far.
    fn num_keys(&self) -> usize;

    /// The non-null keys in id order (id 1 first), as an array.
    fn keys(&self) -> Result<ArrayRef>;

    /// Whether the value type fixes the ids up front (Boolean, 8-bit
    /// integers), so `num_keys` is the whole domain before any value is
    /// seen. A table that grows with the values it sees answers `false`.
    fn has_fixed_domain(&self) -> bool {
        false
    }
}

fn type_mismatch(expected: &DataType, values: &dyn Array) -> Error {
    Error::invalid_input(format!(
        "bitmap index: expected a {expected} value column, got {}",
        values.data_type()
    ))
}

/// Boolean needs no map: ids are fixed at 1 for `false` and 2 for `true`,
/// and both keys are always reported; a build drops the ones with no rows.
struct BooleanKeys;

impl TypedKeys for BooleanKeys {
    fn assign_ids(&mut self, values: &dyn Array, ids: &mut Vec<u32>) -> Result<()> {
        let values = values
            .as_boolean_opt()
            .ok_or_else(|| type_mismatch(&DataType::Boolean, values))?;
        ids.extend(values.iter().map(|value| match value {
            None => NULL_ID,
            Some(false) => 1,
            Some(true) => 2,
        }));
        Ok(())
    }

    fn num_keys(&self) -> usize {
        2
    }

    fn keys(&self) -> Result<ArrayRef> {
        Ok(Arc::new(BooleanArray::from(vec![false, true])))
    }

    fn has_fixed_domain(&self) -> bool {
        true
    }
}

/// Int8 and UInt8 need no map either: the id is the byte value plus one,
/// so the table always reports all 256 keys; a build drops the ones with no
/// rows.
struct SmallIntKeys {
    data_type: DataType,
}

impl TypedKeys for SmallIntKeys {
    fn assign_ids(&mut self, values: &dyn Array, ids: &mut Vec<u32>) -> Result<()> {
        // Int8 keys by their byte pattern, so -1 is id 256 and -128 id 129.
        if let Some(values) = values.as_primitive_opt::<Int8Type>() {
            ids.extend(
                values
                    .iter()
                    .map(|v| v.map_or(NULL_ID, |v| v as u8 as u32 + 1)),
            );
        } else if let Some(values) = values.as_primitive_opt::<UInt8Type>() {
            ids.extend(values.iter().map(|v| v.map_or(NULL_ID, |v| v as u32 + 1)));
        } else {
            return Err(type_mismatch(&self.data_type, values));
        }
        Ok(())
    }

    fn num_keys(&self) -> usize {
        256
    }

    fn keys(&self) -> Result<ArrayRef> {
        Ok(match self.data_type {
            DataType::Int8 => Arc::new(PrimitiveArray::<Int8Type>::from_iter_values(
                (0..=u8::MAX).map(|b| b as i8),
            )),
            _ => Arc::new(PrimitiveArray::<UInt8Type>::from_iter_values(0..=u8::MAX)),
        })
    }

    fn has_fixed_domain(&self) -> bool {
        true
    }
}

struct PrimitiveKeys<T: ArrowPrimitiveType> {
    data_type: DataType,
    map: KeyMap<u64>,
    /// Keys in id order (id 1 at index 0), to rebuild the array at the end.
    values: Vec<T::Native>,
}

impl<T: ArrowPrimitiveType> PrimitiveKeys<T> {
    fn new(data_type: DataType) -> Self {
        Self {
            data_type,
            map: KeyMap::default(),
            values: Vec::new(),
        }
    }
}

impl<T> TypedKeys for PrimitiveKeys<T>
where
    T: ArrowPrimitiveType,
    T::Native: KeyBits,
{
    fn assign_ids(&mut self, values: &dyn Array, ids: &mut Vec<u32>) -> Result<()> {
        let values = values
            .as_primitive_opt::<T>()
            .ok_or_else(|| type_mismatch(&self.data_type, values))?;
        ids.reserve(values.len());
        for value in values {
            let Some(value) = value else {
                ids.push(NULL_ID);
                continue;
            };
            let value = value.canonical();
            let next_id = self.values.len() as u32 + 1;
            let id = *self.map.entry(value.key_bits()).or_insert_with(|| {
                self.values.push(value);
                next_id
            });
            ids.push(id);
        }
        Ok(())
    }

    fn num_keys(&self) -> usize {
        self.values.len()
    }

    fn keys(&self) -> Result<ArrayRef> {
        let array = PrimitiveArray::<T>::new(ScalarBuffer::from(self.values.clone()), None);
        Ok(Arc::new(array.with_data_type(self.data_type.clone())))
    }
}

/// Utf8, LargeUtf8, Binary, LargeBinary and FixedSizeBinary keys, looked up
/// by byte slice so a row allocates only when its value is seen for the
/// first time.
struct BytesKeys<O: OffsetSizeTrait> {
    data_type: DataType,
    map: KeyMap<Box<[u8]>>,
    values: Vec<Box<[u8]>>,
    _offset: std::marker::PhantomData<O>,
}

impl<O: OffsetSizeTrait> BytesKeys<O> {
    fn new(data_type: DataType) -> Self {
        Self {
            data_type,
            map: KeyMap::default(),
            values: Vec::new(),
            _offset: std::marker::PhantomData,
        }
    }

    fn assign_bytes<'a>(
        &mut self,
        rows: impl Iterator<Item = Option<&'a [u8]>>,
        ids: &mut Vec<u32>,
    ) {
        for value in rows {
            let Some(value) = value else {
                ids.push(NULL_ID);
                continue;
            };
            if let Some(id) = self.map.get(value) {
                ids.push(*id);
                continue;
            }
            let id = self.values.len() as u32 + 1;
            let owned: Box<[u8]> = value.into();
            self.values.push(owned.clone());
            self.map.insert(owned, id);
            ids.push(id);
        }
    }
}

/// The rows of a string or binary array of offset width `O`, or of a
/// fixed-width binary array, as byte slices.
fn byte_rows<'a, O: OffsetSizeTrait>(
    data_type: &DataType,
    values: &'a dyn Array,
) -> Result<Box<dyn Iterator<Item = Option<&'a [u8]>> + 'a>> {
    if let Some(strings) = values
        .as_any()
        .downcast_ref::<GenericByteArray<GenericStringType<O>>>()
    {
        return Ok(Box::new(strings.iter().map(|s| s.map(str::as_bytes))));
    }
    if let Some(binary) = values
        .as_any()
        .downcast_ref::<GenericByteArray<GenericBinaryType<O>>>()
    {
        return Ok(Box::new(binary.iter()));
    }
    if let (DataType::FixedSizeBinary(_), Some(fixed)) =
        (data_type, values.as_fixed_size_binary_opt())
    {
        return Ok(Box::new(fixed.iter()));
    }
    Err(type_mismatch(data_type, values))
}

impl<O: OffsetSizeTrait> TypedKeys for BytesKeys<O> {
    fn assign_ids(&mut self, values: &dyn Array, ids: &mut Vec<u32>) -> Result<()> {
        ids.reserve(values.len());
        let rows = byte_rows::<O>(&self.data_type, values)?;
        self.assign_bytes(rows, ids);
        Ok(())
    }

    fn num_keys(&self) -> usize {
        self.values.len()
    }

    fn keys(&self) -> Result<ArrayRef> {
        let values = self.values.iter().map(|v| v.as_ref());
        let array: ArrayRef = match &self.data_type {
            DataType::Utf8 | DataType::LargeUtf8 => {
                let strings = values
                    .map(|v| {
                        std::str::from_utf8(v).map_err(|e| {
                            Error::internal(format!("bitmap index: non-UTF-8 string key: {e}"))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Arc::new(GenericByteArray::<GenericStringType<O>>::from_iter_values(
                    strings,
                ))
            }
            DataType::FixedSizeBinary(width) => Arc::new(
                FixedSizeBinaryArray::try_from_sparse_iter_with_size(values.map(Some), *width)
                    .map_err(|e| {
                        Error::internal(format!(
                            "bitmap index: cannot rebuild the FixedSizeBinary({width}) keys: {e}"
                        ))
                    })?,
            ),
            _ => Arc::new(GenericByteArray::<GenericBinaryType<O>>::from_iter_values(
                values,
            )),
        };
        Ok(array)
    }
}

/// Any other type: a `ScalarValue` per row, as the builder did for every type
/// before the typed tables existed.
struct ScalarKeys {
    data_type: DataType,
    map: HashMap<ScalarValue, u32>,
    values: Vec<ScalarValue>,
}

impl TypedKeys for ScalarKeys {
    fn assign_ids(&mut self, values: &dyn Array, ids: &mut Vec<u32>) -> Result<()> {
        ids.reserve(values.len());
        for i in 0..values.len() {
            if values.is_null(i) {
                ids.push(NULL_ID);
                continue;
            }
            let value = ScalarValue::try_from_array(values, i)?;
            if let Some(id) = self.map.get(&value) {
                ids.push(*id);
                continue;
            }
            let id = self.values.len() as u32 + 1;
            self.values.push(value.clone());
            self.map.insert(value, id);
            ids.push(id);
        }
        Ok(())
    }

    fn num_keys(&self) -> usize {
        self.values.len()
    }

    fn keys(&self) -> Result<ArrayRef> {
        if self.values.is_empty() {
            return Ok(new_null_array(&self.data_type, 0));
        }
        Ok(ScalarValue::iter_to_array(self.values.iter().cloned())?)
    }
}

/// The values of a training column mapped to dense ids, in first-seen order.
pub(super) struct KeyTable {
    data_type: DataType,
    keys: Box<dyn TypedKeys>,
}

/// The keys of a finished [`KeyTable`] in index order, and how first-seen
/// ids map onto that order.
pub(super) struct SortedKeys {
    /// `keys[0]` is the typed null, then the non-null keys ascending.
    pub(super) keys: Vec<ScalarValue>,
    /// `remap[first_seen_id]` is the position of that key in `keys`;
    /// `remap[NULL_ID] == 0`.
    pub(super) remap: Vec<u32>,
}

macro_rules! primitive_keys {
    ($data_type:expr, $($pattern:pat => $t:ty),* $(,)?) => {
        match $data_type {
            $($pattern => Some(Box::new(PrimitiveKeys::<$t>::new($data_type.clone())) as Box<dyn TypedKeys>),)*
            _ => None,
        }
    };
}

impl KeyTable {
    pub(super) fn new(data_type: &DataType) -> Self {
        let primitive = primitive_keys!(
            data_type,
            DataType::Int16 => Int16Type,
            DataType::Int32 => Int32Type,
            DataType::Int64 => Int64Type,
            DataType::UInt16 => UInt16Type,
            DataType::UInt32 => UInt32Type,
            DataType::UInt64 => UInt64Type,
            DataType::Float32 => Float32Type,
            DataType::Float64 => Float64Type,
            DataType::Date32 => Date32Type,
            DataType::Date64 => Date64Type,
            DataType::Timestamp(TimeUnit::Second, _) => TimestampSecondType,
            DataType::Timestamp(TimeUnit::Millisecond, _) => TimestampMillisecondType,
            DataType::Timestamp(TimeUnit::Microsecond, _) => TimestampMicrosecondType,
            DataType::Timestamp(TimeUnit::Nanosecond, _) => TimestampNanosecondType,
            DataType::Time32(TimeUnit::Second) => Time32SecondType,
            DataType::Time32(TimeUnit::Millisecond) => Time32MillisecondType,
            DataType::Time64(TimeUnit::Microsecond) => Time64MicrosecondType,
            DataType::Time64(TimeUnit::Nanosecond) => Time64NanosecondType,
            DataType::Duration(TimeUnit::Second) => DurationSecondType,
            DataType::Duration(TimeUnit::Millisecond) => DurationMillisecondType,
            DataType::Duration(TimeUnit::Microsecond) => DurationMicrosecondType,
            DataType::Duration(TimeUnit::Nanosecond) => DurationNanosecondType,
        );
        let keys: Box<dyn TypedKeys> = match primitive {
            Some(keys) => keys,
            None => match data_type {
                DataType::Boolean => Box::new(BooleanKeys),
                DataType::Int8 | DataType::UInt8 => Box::new(SmallIntKeys {
                    data_type: data_type.clone(),
                }),
                DataType::Utf8 | DataType::Binary | DataType::FixedSizeBinary(_) => {
                    Box::new(BytesKeys::<i32>::new(data_type.clone()))
                }
                DataType::LargeUtf8 | DataType::LargeBinary => {
                    Box::new(BytesKeys::<i64>::new(data_type.clone()))
                }
                _ => Box::new(ScalarKeys {
                    data_type: data_type.clone(),
                    map: HashMap::new(),
                    values: Vec::new(),
                }),
            },
        };
        Self {
            data_type: data_type.clone(),
            keys,
        }
    }

    /// Append one id per row of `values` to `ids`. Ids are dense: in a table
    /// that grows with its values, a value seen for the first time gets
    /// `num_ids()` as its id; a fixed-domain table (see
    /// [`Self::has_fixed_domain`]) has every id from the start.
    pub(super) fn assign_ids(&mut self, values: &dyn Array, ids: &mut Vec<u32>) -> Result<()> {
        self.keys.assign_ids(values, ids)
    }

    /// See [`TypedKeys::has_fixed_domain`]; when true, [`Self::num_ids`] is
    /// the whole domain from the start.
    pub(super) fn has_fixed_domain(&self) -> bool {
        self.keys.has_fixed_domain()
    }

    /// Ids handed out so far, the null slot included.
    pub(super) fn num_ids(&self) -> usize {
        self.keys.num_keys() + 1
    }

    /// The keys in index order and the id remap.
    pub(super) fn sorted_keys(&self) -> Result<SortedKeys> {
        let keys = self.keys.keys()?;
        let null_key = ScalarValue::try_from_array(new_null_array(&self.data_type, 1).as_ref(), 0)?;
        // Sort through `OrderableScalarValue` so the order is exactly the one
        // `BitmapIndex::index_map` keeps, including for floats.
        let mut order = (0..keys.len())
            .map(|i| {
                Ok((
                    OrderableScalarValue(ScalarValue::try_from_array(keys.as_ref(), i)?),
                    i,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        order.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let mut remap = vec![0u32; keys.len() + 1];
        let mut sorted = Vec::with_capacity(keys.len() + 1);
        sorted.push(null_key);
        for (position, (key, first_seen)) in order.into_iter().enumerate() {
            remap[first_seen + 1] = position as u32 + 1;
            sorted.push(key.0);
        }
        Ok(SortedKeys {
            keys: sorted,
            remap,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{
        BinaryArray, Decimal128Array, Float64Array, Int8Array, Int64Array, LargeBinaryArray,
        LargeStringArray, StringArray, UInt8Array,
    };
    use rstest::rstest;

    fn ids_of(table: &mut KeyTable, values: &dyn Array) -> Vec<u32> {
        let mut ids = Vec::new();
        table.assign_ids(values, &mut ids).unwrap();
        ids
    }

    #[test]
    fn test_boolean_ids_are_fixed() {
        let mut table = KeyTable::new(&DataType::Boolean);
        let ids = ids_of(
            &mut table,
            &BooleanArray::from(vec![Some(true), None, Some(false), Some(true)]),
        );
        assert_eq!(ids, vec![2, NULL_ID, 1, 2]);
        assert_eq!(table.num_ids(), 3);
        let sorted = table.sorted_keys().unwrap();
        assert_eq!(
            sorted.keys,
            vec![
                ScalarValue::Boolean(None),
                ScalarValue::Boolean(Some(false)),
                ScalarValue::Boolean(Some(true))
            ]
        );
        assert_eq!(sorted.remap, vec![0, 1, 2]);
    }

    #[test]
    fn test_primitive_ids_in_first_seen_order_and_sorted_remap() {
        let mut table = KeyTable::new(&DataType::Int64);
        let ids = ids_of(
            &mut table,
            &Int64Array::from(vec![Some(30), Some(-5), None, Some(30), Some(7)]),
        );
        assert_eq!(ids, vec![1, 2, NULL_ID, 1, 3]);
        let ids = ids_of(&mut table, &Int64Array::from(vec![Some(7), Some(-5)]));
        assert_eq!(ids, vec![3, 2]);
        assert_eq!(table.num_ids(), 4);
        let sorted = table.sorted_keys().unwrap();
        assert_eq!(
            sorted.keys,
            vec![
                ScalarValue::Int64(None),
                ScalarValue::Int64(Some(-5)),
                ScalarValue::Int64(Some(7)),
                ScalarValue::Int64(Some(30)),
            ]
        );
        // first-seen 1 (30) -> 3, 2 (-5) -> 1, 3 (7) -> 2
        assert_eq!(sorted.remap, vec![0, 3, 1, 2]);
    }

    /// 8-bit integers have fixed ids (the byte plus one) and a fixed domain;
    /// the sorted keys are all 256 values in signed or unsigned order.
    #[test]
    fn test_small_int_keys_have_fixed_ids() {
        let mut table = KeyTable::new(&DataType::UInt8);
        assert!(table.has_fixed_domain());
        assert_eq!(table.num_ids(), 257);
        let ids = ids_of(
            &mut table,
            &UInt8Array::from(vec![Some(255), Some(0), None]),
        );
        assert_eq!(ids, vec![256, 1, NULL_ID]);
        let sorted = table.sorted_keys().unwrap();
        assert_eq!(sorted.keys.len(), 257);
        assert_eq!(sorted.keys[1], ScalarValue::UInt8(Some(0)));
        assert_eq!(sorted.keys[256], ScalarValue::UInt8(Some(255)));
        assert_eq!(sorted.remap[256], 256);

        let mut table = KeyTable::new(&DataType::Int8);
        let ids = ids_of(
            &mut table,
            &Int8Array::from(vec![Some(-1), Some(-128), Some(0), Some(127)]),
        );
        assert_eq!(ids, vec![256, 129, 1, 128]);
        let sorted = table.sorted_keys().unwrap();
        assert_eq!(sorted.keys[1], ScalarValue::Int8(Some(-128)));
        assert_eq!(sorted.keys[256], ScalarValue::Int8(Some(127)));
        // -128 (id 129) sorts first, 127 (id 128) last, 0 (id 1) in the middle.
        assert_eq!(sorted.remap[129], 1);
        assert_eq!(sorted.remap[128], 256);
        assert_eq!(sorted.remap[1], 129);
        assert_eq!(table.num_ids(), 257);
    }

    #[test]
    fn test_boolean_domain_is_fixed() {
        let boolean = KeyTable::new(&DataType::Boolean);
        assert!(boolean.has_fixed_domain());
        assert_eq!(boolean.num_ids(), 3);
        assert!(!KeyTable::new(&DataType::Int64).has_fixed_domain());
        assert!(!KeyTable::new(&DataType::Utf8).has_fixed_domain());
    }

    #[test]
    fn test_float_keys_merge_nans_and_keep_signed_zero() {
        let mut table = KeyTable::new(&DataType::Float64);
        let ids = ids_of(
            &mut table,
            &Float64Array::from(vec![
                // A payload NaN seen first must still be written as the canonical NaN.
                Some(f64::from_bits(f64::NAN.to_bits() | 1)),
                Some(f64::NAN),
                Some(-f64::NAN),
                Some(-0.0),
                Some(0.0),
                Some(1.5),
            ]),
        );
        assert_eq!(ids, vec![1, 1, 1, 2, 3, 4]);
        let sorted = table.sorted_keys().unwrap();
        let as_bits = sorted.keys[1..]
            .iter()
            .map(|k| match k {
                ScalarValue::Float64(Some(f)) => f.to_bits(),
                other => panic!("unexpected key {other:?}"),
            })
            .collect::<Vec<_>>();
        // total_cmp order: -0.0 < 0.0 < 1.5 < NaN
        assert_eq!(
            as_bits,
            vec![
                (-0.0f64).to_bits(),
                0.0f64.to_bits(),
                1.5f64.to_bits(),
                f64::NAN.to_bits()
            ]
        );
    }

    /// A byte-keyed array of `data_type` holding `rows` (`None` is null).
    fn byte_array(data_type: &DataType, rows: &[Option<&str>]) -> ArrayRef {
        match data_type {
            DataType::Utf8 => Arc::new(StringArray::from(rows.to_vec())),
            DataType::LargeUtf8 => Arc::new(LargeStringArray::from(rows.to_vec())),
            DataType::Binary => Arc::new(BinaryArray::from(
                rows.iter()
                    .map(|s| s.map(str::as_bytes))
                    .collect::<Vec<_>>(),
            )),
            DataType::LargeBinary => Arc::new(LargeBinaryArray::from(
                rows.iter()
                    .map(|s| s.map(str::as_bytes))
                    .collect::<Vec<_>>(),
            )),
            DataType::FixedSizeBinary(width) => Arc::new(
                FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                    rows.iter().map(|s| s.map(str::as_bytes)),
                    *width,
                )
                .unwrap(),
            ),
            other => panic!("not a byte-keyed type: {other}"),
        }
    }

    /// The `ScalarValue` of `data_type` holding `s`.
    fn byte_key(data_type: &DataType, s: Option<&str>) -> ScalarValue {
        match data_type {
            DataType::Utf8 => ScalarValue::Utf8(s.map(str::to_string)),
            DataType::LargeUtf8 => ScalarValue::LargeUtf8(s.map(str::to_string)),
            DataType::Binary => ScalarValue::Binary(s.map(|s| s.as_bytes().to_vec())),
            DataType::LargeBinary => ScalarValue::LargeBinary(s.map(|s| s.as_bytes().to_vec())),
            DataType::FixedSizeBinary(width) => {
                ScalarValue::FixedSizeBinary(*width, s.map(|s| s.as_bytes().to_vec()))
            }
            other => panic!("not a byte-keyed type: {other}"),
        }
    }

    /// Every byte-keyed type: ids in first-seen order with a null and a
    /// repeated value, keys back in byte order with the column's type. The
    /// variable-width types also take the empty value; the fixed-width one
    /// takes values of its width.
    #[rstest]
    #[case::utf8(DataType::Utf8, &[Some("pear"), None, Some("apple"), Some("pear"), Some("")])]
    #[case::large_utf8(DataType::LargeUtf8, &[Some("pear"), None, Some("apple"), Some("pear"), Some("")])]
    #[case::binary(DataType::Binary, &[Some("pear"), None, Some("apple"), Some("pear"), Some("")])]
    #[case::large_binary(DataType::LargeBinary, &[Some("pear"), None, Some("apple"), Some("pear"), Some("")])]
    #[case::fixed_size_binary(DataType::FixedSizeBinary(4), &[Some("pear"), None, Some("appl"), Some("pear"), Some("aaaa")])]
    fn test_byte_keys(#[case] data_type: DataType, #[case] rows: &[Option<&str>]) {
        let mut table = KeyTable::new(&data_type);
        let values = byte_array(&data_type, rows);
        let ids = ids_of(&mut table, values.as_ref());
        assert_eq!(ids, vec![1, NULL_ID, 2, 1, 3]);
        let keys = table.keys.keys().unwrap();
        assert_eq!(keys.data_type(), &data_type);
        assert_eq!(keys.len(), 3);
        let sorted = table.sorted_keys().unwrap();
        // Byte order: the shortest or lowest value first, "pear" last.
        let mut expected_values = [rows[2], rows[4]];
        expected_values.sort_unstable();
        assert_eq!(
            sorted.keys,
            vec![
                byte_key(&data_type, None),
                byte_key(&data_type, expected_values[0]),
                byte_key(&data_type, expected_values[1]),
                byte_key(&data_type, Some("pear")),
            ]
        );
        assert_eq!(sorted.remap, vec![0, 3, 2, 1]);
    }

    #[test]
    fn test_scalar_fallback_for_decimal() {
        let data_type = DataType::Decimal128(10, 2);
        let mut table = KeyTable::new(&data_type);
        let values = Decimal128Array::from(vec![Some(250), None, Some(100), Some(250)])
            .with_precision_and_scale(10, 2)
            .unwrap();
        let ids = ids_of(&mut table, &values);
        assert_eq!(ids, vec![1, NULL_ID, 2, 1]);
        let sorted = table.sorted_keys().unwrap();
        assert_eq!(
            sorted.keys,
            vec![
                ScalarValue::Decimal128(None, 10, 2),
                ScalarValue::Decimal128(Some(100), 10, 2),
                ScalarValue::Decimal128(Some(250), 10, 2),
            ]
        );
        assert_eq!(sorted.remap, vec![0, 2, 1]);
    }

    #[rstest]
    #[case::boolean(DataType::Boolean)]
    #[case::int64(DataType::Int64)]
    #[case::utf8(DataType::Utf8)]
    #[case::fixed_size_binary(DataType::FixedSizeBinary(32))]
    #[case::decimal(DataType::Decimal128(10, 2))]
    fn test_empty_input(#[case] data_type: DataType) {
        let mut table = KeyTable::new(&data_type);
        let empty = new_null_array(&data_type, 0);
        assert!(ids_of(&mut table, empty.as_ref()).is_empty());
        let sorted = table.sorted_keys().unwrap();
        // Boolean's two keys are fixed; every other table is just the null.
        let expected_keys = if data_type == DataType::Boolean { 3 } else { 1 };
        assert_eq!(sorted.keys.len(), expected_keys);
        assert!(sorted.keys[0].is_null());
        assert_eq!(sorted.remap.len(), expected_keys);
    }

    /// Keys come back with the column's exact data type, including the
    /// parameters `with_data_type` has to carry over.
    #[rstest]
    #[case::timestamp_tz(DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())))]
    #[case::date32(DataType::Date32)]
    #[case::duration(DataType::Duration(TimeUnit::Nanosecond))]
    #[case::large_binary(DataType::LargeBinary)]
    fn test_keys_keep_the_data_type(#[case] data_type: DataType) {
        let values =
            arrow::compute::cast(&Int64Array::from(vec![Some(3), None, Some(1)]), &data_type)
                .unwrap();
        let mut table = KeyTable::new(&data_type);
        assert_eq!(ids_of(&mut table, values.as_ref()), vec![1, NULL_ID, 2]);
        let sorted = table.sorted_keys().unwrap();
        assert_eq!(sorted.keys.len(), 3);
        for key in &sorted.keys {
            assert_eq!(key.data_type(), data_type, "{key:?}");
        }
        assert_eq!(sorted.remap, vec![0, 2, 1]);
    }

    /// Keys that differ only in their high bits must still spread over the
    /// map's buckets, which hashbrown takes from the low bits of the hash.
    #[test]
    fn test_hasher_spreads_high_bit_keys() {
        use std::hash::{BuildHasher, BuildHasherDefault};
        let hasher = BuildHasherDefault::<KeyHasher>::default();
        let low_bits = |hashes: Vec<u64>| {
            hashes
                .into_iter()
                .map(|h| h & 0xffff)
                .collect::<std::collections::HashSet<_>>()
                .len()
        };
        let integral_floats = (0..10_000u64)
            .map(|i| hasher.hash_one((i as f64).key_bits()))
            .collect();
        let aligned_strings = (0..10_000u32)
            .map(|i| hasher.hash_one(format!("key_{i:04}").into_bytes().into_boxed_slice()))
            .collect();
        // 10k keys into 65k buckets: well above 8k distinct unless they collide.
        assert!(low_bits(integral_floats) > 8_000);
        assert!(low_bits(aligned_strings) > 8_000);
    }

    #[test]
    fn test_type_mismatch_is_an_error() {
        let mut table = KeyTable::new(&DataType::Int64);
        let err = table
            .assign_ids(&StringArray::from(vec!["x"]), &mut Vec::new())
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err:?}");
        assert!(err.to_string().contains("expected a Int64"), "{err}");
    }
}
