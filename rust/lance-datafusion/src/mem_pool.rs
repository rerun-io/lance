// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Instrumentation for the DataFusion memory pools Lance builds.
//!
//! [`InstrumentedMemoryPool`] delegates to an inner pool and records occupancy
//! and `try_grow` rejections against a labelled [`MemoryPoolStats`]. Read the
//! current values with [`memory_pool_stats`].
//!
//! Lance has no metrics facility of its own, so the counters are plain atomics
//! and it is up to the embedder to poll [`memory_pool_stats`] and export them.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicI64, AtomicU64, Ordering::Relaxed},
    },
};

use datafusion::execution::memory_pool::{
    MemoryConsumer, MemoryLimit, MemoryPool, MemoryReservation,
};
use log::warn;

/// Counters for one memory pool, shared by every pool built with the same
/// label and capacity so the totals survive session-cache eviction.
#[derive(Debug)]
pub struct MemoryPoolStats {
    label: &'static str,
    capacity_bytes: u64,
    /// Mirrors the inner pool's `reserved()` without taking its lock. Signed so
    /// that an out-of-order grow/shrink pair cannot wrap.
    reserved_bytes: AtomicI64,
    peak_reserved_bytes: AtomicU64,
    grow_rejected: AtomicU64,
    /// The divisor in `FairSpillPool`'s per-consumer share.
    spillable_consumers: AtomicI64,
}

impl MemoryPoolStats {
    fn new(label: &'static str, capacity_bytes: u64) -> Self {
        Self {
            label,
            capacity_bytes,
            reserved_bytes: AtomicI64::new(0),
            peak_reserved_bytes: AtomicU64::new(0),
            grow_rejected: AtomicU64::new(0),
            spillable_consumers: AtomicI64::new(0),
        }
    }

    #[inline]
    fn record_grow(&self, additional: usize) {
        let reserved =
            self.reserved_bytes.fetch_add(additional as i64, Relaxed) + additional as i64;
        self.peak_reserved_bytes
            .fetch_max(reserved.max(0) as u64, Relaxed);
    }

    #[inline]
    fn record_shrink(&self, shrink: usize) {
        self.reserved_bytes.fetch_sub(shrink as i64, Relaxed);
    }

    #[inline]
    fn record_rejection(&self) {
        if self.grow_rejected.fetch_add(1, Relaxed) == 0 {
            warn!(
                "memory pool '{}' (capacity {} bytes) rejected a reservation; \
                 further rejections are counted, not logged",
                self.label, self.capacity_bytes
            );
        }
    }

    pub fn snapshot(&self) -> MemoryPoolSnapshot {
        MemoryPoolSnapshot {
            label: self.label,
            capacity_bytes: self.capacity_bytes,
            reserved_bytes: self.reserved_bytes.load(Relaxed).max(0) as u64,
            peak_reserved_bytes: self.peak_reserved_bytes.load(Relaxed),
            grow_rejected: self.grow_rejected.load(Relaxed),
            spillable_consumers: self.spillable_consumers.load(Relaxed).max(0) as u64,
        }
    }
}

/// A point-in-time read of one pool's counters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryPoolSnapshot {
    /// Which pool this is, from `MemoryPoolKind::label`.
    pub label: &'static str,
    /// The pool's `memory_limit()`, or `u64::MAX` if unbounded.
    pub capacity_bytes: u64,
    pub reserved_bytes: u64,
    pub peak_reserved_bytes: u64,
    /// Monotonic count of `try_grow` calls the pool refused.
    pub grow_rejected: u64,
    /// Currently registered spillable consumers.
    pub spillable_consumers: u64,
}

type StatsRegistry = Mutex<HashMap<(&'static str, u64), Arc<MemoryPoolStats>>>;

fn registry() -> &'static StatsRegistry {
    static REGISTRY: OnceLock<StatsRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The counters for `(label, capacity_bytes)`, created on first use.
pub fn pool_stats(label: &'static str, capacity_bytes: u64) -> Arc<MemoryPoolStats> {
    registry()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .entry((label, capacity_bytes))
        .or_insert_with(|| Arc::new(MemoryPoolStats::new(label, capacity_bytes)))
        .clone()
}

/// Snapshots every pool Lance has built so far in this process.
pub fn memory_pool_stats() -> Vec<MemoryPoolSnapshot> {
    registry()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .values()
        .map(|stats| stats.snapshot())
        .collect()
}

/// Wraps a [`MemoryPool`], recording occupancy and rejections.
#[derive(Debug)]
pub struct InstrumentedMemoryPool {
    inner: Arc<dyn MemoryPool>,
    stats: Arc<MemoryPoolStats>,
}

impl InstrumentedMemoryPool {
    pub fn new(label: &'static str, inner: Arc<dyn MemoryPool>) -> Self {
        let capacity_bytes = match inner.memory_limit() {
            MemoryLimit::Finite(limit) => limit as u64,
            MemoryLimit::Infinite | MemoryLimit::Unknown => u64::MAX,
        };
        Self {
            stats: pool_stats(label, capacity_bytes),
            inner,
        }
    }

    pub fn stats(&self) -> &Arc<MemoryPoolStats> {
        &self.stats
    }
}

impl MemoryPool for InstrumentedMemoryPool {
    fn register(&self, consumer: &MemoryConsumer) {
        self.inner.register(consumer);
        if consumer.can_spill() {
            self.stats.spillable_consumers.fetch_add(1, Relaxed);
        }
    }

    fn unregister(&self, consumer: &MemoryConsumer) {
        self.inner.unregister(consumer);
        if consumer.can_spill() {
            self.stats.spillable_consumers.fetch_sub(1, Relaxed);
        }
    }

    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        self.inner.grow(reservation, additional);
        self.stats.record_grow(additional);
    }

    fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
        self.inner.shrink(reservation, shrink);
        self.stats.record_shrink(shrink);
    }

    fn try_grow(
        &self,
        reservation: &MemoryReservation,
        additional: usize,
    ) -> datafusion_common::Result<()> {
        match self.inner.try_grow(reservation, additional) {
            Ok(()) => {
                self.stats.record_grow(additional);
                Ok(())
            }
            Err(e) => {
                self.stats.record_rejection();
                Err(e)
            }
        }
    }

    fn reserved(&self) -> usize {
        self.inner.reserved()
    }

    fn memory_limit(&self) -> MemoryLimit {
        self.inner.memory_limit()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::execution::memory_pool::FairSpillPool;

    fn instrumented(
        label: &'static str,
        size: usize,
    ) -> (Arc<dyn MemoryPool>, Arc<MemoryPoolStats>) {
        let pool: Arc<dyn MemoryPool> = Arc::new(InstrumentedMemoryPool::new(
            label,
            Arc::new(FairSpillPool::new(size)),
        ));
        (pool, pool_stats(label, size as u64))
    }

    #[test]
    fn test_tracks_reservations_and_rejections() {
        let (pool, stats) = instrumented("test_tracks", 1024);
        assert_eq!(stats.snapshot().capacity_bytes, 1024);

        let res = MemoryConsumer::new("c")
            .with_can_spill(true)
            .register(&pool);
        assert_eq!(stats.snapshot().spillable_consumers, 1);

        res.try_grow(600).unwrap();
        let snap = stats.snapshot();
        assert_eq!(snap.reserved_bytes, 600);
        assert_eq!(snap.reserved_bytes as usize, pool.reserved());
        assert_eq!(snap.peak_reserved_bytes, 600);
        assert_eq!(snap.grow_rejected, 0);

        // Over the fair share for the single spillable consumer.
        assert!(res.try_grow(600).is_err());
        let snap = stats.snapshot();
        assert_eq!(snap.grow_rejected, 1);
        assert_eq!(snap.reserved_bytes, 600);

        res.shrink(400);
        let snap = stats.snapshot();
        assert_eq!(snap.reserved_bytes, 200);
        assert_eq!(snap.reserved_bytes as usize, pool.reserved());
        // Peak is a high-water mark, not the current value.
        assert_eq!(snap.peak_reserved_bytes, 600);

        drop(res);
        assert_eq!(stats.snapshot().spillable_consumers, 0);
    }

    #[test]
    fn test_stats_are_shared_per_label_and_capacity() {
        let a = InstrumentedMemoryPool::new("test_shared", Arc::new(FairSpillPool::new(4096)));
        let b = InstrumentedMemoryPool::new("test_shared", Arc::new(FairSpillPool::new(4096)));
        let c = InstrumentedMemoryPool::new("test_shared", Arc::new(FairSpillPool::new(8192)));
        assert!(Arc::ptr_eq(a.stats(), b.stats()));
        assert!(!Arc::ptr_eq(a.stats(), c.stats()));

        assert!(
            memory_pool_stats()
                .iter()
                .any(|s| s.label == "test_shared" && s.capacity_bytes == 4096)
        );
    }

    #[test]
    fn test_concurrent_grow_shrink_balances() {
        let (pool, stats) = instrumented("test_concurrent", 1 << 30);
        let threads = (0..8)
            .map(|_| {
                let pool = pool.clone();
                std::thread::spawn(move || {
                    let res = MemoryConsumer::new("c").register(&pool);
                    for _ in 0..1000 {
                        res.try_grow(64).unwrap();
                        res.shrink(64);
                    }
                })
            })
            .collect::<Vec<_>>();
        for t in threads {
            t.join().unwrap();
        }
        let snap = stats.snapshot();
        assert_eq!(snap.reserved_bytes, 0);
        assert_eq!(pool.reserved(), 0);
        assert!(snap.peak_reserved_bytes > 0);
        assert_eq!(snap.grow_rejected, 0);
        assert_eq!(snap.spillable_consumers, 0);
    }
}
