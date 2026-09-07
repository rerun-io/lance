// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Closed-form oracle for the synthetic generator.
//!
//! Holds seven scalars and no collections, and never consults the payload it is
//! checking. One loop, over chain depth; the body is O(1). See "The oracle" in
//! `synthetic_row_ranges.md`.

use crate::generator::{
    BuildParams, FLOOR_ROWS, FRAGS_PER_LEVEL, Layout, is_deleted, layout, position, survivors,
};

pub struct Oracle {
    n: u64,
    b: u64,
    k: u64,
    lmax: u32,
    floor_count: u64,
    fid_max: u32,
    depth: u32,
}

impl Oracle {
    pub fn new(p: BuildParams) -> Self {
        let l = layout(p);
        Self {
            n: p.n,
            b: p.b,
            k: p.k,
            lmax: l.lmax,
            floor_count: l.floor_count,
            fid_max: l.fid_max,
            depth: l.depth(p.k),
        }
    }

    pub fn depth(&self) -> u32 {
        self.depth
    }

    pub fn fid_max(&self) -> u32 {
        self.fid_max
    }

    /// Total initial fragments.
    pub fn fragments(&self) -> u64 {
        FRAGS_PER_LEVEL * self.lmax as u64 + self.floor_count
    }

    /// Row -> initial row address, packed `frag << 32 | offset`.
    pub fn start_addr(&self, row: u64) -> u64 {
        debug_assert!(row < self.n);
        let floor_start = Layout::lstart(self.n, self.lmax);
        if row >= floor_start {
            // Floor level. This guard is mandatory: the geometric formula does not
            // degrade gracefully past here (at N=8192, row 8191 would yield L=13).
            let off = row - floor_start;
            let frag = FRAGS_PER_LEVEL * self.lmax as u64 + off / FLOOR_ROWS;
            return frag << 32 | (off % FLOOR_ROWS);
        }
        let d = self.n - row;
        // ceil_log2 on u64; on u32 this would be off by 32.
        let l = self.n.trailing_zeros() - (u64::BITS - (d - 1).leading_zeros());
        let ls = Layout::lstart(self.n, l);
        let fs = Layout::fsize(self.n, l);
        let frag = FRAGS_PER_LEVEL * l as u64 + (row - ls) / fs;
        frag << 32 | ((row - ls) % fs)
    }

    fn fragment_rows(&self, frag: u32) -> u64 {
        let level = frag as u64 / FRAGS_PER_LEVEL;
        if level >= self.lmax as u64 {
            FLOOR_ROWS
        } else {
            Layout::fsize(self.n, level as u32)
        }
    }

    /// Round (1-based) that first consumes `frag`, plus the surviving rows of the fresh
    /// fragments ahead of it in that round's group.
    fn entry(&self, frag: u32) -> (u32, u64) {
        if self.k == FRAGS_PER_LEVEL {
            let level = frag as u64 / FRAGS_PER_LEVEL;
            let round = if level >= self.lmax as u64 {
                1
            } else {
                (self.lmax as u64 - level + 1) as u32
            };
            let i = frag as u64 % FRAGS_PER_LEVEL;
            let fs = self.fragment_rows(frag);
            // A level's fragments are contiguous in row space, so the survivors of all
            // predecessors collapse to one expression instead of a sum.
            (round, survivors(i * fs, self.b))
        } else {
            // Descending id, two in round 1: ids F-1 and F-2 both enter at round 1,
            // and id x enters at round F-1-x otherwise.
            let f = self.fragments();
            let id = frag as u64;
            if id >= f - 2 {
                // Ascending within the group is [F-2, F-1].
                let before = if id == f - 1 {
                    survivors(self.fragment_rows((f - 2) as u32), self.b)
                } else {
                    0
                };
                (1, before)
            } else {
                ((f - 1 - id) as u32, 0)
            }
        }
    }

    /// Surviving rows contributed by the fresh fragments of `round` (excluding the blob).
    fn fresh_survivors(&self, round: u32) -> u64 {
        if self.k == FRAGS_PER_LEVEL {
            if round == 1 {
                survivors(self.floor_count * FLOOR_ROWS, self.b)
            } else {
                let level = self.lmax - (round - 1);
                survivors(FRAGS_PER_LEVEL * Layout::fsize(self.n, level), self.b)
            }
        } else {
            let f = self.fragments();
            if round == 1 {
                survivors(2 * FLOOR_ROWS, self.b)
            } else {
                survivors(self.fragment_rows((f - 1 - round as u64) as u32), self.b)
            }
        }
    }

    /// Expected result of the full chain walk. `None` means deleted.
    pub fn walk(&self, row: u64) -> Option<u64> {
        let addr = self.start_addr(row);
        let frag = (addr >> 32) as u32;
        let mut off = addr & 0xFFFF_FFFF;

        let (enter, before) = self.entry(frag);
        if is_deleted(off, self.b) {
            return None;
        }
        off = before + position(off, self.b);
        let mut new_id = self.fid_max + enter - 1;

        for round in (enter + 1)..=self.depth {
            if is_deleted(off, self.b) {
                return None;
            }
            off = self.fresh_survivors(round) + position(off, self.b);
            new_id = self.fid_max + round - 1;
        }
        Some((new_id as u64) << 32 | off)
    }

    /// Used by the probe builder's survival filter, which is not swept yet.
    #[allow(dead_code)]
    pub fn survives(&self, row: u64) -> bool {
        self.walk(row).is_some()
    }
}
